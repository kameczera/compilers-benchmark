#!/usr/bin/env python3
"""Comparação entre backends para BERT e GPT-2 com o protocolo K=5 do artigo.

Mesma disciplina da campanha ResNet: cada amostra de compilação vem de um
processo frio independente, com diretórios de cache privados e caches
persistentes desabilitados; a execução é cronometrada por iteração, com
sincronização de GPU. As janelas cronometradas seguem o que já está documentado
no README §8:

* TorchInductor e XLA usam o proxy ``primeira chamada - latência média``;
* TVM cronometra diretamente os passes + ``relax.build``.

Uso (processo pai; ele mesmo dispara os K filhos):

    python scripts/benchmark_transformers.py --backend inductor --model bert \\
        --batch 1 --seq-len 64 --repeats 5 --warmup 10 --iters 50 \\
        --output results/transformers/inductor_bert_1x64_k5.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IR_ROOT = ROOT / "ir_dumps" / "transformers"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------
# TorchInductor
# --------------------------------------------------------------------------
def _run_inductor(args) -> dict[str, Any]:
    import contextlib
    import io

    import torch

    from backends.common import stats_ms
    from backends.pytorch_backend import (
        _count_triton_from_text,
        _new_inductor_files,
        _pycodecache_files,
        use_scratch_debug_dir,
    )
    from models.transformers_common import build_torch_model, torch_inputs

    try:
        import torch._inductor.config as inductor_config

        inductor_config.force_disable_caches = True
    except Exception:
        pass
    use_scratch_debug_dir()
    torch._dynamo.reset()

    device = torch.device("cuda")
    model = build_torch_model(args.model, args.seed, args.seq_len).to(device)
    inputs = (torch_inputs(args.model, args.batch, args.seq_len, args.seed).to(device),)

    files_before = _pycodecache_files()
    scan_start = time.time()
    compiled = torch.compile(
        model, backend="inductor", fullgraph=True, dynamic=False,
        options={"trace.enabled": True, "triton.cudagraphs": False},
    )

    captured = io.StringIO()
    torch.cuda.synchronize()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(
        captured
    ), torch.inference_mode():
        start = time.perf_counter()
        output = compiled(*inputs)
        _ = output.contiguous()
        torch.cuda.synchronize()
        first_call_ms = (time.perf_counter() - start) * 1000.0

        for _ in range(args.warmup):
            compiled(*inputs)
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.iters):
            begin = time.perf_counter()
            compiled(*inputs)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - begin) * 1000.0)

    sources = []
    for source_path in _new_inductor_files(files_before, scan_start):
        try:
            sources.append((source_path, Path(source_path).read_text(encoding="utf-8")))
        except OSError:
            pass
    sources.sort(key=lambda item: (0 if "def call(" in item[1] else 1, item[0]))
    code = "\n\n".join(
        f"# ===== inductor generated file: {path} =====\n{text}"
        for path, text in sources
    )
    capture_method = "inductor_cache_files"
    if not code.strip():
        code = captured.getvalue()
        capture_method = "stdout_fallback"

    artifact_dir = IR_ROOT / "inductor" / (
        f"{int(time.time() * 1000)}_{args.model}_{args.batch}x{args.seq_len}"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    code_path = artifact_dir / "fused_inductor_output_code.py"
    code_path.write_text(code, encoding="utf-8")

    counts = _count_triton_from_text(code)
    exec_stats = stats_ms(samples)
    return {
        "compile_ms": max(0.0, first_call_ms - exec_stats["mean"]),
        "ttfb_ms": first_call_ms,
        "exec_stats": exec_stats,
        "ir": {
            "path": _relative(code_path),
            "capture_method": capture_method,
            "summary": {
                "triton_launches_total": counts.get("triton_launches_total"),
                "triton_kernels_unicos": counts.get("triton_kernels_unicos"),
                "extern_kernels_total": counts.get("extern_kernels_total"),
                "code_size_bytes": len(code.encode("utf-8")),
                "code_lines": code.count("\n") + 1,
            },
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
    }


# --------------------------------------------------------------------------
# XLA / JAX
# --------------------------------------------------------------------------
def _run_xla(args) -> dict[str, Any]:
    import jax

    from backends.common import stats_ms
    from models.transformers_common import build_flax_model

    forward, params, input_ids = build_flax_model(
        args.model, args.batch, args.seq_len, args.seed
    )
    jitted = jax.jit(forward)

    start = time.perf_counter()
    output = jitted(params, input_ids)
    jax.block_until_ready(output)
    first_call_ms = (time.perf_counter() - start) * 1000.0

    for _ in range(args.warmup):
        jax.block_until_ready(jitted(params, input_ids))
    samples = []
    for _ in range(args.iters):
        begin = time.perf_counter()
        jax.block_until_ready(jitted(params, input_ids))
        samples.append((time.perf_counter() - begin) * 1000.0)

    lowered = jitted.lower(params, input_ids)
    optimized = lowered.compile().as_text()
    artifact_dir = IR_ROOT / "xla" / (
        f"{int(time.time() * 1000)}_{args.model}_{args.batch}x{args.seq_len}"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    hlo_path = artifact_dir / "fused_jit_optimized.hlo"
    hlo_path.write_text(optimized, encoding="utf-8")

    fusion_total = optimized.count("fusion(")
    custom_total = optimized.count("custom-call(")
    exec_stats = stats_ms(samples)
    return {
        "compile_ms": max(0.0, first_call_ms - exec_stats["mean"]),
        "ttfb_ms": first_call_ms,
        "exec_stats": exec_stats,
        "ir": {
            "path": _relative(hlo_path),
            "capture_method": "jit_lower_compile",
            "summary": {
                "fusion_total": fusion_total,
                "custom_total": custom_total,
                "kernels_totais": fusion_total + custom_total,
                "code_size_bytes": len(optimized.encode("utf-8")),
                "code_lines": optimized.count("\n") + 1,
            },
        },
        "software": {
            "python": platform.python_version(),
            "jax": jax.__version__,
        },
    }


# --------------------------------------------------------------------------
# TVM Relax
# --------------------------------------------------------------------------
def _run_tvm(args) -> dict[str, Any]:
    import torch
    import tvm
    from tvm import relax
    from tvm.relax.frontend.torch import from_exported_program

    from backends.common import stats_ms
    from models.transformers_common import build_torch_model, torch_inputs

    model = build_torch_model(args.model, args.seed, args.seq_len)
    example = torch_inputs(args.model, args.batch, args.seq_len, args.seed)
    with torch.no_grad():
        exported = torch.export.export(model, (example,))
    # torch.export emits _unsafe_view for GPT-2 attention projections.  It has
    # reshape semantics here (the tensors are not subsequently mutated), while
    # TVM's ExportedProgram importer supports aten.reshape but not this alias.
    # Normalize the alias without changing the graph's values.
    for node in exported.graph_module.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten._unsafe_view.default:
            node.target = torch.ops.aten.reshape.default
    exported.graph_module.graph.lint()
    exported.graph_module.recompile()

    target = tvm.target.Target("cuda")
    device = tvm.cuda(0)

    # Janela cronometrada: importacao + passes + build, como no caminho ResNet.
    start = time.perf_counter()
    mod = from_exported_program(exported, keep_params_as_input=False)
    mod = relax.transform.LegalizeOps()(mod)
    with target:
        mod = tvm.ir.transform.Sequential(
            [
                relax.transform.FoldConstant(),
                relax.transform.FuseOps(),
                relax.transform.FuseTIR(),
                tvm.dlight.ApplyDefaultSchedule(
                    tvm.dlight.gpu.Matmul(),
                    tvm.dlight.gpu.GEMV(),
                    tvm.dlight.gpu.Reduction(),
                    tvm.dlight.gpu.GeneralReduction(),
                    tvm.dlight.gpu.Fallback(),
                ),
                relax.transform.DeadCodeElimination(),
                relax.transform.StaticPlanBlockMemory(),
            ]
        )(mod)
        executable = tvm.compile(mod, target=target)
    compile_ms = (time.perf_counter() - start) * 1000.0

    virtual_machine = relax.VirtualMachine(executable, device)
    tvm_input = tvm.nd.array(example.numpy().astype("int64"), device)

    start = time.perf_counter()
    output = virtual_machine["main"](tvm_input)
    device.sync()
    first_call_ms = (time.perf_counter() - start) * 1000.0
    del output

    for _ in range(args.warmup):
        virtual_machine["main"](tvm_input)
    device.sync()
    samples = []
    for _ in range(args.iters):
        begin = time.perf_counter()
        virtual_machine["main"](tvm_input)
        device.sync()
        samples.append((time.perf_counter() - begin) * 1000.0)

    script = mod.script(show_meta=False)
    artifact_dir = IR_ROOT / "tvm" / (
        f"{int(time.time() * 1000)}_{args.model}_{args.batch}x{args.seq_len}"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    script_path = artifact_dir / "fused_tvmscript.py"
    script_path.write_text(script, encoding="utf-8")

    return {
        "compile_ms": compile_ms,
        "ttfb_ms": compile_ms + first_call_ms,
        "exec_stats": stats_ms(samples),
        "ir": {
            "path": _relative(script_path),
            "capture_method": "tvmscript",
            "summary": {
                "tvm_call_tir_total": script.count("call_tir"),
                "tvm_cls_total": script.count("T.prim_func"),
                "code_size_bytes": len(script.encode("utf-8")),
                "code_lines": script.count("\n") + 1,
            },
        },
        "software": {
            "python": platform.python_version(),
            "tvm": tvm.__version__,
            "torch": torch.__version__,
        },
    }


RUNNERS = {"inductor": _run_inductor, "xla": _run_xla, "tvm": _run_tvm}


def _run_child(args) -> dict[str, Any]:
    os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")
    from models.transformers_common import describe

    payload = RUNNERS[args.backend](args)
    payload["model_info"] = describe(args.model)
    payload["cache_policy"] = {
        "persistent_compiler_cache_disabled": os.environ.get(
            "TORCHINDUCTOR_FORCE_DISABLE_CACHES"
        )
        == "1",
        "torchinductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
        "jax_cache_dir": os.environ.get("JAX_COMPILATION_CACHE_DIR"),
    }
    return payload


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    from backends.common import stats_ms

    compile_summary = stats_ms(
        [record["compile_ms"] for record in records], ci_method="student_t"
    )
    run_means = [record["exec_stats"]["mean"] for record in records]
    exec_between = stats_ms(run_means, ci_method="student_t")
    pooled = [
        sample for record in records for sample in record["exec_stats"]["samples"]
    ]
    exec_pooled = stats_ms(pooled)
    summaries = [json.dumps(r["ir"]["summary"], sort_keys=True) for r in records]
    return {
        "compile_ms": compile_summary["mean"],
        "compile_ms_std": compile_summary["std"],
        "compile_ms_ci95": compile_summary["ci95_halfwidth"],
        "compile_samples_ms": compile_summary["samples"],
        "ttfb_ms": sum(r["ttfb_ms"] for r in records) / len(records),
        "exec_ms": exec_pooled["mean"],
        "exec_ms_std": exec_pooled["std"],
        "exec_run_means_ms": exec_between["samples"],
        "exec_run_means_ms_std": exec_between["std"],
        "exec_ms_ci95": exec_between["ci95_halfwidth"],
        "exec_ms_ci95_method": "student_t_over_process_means",
        "exec_samples_n": len(pooled),
        "software": records[0]["software"],
        "model_info": records[0]["model_info"],
        "cache_policy": records[0]["cache_policy"],
        "ir_dump": {
            "summary": records[0]["ir"]["summary"],
            "capture_method": records[0]["ir"]["capture_method"],
            "replicate_paths": [r["ir"]["path"] for r in records],
            "replicate_summaries_identical": len(set(summaries)) == 1,
        },
    }


def _run_parent(args) -> dict[str, Any]:
    collected: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="cnnbench-transformers-") as scratch:
        for repeat in range(1, args.repeats + 1):
            for attempt in range(1, args.attempts + 1):
                attempt_root = Path(scratch) / f"repeat_{repeat:02d}" / f"a{attempt}"
                attempt_root.mkdir(parents=True, exist_ok=True)
                child_output = attempt_root / "result.json"
                env = os.environ.copy()
                env.update(
                    {
                        "TORCHINDUCTOR_FORCE_DISABLE_CACHES": "1",
                        "TORCHINDUCTOR_CACHE_DIR": str(attempt_root / "torchinductor"),
                        "TRITON_CACHE_DIR": str(attempt_root / "triton"),
                        "JAX_COMPILATION_CACHE_DIR": str(attempt_root / "jax"),
                    }
                )
                print(
                    f"[{args.backend} {args.model} {args.batch}x{args.seq_len}] "
                    f"repeat {repeat}/{args.repeats}, attempt {attempt}/{args.attempts}",
                    flush=True,
                )
                completed = subprocess.run(
                    [
                        sys.executable, str(Path(__file__).resolve()), "--child",
                        "--backend", args.backend, "--model", args.model,
                        "--batch", str(args.batch), "--seq-len", str(args.seq_len),
                        "--warmup", str(args.warmup), "--iters", str(args.iters),
                        "--seed", str(args.seed), "--output", str(child_output),
                    ],
                    cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, timeout=args.timeout, check=False,
                )
                if completed.returncode != 0 or not child_output.exists():
                    tail = (completed.stdout or "").strip().splitlines()[-6:]
                    failed.append(
                        {
                            "repeat": repeat,
                            "attempt": attempt,
                            "error": f"exit {completed.returncode}",
                            "tail": tail,
                        }
                    )
                    print("\n".join(tail), flush=True)
                    continue
                collected.append(json.loads(child_output.read_text(encoding="utf-8")))
                break
            else:
                raise RuntimeError(
                    f"repeat {repeat} falhou apos {args.attempts} tentativas"
                )

    return {
        "meta": {
            "backend": args.backend,
            "model": args.model,
            "input": {"batch": args.batch, "seq_len": args.seq_len},
            "compile_repeats": args.repeats,
            "warmup": args.warmup,
            "iters_per_repeat": args.iters,
            "seed": args.seed,
            "cold_process_isolation": True,
            "private_compiler_cache_per_process": True,
            "failed_attempts": failed,
            "host": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
        },
        "result": _aggregate(collected),
    }


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=tuple(RUNNERS), required=True)
    parser.add_argument("--model", choices=("bert", "gpt2"), required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = _run_child(args) if args.child else _run_parent(args)
    _write_json(args.output, payload)
    print(f"[OK] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
