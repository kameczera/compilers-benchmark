#!/usr/bin/env python3
"""Measure the historical BERT no-fold/fold-control experiment fairly.

This reproduces ``testes/pytorch/bert_script.py``: pretrained
``bert-base-uncased``, an all-ones attention mask, ``fullgraph=True``,
``dynamic=False``, and the same TorchInductor options.  The related historical
fold script is ``testes/pytorch/bert2.py``.

The standard Hugging Face BERT encoder is post-LayerNorm. Every encoder
LayerNorm output is also used by a residual path, so its affine parameters
cannot be absorbed into only the following Linear layers without changing the
model. The historical sibling-based heuristic finds zero adjacent pairs on
this model, as recorded in its original JSON. Consequently, ``fold`` below is
the same explicit zero-transformation control, compiled in a separate cold
process instead of inheriting the base model's compiler cache.

Each K>1 aggregate launches base and fold-control in distinct Python
processes, with private TorchInductor/Triton cache directories and persistent
compiler caches disabled before torch is imported.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
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

# `python scripts/benchmark_bert_fold.py` coloca scripts/ em sys.path[0], nao a
# raiz do repositorio: sem isto o processo pai importa `backends.*` apenas na
# agregacao final e perde toda a campanha ja medida.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _bert_model():
    import torch
    from torch import nn
    from transformers import BertModel

    class BertForBenchmark(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = BertModel.from_pretrained(
                "bert-base-uncased",
                local_files_only=True,
            )
            self.config_dict = self.model.config.to_dict()

        def forward(self, input_ids, attention_mask):
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            ).last_hidden_state

    return BertForBenchmark(), torch


def _fold_control(model):
    """Reproduce the historical heuristic's zero-match result safely."""
    folded = copy.deepcopy(model)
    candidates: list[str] = []

    def walk(module, prefix=""):
        import torch.nn as nn

        siblings = list(module.named_children())
        for index, (name, child) in enumerate(siblings):
            full = f"{prefix}.{name}" if prefix else name
            if (
                isinstance(child, nn.LayerNorm)
                and index + 1 < len(siblings)
                and isinstance(siblings[index + 1][1], nn.Linear)
            ):
                sibling_name = siblings[index + 1][0]
                candidates.append(f"{full} -> {prefix}.{sibling_name}")
            walk(child, full)

    walk(folded)
    if candidates:
        raise RuntimeError(
            "The historical heuristic now finds LayerNorm/Linear siblings; "
            "their residual legality must be audited before rewriting: "
            + "; ".join(candidates)
        )
    return folded, {
        "transformation": "identity_control",
        "folded_pairs": 0,
        "historical_heuristic_candidates": candidates,
        "historical_script": "testes/pytorch/bert2.py",
        "semantics_preserved": True,
        "reason": (
            "The historical sibling-based heuristic finds no LayerNorm "
            "followed by a Linear in the same parent. Hugging Face BertModel "
            "is also post-LayerNorm, so projection-only folding would leave "
            "an uncompensated residual consumer."
        ),
    }


def _compile_historical_protocol(
    model,
    inputs,
    artifact_dir: Path,
    tag: str,
    *,
    warmup: int,
    iters: int,
):
    """Compile with the exact torch.compile options from bert_script.py."""
    import torch
    from backends.common import stats_ms
    from backends.pytorch_backend import (
        _count_triton_from_text,
        _new_inductor_files,
        _pycodecache_files,
        use_scratch_debug_dir,
    )

    try:
        import torch._inductor.config as inductor_config

        inductor_config.force_disable_caches = True
    except Exception:
        pass
    use_scratch_debug_dir()
    torch._dynamo.reset()

    files_before = _pycodecache_files()
    scan_start = time.time()
    compiled = torch.compile(
        model.eval(),
        backend="inductor",
        fullgraph=True,
        dynamic=False,
        options={"trace.enabled": True, "triton.cudagraphs": False},
    )

    captured = io.StringIO()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(
        captured
    ), torch.inference_mode():
        start = time.perf_counter()
        output = compiled(*inputs)
        if hasattr(output, "contiguous"):
            _ = output.contiguous()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        first_call_ms = (time.perf_counter() - start) * 1000.0

    sources = []
    for source_path in _new_inductor_files(files_before, scan_start):
        try:
            sources.append(
                (
                    source_path,
                    Path(source_path).read_text(encoding="utf-8"),
                )
            )
        except OSError:
            pass
    sources.sort(
        key=lambda item: (0 if "def call(" in item[1] else 1, item[0])
    )
    code = "\n\n".join(
        f"# ===== inductor generated file: {path} =====\n{text}"
        for path, text in sources
    )
    if not code.strip():
        code = captured.getvalue()
        sources = [("stdout", code)]

    artifact_dir.mkdir(parents=True, exist_ok=True)
    code_path = artifact_dir / f"{tag}_inductor_output_code.py"
    code_path.write_text(code, encoding="utf-8")

    triton_runs: dict[str, int] = {}
    extern_calls: dict[str, int] = {}
    direct_aten_calls: dict[str, int] = {}
    for _, source in sources:
        count = _count_triton_from_text(source)
        for key, value in count["details"]["triton_runs"].items():
            triton_runs[key] = triton_runs.get(key, 0) + value
        for key, value in count["details"]["extern_calls"].items():
            extern_calls[key] = extern_calls.get(key, 0) + value
        for key, value in count["details"]["direct_aten_calls"].items():
            direct_aten_calls[key] = direct_aten_calls.get(key, 0) + value
    kernel_count = {
        "summary": {
            "triton_launches_total": sum(triton_runs.values()),
            "triton_kernels_unicos": len(triton_runs),
            "extern_kernels_total": sum(extern_calls.values()),
            "extern_funcs_unicas": len(extern_calls),
            "direct_aten_calls_total": sum(direct_aten_calls.values()),
            "code_size_bytes": len(code.encode("utf-8")),
            "code_lines": code.count("\n") + (1 if code else 0),
        }
    }

    with torch.inference_mode():
        for _ in range(max(1, warmup)):
            _ = compiled(*inputs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    samples = []
    with torch.inference_mode():
        for _ in range(iters):
            if torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                _ = compiled(*inputs)
                end_event.record()
                torch.cuda.synchronize()
                samples.append(start_event.elapsed_time(end_event))
            else:
                start = time.perf_counter()
                _ = compiled(*inputs)
                samples.append((time.perf_counter() - start) * 1000.0)
    exec_stats = stats_ms(samples)
    compile_ms = max(0.0, first_call_ms - exec_stats["mean"])
    # Caminho relativo a raiz do repositorio, como nos JSONs do ResNet: o
    # absoluto vazaria o diretorio do host (ou /work, dentro do conteiner).
    try:
        recorded_path = str(code_path.resolve().relative_to(ROOT))
    except ValueError:
        recorded_path = str(code_path)
    return (
        compile_ms,
        first_call_ms,
        exec_stats,
        recorded_path,
        kernel_count,
    )


def _run_child(args) -> dict[str, Any]:
    # These must be set before importing torch or TorchInductor.
    os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import torch
    import transformers

    torch.manual_seed(args.seed)
    model, torch = _bert_model()
    model.eval()

    fold_meta = {
        "transformation": "none",
        "folded_pairs": 0,
        "semantics_preserved": True,
    }
    if args.variant == "fold":
        model, fold_meta = _fold_control(model)

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_ids = torch.randint(
        0,
        model.model.config.vocab_size,
        (args.batch, args.seq_len),
        dtype=torch.int64,
        device=device,
    )
    attention_mask = torch.ones(
        (args.batch, args.seq_len),
        dtype=torch.long,
        device=device,
    )
    inputs = (input_ids, attention_mask)

    # Symmetric eager initialization for both independently launched variants.
    with torch.inference_mode():
        for _ in range(max(1, args.warmup)):
            _ = model(*inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()

    stamp = int(time.time() * 1000)
    artifact_dir = (
        ROOT
        / "ir_dumps"
        / "transformers"
        / "inductor"
        / f"{stamp}_bert_{args.batch}x{args.seq_len}_{args.variant}"
    )
    compile_ms, ttfb_ms, exec_stats, code_path, kernel_count = (
        _compile_historical_protocol(
            model,
            inputs,
            artifact_dir,
            args.variant,
            warmup=args.warmup,
            iters=args.iters,
        )
    )
    return {
        "model": "bert",
        "variant": args.variant,
        "input": {"batch": args.batch, "seq_len": args.seq_len},
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "protocol": {
            "source_script": (
                "/home/kamei/Documentos/projects/tcc/testes/pytorch/"
                "bert_script.py"
            ),
            "pretrained_model": "bert-base-uncased",
            "attention_mask": "all_ones",
            "torch_compile": {
                "backend": "inductor",
                "fullgraph": True,
                "dynamic": False,
                "options": {
                    "trace.enabled": True,
                    "triton.cudagraphs": False,
                },
            },
            "seed": args.seed,
            "warmup": args.warmup,
            "iters": args.iters,
            "torchinductor_force_disable_caches": (
                os.environ.get("TORCHINDUCTOR_FORCE_DISABLE_CACHES") == "1"
            ),
            "torchinductor_cache_dir": os.environ.get(
                "TORCHINDUCTOR_CACHE_DIR"
            ),
            "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
        },
        "fold": fold_meta,
        "timing": {
            "compile_ms": float(compile_ms),
            "ttfb_ms": float(ttfb_ms),
            "exec_ms": float(exec_stats["mean"]),
            "exec_ms_std": float(exec_stats["std"]),
            "exec_ms_ci95_within_process": float(
                exec_stats["ci95_halfwidth"]
            ),
            "exec_samples_ms": exec_stats["samples"],
        },
        "artifact": {
            "path": code_path,
            "kernel_count": kernel_count,
        },
    }


def _aggregate_variant(records: list[dict[str, Any]]) -> dict[str, Any]:
    from backends.common import stats_ms

    compile_summary = stats_ms(
        [record["timing"]["compile_ms"] for record in records],
        ci_method="student_t",
    )
    ttfb_summary = stats_ms(
        [record["timing"]["ttfb_ms"] for record in records],
        ci_method="student_t",
    )
    run_means = [record["timing"]["exec_ms"] for record in records]
    exec_between = stats_ms(run_means, ci_method="student_t")
    pooled_samples = [
        sample
        for record in records
        for sample in record["timing"]["exec_samples_ms"]
    ]
    exec_pooled = stats_ms(pooled_samples)
    return {
        "fold": records[0]["fold"],
        "software": records[0]["software"],
        "compile_ms": compile_summary["mean"],
        "compile_ms_std": compile_summary["std"],
        "compile_ms_ci95": compile_summary["ci95_halfwidth"],
        "compile_samples_ms": compile_summary["samples"],
        "ttfb_ms": ttfb_summary["mean"],
        "ttfb_ms_std": ttfb_summary["std"],
        "ttfb_ms_ci95": ttfb_summary["ci95_halfwidth"],
        "exec_ms": exec_pooled["mean"],
        "exec_ms_std": exec_pooled["std"],
        "exec_ms_ci95": exec_between["ci95_halfwidth"],
        "exec_ms_ci95_method": "student_t_over_process_means",
        "exec_run_means_ms": exec_between["samples"],
        "exec_samples_n": len(pooled_samples),
        "artifacts": [record["artifact"] for record in records],
    }


def _run_parent(args) -> dict[str, Any]:
    variants = ("base", "fold")
    collected: dict[str, list[dict[str, Any]]] = {
        variant: [] for variant in variants
    }
    failed_attempts: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="cnnbench-bert-fold-") as tmp:
        tmp_root = Path(tmp)
        for repeat in range(1, args.repeats + 1):
            for variant in variants:
                for attempt in range(1, args.attempts + 1):
                    attempt_root = (
                        tmp_root
                        / f"repeat_{repeat:02d}"
                        / variant
                        / f"attempt_{attempt:02d}"
                    )
                    attempt_root.mkdir(parents=True, exist_ok=True)
                    child_output = attempt_root / "result.json"
                    env = os.environ.copy()
                    env.update(
                        {
                            "TORCHINDUCTOR_FORCE_DISABLE_CACHES": "1",
                            "TORCHINDUCTOR_CACHE_DIR": str(
                                attempt_root / "torchinductor"
                            ),
                            "TRITON_CACHE_DIR": str(attempt_root / "triton"),
                        }
                    )
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--child",
                        "--variant",
                        variant,
                        "--batch",
                        str(args.batch),
                        "--seq-len",
                        str(args.seq_len),
                        "--warmup",
                        str(args.warmup),
                        "--iters",
                        str(args.iters),
                        "--seed",
                        str(args.seed),
                        "--output",
                        str(child_output),
                    ]
                    print(
                        f"[BERT {args.batch}x{args.seq_len}] "
                        f"repeat {repeat}/{args.repeats}, {variant}, "
                        f"attempt {attempt}/{args.attempts}",
                        flush=True,
                    )
                    try:
                        completed = subprocess.run(
                            command,
                            cwd=ROOT,
                            env=env,
                            text=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            timeout=args.timeout,
                            check=False,
                        )
                    except subprocess.TimeoutExpired:
                        failed_attempts.append(
                            {
                                "repeat": repeat,
                                "variant": variant,
                                "attempt": attempt,
                                "error": f"timeout after {args.timeout}s",
                            }
                        )
                        continue
                    if completed.stdout:
                        print(completed.stdout.rstrip(), flush=True)
                    if completed.returncode != 0 or not child_output.exists():
                        failed_attempts.append(
                            {
                                "repeat": repeat,
                                "variant": variant,
                                "attempt": attempt,
                                "error": f"exit code {completed.returncode}",
                            }
                        )
                        continue
                    collected[variant].append(
                        json.loads(child_output.read_text(encoding="utf-8"))
                    )
                    break
                else:
                    raise RuntimeError(
                        f"{variant} repeat {repeat} failed after "
                        f"{args.attempts} attempts"
                    )

    base = _aggregate_variant(collected["base"])
    fold = _aggregate_variant(collected["fold"])
    return {
        "meta": {
            "model": "bert",
            "input": {"batch": args.batch, "seq_len": args.seq_len},
            "compile_repeats": args.repeats,
            "warmup": args.warmup,
            "iters_per_repeat": args.iters,
            "execution_samples_per_variant": args.repeats * args.iters,
            "base_fold_process_isolation": True,
            "private_compiler_cache_per_process": True,
            "persistent_compiler_cache_disabled": True,
            "seed": args.seed,
            "failed_attempts": failed_attempts,
        },
        "variants": {"base": base, "fold_control": fold},
        "delta": {
            "compile_percent": 100.0
            * (fold["compile_ms"] / base["compile_ms"] - 1.0),
            "exec_percent": 100.0
            * (fold["exec_ms"] / base["exec_ms"] - 1.0),
            "compile_ci95_overlap": (
                abs(fold["compile_ms"] - base["compile_ms"])
                <= fold["compile_ms_ci95"] + base["compile_ms_ci95"]
            ),
            "exec_ci95_overlap": (
                abs(fold["exec_ms"] - base["exec_ms"])
                <= fold["exec_ms_ci95"] + base["exec_ms_ci95"]
            ),
        },
    }


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--variant", choices=("base", "fold"), default="base")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = _run_child(args) if args.child else _run_parent(args)
    _write_json(args.output, payload)
    print(f"[OK] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
