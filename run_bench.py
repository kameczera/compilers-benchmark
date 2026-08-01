if __package__ is None or __package__ == "":
    import os, sys
    this = os.path.abspath(__file__)
    pkg_root = os.path.dirname(this)
    parent = os.path.dirname(pkg_root)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

# Precisa existir ANTES do primeiro `import torch` neste processo: desliga os
# caches persistentes do TorchInductor (/tmp/torchinductor_$USER). Sem isso,
# uma segunda execução reaproveita o cache, o codegen não roda, o output_code
# sai vazio (contagem de kernels = 0) e o compile_ms fica subestimado.
# Para reaproveitar o cache de propósito: TORCHINDUCTOR_FORCE_DISABLE_CACHES=0.
os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")

from backends.common import (
    estimate_energy_j,
    recommend,
    save_json,
    stats_ms,
)


_BACKEND_ONLY_FLAGS = {
    "inductor": ("--no-xla", "--no-tvm"),
    "xla": ("--no-inductor", "--no-tvm"),
    "tvm": ("--no-inductor", "--no-xla"),
}


def _selected_backends(args) -> list[str]:
    selected = []
    if not args.no_inductor:
        selected.append("inductor")
    if not args.no_xla:
        selected.append("xla")
    if not args.no_tvm:
        selected.append("tvm")
    return selected


def _child_command(
    args,
    backend: str,
    output: Path,
    *,
    inductor_variant: str = "both",
) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model", args.model,
        "--device", args.device,
        "--dtype", args.dtype,
        "--batch", str(args.batch),
        "--height", str(args.height),
        "--width", str(args.width),
        "--warmup", str(args.warmup),
        "--iters", str(args.iters),
        "--power-compile", str(args.power_compile),
        "--power-exec", str(args.power_exec),
        "--dataset-images", str(args.dataset_images),
        "--weights", args.weights,
        "--compile-repeats", "1",
        "--skip-recommendation",
        "--output", str(output),
        *_BACKEND_ONLY_FLAGS[backend],
    ]
    if args.compile_budget_ms is not None:
        cmd.extend(["--compile-budget-ms", str(args.compile_budget_ms)])
    if backend == "inductor":
        cmd.extend(["--inductor-variant", inductor_variant])
    elif backend == "xla":
        cmd.extend(["--xla-variant", "fused"])
    elif backend == "tvm":
        cmd.extend(["--tvm-variant", "fused"])
    return cmd


def _aggregate_scalar(
    target: dict[str, Any],
    records: list[dict[str, Any]],
    field: str,
) -> None:
    samples = [
        float(record[field])
        for record in records
        if record.get(field) is not None
    ]
    if not samples:
        return
    summary = stats_ms(samples, ci_method="student_t")
    target[field] = summary["mean"]
    target[f"{field}_std"] = summary["std"]
    target[f"{field}_ci95"] = summary["ci95_halfwidth"]
    target[f"{field}_ci95_method"] = summary["ci95_method"]
    if field.endswith("_ms"):
        samples_field = f"{field[:-3]}_samples_ms"
    elif field.endswith("_j"):
        samples_field = f"{field[:-2]}_samples_j"
    else:
        samples_field = f"{field}_samples"
    target[samples_field] = summary["samples"]
    target[f"{field}_repeats"] = summary["n"]


def _aggregate_variant(
    records: list[dict[str, Any]],
    *,
    iters: int,
    power_compile_w: float,
    power_exec_w: float,
) -> dict[str, Any]:
    target = copy.deepcopy(records[0])
    for field in (
        "compile_ms",
        "fold_preprocess_ms",
        "compile_plus_preprocess_ms",
        "ttfb_ms",
        "first_exec_ms",
        "energy_j",
    ):
        _aggregate_scalar(target, records, field)

    pooled_exec_samples = [
        float(sample)
        for record in records
        for sample in record.get("exec_samples_ms", [])
    ]
    run_means = [
        float(record["exec_ms"])
        for record in records
        if record.get("exec_ms") is not None
    ]
    if pooled_exec_samples:
        pooled = stats_ms(pooled_exec_samples)
        target["exec_ms"] = pooled["mean"]
        target["exec_ms_std"] = pooled["std"]
        target["exec_samples_ms"] = pooled["samples"]
        target["exec_samples_n"] = pooled["n"]
    if run_means:
        between_runs = stats_ms(run_means, ci_method="student_t")
        if not pooled_exec_samples:
            target["exec_ms"] = between_runs["mean"]
        target["exec_ms_ci95"] = between_runs["ci95_halfwidth"]
        target["exec_ms_ci95_method"] = "student_t_over_process_means"
        target["exec_run_means_ms"] = between_runs["samples"]
        target["exec_run_means_ms_std"] = between_runs["std"]
        target["exec_repeats"] = between_runs["n"]

    if target.get("ttfb_ms") is not None and target.get("exec_ms") is not None:
        target["energy_j"] = estimate_energy_j(
            float(target["ttfb_ms"]),
            float(target["exec_ms"]),
            iters,
            power_compile_w,
            power_exec_w,
        )
    return target


def _aggregate_backend_runs(
    backend: str,
    run_payloads: list[dict[str, Any]],
    args,
    attempts_per_repeat: list[Any] | None = None,
    failed_attempts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    blocks = [payload["raw"][backend] for payload in run_payloads]
    target = copy.deepcopy(blocks[0])
    shape_key = f"{args.batch}x3x{args.height}x{args.width}"
    shape_records = [block["shapes"][shape_key] for block in blocks]
    target_shape = {}
    variant_names = sorted({
        name
        for record in shape_records
        for name, value in record.items()
        if isinstance(value, dict)
    })
    for variant in variant_names:
        records = [
            record[variant]
            for record in shape_records
            if isinstance(record.get(variant), dict)
        ]
        target_shape[variant] = _aggregate_variant(
            records,
            iters=args.iters,
            power_compile_w=args.power_compile,
            power_exec_w=args.power_exec,
        )

    speedups = [
        float(record["speedup_exec_x"])
        for record in shape_records
        if record.get("speedup_exec_x") is not None
    ]
    if speedups:
        speedup_stats = stats_ms(speedups, ci_method="student_t")
        target_shape["speedup_exec_x"] = speedup_stats["mean"]
        target_shape["speedup_exec_x_std"] = speedup_stats["std"]

    target["shapes"] = {shape_key: target_shape}
    target.setdefault("meta", {})
    target["meta"].update({
        "compile_repeats_requested": args.compile_repeats,
        "compile_repeats_successful": len(run_payloads),
        "compile_process_isolation": True,
        "compile_ci95_method": "Student t, two-sided, 95%",
        "compile_attempts_per_repeat": attempts_per_repeat or [1] * len(run_payloads),
        "compile_failed_attempts_total": len(failed_attempts or []),
        "compile_failed_attempts": failed_attempts or [],
    })

    ir_runs = [block.get("ir_dump", {}) for block in blocks]
    target["ir_dump"] = copy.deepcopy(ir_runs[0])
    for tag, info in target["ir_dump"].items():
        if not isinstance(info, dict):
            continue
        replicate_paths = [
            run.get(tag, {}).get("path")
            for run in ir_runs
            if isinstance(run.get(tag), dict) and run.get(tag, {}).get("path")
        ]
        info["replicate_paths"] = replicate_paths
        summaries = [
            run.get(tag, {}).get("kernel_count", {}).get("summary")
            for run in ir_runs
            if isinstance(run.get(tag), dict)
        ]
        structural_summaries = [
            {
                key: value
                for key, value in (summary or {}).items()
                if key not in {"code_size_bytes", "code_lines"}
            }
            for summary in summaries
        ]
        info["replicate_kernel_summaries"] = structural_summaries
        info["replicate_kernel_summaries_identical"] = all(
            summary == structural_summaries[0]
            for summary in structural_summaries[1:]
        ) if structural_summaries else None
    return target


def _merge_inductor_variant_payloads(
    base_payload: dict[str, Any],
    fold_payload: dict[str, Any],
    shape_key: str,
) -> dict[str, Any]:
    """Combine independently collected base/fold children into one repeat."""
    merged = copy.deepcopy(base_payload)
    merged_block = merged["raw"]["inductor"]
    fold_block = fold_payload["raw"]["inductor"]
    fold_shape = fold_block["shapes"][shape_key]
    merged_block["shapes"][shape_key]["fused_fold_inductor"] = copy.deepcopy(
        fold_shape["fused_fold_inductor"]
    )
    merged_block.setdefault("ir_dump", {})["fused_fold_inductor"] = copy.deepcopy(
        fold_block.get("ir_dump", {}).get("fused_fold_inductor", {})
    )
    merged_block.setdefault("meta", {})[
        "base_fold_process_isolation"
    ] = True
    return merged


def _run_repeated(args) -> None:
    selected = _selected_backends(args)
    if not selected:
        raise SystemExit("at least one backend must be enabled")
    if args.compile_repeats < 2:
        raise SystemExit("--compile-repeats must be at least 2 in repeated mode")

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    aggregated: Dict[str, Any] = {
        "meta": {
            "device": args.device,
            "dtype": args.dtype,
            "warmup": args.warmup,
            "iters_per_compile": args.iters,
            "compile_repeats": args.compile_repeats,
            "compile_repeat_attempts_max": args.compile_repeat_attempts,
            "compile_repeat_timeout_s": args.compile_repeat_timeout_s,
            "execution_samples_total_per_variant": args.iters * args.compile_repeats,
            "power_compile_W": args.power_compile,
            "power_exec_W": args.power_exec,
            "model": args.model,
            "shape": [args.batch, 3, args.height, args.width],
            "compile_process_isolation": True,
            "inductor_base_fold_process_isolation": True,
            "xla_tvm_repeated_variant": "fused_only",
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "xla_ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
            "xla_python_client_preallocate": os.environ.get(
                "XLA_PYTHON_CLIENT_PREALLOCATE", ""
            ),
            "xla_python_client_mem_fraction": os.environ.get(
                "XLA_PYTHON_CLIENT_MEM_FRACTION", ""
            ),
            "compile_cache_policy": {
                "torchinductor_force_disable_caches": True,
                "private_torchinductor_cache_per_repeat": True,
                "private_triton_cache_per_repeat": True,
                "jax_compilation_cache_enabled": False,
                "compiler_code_cache_reused_between_repeats": False,
                "model_weight_files_shared": True,
                "os_page_cache_flushed": False,
                "model_loading_inside_compile_timer": False,
            },
        },
        "raw": {},
    }

    with tempfile.TemporaryDirectory(prefix="cnnbench-compile-repeats-") as tmp:
        tmp_root = Path(tmp)
        for backend in selected:
            payloads = []
            attempts_per_repeat = []
            failed_attempts = []
            isolated_variants = (
                ("base", "fold") if backend == "inductor" else ("both",)
            )
            for repeat in range(1, args.compile_repeats + 1):
                variant_payloads = {}
                repeat_attempts = {}
                for isolated_variant in isolated_variants:
                    last_error = "unknown error"
                    for attempt in range(1, args.compile_repeat_attempts + 1):
                        repeat_root = (
                            tmp_root
                            / backend
                            / f"repeat_{repeat:02d}"
                            / isolated_variant
                            / f"attempt_{attempt:02d}"
                        )
                        repeat_root.mkdir(parents=True, exist_ok=True)
                        child_output = repeat_root / "result.json"
                        env = os.environ.copy()
                        env.update({
                            "TORCHINDUCTOR_FORCE_DISABLE_CACHES": "1",
                            "TORCHINDUCTOR_CACHE_DIR": str(
                                repeat_root / "torchinductor"
                            ),
                            "TRITON_CACHE_DIR": str(repeat_root / "triton"),
                            "JAX_ENABLE_COMPILATION_CACHE": "0",
                            "JAX_COMPILATION_CACHE_DIR": str(repeat_root / "jax"),
                        })
                        variant_label = (
                            f"/{isolated_variant}"
                            if backend == "inductor"
                            else ""
                        )
                        print(
                            f"[compile repeat {repeat}/{args.compile_repeats}, "
                            f"attempt {attempt}/{args.compile_repeat_attempts}] "
                            f"{backend}{variant_label} {args.model} "
                            f"{args.batch}x3x{args.height}x{args.width}",
                            flush=True,
                        )
                        try:
                            completed = subprocess.run(
                                _child_command(
                                    args,
                                    backend,
                                    child_output,
                                    inductor_variant=isolated_variant,
                                ),
                                cwd=Path(__file__).resolve().parent,
                                env=env,
                                text=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                timeout=args.compile_repeat_timeout_s,
                                check=False,
                            )
                        except subprocess.TimeoutExpired:
                            last_error = (
                                f"timed out after "
                                f"{args.compile_repeat_timeout_s}s"
                            )
                            failed_attempts.append({
                                "repeat": repeat,
                                "variant": isolated_variant,
                                "attempt": attempt,
                                "error": last_error,
                            })
                            print(
                                f"[retry] {backend}{variant_label} repeat "
                                f"{repeat}: {last_error}",
                                flush=True,
                            )
                            continue
                        if completed.stdout:
                            print(completed.stdout.rstrip(), flush=True)
                        if completed.returncode != 0 or not child_output.exists():
                            last_error = f"exit code {completed.returncode}"
                            failed_attempts.append({
                                "repeat": repeat,
                                "variant": isolated_variant,
                                "attempt": attempt,
                                "error": last_error,
                            })
                            continue
                        payload = json.loads(
                            child_output.read_text(encoding="utf-8")
                        )
                        backend_error = payload.get("raw", {}).get(
                            f"{backend}_error"
                        )
                        if backend_error:
                            last_error = backend_error
                            failed_attempts.append({
                                "repeat": repeat,
                                "variant": isolated_variant,
                                "attempt": attempt,
                                "error": backend_error,
                            })
                            print(
                                f"[retry] {backend}{variant_label} repeat "
                                f"{repeat}: {backend_error}",
                                flush=True,
                            )
                            continue
                        variant_payloads[isolated_variant] = payload
                        repeat_attempts[isolated_variant] = attempt
                        break
                    else:
                        raise RuntimeError(
                            f"{backend}{variant_label} repeat {repeat} failed "
                            f"after {args.compile_repeat_attempts} attempts: "
                            f"{last_error}"
                        )
                if backend == "inductor":
                    shape_key = (
                        f"{args.batch}x3x{args.height}x{args.width}"
                    )
                    payloads.append(_merge_inductor_variant_payloads(
                        variant_payloads["base"],
                        variant_payloads["fold"],
                        shape_key,
                    ))
                    attempts_per_repeat.append(repeat_attempts)
                else:
                    payloads.append(variant_payloads["both"])
                    attempts_per_repeat.append(repeat_attempts["both"])
            aggregated["raw"][backend] = _aggregate_backend_runs(
                backend,
                payloads,
                args,
                attempts_per_repeat=attempts_per_repeat,
                failed_attempts=failed_attempts,
            )

    try:
        aggregated["recommendation"] = recommend(
            aggregated["raw"],
            target_shape=(args.batch, 3, args.height, args.width),
            runs_exec=args.iters,
            compile_budget_ms=args.compile_budget_ms,
        )
    except Exception as exc:
        aggregated["recommendation_error"] = repr(exc)
    save_json(str(out_path), aggregated)
    print(f"[OK] Saved K={args.compile_repeats} aggregate: {out_path}")

def main():
    p = argparse.ArgumentParser(description="Benchmark modular: XLA(JAX), TorchInductor, TVM (ResNet)")
    p.add_argument("--model", type=str, default="resnet18", choices=["resnet18","resnet50"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="fp32", choices=["fp32","bf16","fp16"])
    p.add_argument(
        "--inductor-variant",
        choices=("both", "base", "fold"),
        default="both",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--xla-variant",
        choices=("both", "fused"),
        default="both",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--tvm-variant",
        choices=("both", "unfused", "fused"),
        default="both",
        help=argparse.SUPPRESS,
    )
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--height", type=int, default=224)
    p.add_argument("--width", type=int, default=224)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument(
        "--compile-repeats",
        type=int,
        default=1,
        metavar="K",
        help=(
            "Run K cold compilations in independent processes and report "
            "mean, sample sd, and a 95%% Student-t CI (default: 1)."
        ),
    )
    p.add_argument(
        "--compile-repeat-timeout-s",
        type=int,
        default=900,
        help="Timeout for each isolated compilation process (default: 900s).",
    )
    p.add_argument(
        "--compile-repeat-attempts",
        type=int,
        default=3,
        help="Maximum attempts for each successful isolated repeat (default: 3).",
    )
    p.add_argument("--power-compile", type=float, default=25.0)
    p.add_argument("--power-exec", type=float, default=25.0)

    # enable/disable
    p.add_argument("--no-xla", action="store_true")
    p.add_argument("--no-inductor", action="store_true")
    p.add_argument("--no-tvm", action="store_true")

    # dataset + preference
    p.add_argument("--dataset-images", type=int, default=10000)
    p.add_argument("--compile-budget-ms", type=float, default=None)
    p.add_argument("--weights", type=str, default="auto", help="'auto' or 'c,l,e' (e.g., 0.4,0.5,0.1)")

    p.add_argument("--output", type=str, default="cnn_compilers_benchmark.json")
    p.add_argument("--skip-recommendation", action="store_true", help=argparse.SUPPRESS)

    args = p.parse_args()

    if args.compile_repeats > 1:
        _run_repeated(args)
        return

    shape = (args.batch, 3, args.height, args.width)

    results: Dict[str, Any] = {
        "meta": {
            "device": args.device,
            "dtype": args.dtype,
            "warmup": args.warmup,
            "iters": args.iters,
            "power_compile_W": args.power_compile,
            "power_exec_W": args.power_exec,
            "model": args.model,
            "shape": list(shape),
        },
        "raw": {}
    }

    # PyTorch / Inductor
    if not args.no_inductor:
        try:
            from backends.pytorch_backend import run_pytorch
            results["raw"]["inductor"] = run_pytorch(
                model_name=args.model, device=args.device, dtype=args.dtype,
                shape_nchw=shape, warmup=args.warmup, iters=args.iters,
                power_compile_w=args.power_compile, power_exec_w=args.power_exec,
                variant=args.inductor_variant,
            )
        except Exception as e:
            results["raw"]["inductor_error"] = repr(e)

    # XLA (JAX)
    if not args.no_xla:
        try:
            from backends.xla_backend import run_xla
            # Mesmo warmup/iters dos demais backends — o protocolo do artigo
            # (10 warmups, n=50) vale para os três; o corte pela metade que
            # existia aqui fazia o XLA rodar com 5/25 sem registrar isso no texto.
            results["raw"]["xla"] = run_xla(
                model_name=args.model, device=args.device, dtype=args.dtype,
                shape_nchw=shape, warmup=args.warmup, iters=args.iters,
                power_compile_w=args.power_compile, power_exec_w=args.power_exec,
                variant=args.xla_variant,
            )
        except Exception as e:
            results["raw"]["xla_error"] = repr(e)

    # TVM
    if not args.no_tvm:
        try:
            from backends.tvm_backend import run_tvm
            results["raw"]["tvm"] = run_tvm(
                model_name=args.model, device=args.device, dtype=args.dtype,
                shape_nchw=shape, warmup=args.warmup, iters=args.iters,
                power_compile_w=args.power_compile, power_exec_w=args.power_exec,
                variant=args.tvm_variant,
            )
        except Exception as e:
            results["raw"]["tvm_error"] = repr(e)

    try:
        if args.skip_recommendation:
            raise StopIteration
        backends_for_rec = {}
        if "inductor" in results["raw"] and isinstance(results["raw"]["inductor"], dict) and "shapes" in results["raw"]["inductor"]:
            backends_for_rec["inductor"] = results["raw"]["inductor"]
        if "xla" in results["raw"] and isinstance(results["raw"]["xla"], dict) and "shapes" in results["raw"]["xla"]:
            backends_for_rec["xla"] = results["raw"]["xla"]
        if "tvm" in results["raw"] and isinstance(results["raw"]["tvm"], dict) and "shapes" in results["raw"]["tvm"]:
            backends_for_rec["tvm"] = results["raw"]["tvm"]
        rec = recommend(
            backends_for_rec,
            target_shape=shape,
            runs_exec=args.iters,
            compile_budget_ms=args.compile_budget_ms
        )
        results["recommendation"] = rec
    except StopIteration:
        pass
    except Exception as e:
        results["recommendation_error"] = repr(e)

    out_path = Path(args.output).resolve()
    save_json(str(out_path), results)
    best = results.get("recommendation", {}).get("best_at_runs_exec")
    if best:
        print(f"[OK] Saved: {out_path}")
        print(f"Best backend: {best.upper()}")
    else:
        print(f"[OK] Saved: {out_path}")
        print("[warn] Could not select a best backend; check errors in JSON.")

if __name__ == "__main__":
    main()
