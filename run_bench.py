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
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

# Precisa existir ANTES do primeiro `import torch` neste processo: desliga os
# caches persistentes do TorchInductor (/tmp/torchinductor_$USER). Sem isso,
# uma segunda execução reaproveita o cache, o codegen não roda, o output_code
# sai vazio (contagem de kernels = 0) e o compile_ms fica subestimado.
# Para reaproveitar o cache de propósito: TORCHINDUCTOR_FORCE_DISABLE_CACHES=0.
os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")

from backends.common import save_json, pretty_shape, auto_weights, Weights, recommend

def main():
    p = argparse.ArgumentParser(description="Benchmark modular: XLA(JAX), TorchInductor, TVM (ResNet)")
    p.add_argument("--model", type=str, default="resnet18", choices=["resnet18","resnet50"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="fp32", choices=["fp32","bf16","fp16"])
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--height", type=int, default=224)
    p.add_argument("--width", type=int, default=224)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
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

    args = p.parse_args()

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
                power_compile_w=args.power_compile, power_exec_w=args.power_exec
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
                power_compile_w=args.power_compile, power_exec_w=args.power_exec
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
                power_compile_w=args.power_compile, power_exec_w=args.power_exec
            )
        except Exception as e:
            results["raw"]["tvm_error"] = repr(e)

    try:
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
