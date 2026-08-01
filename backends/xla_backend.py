# -*- coding: utf-8 -*-
from __future__ import annotations
import os, time, math, re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jax, jax.numpy as jnp

# --- imports robustos ---
try:
    from .common import pretty_shape, estimate_energy_j, stats_ms
except Exception:
    try:
        from backends.common import pretty_shape, estimate_energy_j, stats_ms
    except Exception:
        from cnnbench.backends.common import pretty_shape, estimate_energy_j, stats_ms

try:
    from ..models.resnet_jax import load_resnet18_from_flaxmodels, load_resnet50_from_flaxmodels
except Exception:
    try:
        from models.resnet_jax import load_resnet18_from_flaxmodels, load_resnet50_from_flaxmodels
    except Exception:
        from cnnbench.models.resnet_jax import load_resnet18_from_flaxmodels, load_resnet50_from_flaxmodels


# ====== HLO kernel counter (texto) ======
class HLOKernelCounter:
    def __init__(self, text: str):
        self.text = text
        self.fusions = {}
        self.fusions_by_kind = {}
        self.custom_calls = {}
        self._parse()

    def _parse(self):
        for line in self.text.splitlines():
            line = line.strip()
            if " fusion(" in line:
                m_name = re.search(r"^%([A-Za-z0-9_.\-]+)\s*=", line)
                if m_name:
                    name = m_name.group(1)
                    m_kind = re.search(r"kind=([A-Za-z]+)", line)
                    kind = m_kind.group(1) if m_kind else "unknown"
                    self.fusions[name] = kind
                    self.fusions_by_kind[kind] = self.fusions_by_kind.get(kind, 0) + 1
            if " custom-call(" in line:
                m_tgt = re.search(r'custom_call_target="([^"]+)"', line)
                if m_tgt:
                    tgt = m_tgt.group(1)
                    self.custom_calls[tgt] = self.custom_calls.get(tgt, 0) + 1

    @property
    def summary(self):
        return {
            "kernels_totais": len(self.fusions) + sum(self.custom_calls.values()),
            "fusion_total": len(self.fusions),
            "fusion_unicos": len(self.fusions),
            "fusion_por_kind": dict(self.fusions_by_kind),
            "custom_total": sum(self.custom_calls.values()),
            "custom_targets_unicos": len(self.custom_calls),
        }

def _keep_unoptimized_hlo() -> bool:
    """Escrever o HLO nao-otimizado do XLA em disco e opt-in.

    O texto embute os pesos como literais: ate ~195 MB por dump e ~7 GB na
    grade K=5 completa. O pacote depositado ja o exclui (`package_artifact.sh`),
    `make cache_audit` o trata como *declared but not archived* usando o
    `code_size_bytes` do JSON, e `make prune` o apagava depois. Escrever por
    padrao so enchia o disco de quem roda a grade. O resumo de kernels e o
    tamanho continuam sendo calculados em memoria, entao o JSON nao muda.

    Ative com CNNBENCH_DUMP_XLA_UNOPT_HLO=1 para regenerar o arquivo.
    """
    return os.environ.get("CNNBENCH_DUMP_XLA_UNOPT_HLO", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _hlo_summary_with_size(txt: str) -> Dict[str, Any]:
    summary = HLOKernelCounter(txt).summary
    summary["code_size_bytes"] = len(txt.encode("utf-8"))
    summary["code_lines"] = txt.count("\n") + (1 if txt else 0)
    return summary

def _dump_hlo(lowered, out_path: Path) -> Dict[str, Any]:
    comp = lowered.compile()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    txt = comp.as_text() if hasattr(comp, "as_text") else ""
    if txt:
        out_path.write_text(txt, encoding="utf-8")
    return {"path": str(out_path), "kernel_count": {"summary": _hlo_summary_with_size(txt)}}

def _dtype_map(dtype: str):
    if dtype == "bf16":
        return jnp.bfloat16
    if dtype == "fp16":
        return jnp.float16
    return jnp.float32


def _cuda_library_versions() -> Dict[str, Any]:
    """Report both the versions used to build and to run the JAX CUDA plugin."""
    versions: Dict[str, Any] = {}
    try:
        from jax._src.lib import cuda_versions

        versions["cudnn_build_version"] = int(cuda_versions.cudnn_build_version())
        versions["cudnn_runtime_version"] = int(cuda_versions.cudnn_get_version())
    except Exception as exc:
        versions["version_probe_error"] = repr(exc)
    return versions

def run_xla(model_name: str = "resnet18",
            device: str = "cuda",   # JAX seleciona o backend disponível
            dtype: str = "fp32",
            shape_nchw: Tuple[int,int,int,int] = (16,3,224,224),
            warmup: int = 10,
            iters: int = 50,
            power_compile_w: float = 25.0,
            power_exec_w: float = 25.0,
            variant: str = "both") -> Dict[str, Any]:
    """
    Mede compilação/execução estilo 'script antigo':
      - 1ª chamada jit = compile + run  -> first_call_ms (TTFB)
      - 2ª chamada jit = 1º run isolado -> first_exec_ms
      - warmup e steady-state por-iter -> avg_exec_ms
      - compile_ms ≈ max(0, first_call_ms - avg_exec_ms)
    Além disso, salva HLO 'unoptimized' e (quando possível) 'optimized', com contagem.
    """
    if variant not in {"both", "fused"}:
        raise ValueError(f"unknown XLA variant: {variant}")

    def _now_s():
        return time.perf_counter()

    def _to_ms(sec: float) -> float:
        return float(sec * 1000.0)

    def _measure_per_iter_ms(fn, runs: int) -> List[float]:
        out = []
        for _ in range(max(1, runs)):
            t0 = _now_s()
            y = fn()
            try:
                jax.block_until_ready(y)
            except Exception:
                # caso y seja pytree/array sem .block_until_ready
                pass
            out.append(_to_ms(_now_s() - t0))
        return out

    def _avg_ms(samples: List[float]) -> float:
        return float(sum(samples) / max(1, len(samples)))

    # ----- Preparação de modelo/entrada (NHWC + dtype FP32) -----
    N, C, H, W = shape_nchw
    nhwc = (N, H, W, C)
    jdtype = _dtype_map(dtype)

    if str(model_name).lower().replace("-", "").replace("_", "") in ("resnet50", "r50"):
        apply_fn, variables, x = load_resnet50_from_flaxmodels(input_shape=nhwc, dtype=jdtype)
    else:
        apply_fn, variables, x = load_resnet18_from_flaxmodels(input_shape=nhwc, dtype=jdtype)

    # força FP32 conforme o script antigo
    if x.dtype != jnp.float32:
        x = x.astype(jnp.float32)
    variables = jax.tree.map(
        lambda a: a.astype(jnp.float32) if hasattr(a, "dtype") and a.dtype != jnp.float32 else a,
        variables,
    )

    def forward(inp):
        return apply_fn(variables, inp)

    # ----- Baseline sem JIT (unfused) -----
    unfused_stats = None
    if variant == "both":
        # warmup completo (mesmo protocolo dos demais backends)
        for _ in range(max(1, warmup)):
            y0 = forward(x)
            try:
                jax.block_until_ready(y0)
            except Exception:
                pass

        # steady-state sem JIT
        unfused_samples = _measure_per_iter_ms(lambda: forward(x), iters)
        unfused_stats = stats_ms(unfused_samples)
        eager_exec_ms = unfused_stats["mean"]
        eager_ttfb_ms = eager_exec_ms

    # ----- JIT -----
    jitted = jax.jit(forward)

    # Diretório para dumps
    stamp = int(time.time() * 1000)
    out_dir = Path("ir_dumps") / "xla" / f"{stamp}_{model_name}_{pretty_shape(shape_nchw)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # HLO (unoptimized) via lower()
    hlo_unopt_info = None
    lowered = None
    try:
        lowered = jitted.lower(x)
        try:
            txt_unopt = lowered.compiler_ir(dialect="hlo").as_text()
        except Exception:
            txt_unopt = lowered.as_text()
        p_unopt = out_dir / "fused_jit_unoptimized.hlo"
        # O arquivo em si é opt-in (ver _keep_unoptimized_hlo); o caminho, o
        # tamanho e a contagem de kernels continuam registrados no JSON.
        if _keep_unoptimized_hlo():
            p_unopt.write_text(txt_unopt, encoding="utf-8")
        # contagem + tamanho do código
        hlo_unopt_info = {"path": str(p_unopt), "kernel_count": {"summary": _hlo_summary_with_size(txt_unopt)}}
    except Exception:
        hlo_unopt_info = None

    # 1ª chamada = compile + run
    t0 = _now_s()
    y1 = jitted(x)
    try:
        jax.block_until_ready(y1)
    except Exception:
        pass
    first_call_ms = _to_ms(_now_s() - t0)  # compile + 1ª execução (TTFB)

    # 1ª execução isolada (segundo run)
    t1 = _now_s()
    y2 = jitted(x)
    try:
        jax.block_until_ready(y2)
    except Exception:
        pass
    first_exec_ms = _to_ms(_now_s() - t1)

    # warmup jit completo
    for _ in range(max(1, warmup)):
        y = jitted(x)
        try:
            jax.block_until_ready(y)
        except Exception:
            pass

    # steady-state jit por iteração
    fused_samples = _measure_per_iter_ms(lambda: jitted(x), iters)
    fused_stats = stats_ms(fused_samples)
    avg_exec_ms = fused_stats["mean"]

    # HLO (optimized), quando possível
    hlo_opt_info = None
    try:
        if lowered is None:
            lowered = jitted.lower(x)
        comp = lowered.compile()
        if hasattr(comp, "as_text"):
            txt_opt = comp.as_text()
            p_opt = out_dir / "fused_jit_optimized.hlo"
            p_opt.write_text(txt_opt, encoding="utf-8")
            hlo_opt_info = {"path": str(p_opt), "kernel_count": {"summary": _hlo_summary_with_size(txt_opt)}}
    except Exception:
        hlo_opt_info = None

    # compile_ms no estilo do script antigo:
    # usar avg steady-state (mais robusto que "exec_ms" medido logo após)
    compile_ms = max(0.0, first_call_ms - (avg_exec_ms if math.isfinite(avg_exec_ms) else 0.0))
    ttfb_ms = first_call_ms
    exec_ms = avg_exec_ms

    energy = estimate_energy_j(ttfb_ms, exec_ms, iters, power_compile_w, power_exec_w)

    # monta retorno
    ir_dump = {}
    if hlo_unopt_info is not None:
        ir_dump["fused_jit_unoptimized"] = hlo_unopt_info
    if hlo_opt_info is not None:
        ir_dump["fused_jit_optimized"] = hlo_opt_info

    shape_out = {
        "fused_jit": {
            "ttfb_ms": float(ttfb_ms),
            "compile_ms": float(compile_ms),
            "first_exec_ms": float(first_exec_ms),
            "exec_ms": float(exec_ms),
            "exec_ms_std": float(fused_stats["std"]),
            "exec_ms_ci95": float(fused_stats["ci95_halfwidth"]),
            "exec_samples_ms": fused_stats["samples"],
            "energy_j": float(energy),
        }
    }
    if unfused_stats is not None:
        shape_out["unfused"] = {
            "ttfb_ms": float(eager_ttfb_ms),
            "exec_ms": float(eager_exec_ms),
            "exec_ms_std": float(unfused_stats["std"]),
            "exec_ms_ci95": float(unfused_stats["ci95_halfwidth"]),
            "exec_samples_ms": unfused_stats["samples"],
        }
        shape_out["speedup_exec_x"] = (
            float(eager_exec_ms / exec_ms) if exec_ms > 0 else None
        )

    return {
        "meta": {
            "device": device,
            "dtype": dtype,
            "warmup": warmup,
            "iters": iters,
            "variant": variant,
            "model_seed": 0,
            "input_pattern": "ones",
            "weights": "random_init",
            "input_normalization_in_model": False,
            "output": "logits",
            **_cuda_library_versions(),
        },
        "shapes": {pretty_shape(shape_nchw): shape_out},
        "ir_dump": ir_dump,
    }
