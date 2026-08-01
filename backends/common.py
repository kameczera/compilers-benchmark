from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

def now_ms() -> float:
    return time.perf_counter() * 1000.0

def sync_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass

def estimate_energy_j(ttfb_ms: float, exec_ms: float, runs: int, p_compile_w: float, p_exec_w: float) -> float:
    return (ttfb_ms/1000.0)*p_compile_w + (exec_ms/1000.0)*p_exec_w*runs

_T_CRITICAL_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def stats_ms(samples: List[float], ci_method: str = "normal") -> Dict[str, Any]:
    """Estatísticas das amostras por iteração (ms): média, desvio-padrão
    amostral (n-1) e meia-largura do IC 95% — os valores "média ± sd" das
    tabelas do artigo saem daqui. As amostras brutas ficam no JSON para
    permitir reanálise sem re-medição."""
    n = len(samples)
    mean = sum(samples) / n if n else 0.0
    if n > 1:
        var = sum((s - mean) ** 2 for s in samples) / (n - 1)
        std = math.sqrt(var)
        if ci_method == "student_t":
            critical = _T_CRITICAL_95.get(n - 1, 1.96)
        elif ci_method == "normal":
            critical = 1.96
        else:
            raise ValueError(f"unknown ci_method: {ci_method}")
        ci95 = critical * std / math.sqrt(n)
    else:
        std = 0.0
        ci95 = 0.0
    return {
        "mean": float(mean),
        "std": float(std),
        "ci95_halfwidth": float(ci95),
        "ci95_method": "student_t" if ci_method == "student_t" else "normal_1.96",
        "n": n,
        "samples": [round(float(s), 4) for s in samples],
    }

def pretty_shape(nchw: Tuple[int,int,int,int]) -> str:
    n,c,h,w = nchw
    return f"{n}x{c}x{h}x{w}"

def save_json(path: str, payload: Dict[str, Any]):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

def nearest_shape(target, available):
    tn, tc, th, tw = target
    best = None
    best_dist = 10**9
    for (n,c,h,w) in available:
        d = abs(n-tn) + abs(c-tc) + abs(h-th) + abs(w-tw)
        if d < best_dist:
            best_dist = d
            best = (n,c,h,w)
    return best

# Variante fundida publicada por backend — é ela que entra no envelope T_b(n).
_FUSED_VARIANT = {"xla": "fused_jit", "tvm": "fused"}
_DEFAULT_FUSED_VARIANT = "fused_inductor"


def _timing(rec: Dict[str, Any], compile_key: str = "compile_ms") -> Dict[str, float]:
    """ttfb/exec/compile de uma variante. O intercepto cai para `compile_ms` e
    depois para `ttfb_ms` nos backends que não separam compilação da primeira
    execução."""
    ttfb = float(rec.get("ttfb_ms", 0.0))
    return {
        "ttfb_ms": ttfb,
        "exec_ms": float(rec.get("exec_ms", 0.0)),
        "compile_ms": float(rec.get(compile_key, rec.get("compile_ms", ttfb))),
    }


def recommend(backends: Dict[str, Dict[str, Any]], *,
              target_shape: Tuple[int,int,int,int],
              runs_exec: int,
              compile_budget_ms: Optional[float] = None) -> Dict[str, Any]:
    """
    Encontra intervalos de n>=0 onde cada backend minimiza
      T_b(n) = a_b * n + b_b
    com:
      a_b = exec_ms   (inclinação)
      b_b = compile_ms (intercepto)

    Considera 'inductor_fold' (quando houver) e salva um gráfico PNG
    marcando apenas as trocas reais de vencedor.
    """
    # ---------- 1) Coleta (inclui inductor_fold) ----------
    per_backend: Dict[str, Dict[str, float]] = {}
    for name, payload in backends.items():
        shapes = payload.get("shapes", {})
        avail = []
        for k in shapes.keys():
            try:
                n, c, h, w = [int(x) for x in k.split("x")]
                avail.append((n, c, h, w))
            except Exception:
                continue
        if not avail:
            continue

        rec = shapes.get(pretty_shape(nearest_shape(target_shape, avail)), {})
        lname = name.lower()

        fused = rec.get(_FUSED_VARIANT.get(lname, _DEFAULT_FUSED_VARIANT), {})
        if fused:
            per_backend[name] = _timing(fused)

        # Só o TorchInductor publica a variante com BN dobrado em Conv, e o
        # custo do próprio fold entra no intercepto.
        fused_fold = {} if lname in _FUSED_VARIANT else rec.get("fused_fold_inductor", {})
        if fused_fold:
            per_backend[f"{name}_fold"] = {
                **_timing(fused_fold, "compile_plus_preprocess_ms"),
                "compiler_only_ms": _timing(fused_fold)["compile_ms"],
                "fold_preprocess_ms": float(fused_fold.get("fold_preprocess_ms", 0.0)),
            }

    if not per_backend:
        return {"error": "no backends collected"}

    # ---------- 2) Linhas T_b(n) = a*n + b  (a=exec, b=compile) ----------
    eps = 1e-12
    lines: List[Tuple[str, float, float]] = []
    for name, v in per_backend.items():
        a = float(v["exec_ms"])      # slope
        b = float(v["compile_ms"])   # intercept
        if compile_budget_ms is not None and b > 1.5 * compile_budget_ms:
            b += 0.15  # penalização suave no intercepto (compilação)
        lines.append((name, a, b))

    # ---------- 3) Interseções par-a-par (domínio n>=0) ----------
    def x_inter(l1, l2):
        # b1 + a1*n = b2 + a2*n  =>  n* = (b1 - b2) / (a2 - a1)
        (_, a1, b1), (_, a2, b2) = l1, l2
        denom = (a2 - a1)
        if abs(denom) < eps:
            return math.inf
        return (b1 - b2) / denom

    pos_breaks = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            xb = x_inter(lines[i], lines[j])
            if xb > 0 and math.isfinite(xb):
                pos_breaks.append(xb)
    pos_breaks = sorted(set(pos_breaks))

    # ---------- 4) Segmentos ótimos em n>=0 ----------
    check_points = [0.0] + [xb + 1e-6 for xb in pos_breaks]

    def eval_line(ln, n):
        # T(n) = b + a*n
        return ln[2] + ln[1] * n

    def best_at(n: float) -> str:
        return min(lines, key=lambda ln: eval_line(ln, n))[0]

    segments: List[Dict[str, Any]] = []
    for cp in check_points:
        best = best_at(cp)
        if not segments:
            segments.append({"n_start_real": 0.0, "n_start_int": 0, "backend": best})
        elif segments[-1]["backend"] != best:
            segments.append({"n_start_real": float(cp),
                             "n_start_int": int(math.ceil(cp)),
                             "backend": best})

    # ---------- 5) Melhor no n = runs_exec ----------
    best_at_runs = best_at(runs_exec) if lines else None

    # ---------- 6) Plot (apenas quebras reais) ----------
    plot_info = None
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np

        if len(segments) > 1:
            first_break = segments[1]["n_start_real"]
            x_max = max(1.0, first_break) * 1.05
        else:
            x_max = 128.0
        x_max = max(x_max, float(runs_exec if runs_exec > 0 else 1.0))

        xs = np.linspace(0.0, x_max, 401)
        plt.figure(figsize=(9, 5.4))
        for (name, a, b) in lines:
            ys = b + a * xs
            plt.plot(xs, ys, label=name, linewidth=2)

        for seg in segments[1:]:
            s = seg["n_start_real"]
            if s <= x_max:
                plt.axvline(s, linestyle="--", linewidth=1)

        plt.xlabel("n (executions)", fontsize=14)
        plt.ylabel("Total time T_b(n) [ms]", fontsize=14)
        plt.title(f"Total-time envelope — shape {pretty_shape(target_shape)}",
                  fontsize=15)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(loc="best", fontsize=12)

        out_dir = Path("plots"); out_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time() * 1000)
        plot_path = out_dir / f"envelope_{stamp}_{pretty_shape(target_shape)}.png"
        plt.tight_layout(); plt.savefig(plot_path, dpi=200); plt.close()
        plot_info = {"path": str(plot_path), "x_max": float(x_max)}
    except Exception as e:
        plot_info = {"error": f"plot_failed: {type(e).__name__}: {e}"}

    # ---------- 7) Retorno ----------
    return {
        "target_shape": list(target_shape),
        "per_backend": per_backend,
        "model_linear": {name: {"a_exec_ms": a, "b_compile_ms": b} for (name, a, b) in lines},
        "breakpoints": [seg["n_start_real"] for seg in segments[1:]],
        "segments": segments,
        "best_at_runs_exec": best_at_runs,
        "plot": plot_info,
    }
