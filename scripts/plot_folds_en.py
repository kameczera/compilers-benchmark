#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate the two paper figures in ENGLISH from repeated-run JSONs.

Outputs (overwrites):
  docs/sblp/images/equivalencia.png    -- fold/original crossover (ResNets)
  docs/sblp/images/envelope_exemplo.png -- total-time envelope example

Run:  ./.venv_xla/bin/python scripts/plot_folds_en.py \
          --results-dir results/k5
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "k5"
IMG = ROOT / "docs" / "sblp" / "images"

plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "legend.fontsize": 13})

SHAPES = ["1x3x224x224", "16x3x224x224", "64x3x224x224", "1x3x512x512", "1x3x1024x1024"]
DISPLAY_NAMES = {
    "inductor": "TorchInductor",
    "inductor_fold": "TorchInductor + fold",
    "xla": "XLA",
    "tvm": "TVM",
}


def pretty(s):  # "1x3x224x224" -> "(1,3,224,224)"
    return "(" + ",".join(s.split("x")) + ")"


def linear(backend, model, shape):
    """Return {name: (a_exec, b_compile)} from a result JSON's model_linear."""
    f = RES / f"{backend}_{model}_{shape}.json"
    if not f.exists():
        return {}
    ml = json.load(open(f)).get("recommendation", {}).get("model_linear", {})
    return {k: (v["a_exec_ms"], v["b_compile_ms"]) for k, v in ml.items()}


def n_eq(model, shape):
    """Point-estimate crossover; zero means that fold dominates for all n."""
    ml = linear("inductor", model, shape)
    if "inductor" not in ml or "inductor_fold" not in ml:
        return None
    a_i, b_i = ml["inductor"]
    a_f, b_f = ml["inductor_fold"]
    da = a_f - a_i
    if abs(da) < 1e-12:
        return float("inf")
    if da <= 0:
        return 0.0
    return (b_i - b_f) / da


# ---------------------------------------------------------------- equivalencia
def plot_equivalence():
    models = [("resnet18", "ResNet-18"), ("resnet50", "ResNet-50")]
    rows = []
    for shape in SHAPES:
        row = []
        for model, _ in models:
            value = n_eq(model, shape)
            if value is None:
                text = "no data"
            elif value == 0:
                text = "fold dominates"
            elif value == float("inf"):
                text = "parallel slopes"
            else:
                text = f"fold until {value:,.0f}"
            row.append(text)
        rows.append(row)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        rowLabels=[pretty(shape) for shape in SHAPES],
        colLabels=[label for _, label in models],
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 1.65)
    ax.set_title(
        "Point-estimate regime: folded vs. original TorchInductor",
        pad=16,
    )
    fig.tight_layout()
    out = IMG / "equivalencia.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------- envelope
def plot_envelope(shape="1x3x224x224"):
    lines = {}
    lines.update(linear("inductor", "resnet18", shape))   # inductor + inductor_fold
    lines.update({"xla": linear("xla", "resnet18", shape).get("xla")})
    lines.update({"tvm": linear("tvm", "resnet18", shape).get("tvm")})
    lines = {k: v for k, v in lines.items() if v}
    order = ["inductor", "inductor_fold", "xla", "tvm"]
    lines = {k: lines[k] for k in order if k in lines}

    # Exact change points of the lower envelope (not every pairwise crossing
    # necessarily appears on that envelope).
    import numpy as np
    candidates = [0.0]
    names = list(lines)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ai, bi = lines[names[i]]
            aj, bj = lines[names[j]]
            if abs(ai - aj) < 1e-12:
                continue
            n = (bj - bi) / (ai - aj)
            if n > 0:
                candidates.append(float(n))
    candidates = sorted(set(candidates))

    def winner_at(n):
        return min(lines, key=lambda name: lines[name][1] + lines[name][0] * n)

    segments = [(0.0, winner_at(0.0))]
    for n in candidates[1:]:
        probe = n + max(1e-6, n * 1e-9)
        winner = winner_at(probe)
        if winner != segments[-1][1]:
            segments.append((n, winner))
    lower_crossings = [n for n, _ in segments[1:]]
    x_max = max(1.0, lower_crossings[-1] if lower_crossings else 0.0) * 1.12

    xs = np.linspace(0.0, x_max, 400)
    fig, ax = plt.subplots(figsize=(9, 5.4))
    env = None
    for name, (a, b) in lines.items():
        ys = b + a * xs
        ax.plot(xs, ys, label=DISPLAY_NAMES.get(name, name), linewidth=2)
        env = ys if env is None else np.minimum(env, ys)
    ax.plot(xs, env, color="black", linewidth=3, alpha=0.35, label="lower envelope")
    for index, x_break in enumerate(lower_crossings):
        ax.axvline(x_break, linestyle="--", color="0.4", linewidth=1.2)
        ax.annotate(
            f"$n_{{{index + 1}}}\\approx$ {x_break:,.0f}",
            (x_break, env.max() * (0.14 + 0.09 * index)),
            xytext=(-8, 0),
            textcoords="offset points",
            ha="right",
            fontsize=11,
        )
    ax.set_xlabel("n (executions)")
    ax.set_ylabel(r"Total time $T_b(n)$ [ms]")
    ax.set_title(f"Total-time envelope example --- ResNet-18, shape {pretty(shape)}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    out = IMG / "envelope_exemplo.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out, "| lower-envelope segments =", segments)
    return segments


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(RES))
    args = parser.parse_args()
    RES = Path(args.results_dir).resolve()
    plot_equivalence()
    plot_envelope()
