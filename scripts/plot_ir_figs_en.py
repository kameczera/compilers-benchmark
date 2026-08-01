#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate the ResNet IR-census figures from the repeated-run JSONs.

The three backends expose different IR concepts, so the figures deliberately
use the neutral term "call-like units":
  * TVM: TIR calls/functions in the optimized TVMScript;
  * TorchInductor: Triton and extern calls in the generated wrapper;
  * XLA: HLO fusions and custom calls in optimized HLO.

These counts provide within-stack structural context; they are not a common
dynamic GPU-launch metric and must not be used as a compiler ranking.

Run:
  ./.venv_xla/bin/python scripts/plot_ir_figs_en.py \
      --results-dir results/k5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs" / "sblp" / "images"
MODELS = [("resnet18", "ResNet-18"), ("resnet50", "ResNet-50")]
BACKENDS = ["TVM", "TorchInductor", "XLA"]
COLORS = {"TVM": "#E1A000", "TorchInductor": "#4C9BD6", "XLA": "#1B9E77"}

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 12,
})


def _summary(payload: dict, backend: str) -> dict:
    ir = payload["raw"][backend]["ir_dump"]
    if backend == "tvm":
        return ir["fused"]["kernel_count"]["summary"]
    if backend == "inductor":
        return ir["fused_inductor"]["kernel_count"]["summary"]
    return ir["fused_jit_optimized"]["kernel_count"]["summary"]


def load_counts(results_dir: Path) -> tuple[dict, dict]:
    internal = {backend: [] for backend in BACKENDS}
    external = {backend: [] for backend in BACKENDS}
    file_backend = {
        "TVM": "tvm",
        "TorchInductor": "inductor",
        "XLA": "xla",
    }
    for model, _ in MODELS:
        for label, backend in file_backend.items():
            path = results_dir / f"{backend}_{model}_1x3x224x224.json"
            if not path.exists():
                raise FileNotFoundError(f"missing repeated result: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("meta", {}).get("compile_repeats") != 5:
                raise ValueError(f"{path} is not a K=5 result")
            summary = _summary(payload, backend)
            if backend == "tvm":
                generated = summary.get(
                    "tvm_cls_total", summary.get("tvm_call_tir_total", 0)
                )
                delegated = 0
            elif backend == "inductor":
                generated = summary["triton_launches_total"]
                delegated = summary["extern_kernels_total"]
            else:
                generated = summary["fusion_total"]
                delegated = summary["custom_total"]
            internal[label].append(int(generated))
            external[label].append(int(delegated))
    return internal, external


def bars(data: dict, title: str, ylabel: str, fname: str) -> None:
    model_labels = [label for _, label in MODELS]
    x = np.arange(len(model_labels))
    width = 0.8 / len(data)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for index, (name, values) in enumerate(data.items()):
        xpos = x + (index - (len(data) - 1) / 2) * width
        containers = ax.bar(
            xpos, values, width, label=name, color=COLORS[name]
        )
        ax.bar_label(containers, padding=2, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    output = IMG / fname
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print("wrote", output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(ROOT / "results" / "k5"))
    args = parser.parse_args()
    internal, external = load_counts(Path(args.results_dir).resolve())
    total = {
        backend: [
            generated + delegated
            for generated, delegated in zip(
                internal[backend], external[backend]
            )
        ]
        for backend in BACKENDS
    }
    bars(
        total,
        "Static call-like units in optimized ResNet artifacts",
        "# call-like units",
        "fusion_rate.png",
    )
    bars(
        internal,
        "Backend-generated units in optimized ResNet artifacts",
        "# backend-generated units",
        "kernels_interno.png",
    )
    bars(
        external,
        "External-library units in optimized ResNet artifacts",
        "# external-library units",
        "kernels_externo.png",
    )
    print("counts:", {"internal": internal, "external": external})


if __name__ == "__main__":
    main()
