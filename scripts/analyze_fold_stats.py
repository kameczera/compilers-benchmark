#!/usr/bin/env python3
"""Welch comparisons for the K=5 TorchInductor ResNet fold experiment.

The independently compiled process is the experimental unit. Three families
are corrected separately with Holm's method: compiler cost after preprocessing,
fixed cost including measured fold preprocessing, and steady-state execution.

Run:
  ./.venv_xla/bin/python scripts/analyze_fold_stats.py \
      --results-dir results/k5 \
      --output results/k5/tables/fold_welch_holm.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy import stats


SHAPES = [
    "1x3x224x224",
    "16x3x224x224",
    "64x3x224x224",
    "1x3x512x512",
    "1x3x1024x1024",
]
MODELS = ["resnet18", "resnet50"]


def holm_adjust(records: list[dict]) -> None:
    ordered = sorted(enumerate(records), key=lambda item: item[1]["p_value"])
    running = 0.0
    total = len(ordered)
    for rank, (original_index, record) in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * record["p_value"])
        running = max(running, adjusted)
        records[original_index]["p_holm"] = running
        records[original_index]["reject_holm_0_05"] = running < 0.05


def compare(
    results_dir: Path,
    model: str,
    shape: str,
    metric: str,
) -> dict:
    path = results_dir / f"inductor_{model}_{shape}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("meta", {}).get("compile_repeats") != 5:
        raise ValueError(f"{path} is not K=5")
    variants = payload["raw"]["inductor"]["shapes"][shape]
    if metric == "compile_after_preprocessing":
        base = variants["fused_inductor"]["compile_samples_ms"]
        fold = variants["fused_fold_inductor"]["compile_samples_ms"]
    elif metric == "fixed_cost_including_fold":
        base = variants["fused_inductor"]["compile_samples_ms"]
        folded = variants["fused_fold_inductor"]
        fold = [
            compile_ms + preprocess_ms
            for compile_ms, preprocess_ms in zip(
                folded["compile_samples_ms"],
                folded["fold_preprocess_samples_ms"],
            )
        ]
    else:
        base = variants["fused_inductor"]["exec_run_means_ms"]
        fold = variants["fused_fold_inductor"]["exec_run_means_ms"]
    if len(base) != 5 or len(fold) != 5:
        raise ValueError(f"{path}: expected five process means per variant")
    test = stats.ttest_ind(fold, base, equal_var=False)
    interval = test.confidence_interval(confidence_level=0.95)
    base_mean = float(stats.tmean(base))
    fold_mean = float(stats.tmean(fold))
    return {
        "model": model,
        "shape": shape,
        "metric": metric,
        "base_process_means": base,
        "fold_process_means": fold,
        "base_mean": base_mean,
        "fold_mean": fold_mean,
        "fold_minus_base": fold_mean - base_mean,
        "relative_change_percent": 100.0 * (fold_mean - base_mean) / base_mean,
        "welch_df": float(test.df),
        "p_value": float(test.pvalue),
        "ci95_fold_minus_base": [float(interval.low), float(interval.high)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    output = Path(args.output)

    families = {}
    for metric in (
        "compile_after_preprocessing",
        "fixed_cost_including_fold",
        "execution",
    ):
        records = [
            compare(results_dir, model, shape, metric)
            for model in MODELS
            for shape in SHAPES
        ]
        holm_adjust(records)
        families[metric] = records

    payload = {
        "method": {
            "experimental_unit": "independent cold-process mean",
            "test": "two-sided Welch independent-samples t-test",
            "confidence_interval": "unadjusted two-sided 95% CI on fold-base",
            "multiple_testing": (
                "Holm correction at alpha=0.05, separately within each family "
                "of ten ResNet input comparisons"
            ),
        },
        "families": families,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("wrote", output)
    for metric, records in families.items():
        resolved = [
            f"{record['model']}:{record['shape']}"
            for record in records
            if record["reject_holm_0_05"]
        ]
        print(metric, "Holm-resolved:", resolved)


if __name__ == "__main__":
    main()
