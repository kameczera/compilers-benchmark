#!/usr/bin/env python3
"""Gera as tabelas LaTeX de BERT e GPT-2 a partir dos JSONs K=5 completos.

Consome `results/transformers/<backend>_<modelo>_<batch>x<seq>_k5.json`,
produzidos por `scripts/benchmark_transformers.py`, e escreve
`table_bert_compile_exec.tex` e `table_gpt2_compile_exec.tex`, consumidos por
`\\input{}` em `docs/sblp/main.tex`. Nenhuma célula é transcrita à mão.

Por padrão, qualquer célula ausente encerra o comando com erro. A opção
``--allow-incomplete`` existe apenas para diagnosticar uma campanha em curso e
nunca deve ser usada para gerar as tabelas consumidas pelo artigo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKENDS = (("inductor", "TorchInductor"), ("tvm", "TVM"), ("xla", "XLA"))
SHAPES = {
    "bert": ((1, 64), (1, 128), (8, 128)),
    "gpt2": ((1, 16), (1, 128), (8, 128), (8, 256)),
}


def load(results_dir: Path, backend: str, model: str, batch: int, seq: int):
    path = results_dir / f"{backend}_{model}_{batch}x{seq}_k5.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("meta", {}).get("compile_repeats") != 5:
        raise SystemExit(f"{path}: compile_repeats != 5")
    return payload["result"]


def build(model: str, results_dir: Path) -> tuple[list[str], list[str]]:
    shapes = SHAPES[model]
    lines = [
        "% Gerado por scripts/make_transformer_tables.py — NÃO editar à mão",
        "\\begin{tabular}{llcc}",
        "\\toprule",
        "\\textbf{Framework} & \\textbf{Input} & \\textbf{Comp.} & \\textbf{Exec.} \\\\",
    ]
    missing: list[str] = []
    for backend, label in BACKENDS:
        lines.append("\\midrule")
        lines.append(f"\\multirow{{{len(shapes)}}}{{*}}{{{label}}}")
        for batch, seq in shapes:
            result = load(results_dir, backend, model, batch, seq)
            if result is None:
                missing.append(f"{backend}/{model}/{batch}x{seq}")
                lines.append(f"& ({batch},\\,{seq}) & -- & -- \\\\")
                continue
            lines.append(
                f"& ({batch},\\,{seq}) & "
                f"{result['compile_ms']:.0f} $\\pm$ {result['compile_ms_std']:.0f} & "
                f"{result['exec_ms']:.2f} $\\pm$ {result['exec_run_means_ms_std']:.2f} \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return lines, missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", type=Path, default=ROOT / "results" / "transformers"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "docs" / "sblp" / "generated"
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="render missing cells as -- for diagnostics instead of failing",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_missing: dict[str, list[str]] = {}
    for model in ("bert", "gpt2"):
        lines, missing = build(model, args.results_dir)
        if missing and not args.allow_incomplete:
            raise SystemExit(
                f"{model}: incomplete K=5 grid ({len(missing)} missing): "
                + ", ".join(missing)
            )
        output = args.out_dir / f"table_{model}_compile_exec.tex"
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        all_missing[model] = missing
        state = f"{len(missing)} célula(s) sem dado" if missing else "completa"
        print(f"[OK] {output} ({state})")
        for cell in missing:
            print(f"      sem dado: {cell}")

    coverage = args.results_dir / "coverage.json"
    if coverage.is_file():
        notes = json.loads(coverage.read_text(encoding="utf-8"))
        for cell in sorted({c for cells in all_missing.values() for c in cells}):
            reason = notes.get(cell)
            if reason:
                print(f"      motivo {cell}: {reason}")


if __name__ == "__main__":
    main()
