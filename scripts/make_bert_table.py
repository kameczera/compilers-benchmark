#!/usr/bin/env python3
"""Gera a tabela LaTeX da auditoria de aplicabilidade do fold no BERT.

Consome os JSONs produzidos por `scripts/benchmark_bert_fold.py` (um por forma
de entrada) e escreve `table_bert_ln_fold.tex`, consumido por `\\input{}` em
`docs/sblp/main.tex`. Nenhuma célula é transcrita à mão.

Uso:
  python3 scripts/make_bert_table.py \\
      --results-dir results/bert_fold --out-dir docs/sblp/generated
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHAPES = ((1, 64), (1, 128), (8, 128))


def load(results_dir: Path, batch: int, seq_len: int) -> dict:
    path = results_dir / f"bert_{batch}x{seq_len}_k5.json"
    if not path.is_file():
        raise SystemExit(f"faltando: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    meta = payload.get("meta", {})
    if meta.get("compile_repeats") != 5:
        raise SystemExit(f"{path}: compile_repeats != 5")
    got = (meta.get("input", {}).get("batch"), meta.get("input", {}).get("seq_len"))
    if got != (batch, seq_len):
        raise SystemExit(f"{path}: entrada {got} != ({batch},{seq_len})")
    if meta.get("failed_attempts"):
        print(f"[aviso] {path.name}: {len(meta['failed_attempts'])} tentativa(s) descartada(s)")
    return payload


def row(payload: dict) -> str:
    batch = payload["meta"]["input"]["batch"]
    seq_len = payload["meta"]["input"]["seq_len"]
    base = payload["variants"]["base"]
    fold = payload["variants"]["fold_control"]
    delta = payload["delta"]
    return (
        f"({batch},{seq_len}) & "
        f"{base['compile_ms']:.0f} $\\pm$ {base['compile_ms_std']:.0f} & "
        f"{fold['compile_ms']:.0f} $\\pm$ {fold['compile_ms_std']:.0f} & "
        f"\\({delta['compile_percent']:+.1f}\\%\\) & "
        f"{base['exec_ms']:.3f} $\\pm$ {base['exec_ms_std']:.3f} & "
        f"{fold['exec_ms']:.3f} $\\pm$ {fold['exec_ms_std']:.3f} & "
        f"\\({delta['exec_percent']:+.1f}\\%\\) \\\\"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "bert_fold")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "sblp" / "generated")
    args = parser.parse_args()

    payloads = [load(args.results_dir, batch, seq_len) for batch, seq_len in SHAPES]
    lines = [
        "% Gerado por scripts/make_bert_table.py — NÃO editar à mão",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "\\textbf{Input} & \\textbf{Comp. base} & \\textbf{Comp. control} & "
        "\\(\\Delta\\)\\textbf{ Comp.} & \\textbf{Exec. base} & "
        "\\textbf{Exec. control} & \\(\\Delta\\)\\textbf{ Exec.} \\\\",
        "\\midrule",
        *(row(payload) for payload in payloads),
        "\\bottomrule",
        "\\end{tabular}",
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "table_bert_ln_fold.tex"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    folded = {
        payload["variants"]["fold_control"]["fold"].get("folded_pairs")
        for payload in payloads
    }
    print(f"[OK] {output} ({len(payloads)} formas; folded_pairs={sorted(folded)})")


if __name__ == "__main__":
    main()
