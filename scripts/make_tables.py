# -*- coding: utf-8 -*-
"""Gera as tabelas do artigo a partir dos JSONs do benchmark — sem transcrição
manual (foi num preenchimento manual que a coluna de compilação do TVM saiu
duplicada na Tabela 1 da versão submetida).

Entrada:  um diretório com JSONs nomeados <backend>_<modelo>_<NxCxHxW>.json
          (como os gerados por scripts/run_full_grid.sh; prefixos com sufixo de
          versão tipo inductor29_* também são aceitos)
Saída:    <out>/table_resnet_compile_exec.tex   (Tabela 1 do artigo)
          <out>/table_fold_resnet18.tex, table_fold_resnet50.tex
          <out>/summary.md                      (tudo em Markdown + n_eq + kernels)

Uso:
  python scripts/make_tables.py --results-dir results/v2 --out-dir results/v2/tables
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

SHAPES = ["1x3x224x224", "16x3x224x224", "64x3x224x224", "1x3x512x512", "1x3x1024x1024"]
MODELS = ["resnet18", "resnet50"]
FNAME_RE = re.compile(r"^(inductor|xla|tvm)\w*_(resnet18|resnet50)_(\d+x\d+x\d+x\d+)\.json$")


def load_results(results_dir: Path) -> dict:
    """{(backend, model, shape): bloco de métricas}"""
    out = {}
    for p in sorted(results_dir.glob("*.json")):
        m = FNAME_RE.match(p.name)
        if not m:
            continue
        backend, model, shape = m.groups()
        data = json.loads(p.read_text())
        raw = data.get("raw", {})
        err = raw.get(f"{backend}_error")
        block = raw.get(backend, {}) if isinstance(raw.get(backend), dict) else {}
        shp = block.get("shapes", {}).get(shape, {})
        variants = {}
        if backend == "inductor":
            variants = {"fused": shp.get("fused_inductor"), "fold": shp.get("fused_fold_inductor")}
        elif backend == "xla":
            variants = {"fused": shp.get("fused_jit")}
        elif backend == "tvm":
            variants = {"fused": shp.get("fused"), "unfused": shp.get("unfused")}
        out[(backend, model, shape)] = {
            "error": err,
            "variants": variants,
            "meta": block.get("meta"),
            "ir_dump": block.get("ir_dump", {}),
            "file": p.name,
        }
    return out


def fmt_exec(v) -> str:
    if not v:
        return "---"
    mean = v.get("exec_ms")
    std = v.get("exec_ms_std")
    if mean is None:
        return "---"
    if std is None:
        return f"{mean:.2f}"
    return f"{mean:.2f} $\\pm$ {std:.2f}"


def fmt_compile(v) -> str:
    if not v or v.get("compile_ms") is None:
        return "---"
    return f"{v['compile_ms']:.0f}"


def cell(results, backend, model, shape, variant="fused"):
    rec = results.get((backend, model, shape))
    if rec is None:
        return None, "sem dado"
    if rec["error"]:
        return None, "erro"
    return rec["variants"].get(variant), None


def table_resnet(results) -> str:
    rows = []
    for backend, label in [("inductor", "TorchInductor"), ("tvm", "TVM"), ("xla", "XLA")]:
        for i, shape in enumerate(SHAPES):
            n, c, h, w = shape.split("x")
            pref = f"\\multirow{{5}}{{*}}{{{label}}}\n" if i == 0 else ""
            cols = []
            for model in MODELS:
                v, why = cell(results, backend, model, shape)
                if v is None:
                    cols += [f"\\multicolumn{{1}}{{c}}{{---}}", f"\\multicolumn{{1}}{{c}}{{({why})}}"]
                else:
                    cols += [fmt_compile(v), fmt_exec(v)]
            rows.append(f"{pref}& ({n},\\,{c},\\,{h},\\,{w}) & " + " & ".join(cols) + " \\\\")
        rows.append("\\midrule")
    body = "\n".join(rows[:-1])
    return (
        "% Gerado por scripts/make_tables.py — NÃO editar à mão\n"
        "\\begin{tabular}{ll cc cc}\n\\toprule\n"
        "& & \\multicolumn{2}{c}{\\textbf{ResNet-18}} & \\multicolumn{2}{c}{\\textbf{ResNet-50}} \\\\\n"
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
        "\\textbf{Framework} & \\textbf{Input (N,C,H,W)} & \\textbf{Comp.} & \\textbf{Exec.} & \\textbf{Comp.} & \\textbf{Exec.} \\\\\n"
        "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n"
    )


def table_fold(results, model: str) -> str:
    rows = []
    for shape in SHAPES:
        fused, _ = cell(results, "inductor", model, shape, "fused")
        fold, _ = cell(results, "inductor", model, shape, "fold")
        n, c, h, w = shape.split("x")
        if not fused or not fold:
            rows.append(f"({n},{c},{h},{w}) & --- & --- & --- & --- & --- & --- \\\\")
            continue
        dc = (fold["compile_ms"] - fused["compile_ms"]) / fused["compile_ms"] * 100
        de = (fold["exec_ms"] - fused["exec_ms"]) / fused["exec_ms"] * 100
        rows.append(
            f"({n},{c},{h},{w}) & {fused['compile_ms']:.0f} & {fold['compile_ms']:.2f} & "
            f"\\({dc:+.1f}\\%\\) & {fused['exec_ms']:.2f} & {fold['exec_ms']:.2f} & \\({de:+.1f}\\%\\) \\\\"
        )
    return (
        "% Gerado por scripts/make_tables.py — NÃO editar à mão\n"
        "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
        "\\textbf{Input} & \\textbf{Comp. w/o fold} & \\textbf{Comp. w/ fold} & \\(\\Delta\\)\\textbf{ Comp.} & "
        "\\textbf{Exec. w/o fold} & \\textbf{Exec. w/ fold} & \\(\\Delta\\)\\textbf{ Exec.} \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )


def n_eq(fused, fold):
    """Ponto de equivalência do fold: n_eq = (b_b - b_f) / (a_f - a_b).
    Negativo (fold melhor em compile E exec) é reportado como 0."""
    if not fused or not fold:
        return None
    da = fold["exec_ms"] - fused["exec_ms"]
    db = fused["compile_ms"] - fold["compile_ms"]
    if abs(da) < 1e-12:
        return math.inf
    return max(0.0, db / da)


def summary_md(results) -> str:
    lines = ["# Resumo gerado por scripts/make_tables.py", ""]
    lines.append("## Compile / Exec (média ± sd, ms)")
    lines.append("")
    lines.append("| backend | modelo | shape | compile_ms | exec_ms ± sd | IC95 (±) | status |")
    lines.append("|---|---|---|---|---|---|---|")
    for backend in ["inductor", "tvm", "xla"]:
        for model in MODELS:
            for shape in SHAPES:
                rec = results.get((backend, model, shape))
                if rec is None:
                    continue
                if rec["error"]:
                    lines.append(f"| {backend} | {model} | {shape} | — | — | — | ERRO: {rec['error'][:60]} |")
                    continue
                v = rec["variants"].get("fused")
                if not v:
                    lines.append(f"| {backend} | {model} | {shape} | — | — | — | sem bloco fused |")
                    continue
                ci = v.get("exec_ms_ci95")
                lines.append(
                    f"| {backend} | {model} | {shape} | {v['compile_ms']:.0f} | "
                    f"{v['exec_ms']:.2f} ± {v.get('exec_ms_std', float('nan')):.2f} | "
                    f"{ci:.3f} |  |" if ci is not None else
                    f"| {backend} | {model} | {shape} | {v['compile_ms']:.0f} | {v['exec_ms']:.2f} | — |  |"
                )
    lines.append("")
    lines.append("## Fold (TorchInductor): deltas e ponto de equivalência n_eq")
    lines.append("")
    lines.append("| modelo | shape | Δcompile | Δexec | n_eq |")
    lines.append("|---|---|---|---|---|")
    for model in MODELS:
        for shape in SHAPES:
            fused, _ = cell(results, "inductor", model, shape, "fused")
            fold, _ = cell(results, "inductor", model, shape, "fold")
            if not fused or not fold:
                continue
            dc = (fold["compile_ms"] - fused["compile_ms"]) / fused["compile_ms"] * 100
            de = (fold["exec_ms"] - fused["exec_ms"]) / fused["exec_ms"] * 100
            ne = n_eq(fused, fold)
            ne_s = "∞" if ne == math.inf else f"{ne:.0f}"
            lines.append(f"| {model} | {shape} | {dc:+.1f}% | {de:+.1f}% | {ne_s} |")
    lines.append("")
    lines.append("## Kernels (análise estática dos artefatos)")
    lines.append("")
    lines.append("| backend | modelo | shape | contagem |")
    lines.append("|---|---|---|---|")
    for (backend, model, shape), rec in sorted(results.items()):
        if rec["error"] or shape != "1x3x224x224":
            continue
        for tag, info in rec["ir_dump"].items():
            s = (info or {}).get("kernel_count", {}).get("summary", {})
            keep = {k: v for k, v in s.items() if k not in ("code_size_bytes", "code_lines") and not isinstance(v, dict)}
            if keep:
                lines.append(f"| {backend} | {model} | {tag} | {keep} |")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    results = load_results(Path(args.results_dir))
    if not results:
        raise SystemExit(f"nenhum JSON reconhecido em {args.results_dir}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "table_resnet_compile_exec.tex").write_text(table_resnet(results))
    (out / "table_fold_resnet18.tex").write_text(table_fold(results, "resnet18"))
    (out / "table_fold_resnet50.tex").write_text(table_fold(results, "resnet50"))
    (out / "summary.md").write_text(summary_md(results))
    print(f"[OK] tabelas escritas em {out} (a partir de {len(results)} células de {args.results_dir})")


if __name__ == "__main__":
    main()
