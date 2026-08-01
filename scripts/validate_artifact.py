#!/usr/bin/env python3
"""Validate the artifact package and the SBLP camera-ready source.

The default mode performs only portable, dependency-free checks.  The stricter
``--submission`` mode also requires the archival items that must be present in
the exact package uploaded to the artifact repository: DOI, paper PDF, K=5 raw
data, and all six generated LaTeX tables.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs" / "sblp" / "main.tex"
README = ROOT / "README.md"
CITATION = ROOT / "CITATION.cff"
EXPECTED_SHAPES = {
    "1x3x224x224",
    "16x3x224x224",
    "64x3x224x224",
    "1x3x512x512",
    "1x3x1024x1024",
}
TRANSFORMER_SHAPES = {
    "bert": ((1, 64), (1, 128), (8, 128)),
    "gpt2": ((1, 16), (1, 128), (8, 128), (8, 256)),
}
# Tabelas geradas que main.tex consome via \input{}: sem qualquer uma delas o
# camera-ready nao compila a partir de uma extracao limpa do pacote.
GENERATED_TABLES = (
    "table_resnet_compile_exec.tex",
    "table_fold_resnet18.tex",
    "table_fold_resnet50.tex",
    "table_bert_ln_fold.tex",
    "table_bert_compile_exec.tex",
    "table_gpt2_compile_exec.tex",
)

DOI_URL_PATTERN = re.compile(
    r"https://doi\.org/(10\.5281/zenodo\.\d+)", re.IGNORECASE
)
CFF_DOI_PATTERN = re.compile(
    r"^doi:\s*[\"']?(10\.5281/zenodo\.\d+)", re.IGNORECASE | re.MULTILINE
)


class Checks:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def require(self, condition: bool, message: str) -> None:
        if condition:
            print(f"[OK] {message}")
        else:
            print(f"[FAIL] {message}")
            self.failures.append(message)

    def warn(self, condition: bool, message: str) -> None:
        if condition:
            print(f"[OK] {message}")
        else:
            print(f"[WARN] {message}")
            self.warnings.append(message)


def balanced_environment(text: str, environment: str) -> list[str]:
    pattern = re.compile(
        rf"\\begin\{{{re.escape(environment)}\}}(.*?)"
        rf"\\end\{{{re.escape(environment)}\}}",
        re.DOTALL,
    )
    return pattern.findall(text)


def paper_checks(checks: Checks, tex: str) -> None:
    checks.require(
        r"\documentclass[sigconf]{acmart}" in tex,
        "paper uses exactly \\documentclass[sigconf]{acmart}",
    )
    checks.require(
        not re.search(r"\\documentclass\[[^]]*(anonymous|review)", tex),
        "paper is not anonymous and not in review mode",
    )
    checks.require(
        not re.search(r"\\(?:begin\{CCSXML\}|ccsdesc\b)", tex),
        "CCS Concepts are absent",
    )
    for command in (
        r"\setcopyright{none}",
        r"\settopmatter{printacmref=false}",
        r"\renewcommand\footnotetextcopyrightpermission[1]{}",
        r"\hypersetup{hidelinks}",
        r"\renewcommand{\shorttitle}",
        r"\renewcommand{\shortauthors}",
    ):
        checks.require(command in tex, f"paper contains {command}")
    checks.require(
        r"\acmConference[SBLP 2026]{30th Brazilian Symposium on Programming Languages}{September 8--11, 2026}{Sao Paulo, SP, Brazil}"
        in tex,
        "paper contains the required SBLP 2026 conference metadata",
    )
    checks.require(tex.count(r"\author[") == 2, "paper restores two separate authors")
    checks.require(tex.count(r"\affiliation{") == 2, "each author has a separate affiliation")
    checks.require(tex.count(r"\email{") == 2, "each author has an email address")
    checks.require(
        not re.search(r"\\(?:received|revised|accepted)\b", tex, re.IGNORECASE),
        "paper contains no publication-history dates",
    )
    checks.require(
        not re.search(r"\\(?:textcolor|color|pagecolor)\b", tex),
        "paper source contains no colored text",
    )
    abstract = balanced_environment(tex, "abstract")
    checks.require(len(abstract) == 1, "paper contains exactly one abstract")
    if abstract:
        paragraphs = [part for part in re.split(r"\n\s*\n", abstract[0].strip()) if part]
        checks.require(len(paragraphs) == 1, "abstract is a single paragraph")
    checks.require(
        bool(re.search(r"\\keywords\{[A-Z][^}]*, [A-Z]", tex)),
        "keywords are comma-separated and capitalized",
    )

    conclusion = tex.find(r"\section{Conclusion}")
    availability = tex.find(r"\section*{Artifact Availability}")
    acknowledgements = tex.find(r"\section*{Acknowledgements}")
    appendix = tex.find(r"\appendix")
    bibliography = tex.find(r"\bibliography{")
    checks.require(
        -1 < conclusion < availability,
        "Artifact Availability follows the Conclusion",
    )
    later = [position for position in (acknowledgements, appendix, bibliography) if position >= 0]
    checks.require(
        availability >= 0 and all(availability < position for position in later),
        "Artifact Availability precedes acknowledgements, appendices, and references",
    )
    checks.require(
        r"\bibliographystyle{ACM-Reference-Format}" in tex
        and r"\bibliography{ref}" in tex,
        "references use BibTeX",
    )

    for body in balanced_environment(tex, "figure") + balanced_environment(tex, "figure*"):
        graphic = body.find(r"\includegraphics")
        caption = body.find(r"\caption{")
        checks.require(
            graphic >= 0 and caption > graphic,
            "figure caption is below its graphic",
        )
    for body in balanced_environment(tex, "table") + balanced_environment(tex, "table*"):
        caption = body.find(r"\caption{")
        content_positions = [
            position
            for position in (body.find(r"\begin{tabular}"), body.find(r"\input{"))
            if position >= 0
        ]
        checks.require(
            caption >= 0 and content_positions and caption < min(content_positions),
            "table caption is above its contents",
        )


# Árvores geradas pelo benchmark: contêm Python emitido por compiladores (dumps
# do Inductor, traces de depuração), que não é código mantido do artefato e nem
# sempre é sintaticamente válido para a versão de Python que roda o validador.
GENERATED_TREES = {
    "artifacts",
    "ir_dumps",
    "plots",
    "torch_compile_debug",
}


def portable_source_checks(checks: Checks) -> None:
    python_files = sorted(
        path
        for path in ROOT.rglob("*.py")
        if not any(part.startswith(".venv") for part in path.parts)
        and not GENERATED_TREES.intersection(path.parts)
    )
    syntax_errors = []
    for path in python_files:
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except (SyntaxError, UnicodeDecodeError) as exc:
            syntax_errors.append(f"{path.relative_to(ROOT)}: {exc}")
    checks.require(not syntax_errors, "all maintained Python files parse")
    if syntax_errors:
        for error in syntax_errors:
            print(f"      {error}")

    shell_files = [
        ROOT / "docker" / "entrypoint.sh",
        ROOT / "scripts" / "package_artifact.sh",
        ROOT / "scripts" / "run_full_grid.sh",
        ROOT / "scripts" / "run_transformer_grid.sh",
    ]
    result = subprocess.run(
        ["bash", "-n", *map(str, shell_files)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    checks.require(result.returncode == 0, "maintained shell scripts pass bash -n")
    if result.stderr:
        print(result.stderr.rstrip())


def expected_result_names() -> set[str]:
    return {
        f"{backend}_{model}_{shape}.json"
        for backend in ("inductor", "tvm", "xla")
        for model in ("resnet18", "resnet50")
        for shape in EXPECTED_SHAPES
    }


def bert_result_paths() -> tuple[Path, ...]:
    directory = ROOT / "results" / "bert_fold"
    return tuple(
        directory / f"bert_{batch}x{seq_len}_k5.json"
        for batch, seq_len in ((1, 64), (1, 128), (8, 128))
    )


def transformer_result_paths() -> tuple[Path, ...]:
    directory = ROOT / "results" / "transformers"
    return tuple(
        directory / f"{backend}_{model}_{batch}x{seq_len}_k5.json"
        for backend in ("inductor", "tvm", "xla")
        for model, shapes in TRANSFORMER_SHAPES.items()
        for batch, seq_len in shapes
    )


def validate_transformer_results(paths: tuple[Path, ...]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        if not path.is_file():
            errors.append(f"{path.relative_to(ROOT)}: missing")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
            continue
        meta = payload.get("meta", {})
        result = payload.get("result", {})
        ir_dump = result.get("ir_dump", {})
        checks = {
            "compile_repeats != 5": meta.get("compile_repeats") == 5,
            "warmup != 10": meta.get("warmup") == 10,
            "iters_per_repeat != 50": meta.get("iters_per_repeat") == 50,
            "compile sample count != 5": len(result.get("compile_samples_ms", [])) == 5,
            "process-mean count != 5": len(result.get("exec_run_means_ms", [])) == 5,
            "exec_samples_n != 250": result.get("exec_samples_n") == 250,
            "IR replicate count != 5": len(ir_dump.get("replicate_paths", [])) == 5,
            "referenced IR file missing": all(
                (ROOT / relative).is_file()
                for relative in ir_dump.get("replicate_paths", [])
            ),
        }
        errors.extend(
            f"{path.relative_to(ROOT)}: {message}"
            for message, valid in checks.items()
            if not valid
        )
    return errors


def supporting_evidence_paths() -> tuple[Path, ...]:
    return (
        ROOT / "results" / "k5" / "cache_audit.md",
        ROOT / "results" / "k5" / "tables" / "fold_welch_holm.json",
        ROOT / "env_reports" / "environment_report.json",
    )


def shared_artifact_dois(readme: str, tex: str, citation: str) -> set[str]:
    """Return DOI identifiers present in the README, paper, and CFF metadata."""
    readme_dois = {match.lower() for match in DOI_URL_PATTERN.findall(readme)}
    paper_dois = {match.lower() for match in DOI_URL_PATTERN.findall(tex)}
    citation_dois = {match.lower() for match in CFF_DOI_PATTERN.findall(citation)}
    return readme_dois & paper_dois & citation_dois


def submission_checks(checks: Checks, readme: str, tex: str, citation: str) -> None:
    checks.require(
        bool(shared_artifact_dois(readme, tex, citation)),
        "README, paper, and CITATION.cff contain the same persistent artifact DOI",
    )
    paper_pdf = ROOT / "docs" / "sblp" / "main.pdf"
    checks.require(
        paper_pdf.is_file() and paper_pdf.stat().st_size > 0,
        "camera-ready paper PDF is included",
    )
    checks.require(
        paper_pdf.is_file() and paper_pdf.stat().st_mtime_ns >= PAPER.stat().st_mtime_ns,
        "camera-ready PDF was rebuilt after the latest main.tex change",
    )
    checks.require(
        all(
            (ROOT / "docs" / "sblp" / "generated" / name).is_file()
            for name in GENERATED_TABLES
        ),
        "every generated table consumed by main.tex is included",
    )
    checks.require(
        all(path.is_file() for path in bert_result_paths()),
        "the three K=5 BERT fold-audit JSON files are included",
    )
    transformer_paths = transformer_result_paths()
    transformer_errors = validate_transformer_results(transformer_paths)
    checks.require(
        not transformer_errors,
        "all 21 Transformer K=5 JSONs and their 105 IR files are complete",
    )
    for error in transformer_errors:
        print(f"      {error}")
    transformer_tables = (
        ROOT / "docs" / "sblp" / "generated" / "table_bert_compile_exec.tex",
        ROOT / "docs" / "sblp" / "generated" / "table_gpt2_compile_exec.tex",
    )
    checks.require(
        all(path.is_file() and "& -- & --" not in path.read_text(encoding="utf-8") for path in transformer_tables),
        "Transformer paper tables contain no missing cells",
    )

    result_dir = ROOT / "results" / "k5"
    present = {path.name for path in result_dir.glob("*.json")} if result_dir.exists() else set()
    expected = expected_result_names()
    checks.require(expected <= present, "all 30 K=5 raw result JSON files are included")
    malformed = []
    for name in sorted(expected & present):
        try:
            payload = json.loads((result_dir / name).read_text(encoding="utf-8"))
            if payload.get("meta", {}).get("compile_repeats") != 5:
                malformed.append(f"{name}: compile_repeats != 5")
        except (OSError, json.JSONDecodeError) as exc:
            malformed.append(f"{name}: {exc}")
    checks.require(not malformed, "included K=5 JSON files parse and declare five repeats")
    for error in malformed:
        print(f"      {error}")
    checks.require(
        all(path.is_file() and path.stat().st_size > 0 for path in supporting_evidence_paths()),
        "cache audit, fold statistics, and environment report are included",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submission",
        action="store_true",
        help="also require DOI, paper PDF, generated tables, and complete K=5 data",
    )
    args = parser.parse_args()

    checks = Checks()
    for relative in (
        "README.md",
        "LICENSE",
        "CITATION.cff",
        ".zenodo.json",
        "ARTIFACT_EVALUATION.md",
    ):
        checks.require((ROOT / relative).is_file(), f"root contains {relative}")

    try:
        json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        checks.require(False, f".zenodo.json is valid JSON ({exc})")
    else:
        checks.require(True, ".zenodo.json is valid JSON")

    readme = README.read_text(encoding="utf-8")
    tex = PAPER.read_text(encoding="utf-8")
    citation = CITATION.read_text(encoding="utf-8")
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    checks.require("MIT License" in license_text, "LICENSE grants the MIT License")
    for heading in ("## Requirements", "## Installation"):
        checks.require(heading in readme, f"README contains the required {heading[3:]} section")
    checks.require(
        "docs/sblp/main.pdf" in readme,
        "README links to the accepted paper PDF",
    )
    checks.require("Docker" in readme and "docker_smoke" in readme, "README documents a Docker smoke test")

    paper_checks(checks, tex)
    portable_source_checks(checks)

    if args.submission:
        submission_checks(checks, readme, tex, citation)
    else:
        checks.warn(
            bool(shared_artifact_dois(readme, tex, citation)),
            "same persistent artifact DOI in README, paper, and CITATION.cff "
            "(required for Available)",
        )
        checks.warn(
            all(
                (ROOT / "docs" / "sblp" / "generated" / name).is_file()
                for name in GENERATED_TABLES
            ),
            "every generated table consumed by main.tex",
        )
        checks.warn(
            all(path.is_file() for path in bert_result_paths()),
            "K=5 BERT fold-audit JSON files",
        )
        checks.warn(
            not validate_transformer_results(transformer_result_paths()),
            "complete 21-cell K=5 Transformer grid and referenced IR files",
        )
        checks.warn(
            expected_result_names()
            <= {path.name for path in (ROOT / "results" / "k5").glob("*.json")},
            "complete K=5 raw result grid required by the paper",
        )
        checks.warn(
            all(
                path.is_file() and path.stat().st_size > 0
                for path in supporting_evidence_paths()
            ),
            "cache audit, fold statistics, and environment report",
        )

    print()
    print(
        f"Summary: {len(checks.failures)} failure(s), "
        f"{len(checks.warnings)} warning(s)"
    )
    return 1 if checks.failures else 0


if __name__ == "__main__":
    sys.exit(main())
