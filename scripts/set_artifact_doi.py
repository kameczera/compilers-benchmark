#!/usr/bin/env python3
"""Insert a reserved Zenodo DOI into all artifact-facing metadata.

Run this only after Zenodo has reserved the version-specific DOI for the draft
record.  The command is intentionally limited to the artifact DOI locations;
it does not rewrite bibliography identifiers.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ZENODO_DOI = re.compile(r"10\.5281/zenodo\.\d+", re.IGNORECASE)
ARTIFACT_URL = re.compile(
    r"https://doi\.org/(?:ZENODO_DOI_TBD|10\.5281/zenodo\.\d+)",
    re.IGNORECASE,
)
CFF_DOI = re.compile(r"^doi:\s*[^\n]+$", re.MULTILINE)


def normalized_doi(value: str) -> str:
    value = value.strip().removeprefix("https://doi.org/")
    if not ZENODO_DOI.fullmatch(value):
        raise ValueError(
            "expected a reserved version DOI such as 10.5281/zenodo.12345678"
        )
    return value.lower()


def update_artifact_url(path: Path, doi: str, expected_replacements: int) -> None:
    text = path.read_text(encoding="utf-8")
    updated, replacements = ARTIFACT_URL.subn(f"https://doi.org/{doi}", text)
    if replacements != expected_replacements:
        raise RuntimeError(
            f"{path.relative_to(ROOT)}: expected {expected_replacements} artifact "
            f"DOI location(s), found {replacements}"
        )
    path.write_text(updated, encoding="utf-8")


def update_citation(doi: str) -> None:
    path = ROOT / "CITATION.cff"
    text = path.read_text(encoding="utf-8")
    line = f"doi: {doi}"
    if CFF_DOI.search(text):
        updated = CFF_DOI.sub(line, text, count=1)
    else:
        anchor = "license: MIT\n"
        if text.count(anchor) != 1:
            raise RuntimeError("CITATION.cff: could not locate the license field")
        updated = text.replace(anchor, f"{anchor}{line}\n", 1)
    path.write_text(updated, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="replace the pre-deposit marker with a reserved Zenodo DOI"
    )
    parser.add_argument("doi", help="10.5281/zenodo.NUMBER or its doi.org URL")
    args = parser.parse_args()

    try:
        doi = normalized_doi(args.doi)
        update_artifact_url(ROOT / "README.md", doi, expected_replacements=2)
        update_artifact_url(
            ROOT / "docs" / "sblp" / "main.tex", doi, expected_replacements=1
        )
        update_citation(doi)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    print(f"[OK] Artifact DOI set to https://doi.org/{doi}")
    print("[NEXT] Rebuild docs/sblp/main.pdf, then run make artifact_package")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
