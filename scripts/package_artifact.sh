#!/usr/bin/env bash
# Build and validate the exact source archive intended for the Zenodo record.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="${1:-$repo_root/dist/cnnbench-source-v1.0.0.tar.gz}"
work_dir="$(mktemp -d -t cnnbench-package.XXXXXXXX)"
stage="$work_dir/stage"
extracted="$work_dir/extracted"
trap 'rm -rf -- "$work_dir"' EXIT

mkdir -p "$stage" "$extracted" "$(dirname "$output")"

cd "$repo_root"
includes=(
  .dockerignore .gitignore .zenodo.json
  ARTIFACT_EVALUATION.md CITATION.cff Dockerfile LICENSE Makefile README.md
  __init__.py collect.py run_bench.py
  backends docker docs env_reports envs ir_dumps models plots results scripts
)

rsync -aR \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.aux' \
  --exclude='*.bbl' \
  --exclude='*.blg' \
  --exclude='*.fdb_latexmk' \
  --exclude='*.fls' \
  --exclude='*.log' \
  --exclude='*.out' \
  --exclude='*unoptimized.hlo' \
  --exclude='ir_dumps/transformers/' \
  --exclude='results/transformers/coverage.json' \
  --exclude='MUDANCAS_DA_VERSAO_ANTIGA_PARA_A_NOVA.md' \
  --exclude='reviews.md' \
  --exclude='old.tex' \
  "${includes[@]}" "$stage/"

# Only archive Transformer IR files referenced by the 21 final K=5 JSONs.
# Failed/intermediate campaign attempts may remain in the working tree, but
# cannot enter the immutable release by accident.
ir_manifest="$work_dir/transformer-ir-files.txt"
python3 - "$repo_root" "$ir_manifest" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = Path(sys.argv[2])
paths = set()
for result_path in sorted((root / "results" / "transformers").glob("*_k5.json")):
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    for relative in payload.get("result", {}).get("ir_dump", {}).get("replicate_paths", []):
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise SystemExit(f"unsafe Transformer IR path: {relative}")
        if candidate.parts[:2] != ("ir_dumps", "transformers"):
            raise SystemExit(f"unexpected Transformer IR path: {relative}")
        paths.add(candidate.as_posix())
if len(paths) != 105:
    raise SystemExit(f"expected 105 referenced Transformer IR files, found {len(paths)}")
manifest.write_text("\n".join(sorted(paths)) + "\n", encoding="utf-8")
PY
rsync -aR --files-from="$ir_manifest" ./ "$stage/"

echo "[CHECK] Validating staged release"
python3 "$stage/scripts/validate_artifact.py" --submission

tar --sort=name --mtime='UTC 2026-08-01' \
  --owner=0 --group=0 --numeric-owner \
  -C "$stage" -czf "$output" .
(
  cd "$(dirname "$output")"
  sha256sum "$(basename "$output")" > "$(basename "$output").sha256"
)

tar -xzf "$output" -C "$extracted"
echo "[CHECK] Validating a clean extraction"
python3 "$extracted/scripts/validate_artifact.py" --submission
(
  cd "$(dirname "$output")"
  sha256sum --check "$(basename "$output").sha256"
)

echo "[OK] Release archive: $output"
echo "[OK] Checksum: $output.sha256"
du -h "$output" "$output.sha256"
