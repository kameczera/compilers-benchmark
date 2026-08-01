#!/usr/bin/env python3
"""Remove do diretório de trabalho o que a grade regenera e o depósito não arquiva.

O que sai:

  * ``ir_dumps/**/fused_jit_unoptimized.hlo`` — o HLO não-otimizado do XLA embute
    os pesos como literais (~7 GB na grade completa). ``package_artifact.sh`` já o
    exclui e ``make cache_audit`` o trata como *declared but not archived*, então
    apagá-lo não muda o pacote nem a auditoria.
  * diretórios de IR órfãos — tentativas de campanha que falharam e que nenhum
    JSON de resultado referencia.
  * ``torch_compile_debug/``, ``artifacts/``, ``__pycache__/`` e ``*.pyc``, todos
    regeneráveis e já ignorados pelo git.

O que nunca sai: JSON de resultado, tabela gerada, relatório de ambiente e
qualquer arquivo de IR referenciado por algum JSON. Antes de apagar, a
ferramenta recalcula o conjunto referenciado e aborta se a remoção fosse
derrubar algo além da exceção documentada acima.

    python3 scripts/prune_regenerable.py --check    # só relata
    python3 scripts/prune_regenerable.py --apply    # remove
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable, Set

ROOT = Path(__file__).resolve().parents[1]

# Único artefato que os JSONs referenciam mas o depósito declara como não
# arquivado: é regenerável pelo comando da seção 7 do README.
UNARCHIVED_SUFFIX = "fused_jit_unoptimized.hlo"

# Diretórios inteiros que a execução recria.
REGENERABLE_DIRS = ("torch_compile_debug", "artifacts")


def referenced_ir_paths() -> Set[str]:
    """Todo caminho sob ir_dumps/ citado por algum JSON em results/."""
    referenced: Set[str] = set()
    for json_path in sorted(ROOT.glob("results/**/*.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        stack = [payload]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
            elif isinstance(node, str) and "ir_dumps/" in node:
                referenced.add(node.lstrip("./"))
    return referenced


def human(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TiB"


def tree_size(paths: Iterable[Path]) -> int:
    total = 0
    for path in paths:
        if path.is_file():
            total += path.stat().st_size
        elif path.is_dir():
            total += sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total


def collect() -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    referenced = referenced_ir_paths()
    referenced_dirs = {str(Path(p).parent) for p in referenced}

    unoptimized = sorted(ROOT.glob(f"ir_dumps/**/{UNARCHIVED_SUFFIX}"))

    # Diretório de IR é órfão quando contém arquivos e nenhum JSON o cita.
    leaf_dirs = {p.parent for p in ROOT.glob("ir_dumps/**/*") if p.is_file()}
    orphans = sorted(
        d for d in leaf_dirs
        if str(d.relative_to(ROOT)) not in referenced_dirs
    )

    regenerable = [ROOT / name for name in REGENERABLE_DIRS if (ROOT / name).exists()]

    # `.claude/` pode conter worktrees de outra sessão; não é nosso para podar.
    caches = sorted(
        p for p in ROOT.rglob("__pycache__")
        if p.is_dir() and not {".git", ".claude"} & set(p.parts)
    )
    return unoptimized, orphans, regenerable, caches


def verify_safe(removing: Set[Path]) -> list[str]:
    """Nenhum arquivo referenciado pode sumir, fora a exceção documentada."""
    violations = []
    for relative in referenced_ir_paths():
        if relative.endswith(UNARCHIVED_SUFFIX):
            continue
        target = ROOT / relative
        if not target.exists():
            violations.append(f"já ausente: {relative}")
            continue
        if any(target == r or r in target.parents for r in removing):
            violations.append(f"seria removido: {relative}")
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="apenas relata")
    mode.add_argument("--apply", action="store_true", help="remove de fato")
    args = parser.parse_args()

    unoptimized, orphans, regenerable, caches = collect()
    groups = [
        ("HLO nao-otimizado do XLA (regeneravel)", unoptimized),
        ("diretorios de IR orfaos", orphans),
        ("saidas regeneraveis", regenerable),
        ("caches __pycache__", caches),
    ]

    total = 0
    for label, paths in groups:
        if not paths:
            continue
        size = tree_size(paths)
        total += size
        print(f"[{label}] {len(paths)} item(ns), {human(size)}")
        for path in paths[:5]:
            print(f"    {path.relative_to(ROOT)}")
        if len(paths) > 5:
            print(f"    ... mais {len(paths) - 5}")

    if total == 0:
        print("[OK] nada a podar")
        return 0

    removing = {p for _, paths in groups for p in paths}
    violations = verify_safe(removing)
    if violations:
        print("\n[FAIL] a poda derrubaria arquivos referenciados por JSONs:")
        for item in violations[:10]:
            print(f"    {item}")
        return 1

    print(f"\nTotal recuperavel: {human(total)}")
    if args.check:
        print("[OK] verificacao apenas; rode com --apply para remover")
        return 0

    for path in removing:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.is_file():
            path.unlink(missing_ok=True)

    remaining = verify_safe(set())
    if remaining:
        print("\n[FAIL] apos a poda, arquivos referenciados estao ausentes:")
        for item in remaining[:10]:
            print(f"    {item}")
        return 1

    print(f"[OK] podado: {human(total)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
