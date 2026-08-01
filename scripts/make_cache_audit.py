#!/usr/bin/env python3
"""Build a self-checking cold-process/cache audit from the K=5 result grid.

The generated Markdown is archival evidence, not an additional measurement. It
copies the raw compile samples from the result JSONs and verifies every IR path
recorded by the five isolated compiler processes. The command fails before
writing the report if a cell, repeat, cache-policy field, or IR file is missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ("inductor", "xla", "tvm")
MODELS = ("resnet18", "resnet50")
SHAPES = (
    "1x3x224x224",
    "16x3x224x224",
    "64x3x224x224",
    "1x3x512x512",
    "1x3x1024x1024",
)
METRIC_VARIANTS = {
    "inductor": ("fused_inductor", "fused_fold_inductor"),
    "xla": ("fused_jit",),
    "tvm": ("fused",),
}
IR_VARIANTS = {
    "inductor": ("fused_inductor", "fused_fold_inductor"),
    "xla": ("fused_jit_unoptimized", "fused_jit_optimized"),
    "tvm": ("fused",),
}
# O HLO nao-otimizado do XLA embute os pesos como literais (ate ~195 MB por
# dump, ~7 GB na grade completa) e e regeneravel pelo comando documentado no
# README §7, que autoriza deixa-lo fora do pacote depositado. Quando um desses
# arquivos esta ausente, a auditoria exige que o JSON ainda declare um tamanho
# positivo e marca a linha como declarada-e-omitida em vez de falhar. Todos os
# demais artefatos continuam obrigatorios.
OMITTABLE_IR_VARIANTS = {("xla", "fused_jit_unoptimized")}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def format_samples(values: list[float]) -> str:
    return ", ".join(f"{float(value):.3f}" for value in values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/k5"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/k5/cache_audit.md")
    )
    args = parser.parse_args()

    root = ROOT.resolve()
    metric_rows: list[str] = []
    ir_rows: list[str] = []
    detail_rows: list[str] = []
    policies: set[str] = set()
    artifact_count = 0
    artifact_bytes = 0
    omitted_artifacts = 0
    stable_ir_summaries = 0
    ir_summary_groups = 0
    compile_observations = 0
    cell_repeat_units = 0
    failed_attempts = 0

    for backend in BACKENDS:
        for model in MODELS:
            for shape in SHAPES:
                result_path = args.results_dir / f"{backend}_{model}_{shape}.json"
                require(result_path.is_file(), f"missing result: {result_path}")
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                raw = payload.get("raw", {})
                require(
                    f"{backend}_error" not in raw,
                    f"backend error recorded in {result_path}",
                )
                require(
                    payload.get("meta", {}).get("compile_repeats") == 5,
                    f"compile_repeats != 5 in {result_path}",
                )
                policy = payload.get("meta", {}).get("compile_cache_policy", {})
                require(
                    policy.get("torchinductor_force_disable_caches") is True,
                    f"TorchInductor caches not disabled in {result_path}",
                )
                require(
                    policy.get("private_torchinductor_cache_per_repeat") is True,
                    f"private TorchInductor cache missing in {result_path}",
                )
                require(
                    policy.get("private_triton_cache_per_repeat") is True,
                    f"private Triton cache missing in {result_path}",
                )
                require(
                    policy.get("jax_compilation_cache_enabled") is False,
                    f"JAX persistent compilation cache enabled in {result_path}",
                )
                require(
                    policy.get("compiler_code_cache_reused_between_repeats") is False,
                    f"compiler cache reuse recorded in {result_path}",
                )
                policies.add(json.dumps(policy, sort_keys=True))

                block = raw.get(backend, {})
                backend_meta = block.get("meta", {})
                require(
                    backend_meta.get("compile_process_isolation") is True,
                    f"process isolation missing in {result_path}",
                )
                require(
                    backend_meta.get("compile_repeats_requested") == 5
                    and backend_meta.get("compile_repeats_successful") == 5,
                    f"incomplete repeats in {result_path}",
                )
                current_failed = backend_meta.get("compile_failed_attempts_total", 0)
                require(current_failed == 0, f"failed attempt in {result_path}")
                failed_attempts += int(current_failed)
                cell_repeat_units += 5

                shape_block = block.get("shapes", {}).get(shape, {})
                for variant in METRIC_VARIANTS[backend]:
                    samples = shape_block.get(variant, {}).get("compile_samples_ms")
                    require(
                        isinstance(samples, list) and len(samples) == 5,
                        f"five compile samples not found for {result_path}:{variant}",
                    )
                    require(
                        all(isinstance(value, (int, float)) and value >= 0 for value in samples),
                        f"invalid compile sample in {result_path}:{variant}",
                    )
                    compile_observations += len(samples)
                    metric_rows.append(
                        f"| {backend} | {model} | `{shape}` | `{variant}` | "
                        f"{format_samples(samples)} |"
                    )

                for ir_variant in IR_VARIANTS[backend]:
                    ir_record = block.get("ir_dump", {}).get(ir_variant, {})
                    paths = ir_record.get("replicate_paths")
                    require(
                        isinstance(paths, list) and len(paths) == 5,
                        f"five IR paths not found for {result_path}:{ir_variant}",
                    )
                    require(
                        len(set(paths)) == 5,
                        f"IR paths are not unique for {result_path}:{ir_variant}",
                    )
                    if backend == "inductor":
                        require(
                            ir_record.get("kernel_count", {}).get("capture_method")
                            == "inductor_cache_files",
                            f"degraded Inductor capture in {result_path}:{ir_variant}",
                        )

                    omittable = (backend, ir_variant) in OMITTABLE_IR_VARIANTS
                    sizes: list[int] = []
                    omitted = 0
                    for repeat, relative in enumerate(paths, start=1):
                        artifact = (ROOT / relative).resolve()
                        require(
                            artifact.is_relative_to(root),
                            f"IR path leaves repository: {relative}",
                        )
                        present = artifact.is_file() and artifact.stat().st_size > 0
                        if not present and omittable:
                            declared = ir_record.get("kernel_count", {}).get(
                                "summary", {}
                            ).get("code_size_bytes")
                            require(
                                isinstance(declared, int) and declared > 0,
                                f"omitted IR artifact without recorded size: {relative}",
                            )
                            omitted += 1
                            omitted_artifacts += 1
                            detail_rows.append(
                                f"| {backend} | {model} | `{shape}` | `{ir_variant}` | "
                                f"{repeat} | `{relative}` | declared {declared} | "
                                "`omitted from archive (regenerable)` |"
                            )
                            continue
                        require(
                            present,
                            f"missing/empty IR artifact: {relative}",
                        )
                        size = artifact.stat().st_size
                        checksum = sha256(artifact)
                        sizes.append(size)
                        artifact_count += 1
                        artifact_bytes += size
                        detail_rows.append(
                            f"| {backend} | {model} | `{shape}` | `{ir_variant}` | "
                            f"{repeat} | `{relative}` | {size} | `{checksum}` |"
                        )
                    require(
                        omitted in (0, len(paths)),
                        f"partially omitted IR variant in {result_path}:{ir_variant}",
                    )

                    stable = ir_record.get("replicate_kernel_summaries_identical")
                    ir_summary_groups += 1
                    stable_ir_summaries += int(stable is True)
                    verified_label = (
                        "0/5 (omitted, regenerable)" if omitted else "5/5"
                    )
                    size_label = (
                        "not archived" if omitted else f"{min(sizes)}--{max(sizes)}"
                    )
                    ir_rows.append(
                        f"| {backend} | {model} | `{shape}` | `{ir_variant}` | "
                        f"{verified_label} | {size_label} | "
                        f"{'yes' if stable else 'no (recorded variation)'} |"
                    )

    require(len(policies) == 1, "cache-policy metadata differs across cells")
    policy = json.loads(next(iter(policies)))
    policy_rows = [
        f"| `{key}` | `{str(value).lower() if isinstance(value, bool) else value}` |"
        for key, value in sorted(policy.items())
    ]

    lines = [
        "# Cold-Process and Compiler-Cache Audit",
        "",
        "**Verdict: PASS.** The complete 30-cell K=5 grid was checked against "
        "the raw JSON and every compiler-IR path retained by the isolated "
        "processes"
        + (
            ", except the regenerable XLA `fused_jit_unoptimized` dumps listed "
            "below as declared but not archived."
            if omitted_artifacts
            else "."
        ),
        "",
        "This report is generated by `scripts/make_cache_audit.py`; it does not "
        "add, remove, or transform timing observations. Regenerate it with:",
        "",
        "```bash",
        "python3 scripts/make_cache_audit.py --results-dir results/k5 \\",
        "  --output results/k5/cache_audit.md",
        "```",
        "",
        "## Audit Summary",
        "",
        "- Complete backend/model/input cells: **30/30**.",
        f"- Cell/backend repeat units: **{cell_repeat_units}** (30 x 5).",
        f"- Raw compile observations: **{compile_observations}** (the Inductor "
        "base and folded variants are separate isolated processes).",
        f"- Failed compile attempts admitted to the data: **{failed_attempts}**.",
        f"- Non-empty per-process IR artifacts verified: **{artifact_count}**, "
        f"totalling **{artifact_bytes} bytes**.",
        (
            f"- Declared but not archived: **{omitted_artifacts}** XLA "
            "`fused_jit_unoptimized` dumps. These embed the weights as literals "
            "(about 7 GB across the grid) and are regenerable with the command "
            "in README section 7; the JSON still records their size and kernel "
            "summary."
            if omitted_artifacts
            else "- Every IR path recorded by the isolated processes is archived "
            "and was hashed."
        ),
        f"- Structurally identical replicate summaries: "
        f"**{stable_ir_summaries}/{ir_summary_groups}** groups. A `no` below is "
        "retained evidence of compiler variation, not a missing artifact.",
        "- Every TorchInductor capture uses `inductor_cache_files`; no degraded "
        "stdout fallback was accepted.",
        "",
        "The compiler code caches are isolated, but the operating-system page "
        "cache is deliberately not flushed and model weight files may be shared. "
        "Those two limitations are explicitly retained in the policy metadata.",
        "",
        "## Recorded Cache Policy",
        "",
        "| Field | Value |",
        "|---|---|",
        *policy_rows,
        "",
        "## Raw Compile Samples (ms)",
        "",
        "Each row contains the five process-level observations used by the "
        "reported mean, standard deviation, and Student-t confidence interval.",
        "",
        "| Backend | Model | Input | Variant | Five raw samples (ms) |",
        "|---|---|---|---|---|",
        *metric_rows,
        "",
        "## Per-Process IR Verification",
        "",
        "| Backend | Model | Input | IR record | Non-empty files | Bytes (min--max) | Structural summaries identical |",
        "|---|---|---|---|---:|---:|---|",
        *ir_rows,
        "",
        "## IR File Integrity Details",
        "",
        "| Backend | Model | Input | IR record | Repeat | Path | Bytes | SHA-256 |",
        "|---|---|---|---|---:|---|---:|---|",
        *detail_rows,
        "",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"[OK] wrote {args.output}: 30 cells, {compile_observations} compile "
        f"observations, {artifact_count} verified IR files"
    )


if __name__ == "__main__":
    main()
