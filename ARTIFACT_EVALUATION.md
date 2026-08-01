# Artifact Evaluation Guide

This guide gives evaluators the shortest reliable path from the archived
artifact to evidence for the claimed results. The artifact targets the
**Available** and **Functional** badges. It is complete for submission only
when the command below exits with status 0:

```bash
make artifact_submission_check
```

That strict check requires the exact persistent DOI in the README, paper, and
`CITATION.cff`; the camera-ready PDF; the complete 30-file ResNet and 21-file
Transformer K=5 raw result grids; all six generated LaTeX tables consumed by
the paper; the three-file BERT fold audit; and the cache audit, fold-statistics
output, and environment report.

## 1. Evaluation Scope

The artifact supports the following claims:

| Claim | Evidence | Command |
|---|---|---|
| The package is structurally complete and the camera-ready source follows the SBLP rules | Portable preflight checks | `make artifact_check` |
| Conv--BatchNorm folding removes every BatchNorm and preserves FP32 outputs within the stated tolerance | CPU correctness test on ResNet-18/50 | `make check_fold` or container `verify` |
| TorchInductor, XLA, and TVM can execute on the evaluator's CUDA GPU | One small run per backend | `make docker_smoke` |
| The published ResNet protocol uses five isolated cold processes for every backend/model/input cell | Raw JSON metadata and complete-grid validator | `make artifact_submission_check` |
| The paper tables and fold statistics are derived from raw JSON without manual transcription | Table generator and Welch/Holm script | container `tables`, then `make fold_stats` in the Python environment |
| The five paper figures are regenerated from the same raw JSON grid | Figure generators reading `results/k5` | `make figures` |
| The LayerNorm fold has no legal instance in post-LayerNorm BERT, so the control variant matches the base | Three K=5 audit JSONs in `results/bert_fold/` and the table they generate | `make bert_audit` then `make bert_table` |
| The BERT and GPT-2 cross-backend tables come from the same repeated cold-process protocol as the ResNet grid | All 21 K=5 raw JSONs in `results/transformers/`; the table generator refuses incomplete coverage | `make transformers_audit` then `make transformers_tables` |

The cross-backend Transformer campaign is independent from the negative BERT
fold-applicability audit. The former contains 21 backend/model/input cells; the
latter compares an unchanged base/control pair in three BERT inputs. Both use
five isolated cold processes and archive their raw process-level observations.

`make bert_audit` measures with `local_files_only=True` and never downloads
inside the timed interval. Fetch `bert-base-uncased` into the Hugging Face
cache once before running it, as described in README section 5.2a; otherwise
the script stops with an `OSError` about connecting to huggingface.co.

### 1.1 Badge claims and justification

This artifact claims two badges. Each item names evidence a reviewer can check
without taking the authors' word for it.

**Available.** The complete artifact is archived in a public repository under a
version-specific DOI and the MIT license (`LICENSE` at the archive root, with the
same license declared in `.zenodo.json` and `CITATION.cff`). The deposit is
self-contained: benchmark source, pinned Python environments, the `Dockerfile`
recipe, the complete raw K=5 grids, the compiler-IR evidence, the analysis and
table/figure generators, and both the camera-ready PDF and its LaTeX source. The
same persistent DOI appears in the paper, in `README.md` and in `CITATION.cff`,
and `make artifact_submission_check` fails if the three ever disagree. The GitHub
repository is a development mirror; the archived record is the citable copy.

**Functional.** The artifact is documented, consistent, complete and exercisable:

- *Documented.* `README.md` states the requirements, the installation, the expected outputs, the failure diagnosis and the resource costs. This guide maps every claim of the paper to its evidence and to the command that produces it (Section 1), estimates the cost of each task (Section 2) and gives a recommended order (Section 3), including a CPU-only path for a reviewer without a CUDA GPU.
- *Consistent.* No number in the paper is transcribed by hand. The six tables consumed by `\input{}` in `main.tex` and the five figures are regenerated from the archived JSONs (`make tables`, `make bert_table`, `make transformers_tables`, `make figures`); rerunning those generators over the deposited data reproduces the deposited files. The withdrawn BERT fold claim is documented together with its isolated control experiment, and no value from the superseded run is used in the paper.
- *Complete.* The deposit carries the entire measurement campaign: the 30 ResNet cells, the 21 Transformer cells with their 105 referenced IR files, the three BERT audit JSONs, the cache audit, the Welch/Holm statistics and the environment report. `make artifact_submission_check` refuses a package missing any of them, and it currently reports 0 failures and 0 warnings.
- *Exercisable.* `make artifact_check` and the table generators run with `python3` and the standard library alone; `make check_fold` validates the transformation numerically on CPU; `make docker_build`, `make docker_verify` and `make docker_smoke` exercise the three backends end to end on a CUDA GPU.
- *Verifiable integrity.* `make cache_audit` verifies and hashes the archived compiler artifacts against what each JSON declares, and the deposit publishes a SHA-256 file next to the source archive.

**Not claimed.** The absolute timings are bound to the reference host (RTX 3050
with the pinned driver, CUDA and framework versions), so the artifact does not
claim that a re-execution reproduces the published numbers. The tolerated
variation is stated in README section 5.2b and the measurement limits in README
section 8.

## 2. Resource Estimate

Reference host: Linux x86_64, NVIDIA RTX 3050 (8 GiB), 16 GiB system RAM, and
an NVIDIA driver compatible with CUDA 12.8.

| Task | Typical wall time | Storage | GPU |
|---|---:|---:|---|
| Portable preflight | under 1 minute | negligible | no |
| CPU fold correctness | about 1 minute | under 1 GiB beyond dependencies | no |
| Docker build | 45--120 minutes | 25--40 GiB image plus build cache | no |
| Three-backend smoke test | 10--30 minutes | about 1 GiB of outputs | yes |
| Complete K=5 ResNet grid | several hours | about 1 GiB of JSON, IR, plots, and logs | yes |
| Complete K=5 Transformer grid | about 30--60 minutes on the reference host | about 100 MiB of JSON, IR, and logs | yes |

Each isolated process writes its TorchInductor, Triton and JAX caches under
`$TMPDIR`. On hosts where `/tmp` is a tmpfs — the Fedora and RHEL default — that
space is physical memory, so export `TMPDIR` to a disk-backed directory before
the full grids to keep a 16 GiB host from swapping.

Times vary substantially with the GPU, network, Docker cache, compiler
versions, and autotuning. The full grid should be scheduled as a long-running
experiment; the smoke test is the appropriate first evaluation.

### Disk-space management

Docker dominates the storage cost, and it does not reclaim space on its own:
each repeated `make docker_build` leaves the previous build cache and untagged
layers behind, which can reach tens of gigabytes. Check and reclaim with:

```bash
make docker_disk      # reports total and reclaimable space
make docker_reclaim   # drops the build cache and dangling images
```

Inside the working tree, `make prune_check` reports and `make prune` removes
everything the grid regenerates and the deposit does not archive. The XLA
`fused_jit_unoptimized.hlo` dumps embed the weights as literals (~195 MB each,
~7 GB per full grid); they are no longer written by default, so a repeated grid
no longer accumulates them. Set `CNNBENCH_DUMP_XLA_UNOPT_HLO=1` to regenerate
them on demand. The JSON still records their path, size, and kernel summary, so
`make cache_audit` behaves exactly as before, reporting them as *declared but
not archived*.

If you also export the image with `make docker_export` to move it to another
host, budget a further ~13 GiB for `dist/cnnbench-artifact.tar.gz`.

## 3. Recommended Evaluation

From the root of the exact archived release:

```bash
make artifact_check
make docker_build
make docker_verify
make docker_smoke
```

`make docker_build` compiles TVM from source and takes 45--120 minutes; it is
the only step with that cost, and the Zenodo record ships the `Dockerfile`
rather than a prebuilt image. The build produces the `cnnbench:artifact` tag the
remaining commands expect, so `make docker_verify` and `make docker_smoke`
proceed unchanged. On a machine without network access, export the image on a
connected host with `make docker_export` and load it with
`docker load < cnnbench-artifact.tar.gz` after checking its `.sha256`.

### Evaluation without a CUDA GPU

Everything except re-measuring the timings can be exercised on CPU. From the
root of the archived release:

```bash
make artifact_check                          # structural and camera-ready preflight
make artifact_submission_check               # complete K=5 grid, DOI, tables, PDF
make tables bert_table transformers_tables   # rewrite the six paper tables from the raw JSONs
make check_fold                              # CPU correctness of the Conv--BatchNorm fold
```

The first three targets need only `python3` and the standard library; no
container, no GPU, and no third-party package. `make check_fold` additionally
builds the pinned Python environment and runs on CPU. Together they exercise the
structural-completeness, table-provenance, and fold-correctness claims of
Section 1: the six tables consumed by `main.tex` are rewritten from the archived
`results/` JSONs, so any divergence between the paper and the raw data surfaces
here. Re-measuring compilation and execution times --- `make docker_smoke` and
the two full grids --- still requires the CUDA GPU described in Section 2.

Successful validation includes lines similar to:

```text
[OK ] resnet18: BN restantes=0 ...
[OK ] resnet50: BN restantes=0 ...
[OK] Imagem e GPU validadas
[OK] Smoke dos três backends concluído em /artifacts
```

The smoke test writes one JSON per backend under
`artifacts/results/` and compiler artifacts under `artifacts/ir_dumps/`.
Confirm that all three JSON files exist and do not contain a top-level backend
error before starting the complete experiment.

## 4. Full Reproduction

```bash
make docker_grid

docker run --gpus all --rm \
  -v "$PWD/artifacts:/artifacts:Z" \
  cnnbench:artifact transformers

docker run --rm \
  -v "$PWD/artifacts:/artifacts:Z" \
  cnnbench:artifact tables --require-complete-k5
```

The first command runs the three backends for ResNet-18/50 and all five input
shapes with five cold processes per cell. The second runs all 21 BERT/GPT-2
cells. The table generators refuse incomplete K=5 metadata. Compare the new
tables with the archived generated tables. Performance values need not be
bit-identical on different hardware; use the tolerances and interpretation
rules in the README.

The reference host loads cuDNN 9.11 from `/usr/local/lib64` for the XLA cells.
That library is not inside the image, so the container grid reports the
substitution and continues with the environment's own cuDNN, recording
`cudnn_runtime_version` in every JSON. XLA numbers from the container may
therefore differ from the published ones; `XLA_STRICT_CUDNN=1` aborts instead
of substituting.

## 5. Expected Output and Failure Diagnosis

- `Failed to initialize NVML: Insufficient Permissions` on Fedora, RHEL, or
  CentOS is SELinux blocking `/dev/nvidia*` for the container label, not a
  missing GPU. The `make docker_*` targets detect `getenforce` reporting
  `Enforcing` and add `--security-opt label=disable`; add the same flag to any
  manual `docker run`.
- `Failed to initialize NVML`, no `/dev/nvidia*`, or an empty
  `nvidia-smi` result indicates that the container runtime did not receive the
  GPU. Check the NVIDIA Container Toolkit and `--gpus all`.
- A JAX CPU device invalidates the GPU comparison. The `verify` command fails
  instead of silently accepting this fallback.
- An empty TorchInductor dump indicates a cache hit or degraded capture. Follow
  the cache-cleaning procedure in the README and repeat the cell.
- XLA may reject individual autotuning plans that exceed 8 GiB; a cell is valid
  only if the process completes and the JSON records a successful observation.
- Smoke results demonstrate functionality only and must not be used to replace
  the full experimental measurements.

## 6. Archival Release Checklist

Before uploading to Zenodo, Figshare, or OSF:

1. Include `README.md` and `LICENSE` directly at the archive root.
2. Include `docs/sblp/main.pdf` and all six generated tables consumed by
   `\input{}` in `main.tex`: `table_resnet_compile_exec.tex`,
   `table_fold_resnet18.tex`, `table_fold_resnet50.tex`,
   `table_bert_ln_fold.tex`, `table_bert_compile_exec.tex`, and
   `table_gpt2_compile_exec.tex`; without them the camera-ready source does not
   compile from a clean extraction.
3. Include the complete `results/k5/` ResNet grid, all 21
   `results/transformers/*_k5.json` files, the three `results/bert_fold/` audit
   JSONs, `cache_audit.md`, the corresponding statistical output,
   `env_reports/environment_report.json`, and the `ir_dumps/` evidence except the regenerable XLA
   `fused_jit_unoptimized.hlo` files; do not rely on unarchived local files.
4. Create the Zenodo draft and reserve its version-specific DOI. Run
   `make set_artifact_doi DOI=10.5281/zenodo.NUMBER`, then `make paper_pdf` so
   that the exact identifier appears in the paper, README, and `CITATION.cff`.
5. Run `make artifact_package`. It performs the strict check in staging and
   again after a clean extraction, then creates
   `cnnbench-source-v1.0.0.tar.gz` and its SHA-256 file. Upload both.
6. Do not upload the ~13 GiB `cnnbench-artifact.tar.gz`: the record ships the
   `Dockerfile`, and evaluators build the image with `make docker_build`.
7. Ensure the repository license and the archive metadata both say MIT.
8. Inspect the final PDF page break: excluding references, the camera-ready
   limit is 9 pages for a full paper or 5 pages for a short paper (the CFP
   limit plus the permitted extra camera-ready page).
9. In JEMS, claim both badges and point evaluators to this guide and the Docker
   smoke test.
