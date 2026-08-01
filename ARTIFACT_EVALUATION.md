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

## 2. Resource Estimate

Reference host: Linux x86_64, NVIDIA RTX 3050 (8 GiB), 16 GiB system RAM, and
an NVIDIA driver compatible with CUDA 12.8.

| Task | Typical wall time | Storage | GPU |
|---|---:|---:|---|
| Portable preflight | under 1 minute | negligible | no |
| CPU fold correctness | about 1 minute | under 1 GiB beyond dependencies | no |
| Docker build | 45--120 minutes | 25--40 GiB image plus build cache | no |
| Three-backend smoke test | 10--30 minutes | about 1 GiB of outputs | yes |
| Complete K=5 ResNet grid | several hours | allow at least 20 GiB for JSON, IR, plots, and logs | yes |
| Complete K=5 Transformer grid | about 30--60 minutes on the reference host | allow at least 2 GiB for JSON, IR, and logs | yes |

Times vary substantially with the GPU, network, Docker cache, compiler
versions, and autotuning. The full grid should be scheduled as a long-running
experiment; the smoke test is the appropriate first evaluation.

## 3. Recommended Evaluation

From the root of the exact archived release:

```bash
make artifact_check
make docker_build
make docker_verify
make docker_smoke
```

`make docker_build` compiles TVM from source and takes 45--120 minutes. To skip
it, download `cnnbench-artifact.tar.gz` — published as a separate file in the
same Zenodo record — and load the prebuilt image instead of building:

```bash
sha256sum --check cnnbench-artifact.tar.gz.sha256
docker load < cnnbench-artifact.tar.gz
```

That produces the same `cnnbench:artifact` tag the remaining commands expect, so
`make docker_verify` and `make docker_smoke` proceed unchanged.

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
6. Optionally upload `cnnbench-artifact.tar.gz` and its `.sha256` as separate
   files in the same record so evaluators can skip the TVM build; keep the
   source archive independently downloadable.
7. Ensure the repository license and the archive metadata both say MIT.
8. Inspect the final PDF page break: excluding references, the camera-ready
   limit is 9 pages for a full paper or 5 pages for a short paper (the CFP
   limit plus the permitted extra camera-ready page).
9. In JEMS, claim both badges and point evaluators to this guide and the Docker
   smoke test.
