# CNN Compilers Benchmark

> Benchmark of the *trade-off* between **compilation time** and **execution time** in machine-learning compilers — **TorchInductor (PyTorch 2)**, **XLA (JAX)** and **TVM (Relax)** — including the TorchInductor variant with **BatchNorm folded into Conv** before operator fusion. The results feed the cost model \(T_b(n) = a_b\,n + b_b\) described in the paper (`docs/sblp/main.tex`).

## Artifact Status

- **Accepted paper:** [camera-ready PDF of the SBLP 2026 paper](docs/sblp/main.pdf); the [LaTeX source](docs/sblp/main.tex) ships with the artifact as well.
- **Persistent version:** <https://doi.org/10.5281/zenodo.21731237> — the version-specific DOI of this artifact, not the "all versions" record.
- **Development repository:** <https://github.com/kameczera/compilers-benchmark>.
- **License:** [MIT](LICENSE), allowing use, modification and redistribution.
- **Intended badges:** *Available* (after the DOI deposit) and *Functional* (reproducible installation, automated checks, three-backend smoke test and full reproduction).

The short guide for the evaluation committee is
[ARTIFACT_EVALUATION.md](ARTIFACT_EVALUATION.md). Before building the archive to
deposit, run `make artifact_submission_check`: that target fails if the DOI, the
PDF, the K=5 data and supporting evidence, the environment report or the six
generated tables consumed by the paper are missing.

## Requirements

The recommended path uses Docker, which avoids installing two mutually
incompatible Python stacks by hand. Host requirements:

- Linux x86_64, Docker Engine and the NVIDIA Container Toolkit;
- an NVIDIA GPU with at least 8 GiB of VRAM and a driver compatible with CUDA 12.8;
- 16 GiB of RAM and at least 60 GiB free for the image, the build cache and one short run;
- network access during the build, to fetch TVM at the pinned commit and the Python packages.

**Hosts with SELinux (Fedora, RHEL, CentOS).** With SELinux in `Enforcing`, the
default container label blocks `/dev/nvidia*` and `nvidia-smi` fails inside the
image with `Failed to initialize NVML: Insufficient Permissions`, even with
`--gpus all`. The `docker_*` targets detect this through `getenforce` and add
`--security-opt label=disable` on their own; when calling `docker run` by hand on
such hosts, add the same flag.

The preflight checks and the algebraic validation of the fold run on CPU. The
smoke test and any reproduction of the timings require a CUDA GPU. Budget
25--40 GiB for the image, up to 30 minutes for the smoke test, and several hours
plus at least 20 GiB more for the complete K=5 grid. Estimates and limitations are
detailed in the [evaluation guide](ARTIFACT_EVALUATION.md#2-resource-estimate).

For a native installation, use Python 3.10/3.11 with the pinned files under
`envs/`, and see section 3 for the system-level dependencies that `pip` does not
install.

## Installation

From the root of a copy of the **exact archived version**:

```bash
# portable checks: structure, paper, Python and shell syntax
make artifact_check

# recommended path to install the three backends
make docker_build
make docker_verify
make docker_smoke
```

`docker_verify` should list the GPU and the Torch/JAX/TVM versions and end with
`[OK] Imagem e GPU validadas`. `docker_smoke` runs a small ResNet-18 on the three
backends and ends with `[OK] Smoke dos três backends concluído em /artifacts`.
Outputs land in `artifacts/results/`, `artifacts/ir_dumps/`, `artifacts/plots/`
and `artifacts/env_reports/`.

If the machine has no GPU available yet, the CPU check that underpins the
correctness of the fold can be run in any environment with PyTorch/torchvision:

```bash
python scripts/check_fold.py
```

It should report `BN restantes=0` and `[OK]` for `resnet18` and `resnet50`. The
commands above validate installation and functionality; they do not replace the
full experimental grid described in section 5.2.

## 1) What the benchmark collects

For each *(model, input, backend)* triple, `run_bench.py` writes a JSON with three
blocks: `meta` (experiment context), `raw` (per-backend measurements) and
`recommendation` (cost model and best backend per usage regime).

### 1.1 Timing metrics (`raw.<backend>.shapes.<NxCxHxW>`)

| Metric | Field | How it is measured |
|---|---|---|
| Compilation time | `compile_ms`, `compile_ms_std`, `compile_ms_ci95`, `compile_samples_ms` | With `--compile-repeats K`, each sample comes from a cold process with its own caches; `compile_ms` is the mean, the deviation is the sample sd and the 95% CI uses Student's *t*. **TVM** is measured directly (passes + `relax.build`). **TorchInductor and XLA** use the proxy `first_call_ms − exec_ms`. |
| *Time to first batch* | `ttfb_ms` | Time of the first compiled call (compilation + first execution). |
| Latency per execution | `exec_ms` | Mean over the per-iteration samples (`--iters`, default 50) after warmup. **Every iteration is timed individually with GPU synchronization** (`torch.cuda.synchronize` / `block_until_ready` / `dev.sync`) — the same semantics in all three backends. |
| Execution dispersion | `exec_ms_std`, `exec_run_means_ms_std`, `exec_ms_ci95`, `exec_samples_ms`, `exec_run_means_ms` | `exec_ms_std` describes the individual samples; the tables use `exec_run_means_ms_std`, and the 95% CI uses Student's *t* over the K process means, which avoids pseudoreplication. |
| Isolated first execution | `first_exec_ms` | XLA only: the second call of the already compiled JIT. |
| Energy (estimate) | `energy_j` | `ttfb/1000·P_compile + exec/1000·P_exec·iters` with fixed powers (`--power-compile`, `--power-exec`). **Not a real measurement** — a parametric estimate. |
| Speedup | `speedup_exec_x` | Ratio of eager (or unfused) to compiled. |

Variants per backend:

- **TorchInductor**: `eager` (baseline), `fused_inductor` (`torch.compile` with `mode="max-autotune"`) and `fused_fold_inductor` (same compilation, but with **BN folded into Conv** through `fold_bn_inplace` before `torch.compile`). Base and fold start from the same `state_dict` with seed 0. The fold covers BasicBlock (`conv1/bn1`, `conv2/bn2`) **and Bottleneck (`conv3/bn3`)** — validate it with `make check_fold` (0 remaining BNs plus numerical equivalence). The PyTorch path runs with **TF32 enabled** (`set_float32_matmul_precision("high")`) and the **`channels_last` layout** on CUDA.
- **XLA**: `unfused` (forward without JIT) and `fused_jit` (`jax.jit`). It uses the native `flaxmodels` implementation and layout, but pins `pretrained=None`, removes the built-in input normalization and returns logits, like the Torch/TVM paths; even so, the comparison is between complete stacks, not between numerically identical graphs.
- **TVM**: `unfused` (legalization plus basic `FuseOps`/`FuseTIR`, without `FoldConstant`, dlight or memory planning) and `fused` (full pipeline with `FoldConstant`, `FuseOps`, `FuseTIR`, dlight, DCE, static memory and so on). Note that the `unfused` label is historical — the minimal executable pipeline still includes basic fusion.

### 1.2 Call-like unit counts and code size (`raw.<backend>.ir_dump`)

Each backend exports its compilation artifact to
`ir_dumps/<backend>/<timestamp>_<model>_<shape>/` and counts units by **static
analysis** of that artifact (this is not a profiler launch count):

| Backend | Artifact | Counts in `kernel_count.summary` |
|---|---|---|
| TorchInductor | `fused[_fold]_inductor_output_code.py` — the wrapper with `call()` plus the generated Triton kernels | `triton_launches_total`, `triton_kernels_unicos`, `extern_kernels_total` (cuDNN/cuBLAS), `extern_funcs_unicas` |
| XLA | `fused_jit_unoptimized.hlo` and `fused_jit_optimized.hlo` | `fusion_total`, `fusion_por_kind` (kLoop/kInput/kCustom…), `custom_total` (custom calls, e.g. cuDNN), `kernels_totais` |
| TVM | `unfused_tvmscript.py` and `fused_tvmscript.py` | `tvm_call_tir_total`, `tvm_call_tir_kernels_unicos`, `tvm_cls_total` |

Every summary also includes **`code_size_bytes`** and **`code_lines`** (size of the
generated code). `kernel_count.details` holds the kernel names, the per-function
counts and, for Inductor, the `fused_ops` map (which ATen ops were fused into each
Triton kernel, extracted from the `Original ATen: [...]` comments).

Notes:

- XLA's **unoptimized** HLO embeds the model weights as literals and can exceed **90 MB** per dump. Use `fused_jit_optimized.hlo` (tens of KB) for the unit analysis.
- The Inductor dump concatenates the wrapper (which holds the `triton_*.run(...)` and `extern_kernels.*` calls) with the Triton kernel modules, one `# ===== inductor generated file: ...` header per file. The `capture_method` field records how the code was captured (`inductor_cache_files` is the expected value).

### 1.3 Recommendation (`recommendation`)

A linear model per backend, \(T_b(n) = a_b\,n + b_b\), with `a_exec_ms` as the mean
latency and `b_compile_ms` as the compilation cost:

```json
{
  "model_linear": {
    "inductor":      { "a_exec_ms": 4.74, "b_compile_ms": 21358.4 },
    "inductor_fold": { "a_exec_ms": 3.29, "b_compile_ms": 8548.6 }
  },
  "breakpoints": [],
  "segments": [ { "n_start_int": 0, "backend": "inductor_fold" } ],
  "best_at_runs_exec": "inductor_fold",
  "plot": { "path": "plots/envelope_<timestamp>_<shape>.png" }
}
```

`segments` are the ranges of \(n\) (number of executions) in which each backend
dominates (the lower envelope); `breakpoints` are the switching points; and
`best_at_runs_exec` is the best backend for the \(n\) passed in `--iters`. The
envelope plot is saved under `plots/`.

## 2) Repository layout

```text
.
├── run_bench.py               # Benchmark orchestrator
├── collect.py                 # Environment report (versions, GPU, driver) → env_reports/
├── backends/
│   ├── pytorch_backend.py     # eager + Inductor + Inductor with BN fold, output_code dump
│   ├── xla_backend.py         # JAX/XLA, HLO dump (unopt/opt)
│   ├── tvm_backend.py         # TVM Relax, TVMScript dump
│   └── common.py              # metrics, T(n)=a·n+b model, envelope and plot
├── models/
│   ├── resnet_torch.py        # torchvision ResNet-18/50
│   └── resnet_jax.py          # JAX implementations of ResNet-18/50
├── envs/
│   ├── requirements_xla.txt   # XLA/JAX + PyTorch environment (venv 1)
│   └── requirements_tvm.txt   # TVM environment (venv 2; TVM is built from source, see §4.2)
├── scripts/
│   ├── run_full_grid.sh       # Full paper grid (all backends × shapes) → results/
│   ├── make_tables.py         # JSONs → paper LaTeX tables + summary.md (no manual transcription)
│   ├── check_fold.py          # Validates the BN→Conv fold (0 remaining BNs + numerical equivalence)
│   ├── plot_folds_en.py       # JSONs → equivalencia.png and envelope_exemplo.png
│   ├── plot_ir_figs_en.py     # JSONs → fusion_rate.png and the two unit-count figures
│   ├── prune_regenerable.py   # Removes what is regenerable without touching referenced data
│   └── validate_artifact.py   # Portable preflight and strict checklist of the archived version
├── ARTIFACT_EVALUATION.md     # Short guide for the evaluators
├── CITATION.cff               # Citation metadata
├── Makefile                   # Venvs, smoke tests, check_fold, grid, tables, cache cleanup
├── ir_dumps/                  # Generated: compilation artifacts (Triton/HLO/TVMScript)
├── plots/                     # Generated: T(n) envelope plots
├── results/                   # Archived version: measured JSONs, per campaign
├── env_reports/               # Generated: environment reports and locks
└── docs/sblp/main.tex         # Paper
```

## 3) Native-installation prerequisites

These apply only if you install without Docker; the container image already
provides all of them.

- Python 3.10 for the XLA/PyTorch venv (3.11 also works) and Python 3.11 for the TVM venv.
- A TVM checkout built from source with `USE_CUDA=ON` and `USE_LLVM=ON` (see §4.2).
- The NVIDIA driver, `nvcc` and LLVM are system dependencies — they are **not** installed by the requirement files.
- Reference host of the paper: RTX 3050 8 GB, driver 580.x (CUDA 13.0 as reported by the driver), `nvcc` 12.6.

## 4) Python environments

The benchmark uses **two separate venvs**, because the torch versions required by
XLA/Inductor and by TVM conflict.

### 4.1 XLA + PyTorch venv (`.venv_xla`)

```bash
make xla_env                 # creates .venv_xla from envs/requirements_xla.txt
make check_xla               # runs pip check and confirms GPU visibility for torch and jax
```

If `python3.10` is not on the `PATH`, point at the interpreter:
`make xla_env PYTHON_BIN=python3.11`.

### 4.2 TVM venv (`.venv_tvm`)

The TVM used in the study (**0.22.dev0, commit `3b60f1c`**) cannot be installed
with pip — the PyPI `apache-tvm` package stopped at 0.14 and lacks the `relax`
module the backend uses. Build TVM from source and point `TVM_HOME` at the
checkout:

```bash
git clone --recursive https://github.com/apache/tvm && cd tvm
git checkout 3b60f1c9b
# build with USE_CUDA=ON and USE_LLVM=ON:
# https://tvm.apache.org/docs/install/from_source.html
```

```bash
make tvm_env                                # creates .venv_tvm (runtime deps + torch for the FX trace)
make smoke_tvm TVM_HOME=/path/to/tvm        # validates
```

The Makefile injects `PYTHONPATH=$TVM_HOME/python:$TVM_HOME/build` when running TVM.

### 4.3 Environment variables (exported by the Makefile)

| Variable | Value | Why |
|---|---|---|
| `XLA_PYTHON_CLIENT_PREALLOCATE` | `false` | Keeps JAX from preallocating all of the VRAM, which avoids `Failed to allocate ... bytes` when PyTorch and JAX share the GPU. |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | `0.75` | Caps the VRAM fraction JAX may take. |
| `XLA_LIBRARY_PATH` | `/usr/local/lib64` | Makes the JAX CUDA plugin load the host cuDNN 9.11.0; the effective version is recorded in the JSON. |
| `XLA_FLAGS` | empty (XLA default) | Allows overriding XLA flags; the main campaign does not change the autotuner. |
| `TORCHINDUCTOR_FORCE_DISABLE_CACHES` | `1` | **Required for a correct collection** — see §7. `run_bench.py` also sets this default on its own. |

They must exist **before the first `import jax` / `import torch`** in the process.
When running by hand outside the Makefile, export them first.

## 5) Running the benchmark

### 5.1 Smoke tests (quick validation)

```bash
make smoke_torch                            # TorchInductor only (eager + fused + fold)
make smoke_xla                              # XLA/JAX only
make smoke_xla_torch                        # XLA + Inductor in the same process
make smoke_tvm TVM_HOME=/path/to/tvm        # TVM only
```

Each target uses ResNet-18, batch 1, 224×224, 3 warmups and 5 iterations, and
writes `out_smoke_*.json`.

### 5.2 Full measurement (the paper's ResNet protocol: K=5, warmup 10, iters 50)

`--compile-repeats 5` automatically creates five isolated processes per backend,
each with its own cache directories. There is no need to clean caches by hand
between those repetitions. "Cold" here refers to the compiler code cache; model
weights and the operating-system page cache stay shared and remain outside the
timed interval. In TorchInductor, the original and folded variants are also
compiled in separate processes, which prevents in-memory reuse between them. In
the repeated XLA/TVM processes only the published JIT/fused variant runs, so the
unfused baseline never executes first and cannot warm the compiler state:

**`/tmp` on tmpfs (the Fedora, RHEL and Arch default).** Each isolated process
writes its own TorchInductor, Triton and JAX cache under a scratch directory
created with `tempfile`, that is, under `$TMPDIR`, which defaults to `/tmp`.
Where `/tmp` is a tmpfs those caches live in **RAM** — the default tmpfs size is
half of the physical memory, 7.8 GiB on the 16 GiB reference host — so a full
K=5 campaign can push a desktop into swap and make it unresponsive. Point
`TMPDIR` at a disk-backed directory before the long runs:

```bash
export TMPDIR=/var/tmp/cnnbench && mkdir -p "$TMPDIR"
```

The archived Transformer JSONs record their cache directories under `/tmp`,
which is the distribution default used for the published campaigns; the variable
changes only where the compiler caches are written, not the protocol, although
the compilation-cost proxy can shift slightly because a tmpfs is faster than a
disk. Runs inside the container are unaffected unless you mount a tmpfs over its
`/tmp`.

```bash
# --- TorchInductor (eager + fused + fused_fold) ---
make clean_inductor_cache
. .venv_xla/bin/activate
python run_bench.py --no-tvm --no-xla --device cuda --model resnet18 \
  --dtype fp32 --batch 1 --height 224 --width 224 \
  --warmup 10 --iters 50 --compile-repeats 5 \
  --output out_inductor_resnet18_1x3x224x224.json
deactivate

# --- XLA ---
. .venv_xla/bin/activate
export LD_LIBRARY_PATH=/usr/local/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
export XLA_FLAGS=''
python run_bench.py --no-tvm --no-inductor --device cuda --model resnet18 \
  --dtype fp32 --batch 1 --height 224 --width 224 \
  --warmup 10 --iters 50 --compile-repeats 5 \
  --output out_xla_resnet18_1x3x224x224.json
deactivate

# --- TVM ---
. .venv_tvm/bin/activate
export TVM_HOME=/path/to/tvm
PYTHONPATH=$TVM_HOME/python:$TVM_HOME/build python run_bench.py \
  --no-xla --no-inductor --device cuda --model resnet18 \
  --dtype fp32 --batch 1 --height 224 --width 224 \
  --warmup 10 --iters 50 --compile-repeats 5 \
  --output out_tvm_resnet18_1x3x224x224.json
deactivate
```

Repeat while varying `--batch/--height/--width` (the paper uses 1/16/64×224² plus
1×512² and 1×1024²) and `--model resnet50`.

### 5.2a BERT audit: fold applicability

The standard Hugging Face BERT is post-LayerNorm: the normalized output also feeds
the residual path, so absorbing the LayerNorm affine part into the linear
projections alone would change the model's function. The script below measures a
semantically identical control variant, in separate processes and caches, to audit
the earlier fold claim without applying an illegal transformation.

The script measures with `local_files_only=True`, that is, it **does not download
the model** — no network access happens inside the timed interval. Fetch the
checkpoint once before the first collection:

```bash
. .venv_xla/bin/activate
python -c "from transformers import BertModel; BertModel.from_pretrained('bert-base-uncased')"
```

Without that step the script fails with
`OSError: We couldn't connect to 'https://huggingface.co' ...`. The default cache
lives in `~/.cache/huggingface` (about 1.3 GB for `bert-base-uncased`); use
`HF_HOME` to choose another directory. Then:

```bash
python scripts/benchmark_bert_fold.py \
  --batch 1 --seq-len 64 \
  --repeats 5 --warmup 10 --iters 50 \
  --output results/bert_fold/bert_1x64_k5.json
deactivate
```

Inside the container, mount the host cache
(`-v "$HOME/.cache/huggingface:/hf" -e HF_HOME=/hf`). The script uses `torch` and
`transformers`, both pinned in `envs/requirements_xla.txt`. The three JSONs of the
published campaign ship with the artifact under `results/bert_fold/`.

Repeat for `(1,128)` and `(8,128)`. Each JSON holds the five cold compilation
samples, 250 execution samples per variant, standard deviations, 95% CIs over the
process means, cache metadata and the generated artifacts.

### 5.2b Full grid and paper tables (one command each)

```bash
make grid TVM_HOME=/path/to/tvm COMPILE_REPEATS=5
make tables                                      # results/k5/tables/{table_*.tex, summary.md} from the JSONs
make fold_stats                                  # Welch plus Holm correction over the process means
make bert_audit                                  # BERT fold audit in the three shapes → results/bert_fold/
make bert_table                                  # docs/sblp/generated/table_bert_ln_fold.tex from the JSONs
make transformers_audit TVM_HOME=/path/to/tvm    # 21 BERT/GPT-2 K=5 cells
make transformers_tables                         # the two Transformer tables from the JSONs
```

`make tables` requires the complete 30-cell K=5 ResNet grid. `make transformers_tables`
requires the 21 K=5 Transformer cells and fails rather than filling a missing cell
with `--`. Together, the generators produce the paper tables **directly from the
JSONs**, into `results/k5/tables/` and `docs/sblp/generated/`. No cell is filled in
by hand.

The Transformer grid uses the BERT/GPT-2 base configurations with deterministic
random weights, so no download is needed. For GPT-2, the PyTorch paths receive a
fixed 4-D causal mask. Before the TVM import, the exported `aten._unsafe_view`
alias is normalized to the equivalent supported `aten.reshape`; shapes and values
do not change. Each of the 21 archived JSONs holds five cold compilations, five
process means, 250 synchronized samples and five IR paths.

To resume only some backends, use `RUN_INDUCTOR=0|1`, `RUN_XLA=0|1` and
`RUN_TVM=0|1`; JSONs that already contain the requested K are skipped. For example:

```bash
RUN_INDUCTOR=0 RUN_XLA=0 RUN_TVM=1 \
  make grid TVM_HOME=/path/to/tvm COMPILE_REPEATS=5
```

**Expected tolerances when reproducing** (same GPU class, RTX 3050):

| Metric | Typical tolerance | Note |
|---|---|---|
| `exec_ms`, batch > 1 or large inputs | ±5% | stable across days and re-runs |
| `exec_ms`, batch 1 | ±20% | sensitive to torch version × layout; see §8 |
| `compile_ms`, XLA and TVM | ±10% | |
| `compile_ms`, TorchInductor | depends on the **torch version** | torch 2.9 compiles about 1.3–2.6× slower than 2.5.1 on this GPU; compare only within the same version (the one pinned in `envs/requirements_xla.txt`) |
| Unit counts | exact | same torch/jax/TVM version ⇒ same artifacts |

### 5.3 Flags

- `--model resnet18|resnet50` · `--device cuda|cpu` · `--dtype fp32|bf16|fp16`
- `--batch`, `--height`, `--width` — NCHW shape of the input
- `--warmup`, `--iters` — measurement protocol
- `--compile-repeats K` — K cold compilations in independent processes (`K=5` in the paper's repeated ResNet measurements)
- `--compile-repeat-attempts` — maximum attempts to obtain each complete repetition; failures are discarded and recorded in the JSON
- `--compile-repeat-timeout-s` — per-process timeout
- `--no-xla`, `--no-inductor`, `--no-tvm` — enable/disable backends
- `--power-compile`, `--power-exec` — powers (W) used by the energy estimate
- `--compile-budget-ms` — penalizes backends above the compilation budget in the recommendation
- `--output file.json`

## 6) Checking that a collection is complete

After a run, check the JSON for:

1. **No errors**: there must be no `inductor_error`, `xla_error` or `tvm_error` under `raw`.
2. **Units counted**: in `raw.inductor.ir_dump.fused_inductor.kernel_count.summary`, `triton_launches_total` and `extern_kernels_total` must be **> 0** (ResNet-18 with torch 2.9: 18 Triton launches and 21 extern calls; ResNet-50: 50 and 54). If they come back as zero, see §7.
3. **`capture_method`**: must be `inductor_cache_files`. A `stdout_fallback` value indicates a degraded capture.
4. **Non-empty dumps** under `ir_dumps/` (the `*_inductor_output_code.py` of a ResNet-18 is hundreds of KB).
5. **A real GPU**: `make check_xla` confirms `torch.cuda.is_available()` and `jax.devices('gpu')`. If JAX falls back to CPU, the comparison is void.

The archived version must also include `results/k5/cache_audit.md`, with the raw
samples and the verification of the artifacts generated in each process. The
`make artifact_submission_check` target prevents a package without the complete
K=5 JSON grid from being treated as submission-ready.

## 7) ⚠️ TorchInductor cache: read this before measuring PyTorch

**Symptom.** An empty `*_inductor_output_code.py` under `ir_dumps/inductor/`, a
zeroed unit count (with a `warning` field in the JSON), and an Inductor
`compile_ms` far below the real value.

**Cause.** TorchInductor keeps a persistent compilation cache (the FX graph cache)
in `/tmp/torchinductor_$USER`, plus the Triton cache in `~/.triton/cache`. From the
**second** run onwards with the same model, shape and versions, Inductor reuses the
cache: code generation does not run, no new `output_code` is produced — so the
artifact needed to count units disappears — and the measured time becomes a *cache
hit*, not a compilation.

**How this repository handles it.**

1. `run_bench.py` and the `Makefile` set `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` by default (it must exist before `import torch`, which is why it sits at the top of `run_bench.py`).
2. The backend also forces `torch._inductor.config.force_disable_caches = True` at runtime and captures the generated code straight from the files Inductor produces (wrapper plus Triton kernels), without relying on `TORCH_LOGS`.
3. To guarantee a "cold" compilation (the number reported as `compile_ms`), **always** run this before each PyTorch measurement:

```bash
make clean_inductor_cache
# removes /tmp/torchinductor_$USER; K>1 uses per-process private caches
```

**If it still comes back empty** (for example, another torch version with a
different cache layout), the JSON records `kernel_count.warning` and
`capture_method: stdout_fallback`. In that case run `make clean_inductor_cache`,
confirm `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` in the environment and repeat the
measurement. This procedure is part of the artifact precisely so that evaluators
can reproduce the PyTorch unit counts. The deposited version carries about 130 MB
of optimized dumps and generated code, including the 105 referenced Transformer
IRs, **except** XLA's unoptimized HLO: it embeds the weights as literals (~195 MB
per dump, ~7 GB across the grid). `make cache_audit` knows about that exception in
the ResNet grid — it verifies and hashes the 200 archived artifacts and marks the
50 declared unoptimized HLOs as *declared but not archived*, using the
`code_size_bytes` recorded in the JSON. Any other missing expected artifact makes
the audit fail.

### 7.1) XLA unoptimized HLO: opt-in dumping

Because it is bulky, regenerable and never archived, **this file is not written to
disk by default**. The backend still computes the text in memory, so the JSON
records the `path`, the `code_size_bytes` and the unit summary as usual — exactly
what `make cache_audit` consumes to mark the row as *declared but not archived*.
Without this behaviour, every complete K=5 grid left about 7 GB behind, and
repeating the grid filled the disk.

To regenerate the files (for example, to inspect XLA's input HLO), turn the
variable on and run the cell you want:

```bash
CNNBENCH_DUMP_XLA_UNOPT_HLO=1 python run_bench.py --backends xla \
  --model resnet50 --batch 1 --height 224 --width 224
# budget ~195 MB per dump; run make prune afterwards
```

Writing happens **outside** the timed window (before the first compiled call), so
turning the variable on or off does not change `compile_ms`.

*Warm-cache measurement (optional).* To deliberately measure the time with a reused
cache, run with `TORCHINDUCTOR_FORCE_DISABLE_CACHES=0` **without** cleaning the
cache — and report it separately, since it is not comparable to the `compile_ms` of
the other backends.

## 8) Known limitations of the measurement

- **The Inductor and XLA `compile_ms` is a proxy** (`first call − mean latency`); TVM times the passes plus `relax.build` after the FX trace/import. The windows do not include the same frontend work, so the comparison is an operational one between native stacks, not an isonomic comparison of the compilers alone.
- **Unit counting is static** (analysis of the compilation artifact). It does not count real GPU launches; use a profiler (Nsight Systems, for example) separately for that.
- **`energy_j` is a parametric estimate**, not a measurement (it does not use NVML or RAPL).
- **TVM starts from a PyTorch FX trace** — the export fragments operators that PyTorch treats as atomic, which inflates TVM's unit count (discussed in the paper's *Evaluation* section).
- **Asymmetric numerical precision at execution**: torch (cuDNN) and XLA use TF32 on Ampere GPUs, whereas the CUDA kernels generated by TVM (dlight) are pure FP32 — a slight execution advantage for the first two.
- **JAX/cuDNN compatibility**: the cuDNN 9.10.2 installed in the Python environment produced intermittent `XlaRuntimeError: INTERNAL` at batch \(>1\) on this RTX 3050. The main campaign explicitly loads the host cuDNN 9.11.0 (`XLA_LIBRARY_PATH=/usr/local/lib64`) and records `cudnn_build_version` and `cudnn_runtime_version`; a ResNet-18 batch-64 control completed \(K=5\) with no retry using the default autotuner. The collector still records any failed attempt and never includes it in the statistics.
- **XLA plans on 1024×1024 inputs**: during autotuning, some candidate plans request workspaces larger than the available 8 GiB and are rejected with non-fatal allocator warnings; the five compilations of each cell still finish and select an executable plan. This may limit the best latency XLA can reach on this GPU.
- **Transformers use random weights and fixed shapes.** The BERT/GPT-2 grid is complete and repeated at K=5, but it measures the base configurations initialized with seed 0, not the accuracy of trained checkpoints. GPT-2 uses the 4-D causal mask and the export-alias normalization described in section 5.2b.
- **TorchInductor's `compile_ms` depends heavily on the torch version** (about 2× between 2.5.1 and 2.9 on this GPU). The artifact pins the version in `envs/requirements_xla.txt`; compilation numbers are comparable only within the same version.
- The smoke tests (batch 1, few iterations) exist to validate the environment, **not** to compare backends — use the full protocol (§5.2).

## 9) Reproducibility

```bash
# environment report (GPU, driver, versions) → env_reports/environment_report.json
. .venv_xla/bin/activate && python collect.py

# lock file for the XLA venv → env_reports/requirements_xla.lock.txt
make lock_xla
```

The required dependencies of the TVM environment are pinned in
`envs/requirements_tvm.txt`; `collect.py` produces a full report of the versions
actually loaded in each reproduction.

## 10) Cleanup

```bash
make clean_inductor_cache   # TorchInductor/Triton caches (see §7)
make clean                  # removes .venv_xla, .venv_tvm, out_*.json and locks
make prune_check            # reports what is regenerable, without deleting
make prune                  # removes the regenerable files from the working directory
make docker_disk            # how much Docker uses and how much can be reclaimed
make docker_reclaim         # discards the build cache plus dangling images
```

**Where the RAM goes.** Besides the frameworks themselves, the compiler caches of
every isolated process land under `$TMPDIR` (§5.2). On a host where `/tmp` is a
tmpfs, that is physical memory; redirect `TMPDIR` to a disk-backed path for the
full grids.

**Where the disk actually goes.** In a full reproduction the largest consumers are,
in order: (1) the Docker build cache and images — rebuilding TVM from source
accumulates tens of GB that Docker does not reclaim on its own, hence
`make docker_disk` / `make docker_reclaim`; (2) the image export,
`dist/cnnbench-artifact.tar.gz` (~13 GB, produced only by `make docker_export` and
disposable afterwards); (3) whatever `make prune` removes. XLA's unoptimized HLOs,
which used to add up to ~7 GB per grid, are no longer written by default (§7.1).

`prune` removes only what the grid recreates and the deposit does not archive:
XLA's unoptimized HLOs (§7.1, if you regenerated them), IR directories orphaned by
failed attempts, `torch_compile_debug/`, `artifacts/` and `__pycache__/`. It
**never** touches a result JSON, a generated table, an environment report or an IR
referenced by some JSON: before deleting, it recomputes the referenced set and
aborts if the removal would drop any cited file — except the unoptimized HLO, which
the JSONs cite but the package declares as *declared but not archived*. Run
`make prune_check` first to see what would go.

## 10.1) Paper figures

The five figures come from the same K=5 JSONs that feed the tables:

```bash
make figures    # equivalencia.png, envelope_exemplo.png, fusion_rate.png,
                # kernels_interno.png, kernels_externo.png
```

The target runs in the XLA venv because the scripts depend on matplotlib. As with
the tables, no paper figure is assembled by hand.

## 11) Docker artifact

The image contains the three backends and preserves the two separate Python
environments required by the incompatible PyTorch versions. TVM is built during the
image build from commit `3b60f1c9b8907dcf5d39a033876020e96e6915b2`. Host
prerequisites are the ones listed under [Requirements](#requirements); the GPU must
be passed to the container with `--gpus all`.

```bash
make docker_build             # cnnbench:artifact image
make docker_verify            # imports, GPU and fold equivalence
make docker_smoke             # real smoke run on Inductor, XLA and TVM
```

All outputs are written to the local `artifacts/` directory, mounted as
`/artifacts` inside the container. To run a single backend (on a host with SELinux
in `Enforcing`, add `--security-opt label=disable`):

```bash
mkdir -p artifacts
docker run --gpus all --rm \
  -v "$PWD/artifacts:/artifacts:Z" \
  cnnbench:artifact inductor \
  --model resnet18 --device cuda --dtype fp32 \
  --batch 1 --height 224 --width 224 \
  --warmup 10 --iters 50 --compile-repeats 5
```

Replace `inductor` with `xla` or `tvm`. The full grid is available with:

```bash
make docker_grid              # equivalent to `docker run ... cnnbench:artifact grid`
```

Inside the container there is no host cuDNN 9.11: the grid detects its absence,
proceeds with the environment's own cuDNN (9.10.2) and records
`cudnn_runtime_version` in every JSON. That reproduces the protocol, but the XLA
numbers may diverge from the published ones — see §8. To require the reference
library and abort without it, set `XLA_STRICT_CUDNN=1`.

### 11.1 Image: build it locally

The Zenodo record publishes **only the source package**; the image is not
distributed prebuilt. Build it from the `Dockerfile` included in the package:

```bash
make docker_build           # builds TVM from source: 45--120 minutes
make docker_verify          # continue with the normal script from here
```

The build produces the `cnnbench:artifact` tag used by every `docker_*` target.

If you want to move the built image to a machine without network access, export it
locally (needs about 10 GiB free) and load it on the other host:

```bash
make docker_export          # writes dist/cnnbench-artifact.tar.gz (~13 GB) + .sha256
sha256sum --check cnnbench-artifact.tar.gz.sha256
docker load < cnnbench-artifact.tar.gz
docker image ls cnnbench:artifact          # confirms the tag
```

## 12) Preparing the Zenodo deposit

Create the Zenodo draft first and use **Reserve DOI**. Do not invent the number,
and do not publish the draft yet. With the version-specific DOI reserved, finalize
the local files:

```bash
make set_artifact_doi DOI=10.5281/zenodo.NUMBER
make paper_pdf
make artifact_submission_check
make artifact_package
```

The last command builds and verifies, including after a clean extraction:

- `dist/cnnbench-source-v1.0.0.tar.gz`;
- `dist/cnnbench-source-v1.0.0.tar.gz.sha256`.

Upload those two files to the record; they are the complete deposit. The Docker
image is not deposited: the evaluator builds it with `make docker_build` from the
included `Dockerfile`. The source package omits only the regenerable unoptimized
XLA HLOs (about 7 GiB); the JSONs, the 105 referenced Transformer IRs, the
remaining IRs, the six tables and the PDF all stay in the archive.

In the Zenodo form, select type **Software**, access **Open**, version `1.0.0` and
license **MIT**; copy the title, authors, description and keywords from
`.zenodo.json`. Check that the DOI inside the PDF, the README and `CITATION.cff`
matches the reserved one — for this version, <https://doi.org/10.5281/zenodo.21731237> —
before pressing **Publish**. After publication, treat the record as immutable and
create a new Zenodo version for any later correction.
