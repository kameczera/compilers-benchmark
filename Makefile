# ========= CNN Compilers Benchmark: envs + smoke tests =========
# Quick commands:
#   make conda_xla_env CONDA_ENV_XLA=teste_xla
#   make conda_check_xla CONDA_ENV_XLA=teste_xla
#   make conda_smoke_xla_torch CONDA_ENV_XLA=teste_xla
#
# The JAX/XLA memory exports below are intentionally enabled by default.
# They reduce GPU memory preallocation and avoid failures when JAX and PyTorch
# are tested in the same machine/session.

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

# -------------------- Files/env names --------------------
PYTHON_VERSION ?= 3.10.18
# Usa python3.10 se existir; senão cai para python3.11/python3.
PYTHON_BIN ?= $(shell command -v python3.10 || command -v python3.11 || command -v python3)
PYTHON_TVM_BIN ?= $(PYTHON_BIN)

CONDA ?= conda
CONDA_ENV_XLA ?= tcc_xla
CONDA_ENV_TVM ?= tcc_tvm

VENV_XLA ?= .venv_xla
VENV_TVM ?= .venv_tvm

REQ_XLA ?= envs/requirements_xla.txt
REQ_TVM ?= envs/requirements_tvm.txt

# Checkout do TVM compilado do fonte (0.22.dev0, commit 3b60f1c).
# O pacote pip `apache-tvm` (0.14) NAO serve: nao tem o modulo relax.
TVM_HOME ?=

# -------------------- JAX/XLA GPU memory policy --------------------
# Must be visible before the first `import jax` in each Python process.
XLA_PYTHON_CLIENT_PREALLOCATE ?= false
XLA_PYTHON_CLIENT_MEM_FRACTION ?= 0.40
export XLA_PYTHON_CLIENT_PREALLOCATE
export XLA_PYTHON_CLIENT_MEM_FRACTION

# -------------------- TorchInductor cache policy --------------------
# Sem isso o Inductor reaproveita o cache persistente (/tmp/torchinductor_$USER):
# o codegen nao roda, o dump do output_code sai vazio (kernels = 0) e o
# compile_ms fica subestimado. run_bench.py tambem seta este default.
TORCHINDUCTOR_FORCE_DISABLE_CACHES ?= 1
export TORCHINDUCTOR_FORCE_DISABLE_CACHES

# Small default smoke-test shape to make validation fast.
BENCH_QUICK_ARGS ?= --device cuda --model resnet18 --dtype fp32 --batch 1 --height 224 --width 224 --warmup 3 --iters 5

.PHONY: help \
        conda_xla_env conda_check_xla conda_smoke_xla conda_smoke_torch conda_smoke_xla_torch conda_clean_xla \
        xla_env check_xla smoke_xla smoke_torch smoke_xla_torch \
        tvm_env smoke_tvm show_versions_xla show_versions_conda_xla lock_xla \
        clean_inductor_cache clean

help:
	@echo "CNN Compilers Benchmark"
	@echo ""
	@echo "Venv workflow (recomendado, ver README):"
	@echo "  make xla_env"
	@echo "  make check_xla"
	@echo "  make smoke_torch / smoke_xla / smoke_xla_torch"
	@echo "  make tvm_env && make smoke_tvm TVM_HOME=/caminho/para/tvm"
	@echo "  make clean_inductor_cache   # antes de cada medicao do PyTorch!"
	@echo ""
	@echo "Conda workflow alternativo:"
	@echo "  make conda_xla_env CONDA_ENV_XLA=teste_xla"
	@echo "  make conda_check_xla CONDA_ENV_XLA=teste_xla"
	@echo "  make conda_smoke_xla_torch CONDA_ENV_XLA=teste_xla"
	@echo ""
	@echo "Variáveis úteis:"
	@echo "  REQ_XLA=$(REQ_XLA)"
	@echo "  CONDA_ENV_XLA=$(CONDA_ENV_XLA)"
	@echo "  XLA_PYTHON_CLIENT_PREALLOCATE=$(XLA_PYTHON_CLIENT_PREALLOCATE)"
	@echo "  XLA_PYTHON_CLIENT_MEM_FRACTION=$(XLA_PYTHON_CLIENT_MEM_FRACTION)"
	@echo "  BENCH_QUICK_ARGS=$(BENCH_QUICK_ARGS)"

# -------------------- Conda: XLA + PyTorch --------------------
conda_xla_env:
	$(CONDA) create -n $(CONDA_ENV_XLA) python=$(PYTHON_VERSION) pip -y
	$(CONDA) run -n $(CONDA_ENV_XLA) python -m pip install --upgrade pip wheel setuptools
	$(CONDA) run -n $(CONDA_ENV_XLA) python -m pip install -r $(REQ_XLA)
	@echo "[OK] Conda env criado: $(CONDA_ENV_XLA)"
	@echo "Ative com: conda activate $(CONDA_ENV_XLA)"

conda_check_xla:
	$(CONDA) run -n $(CONDA_ENV_XLA) python -m pip check
	$(CONDA) run -n $(CONDA_ENV_XLA) python -c "import os, torch, jax; print('XLA_PYTHON_CLIENT_PREALLOCATE=', os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE')); print('XLA_PYTHON_CLIENT_MEM_FRACTION=', os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION')); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cuda_available', torch.cuda.is_available(), 'cudnn', torch.backends.cudnn.version()); assert torch.cuda.is_available(), 'PyTorch nao encontrou CUDA'; print('jax', jax.__version__, 'backend', jax.default_backend(), 'devices', jax.devices()); assert jax.devices('gpu'), 'JAX nao encontrou GPU'"

conda_smoke_xla:
	$(CONDA) run -n $(CONDA_ENV_XLA) python run_bench.py $(BENCH_QUICK_ARGS) --no-tvm --no-inductor --output out_smoke_xla.json
	@echo "[OK] Gerado: out_smoke_xla.json"

conda_smoke_torch:
	$(CONDA) run -n $(CONDA_ENV_XLA) python run_bench.py $(BENCH_QUICK_ARGS) --no-tvm --no-xla --output out_smoke_torch.json
	@echo "[OK] Gerado: out_smoke_torch.json"

conda_smoke_xla_torch:
	$(CONDA) run -n $(CONDA_ENV_XLA) python run_bench.py $(BENCH_QUICK_ARGS) --no-tvm --output out_smoke_xla_torch.json
	@echo "[OK] Gerado: out_smoke_xla_torch.json"

conda_clean_xla:
	$(CONDA) env remove -n $(CONDA_ENV_XLA) -y

# -------------------- venv alternative: XLA + PyTorch --------------------
xla_env: $(VENV_XLA)/bin/python

$(VENV_XLA)/bin/python: $(REQ_XLA)
	$(PYTHON_BIN) -m venv $(VENV_XLA)
	. $(VENV_XLA)/bin/activate && python -m pip install --upgrade pip wheel setuptools
	. $(VENV_XLA)/bin/activate && python -m pip install -r $(REQ_XLA)
	@echo "[OK] venv criado: $(VENV_XLA)"

check_xla: xla_env
	. $(VENV_XLA)/bin/activate && python -m pip check
	. $(VENV_XLA)/bin/activate && python -c "import os, torch, jax; print('XLA_PYTHON_CLIENT_PREALLOCATE=', os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE')); print('XLA_PYTHON_CLIENT_MEM_FRACTION=', os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION')); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cuda_available', torch.cuda.is_available(), 'cudnn', torch.backends.cudnn.version()); assert torch.cuda.is_available(), 'PyTorch nao encontrou CUDA'; print('jax', jax.__version__, 'backend', jax.default_backend(), 'devices', jax.devices()); assert jax.devices('gpu'), 'JAX nao encontrou GPU'"

smoke_xla: xla_env
	. $(VENV_XLA)/bin/activate && python run_bench.py $(BENCH_QUICK_ARGS) --no-tvm --no-inductor --output out_smoke_xla.json
	@echo "[OK] Gerado: out_smoke_xla.json"

smoke_torch: xla_env
	. $(VENV_XLA)/bin/activate && python run_bench.py $(BENCH_QUICK_ARGS) --no-tvm --no-xla --output out_smoke_torch.json
	@echo "[OK] Gerado: out_smoke_torch.json"

smoke_xla_torch: xla_env
	. $(VENV_XLA)/bin/activate && python run_bench.py $(BENCH_QUICK_ARGS) --no-tvm --output out_smoke_xla_torch.json
	@echo "[OK] Gerado: out_smoke_xla_torch.json"

# -------------------- TVM, optional/separate --------------------
tvm_env: $(VENV_TVM)/bin/python

$(VENV_TVM)/bin/python: $(REQ_TVM)
	$(PYTHON_TVM_BIN) -m venv $(VENV_TVM)
	. $(VENV_TVM)/bin/activate && python -m pip install --upgrade pip wheel setuptools
	. $(VENV_TVM)/bin/activate && python -m pip install -r $(REQ_TVM)
	@echo "[OK] venv TVM criado: $(VENV_TVM)"

smoke_tvm: tvm_env
	@test -n "$(TVM_HOME)" || { echo "ERRO: defina TVM_HOME (checkout do TVM compilado do fonte), ex.: make smoke_tvm TVM_HOME=$$HOME/tvm"; exit 1; }
	. $(VENV_TVM)/bin/activate && PYTHONPATH=$(TVM_HOME)/python:$(TVM_HOME)/build:$$PYTHONPATH python run_bench.py $(BENCH_QUICK_ARGS) --no-xla --no-inductor --output out_smoke_tvm.json
	@echo "[OK] Gerado: out_smoke_tvm.json"

# -------------------- Diagnostics/locks --------------------
show_versions_xla: xla_env
	. $(VENV_XLA)/bin/activate && python -c "import importlib.metadata as md, platform, torch, jax; print('python', platform.python_version()); print('torch', torch.__version__, 'torch_cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version()); print('jax', jax.__version__, 'backend', jax.default_backend(), 'devices', jax.devices()); [print(p, md.version(p)) for p in ('jaxlib','jax-cuda12-pjrt','jax-cuda12-plugin','nvidia-cublas-cu12','nvidia-cudnn-cu12','triton') if md.version(p)]"

show_versions_conda_xla:
	$(CONDA) run -n $(CONDA_ENV_XLA) python -c "import importlib.metadata as md, platform, torch, jax; print('python', platform.python_version()); print('torch', torch.__version__, 'torch_cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version()); print('jax', jax.__version__, 'backend', jax.default_backend(), 'devices', jax.devices()); [print(p, md.version(p)) for p in ('jaxlib','jax-cuda12-pjrt','jax-cuda12-plugin','nvidia-cublas-cu12','nvidia-cudnn-cu12','triton') if md.version(p)]"

lock_xla: xla_env
	mkdir -p env_reports
	. $(VENV_XLA)/bin/activate && python -m pip freeze > env_reports/requirements_xla.lock.txt
	@echo "[OK] Wrote env_reports/requirements_xla.lock.txt"

# Remove os caches persistentes do TorchInductor/Triton. Rode antes de cada
# medicao de compile_ms/kernels do PyTorch para garantir codegen "frio".
clean_inductor_cache:
	rm -rf /tmp/torchinductor_$$USER
	rm -rf $${XDG_CACHE_HOME:-$$HOME/.cache}/torch/inductor
	rm -rf $$HOME/.triton/cache
	@echo "[OK] Caches do TorchInductor/Triton removidos"

clean:
	rm -rf $(VENV_XLA) $(VENV_TVM) out_*.json env_reports/requirements_xla.lock.txt
