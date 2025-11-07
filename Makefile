
# ========= Benchmark Envs (XLA+PyTorch together) and (TVM separate) =========
# Usage:
#   make xla_env         # creates .venv_xla and installs requirements_xla + CUDA-enabled PyTorch
#   make tvm_env         # creates .venv_tvm and installs requirements_tvm
#   make test_xla        # runs run_bench.py for XLA+Torch (no TVM)
#   make test_tvm        # runs run_bench.py for TVM (no XLA/Inductor)
#   make show_versions_xla / show_versions_tvm  # print key libs versions
#   make lock_xla / lock_tvm  # export exact resolved envs
#
# Notes:
#  - XLA (JAX) and PyTorch share the same env to ensure identical versions of Python/CUDA/cuDNN/etc.
#  - TVM runs in its own env (as requested), using the working combo you provided.
#  - The Makefile tries multiple CUDA channels for PyTorch (cu128→cu126→cu124→cu121).
#    You can override with: make TORCH_CU_CHOICES="cu126 cu124"
#
# Variables you may override:
#   PYTHON_BIN=python3.11        # interpreter for XLA+Torch env
#   PYTHON_TVM_BIN=python3.10    # interpreter for TVM env (defaults to PYTHON_BIN)
#   TORCH_VERSION=2.9.0
#   TORCHVISION_VERSION=0.24.0
#   TORCH_CU_CHOICES="cu128 cu126 cu124 cu121"

PYTHON_BIN ?= python3.11
PYTHON_TVM_BIN ?= $(PYTHON_BIN)
VENV_XLA := .venv_xla
VENV_TVM := .venv_tvm
REQ_XLA := envs/requirements_xla.txt
REQ_TVM := envs/requirements_tvm.txt

TORCH_VERSION ?= 2.9.0
TORCHVISION_VERSION ?= 0.24.0
TORCH_CU_CHOICES ?= cu128 cu126 cu124 cu121

.PHONY: all xla_env tvm_env test_xla test_tvm show_versions_xla show_versions_tvm lock_xla lock_tvm clean

all: xla_env tvm_env

# -------------------- XLA + PyTorch (same env) --------------------
xla_env: $(VENV_XLA)/bin/python

$(VENV_XLA)/bin/python: $(REQ_XLA)
	$(PYTHON_BIN) -m venv $(VENV_XLA)
	. $(VENV_XLA)/bin/activate && pip install -U pip wheel setuptools
	. $(VENV_XLA)/bin/activate && pip install -r $(REQ_XLA)
	@set -e; \
	for CU in $(TORCH_CU_CHOICES); do \
	  echo ">> Trying PyTorch CUDA channel $$CU"; \
	  if . $(VENV_XLA)/bin/activate && pip install --index-url https://download.pytorch.org/whl/$$CU torch==$(TORCH_VERSION) torchvision==$(TORCHVISION_VERSION); then \
	    echo ">> Installed PyTorch $(TORCH_VERSION) / torchvision $(TORCHVISION_VERSION) from $$CU"; \
	    break; \
	  fi; \
	done
	# Sanity checks (GPU presence for both frameworks)
	. $(VENV_XLA)/bin/activate && python - <<'PY'
import torch, jax
assert torch.cuda.is_available(), "PyTorch não encontrou CUDA"
print("PyTorch OK | CUDA:", torch.version.cuda, "| cuDNN:", torch.backends.cudnn.version())
d = jax.devices('gpu')
assert d, "JAX não encontrou GPU"
print("JAX OK | GPUs:", d)
PY

# -------------------- TVM (separate env) --------------------
tvm_env: $(VENV_TVM)/bin/python

$(VENV_TVM)/bin/python: $(REQ_TVM)
	$(PYTHON_TVM_BIN) -m venv $(VENV_TVM)
	. $(VENV_TVM)/bin/activate && pip install -U pip wheel setuptools
	. $(VENV_TVM)/bin/activate && pip install -r $(REQ_TVM)

# -------------------- Quick smoke tests --------------------
test_xla: xla_env
	. $(VENV_XLA)/bin/activate && python run_bench.py --no-tvm --device cuda --model resnet18 --output out_xla_torch_resnet18.json
	. $(VENV_XLA)/bin/activate && python run_bench.py --no-tvm --device cuda --model resnet50 --output out_xla_torch_resnet50.json

test_tvm: tvm_env
	. $(VENV_TVM)/bin/activate && python run_bench.py --no-xla --no-inductor --device cuda --model resnet18 --output out_tvm_resnet18.json
	. $(VENV_TVM)/bin/activate && python run_bench.py --no-xla --no-inductor --device cuda --model resnet50 --output out_tvm_resnet50.json

# -------------------- Diagnostics --------------------
show_versions_xla: xla_env
	. $(VENV_XLA)/bin/activate && python - <<'PY'
import importlib.metadata as md, platform, torch
print("python", platform.python_version())
for p in ("torch","torchvision","jax","jaxlib","jax-cuda12-pjrt","nvidia-cublas-cu12","nvidia-cudnn-cu12","nvidia-cusparse-cu12","nvidia-nvjitlink-cu12"):
    try: print(p, md.version(p))
    except: print(p, "N/A")
print("torch.cuda", torch.version.cuda)
print("cudnn_version", torch.backends.cudnn.version())
PY

show_versions_tvm: tvm_env
	. $(VENV_TVM)/bin/activate && python - <<'PY'
import importlib.metadata as md, platform
print("python", platform.python_version())
for p in ("apache-tvm","torch","torchvision","nvidia-cublas-cu12","nvidia-cudnn-cu12"):
    try: print(p, md.version(p))
    except: print(p, "N/A")
PY

# -------------------- Lock exact envs --------------------
lock_xla: xla_env
	. $(VENV_XLA)/bin/activate && pip freeze > envs/requirements_xla.lock.txt
	@echo "Wrote requirements_xla.lock.txt"

lock_tvm: tvm_env
	. $(VENV_TVM)/bin/activate && pip freeze > envs/requirements_tvm.lock.txt
	@echo "Wrote requirements_tvm.lock.txt"

clean:
	rm -rf $(VENV_XLA) $(VENV_TVM) out_*.json requirements_xla.lock.txt requirements_tvm.lock.txt
