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
XLA_PYTHON_CLIENT_MEM_FRACTION ?= 0.75
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
        artifact_check artifact_submission_check artifact_package set_artifact_doi check_fold grid tables fold_stats cache_audit \
        bert_audit bert_table transformers_audit transformers_tables \
        paper_tables paper_image paper_pdf \
        docker_build docker_verify docker_smoke docker_grid docker_export docker_verify_export \
        clean_inductor_cache clean prune prune_check figures

# Diretório dos JSONs medidos (entrada do make tables / saída do make grid)
RESULTS_DIR ?= results/k5
BERT_RESULTS_DIR ?= results/bert_fold
TRANSFORMER_RESULTS_DIR ?= results/transformers
PAPER_TABLE_DIR ?= docs/sblp/generated
COMPILE_REPEATS ?= 5
RUN_INDUCTOR ?= 1
RUN_XLA ?= 1
RUN_TVM ?= 1
DOCKER_IMAGE ?= cnnbench:artifact
DOCKER_ARTIFACTS ?= $(CURDIR)/artifacts
DOCKER_EXPORT ?= dist/cnnbench-artifact.tar.gz
PAPER_IMAGE ?= cnnbench-paper:artifact
ARTIFACT_ARCHIVE ?= dist/cnnbench-source-v1.0.0.tar.gz

# Em hosts com SELinux em Enforcing (Fedora/RHEL/CentOS) o rotulo padrao do
# contêiner bloqueia /dev/nvidia*, e a GPU aparece como
# "Failed to initialize NVML: Insufficient Permissions" mesmo com --gpus all.
# Detectado automaticamente; sobrescreva com DOCKER_SECURITY_OPTS= para desligar.
DOCKER_SECURITY_OPTS ?= $(shell if command -v getenforce >/dev/null 2>&1 && \
	[ "$$(getenforce 2>/dev/null)" = "Enforcing" ]; then \
	echo --security-opt label=disable; fi)

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
	@echo "Reproducao do artigo (dado -> tabela, sem transcricao manual):"
	@echo "  make artifact_check                           # preflight portátil"
	@echo "  make artifact_submission_check                # DOI + PDF + K=5 + tabela"
	@echo "  make set_artifact_doi DOI=10.5281/zenodo.N   # insere o DOI reservado"
	@echo "  make paper_pdf                               # regenera tabelas + PDF em contêiner"
	@echo "  make artifact_package                        # pacote Zenodo + SHA-256 + validação limpa"
	@echo "  make check_fold                              # valida o fold BN->Conv (CPU)"
	@echo "  make grid TVM_HOME=/caminho/para/tvm         # K=$(COMPILE_REPEATS) compilações frias -> $(RESULTS_DIR)"
	@echo "  make tables                                  # gera tabelas LaTeX/MD de $(RESULTS_DIR)"
	@echo "  make fold_stats                              # Welch + Holm do fold ResNet"
	@echo "  make bert_audit                              # auditoria do fold no BERT (K=$(COMPILE_REPEATS))"
	@echo "  make bert_table                              # tabela do BERT a partir de $(BERT_RESULTS_DIR)"
	@echo "  make transformers_audit                      # BERT/GPT-2 nos tres backends (K=$(COMPILE_REPEATS))"
	@echo "  make transformers_tables                     # tabelas BERT/GPT-2 dos JSONs"
	@echo "  make cache_audit                            # audita caches + dumps K=5"
	@echo "  make figures                                 # JSONs -> as cinco figuras do artigo"
	@echo ""
	@echo "Limpeza:"
	@echo "  make prune_check                             # relata o regeneravel, sem apagar"
	@echo "  make prune                                   # remove o regeneravel (~7 GB)"
	@echo ""
	@echo "Contêiner reproduzível:"
	@echo "  make docker_build                            # cria $(DOCKER_IMAGE)"
	@echo "  make docker_verify                           # valida imagem + GPU"
	@echo "  make docker_smoke                            # smoke nos três backends"
	@echo "  make docker_grid                             # grade completa no contêiner"
	@echo "  make docker_export                           # exporta $(DOCKER_EXPORT) + checksum"
	@echo "  make docker_verify_export                    # confere o checksum do arquivo baixado"
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

# -------------------- Reproducao do artigo --------------------
# Sem dependências Python externas: valida pacote, paper e sintaxe dos fontes.
artifact_check:
	python3 scripts/validate_artifact.py

# Checklist estrito para executar sobre uma extração limpa da versão que será
# depositada no repositório persistente.
artifact_submission_check:
	python3 scripts/validate_artifact.py --submission

# Valida o fold BN->Conv (sem BN residual + equivalencia numerica). Roda em CPU.
check_fold: xla_env
	. $(VENV_XLA)/bin/activate && python scripts/check_fold.py

# Grade completa do artigo (warmup 10, iters 50): três backends,
# ResNet-18/50 x 5 shapes. Grava <backend>_<modelo>_<shape>.json em RESULTS_DIR.
grid: xla_env tvm_env
	@test -n "$(TVM_HOME)" || { echo "ERRO: defina TVM_HOME, ex.: make grid TVM_HOME=$$HOME/tvm"; exit 1; }
	VENV_XLA=$(VENV_XLA) VENV_TVM=$(VENV_TVM) TVM_HOME=$(TVM_HOME) \
		COMPILE_REPEATS=$(COMPILE_REPEATS) RUN_INDUCTOR=$(RUN_INDUCTOR) \
		RUN_XLA=$(RUN_XLA) RUN_TVM=$(RUN_TVM) \
		scripts/run_full_grid.sh $(RESULTS_DIR)

# Auditoria de aplicabilidade do fold no BERT (K=5 processos frios por forma).
# Exige o checkpoint bert-base-uncased ja no cache do Hugging Face — ver README §5.2a.
bert_audit: xla_env
	. $(VENV_XLA)/bin/activate && python scripts/benchmark_bert_fold.py --batch 1 --seq-len 64 \
		--repeats $(COMPILE_REPEATS) --warmup 10 --iters 50 \
		--output $(BERT_RESULTS_DIR)/bert_1x64_k5.json
	. $(VENV_XLA)/bin/activate && python scripts/benchmark_bert_fold.py --batch 1 --seq-len 128 \
		--repeats $(COMPILE_REPEATS) --warmup 10 --iters 50 \
		--output $(BERT_RESULTS_DIR)/bert_1x128_k5.json
	. $(VENV_XLA)/bin/activate && python scripts/benchmark_bert_fold.py --batch 8 --seq-len 128 \
		--repeats $(COMPILE_REPEATS) --warmup 10 --iters 50 \
		--output $(BERT_RESULTS_DIR)/bert_8x128_k5.json

# Comparacao BERT/GPT-2 entre os tres backends (K=5 processos frios por celula).
transformers_audit: xla_env tvm_env
	VENV_XLA=$(VENV_XLA) VENV_TVM=$(VENV_TVM) TVM_HOME=$(TVM_HOME) \
		COMPILE_REPEATS=$(COMPILE_REPEATS) RUN_INDUCTOR=$(RUN_INDUCTOR) \
		RUN_XLA=$(RUN_XLA) RUN_TVM=$(RUN_TVM) \
		scripts/run_transformer_grid.sh $(TRANSFORMER_RESULTS_DIR)

# Tabelas de BERT e GPT-2 geradas dos JSONs
transformers_tables:
	python3 scripts/make_transformer_tables.py --results-dir $(TRANSFORMER_RESULTS_DIR) --out-dir $(PAPER_TABLE_DIR)

# Tabela do BERT gerada dos JSONs (consumida por \input{} em main.tex)
bert_table:
	python3 scripts/make_bert_table.py --results-dir $(BERT_RESULTS_DIR) --out-dir $(PAPER_TABLE_DIR)

# Tabelas do artigo geradas dos JSONs (LaTeX + summary.md com crossover e kernels)
tables:
	python3 scripts/make_tables.py --results-dir $(RESULTS_DIR) --out-dir $(RESULTS_DIR)/tables --require-complete-k5
	python3 scripts/make_tables.py --results-dir $(RESULTS_DIR) --out-dir $(PAPER_TABLE_DIR) --require-complete-k5

# Fonte de verdade do camera-ready: todos os seis arquivos consumidos via
# \input{} são regenerados diretamente dos JSONs K=5.
paper_tables: tables bert_table transformers_tables

# Ambiente LaTeX separado do contêiner GPU; o digest e os pacotes estão
# documentados em docker/Dockerfile.paper.
paper_image:
	docker build --file docker/Dockerfile.paper --tag $(PAPER_IMAGE) .

paper_pdf: paper_tables paper_image
	docker run --rm $(DOCKER_SECURITY_OPTS) \
		--user "$$(id -u):$$(id -g)" --env HOME=/tmp \
		--volume "$(CURDIR):/workspace" \
		--workdir /workspace/docs/sblp \
		$(PAPER_IMAGE) latexmk -g -pdf -interaction=nonstopmode -halt-on-error main.tex
	@echo "[OK] Gerado: docs/sblp/main.pdf"

# O DOI deve ser reservado no rascunho do Zenodo antes deste comando.
set_artifact_doi:
	@test -n "$(DOI)" || { echo "ERRO: use make set_artifact_doi DOI=10.5281/zenodo.NUMERO"; exit 1; }
	python3 scripts/set_artifact_doi.py "$(DOI)"

# Inclui somente arquivos de release, omite HLOs não otimizados regeneráveis
# (~7 GiB), valida a área de staging e repete a validação após extrair o tar.
artifact_package: artifact_submission_check
	bash scripts/package_artifact.sh "$(ARTIFACT_ARCHIVE)"

fold_stats: xla_env
	. $(VENV_XLA)/bin/activate && python scripts/analyze_fold_stats.py \
		--results-dir $(RESULTS_DIR) \
		--output $(RESULTS_DIR)/tables/fold_welch_holm.json

# Reproduz a evidência de processos frios e verifica todos os dumps registrados.
cache_audit:
	python3 scripts/make_cache_audit.py \
		--results-dir $(RESULTS_DIR) \
		--output $(RESULTS_DIR)/cache_audit.md

# -------------------- Docker artifact --------------------
docker_build:
	docker build --tag $(DOCKER_IMAGE) .

docker_verify:
	mkdir -p $(DOCKER_ARTIFACTS)
	docker run --gpus all --rm $(DOCKER_SECURITY_OPTS) \
		--volume $(DOCKER_ARTIFACTS):/artifacts:Z \
		$(DOCKER_IMAGE) verify

docker_smoke:
	mkdir -p $(DOCKER_ARTIFACTS)
	docker run --gpus all --rm $(DOCKER_SECURITY_OPTS) \
		--volume $(DOCKER_ARTIFACTS):/artifacts:Z \
		$(DOCKER_IMAGE) smoke

# Grade completa dentro do contêiner (varias horas).
docker_grid:
	mkdir -p $(DOCKER_ARTIFACTS)
	docker run --gpus all --rm $(DOCKER_SECURITY_OPTS) \
		--volume $(DOCKER_ARTIFACTS):/artifacts:Z \
		$(DOCKER_IMAGE) grid

# Exporta a imagem para depósito como arquivo separado no mesmo registro do
# Zenodo, permitindo ao avaliador pular o build (45--120 min) com `docker load`.
# Requer espaço livre da ordem do tamanho comprimido da imagem (~10 GiB).
docker_export:
	mkdir -p $(dir $(DOCKER_EXPORT))
	docker image inspect $(DOCKER_IMAGE) >/dev/null
	docker save $(DOCKER_IMAGE) | gzip -1 > $(DOCKER_EXPORT)
	cd $(dir $(DOCKER_EXPORT)) && sha256sum $(notdir $(DOCKER_EXPORT)) > $(notdir $(DOCKER_EXPORT)).sha256
	@echo "[OK] Imagem exportada: $(DOCKER_EXPORT)"
	@echo "[OK] Checksum: $(DOCKER_EXPORT).sha256"
	@ls -lh $(DOCKER_EXPORT)

# Confere que um arquivo de imagem baixado corresponde ao checksum publicado.
docker_verify_export:
	cd $(dir $(DOCKER_EXPORT)) && sha256sum --check $(notdir $(DOCKER_EXPORT)).sha256
	@echo "[OK] Arquivo da imagem íntegro"

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
	@cache_path="/tmp/torchinductor_$$USER"; \
	case "$$cache_path" in /tmp/torchinductor_?*) rm -rf -- "$$cache_path" ;; \
	*) echo "ERRO: caminho de cache inseguro: $$cache_path"; exit 1 ;; esac
	@echo "[OK] Cache temporário do TorchInductor removido; K>1 usa caches privados"

clean:
	rm -rf $(VENV_XLA) $(VENV_TVM) out_*.json env_reports/requirements_xla.lock.txt

# Remove o que a grade regenera e o deposito nao arquiva. Nao toca em JSON de
# resultado, tabela gerada nem IR referenciado por algum JSON: `prune` roda
# `--check` antes de apagar e aborta se algo referenciado sumiria.
#   - HLO nao-otimizado do XLA (~7 GB; regeneravel, ver README secao 7)
#   - torch_compile_debug/, artifacts/, __pycache__/
#   - diretorios de IR orfaos, de tentativas de campanha que falharam
prune:
	python3 scripts/prune_regenerable.py --apply

prune_check:
	python3 scripts/prune_regenerable.py --check

# As cinco figuras do artigo saem dos mesmos JSONs K=5 que as tabelas. Rodam no
# venv XLA porque dependem de matplotlib, ausente no Python do sistema.
figures: xla_env
	. $(VENV_XLA)/bin/activate && python scripts/plot_folds_en.py --results-dir $(RESULTS_DIR)
	. $(VENV_XLA)/bin/activate && python scripts/plot_ir_figs_en.py --results-dir $(RESULTS_DIR)
