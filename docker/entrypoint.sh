#!/usr/bin/env bash
set -Eeuo pipefail

APP_DIR="${CNNBENCH_APP_DIR:-/opt/cnnbench}"
ARTIFACT_DIR="${CNNBENCH_ARTIFACT_DIR:-/artifacts}"
XLA_PYTHON="/opt/venvs/xla/bin/python"
TVM_PYTHON="/opt/venvs/tvm/bin/python"

mkdir -p \
  "$ARTIFACT_DIR/results" \
  "$ARTIFACT_DIR/plots" \
  "$ARTIFACT_DIR/ir_dumps" \
  "$ARTIFACT_DIR/env_reports" \
  "$ARTIFACT_DIR/tables"

# Processos filhos usam APP_DIR como cwd. Estes links garantem que inclusive
# dumps e gráficos desses processos sejam persistidos no volume do usuário.
for output_dir in plots ir_dumps env_reports; do
  if [[ ! -e "$APP_DIR/$output_dir" ]]; then
    ln -s "$ARTIFACT_DIR/$output_dir" "$APP_DIR/$output_dir"
  fi
done

has_output_arg() {
  local arg
  for arg in "$@"; do
    if [[ "$arg" == "--output" || "$arg" == --output=* ]]; then
      return 0
    fi
  done
  return 1
}

run_backend() {
  local backend="$1"
  local python_bin
  local -a disabled_flags
  shift

  case "$backend" in
    inductor)
      python_bin="$XLA_PYTHON"
      disabled_flags=(--no-xla --no-tvm)
      ;;
    xla)
      python_bin="$XLA_PYTHON"
      disabled_flags=(--no-inductor --no-tvm)
      ;;
    tvm)
      python_bin="$TVM_PYTHON"
      disabled_flags=(--no-inductor --no-xla)
      ;;
    *)
      echo "Backend desconhecido: $backend" >&2
      return 2
      ;;
  esac

  local -a user_args=("$@")
  if ! has_output_arg "${user_args[@]}"; then
    user_args+=(--output "$ARTIFACT_DIR/results/${backend}.json")
  fi

  cd "$ARTIFACT_DIR"
  "$python_bin" "$APP_DIR/run_bench.py" \
    "${disabled_flags[@]}" \
    "${user_args[@]}"
}

verify() {
  "$XLA_PYTHON" "$APP_DIR/scripts/validate_artifact.py"
  command -v nvidia-smi >/dev/null
  nvidia-smi --query-gpu=name,driver_version,memory.total \
    --format=csv,noheader

  "$XLA_PYTHON" - <<'PY'
import jax
import torch

assert torch.cuda.is_available(), "PyTorch não encontrou uma GPU CUDA"
assert jax.devices("gpu"), "JAX não encontrou uma GPU CUDA"
print("torch", torch.__version__, torch.cuda.get_device_name(0))
print("jax", jax.__version__, jax.devices("gpu"))
PY

  "$TVM_PYTHON" - <<'PY'
import tvm

assert tvm.cuda(0).exist, "TVM não encontrou uma GPU CUDA"
print("tvm", tvm.__version__, tvm.cuda(0))
PY

  "$XLA_PYTHON" "$APP_DIR/scripts/check_fold.py"
  echo "[OK] Imagem e GPU validadas"
}

smoke() {
  local -a quick_args=(
    --device cuda
    --model resnet18
    --dtype fp32
    --batch 1
    --height 224
    --width 224
    --warmup 1
    --iters 2
    --compile-repeats 1
  )

  verify
  run_backend inductor "${quick_args[@]}" \
    --output "$ARTIFACT_DIR/results/smoke_inductor.json"
  run_backend xla "${quick_args[@]}" \
    --output "$ARTIFACT_DIR/results/smoke_xla.json"
  run_backend tvm "${quick_args[@]}" \
    --output "$ARTIFACT_DIR/results/smoke_tvm.json"
  echo "[OK] Smoke dos três backends concluído em $ARTIFACT_DIR"
}

usage() {
  cat <<'EOF'
CNN Compilers Benchmark

Uso:
  docker run --gpus all --rm -v "$PWD/artifacts:/artifacts:Z" IMAGE COMANDO [ARGS]

Comandos:
  verify      valida GPU, imports Torch/JAX/TVM e o fold BN->Conv
  artifact-check executa o preflight portátil do pacote e do paper
  smoke       executa um smoke curto nos três backends
  inductor    executa somente TorchInductor; aceita flags do run_bench.py
  xla         executa somente XLA; aceita flags do run_bench.py
  tvm         executa somente TVM; aceita flags do run_bench.py
  grid        executa a grade completa; saída em /artifacts/results/k5
  transformers executa a comparação BERT/GPT-2; saída em /artifacts/results/transformers
  tables      gera tabelas a partir de /artifacts/results/k5
  collect     grava o relatório do ambiente em /artifacts/env_reports
  help        mostra esta ajuda
  shell       abre um shell dentro do contêiner

Exemplo:
  docker run --gpus all --rm -v "$PWD/artifacts:/artifacts:Z" \
    cnnbench:artifact inductor --model resnet18 --warmup 10 --iters 50
EOF
}

command_name="${1:-help}"
if (( $# > 0 )); then
  shift
fi

case "$command_name" in
  artifact-check)
    exec "$XLA_PYTHON" "$APP_DIR/scripts/validate_artifact.py"
    ;;
  verify)
    verify
    ;;
  smoke)
    smoke
    ;;
  inductor|xla|tvm)
    run_backend "$command_name" "$@"
    ;;
  grid)
    cd "$APP_DIR"
    VENV_XLA=/opt/venvs/xla \
    VENV_TVM=/opt/venvs/tvm \
    TVM_HOME=/opt/tvm \
      exec scripts/run_full_grid.sh "$ARTIFACT_DIR/results/k5"
    ;;
  transformers)
    cd "$APP_DIR"
    VENV_XLA=/opt/venvs/xla \
    VENV_TVM=/opt/venvs/tvm \
    TVM_HOME=/opt/tvm \
      exec scripts/run_transformer_grid.sh "$ARTIFACT_DIR/results/transformers"
    ;;
  tables)
    exec "$XLA_PYTHON" "$APP_DIR/scripts/make_tables.py" \
      --results-dir "$ARTIFACT_DIR/results/k5" \
      --out-dir "$ARTIFACT_DIR/tables" \
      "$@"
    ;;
  collect)
    cd "$ARTIFACT_DIR"
    exec "$XLA_PYTHON" "$APP_DIR/collect.py" "$@"
    ;;
  shell)
    cd "$ARTIFACT_DIR"
    exec /bin/bash "$@"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Comando desconhecido: $command_name" >&2
    usage >&2
    exit 2
    ;;
esac
