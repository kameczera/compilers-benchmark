#!/usr/bin/env bash
# Re-mede a grade completa do artigo com o protocolo padrão (warmup 10, iters 50):
#   TorchInductor: resnet18 + resnet50 x 5 shapes  (cache limpo antes de cada run)
#   TVM:           resnet18 + resnet50 x 5 shapes  (requer TVM_HOME)
#   XLA:           resnet18 x 5 shapes             (harness só suporta ResNet-18 no XLA;
#                                                   batch>1 pode falhar por driver/cuDNN — o
#                                                   JSON registra o erro e o grid continua)
#
# Uso:
#   TVM_HOME=/caminho/para/tvm scripts/run_full_grid.sh [dir_de_saida]
#
# Saída: <dir_de_saida>/<backend>_<modelo>_<NxCxHxW>.json  (padrão: results/v2)
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-results/v2}"
VENV_XLA="${VENV_XLA:-$ROOT/.venv_xla}"
VENV_TVM="${VENV_TVM:-$ROOT/.venv_tvm}"
PER_RUN_TIMEOUT="${PER_RUN_TIMEOUT:-900}"

cd "$ROOT"
mkdir -p "$OUT"

SHAPES=("1 224 224" "16 224 224" "64 224 224" "1 512 512" "1 1024 1024")

clean_inductor_cache() {
  rm -rf "/tmp/torchinductor_${USER}" "$HOME/.cache/torch/inductor" "$HOME/.triton/cache"
}

run_one() { # $1=cmd-prefix... executa com timeout e loga
  local label="$1"; shift
  echo "=== [$(date +%H:%M:%S)] $label"
  if ! timeout "$PER_RUN_TIMEOUT" "$@"; then
    echo "!!! $label saiu com erro/timeout (seguindo para o próximo)"
  fi
}

# ---------- TorchInductor ----------
for model in resnet18 resnet50; do
  for s in "${SHAPES[@]}"; do
    read -r B H W <<<"$s"
    out_json="$OUT/inductor_${model}_${B}x3x${H}x${W}.json"
    clean_inductor_cache
    run_one "inductor $model ${B}x3x${H}x${W}" \
      env TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 "$VENV_XLA/bin/python" run_bench.py \
        --no-tvm --no-xla --device cuda --model "$model" --dtype fp32 \
        --batch "$B" --height "$H" --width "$W" --warmup 10 --iters 50 \
        --output "$out_json"
  done
done

# ---------- XLA (só ResNet-18; ver README §8) ----------
# O XlaRuntimeError INTERNAL em batch>1 (driver 580.105 × cuDNN 9.10) é
# INTERMITENTE: re-tentar a mesma medição costuma passar. Até 3 tentativas.
xla_ok() { python3 -c "import json,sys; d=json.load(open('$1')); sys.exit(0 if 'xla_error' not in d.get('raw',{}) else 1)" 2>/dev/null; }
for s in "${SHAPES[@]}"; do
  read -r B H W <<<"$s"
  out_json="$OUT/xla_resnet18_${B}x3x${H}x${W}.json"
  for attempt in 1 2 3; do
    run_one "xla resnet18 ${B}x3x${H}x${W} (tentativa $attempt)" \
      env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.40 \
        "$VENV_XLA/bin/python" run_bench.py \
        --no-tvm --no-inductor --device cuda --model resnet18 --dtype fp32 \
        --batch "$B" --height "$H" --width "$W" --warmup 10 --iters 50 \
        --output "$out_json"
    xla_ok "$out_json" && break
    echo "!!! xla ${B}x3x${H}x${W}: XlaRuntimeError intermitente — repetindo"
  done
done

# ---------- TVM ----------
if [ -z "${TVM_HOME:-}" ]; then
  echo "!!! TVM_HOME não definido — pulando as medições do TVM"
else
  for model in resnet18 resnet50; do
    for s in "${SHAPES[@]}"; do
      read -r B H W <<<"$s"
      out_json="$OUT/tvm_${model}_${B}x3x${H}x${W}.json"
      run_one "tvm $model ${B}x3x${H}x${W}" \
        env PYTHONPATH="$TVM_HOME/python:$TVM_HOME/build" "$VENV_TVM/bin/python" run_bench.py \
          --no-xla --no-inductor --device cuda --model "$model" --dtype fp32 \
          --batch "$B" --height "$H" --width "$W" --warmup 10 --iters 50 \
          --output "$out_json"
    done
  done
fi

echo "=== [$(date +%H:%M:%S)] grade completa em $OUT"
