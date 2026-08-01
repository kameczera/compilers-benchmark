#!/usr/bin/env bash
# Re-mede a grade completa do artigo com o protocolo padrão (K=5 compilações
# frias em processos isolados; em cada processo: warmup 10, iters 50):
#   TorchInductor: resnet18 + resnet50 x 5 shapes
#   TVM:           resnet18 + resnet50 x 5 shapes  (requer TVM_HOME)
#   XLA:           resnet18 + resnet50 x 5 shapes  (cuDNN 9.11 do host)
#
# Uso:
#   TVM_HOME=/caminho/para/tvm scripts/run_full_grid.sh [dir_de_saida]
#
# Saída: <dir_de_saida>/<backend>_<modelo>_<NxCxHxW>.json  (padrão: results/k5)
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-results/k5}"
VENV_XLA="${VENV_XLA:-$ROOT/.venv_xla}"
VENV_TVM="${VENV_TVM:-$ROOT/.venv_tvm}"
COMPILE_REPEATS="${COMPILE_REPEATS:-5}"
PER_REPEAT_TIMEOUT="${PER_REPEAT_TIMEOUT:-900}"
PER_RUN_TIMEOUT="${PER_RUN_TIMEOUT:-$((PER_REPEAT_TIMEOUT * COMPILE_REPEATS + 60))}"
XLA_REPEAT_ATTEMPTS="${XLA_REPEAT_ATTEMPTS:-3}"
XLA_FLAGS_VALUE="${XLA_FLAGS:-}"
XLA_LIBRARY_PATH="${XLA_LIBRARY_PATH:-/usr/local/lib64}"
XLA_CUDNN_RUNTIME_VERSION="${XLA_CUDNN_RUNTIME_VERSION:-91100}"
RUN_INDUCTOR="${RUN_INDUCTOR:-1}"
RUN_XLA="${RUN_XLA:-1}"
RUN_TVM="${RUN_TVM:-1}"

cd "$ROOT"
mkdir -p "$OUT"

SHAPES=("1 224 224" "16 224 224" "64 224 224" "1 512 512" "1 1024 1024")

clean_inductor_cache() {
  local cache_path="/tmp/torchinductor_${USER}"
  case "$cache_path" in
    /tmp/torchinductor_?*) rm -rf -- "$cache_path" ;;
    *) echo "refusing unsafe cache path: $cache_path" >&2; return 1 ;;
  esac
}

run_one() { # $1=cmd-prefix... executa com timeout e loga
  local label="$1"; shift
  echo "=== [$(date +%H:%M:%S)] $label"
  if ! timeout "$PER_RUN_TIMEOUT" "$@"; then
    echo "!!! $label saiu com erro/timeout (seguindo para o próximo)"
  fi
}

has_requested_repeats() {
  local expected_xla_flags="${2:-}"
  local expected_cudnn_runtime="${3:-}"
  python3 - "$1" "$COMPILE_REPEATS" "$expected_xla_flags" "$expected_cudnn_runtime" <<'PY' 2>/dev/null
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
actual = payload.get("meta", {}).get("compile_repeats", 1)
valid = int(actual) == int(sys.argv[2])
if "inductor_" in sys.argv[1]:
    valid = valid and bool(
        payload.get("meta", {}).get("inductor_base_fold_process_isolation")
    )
    valid = valid and bool(
        payload.get("raw", {})
        .get("inductor", {})
        .get("meta", {})
        .get("base_fold_same_initial_state")
    )
elif "xla_" in sys.argv[1] or "tvm_" in sys.argv[1]:
    valid = valid and (
        payload.get("meta", {}).get("xla_tvm_repeated_variant")
        == "fused_only"
    )
if "tvm_" in sys.argv[1]:
    valid = valid and (
        payload.get("raw", {}).get("tvm", {}).get("meta", {}).get("model_seed")
        == 0
    )
if "xla_" in sys.argv[1]:
    valid = valid and payload.get("meta", {}).get("xla_flags") == sys.argv[3]
    valid = valid and (
        payload.get("raw", {})
        .get("xla", {})
        .get("meta", {})
        .get("cudnn_runtime_version")
        == int(sys.argv[4])
    )
    valid = valid and (
        payload.get("raw", {})
        .get("xla", {})
        .get("meta", {})
        .get("weights")
        == "random_init"
    )
raise SystemExit(0 if valid else 1)
PY
}

# ---------- TorchInductor ----------
if [ "$RUN_INDUCTOR" -eq 1 ]; then
  for model in resnet18 resnet50; do
    for s in "${SHAPES[@]}"; do
      read -r B H W <<<"$s"
      out_json="$OUT/inductor_${model}_${B}x3x${H}x${W}.json"
      if has_requested_repeats "$out_json"; then
        echo "=== [skip] $out_json já contém K=$COMPILE_REPEATS"
        continue
      fi
      if [ "$COMPILE_REPEATS" -eq 1 ]; then
        clean_inductor_cache
      fi
      run_one "inductor $model ${B}x3x${H}x${W}" \
        env TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 "$VENV_XLA/bin/python" run_bench.py \
          --no-tvm --no-xla --device cuda --model "$model" --dtype fp32 \
          --batch "$B" --height "$H" --width "$W" --warmup 10 --iters 50 \
          --compile-repeats "$COMPILE_REPEATS" --compile-repeat-timeout-s "$PER_REPEAT_TIMEOUT" \
          --output "$out_json"
    done
  done
fi

# ---------- XLA ----------
# A cuDNN 9.10.2 empacotada no venv falha de forma intermitente nesta máquina.
# A campanha carrega explicitamente a cuDNN 9.11 do host, validada no JSON pelo
# runtime version 91100. Falhas ainda são registradas e repetidas, nunca medidas.
xla_ok() { python3 -c "import json,sys; d=json.load(open('$1')); sys.exit(0 if 'xla_error' not in d.get('raw',{}) else 1)" 2>/dev/null; }
if [ "$RUN_XLA" -eq 1 ]; then
  # O host de referência do artigo carrega a cuDNN 9.11 de $XLA_LIBRARY_PATH.
  # Onde essa biblioteca existe, a versão é exigida para que a campanha
  # publicada nunca seja medida com a cuDNN 9.10.2 (falhas intermitentes, ver
  # README §8). Em outros hosts — inclusive no contêiner — o grid continua com
  # a cuDNN do próprio ambiente; a versão efetiva fica registrada no JSON.
  # XLA_STRICT_CUDNN=1 restaura a exigência absoluta.
  XLA_STRICT_CUDNN="${XLA_STRICT_CUDNN:-0}"
  if [ -e "$XLA_LIBRARY_PATH/libcudnn.so.9" ]; then
    actual_cudnn_runtime="$(
      env LD_LIBRARY_PATH="$XLA_LIBRARY_PATH" "$VENV_XLA/bin/python" -c \
        'from jax._src.lib import cuda_versions; print(cuda_versions.cudnn_get_version())'
    )"
    if [ "$actual_cudnn_runtime" != "$XLA_CUDNN_RUNTIME_VERSION" ]; then
      echo "!!! cuDNN runtime: esperado $XLA_CUDNN_RUNTIME_VERSION, obtido $actual_cudnn_runtime" >&2
      exit 1
    fi
  elif [ "$XLA_STRICT_CUDNN" -eq 1 ]; then
    echo "!!! cuDNN 9.11 não encontrada em $XLA_LIBRARY_PATH (XLA_STRICT_CUDNN=1)" >&2
    exit 1
  else
    XLA_LIBRARY_PATH=""
    actual_cudnn_runtime="$(
      "$VENV_XLA/bin/python" -c \
        'from jax._src.lib import cuda_versions; print(cuda_versions.cudnn_get_version())'
    )"
    # A retomada compara o JSON existente com a cuDNN realmente usada aqui.
    XLA_CUDNN_RUNTIME_VERSION="$actual_cudnn_runtime"
    echo "### cuDNN de referência ausente; usando a cuDNN do ambiente" \
         "(runtime $actual_cudnn_runtime, registrada em cudnn_runtime_version no JSON)"
  fi
  # Sem a biblioteca de referência, não prefixe LD_LIBRARY_PATH com ':' vazio.
  xla_ld_library_path="${XLA_LIBRARY_PATH}"
  if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    xla_ld_library_path="${XLA_LIBRARY_PATH:+$XLA_LIBRARY_PATH:}$LD_LIBRARY_PATH"
  fi
  for model in resnet18 resnet50; do
    for s in "${SHAPES[@]}"; do
      read -r B H W <<<"$s"
      out_json="$OUT/xla_${model}_${B}x3x${H}x${W}.json"
      if has_requested_repeats "$out_json" "$XLA_FLAGS_VALUE" "$XLA_CUDNN_RUNTIME_VERSION"; then
        echo "=== [skip] $out_json já contém K=$COMPILE_REPEATS"
        continue
      fi
      for attempt in 1 2 3; do
        run_one "xla $model ${B}x3x${H}x${W} (tentativa $attempt)" \
          env LD_LIBRARY_PATH="$xla_ld_library_path" \
            XLA_FLAGS="$XLA_FLAGS_VALUE" \
            XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 \
            "$VENV_XLA/bin/python" run_bench.py \
            --no-tvm --no-inductor --device cuda --model "$model" --dtype fp32 \
            --batch "$B" --height "$H" --width "$W" --warmup 10 --iters 50 \
            --compile-repeats "$COMPILE_REPEATS" --compile-repeat-timeout-s "$PER_REPEAT_TIMEOUT" \
            --compile-repeat-attempts "$XLA_REPEAT_ATTEMPTS" \
            --output "$out_json"
        xla_ok "$out_json" && break
        echo "!!! xla $model ${B}x3x${H}x${W}: XlaRuntimeError intermitente — repetindo"
      done
    done
  done
fi

# ---------- TVM ----------
if [ "$RUN_TVM" -ne 1 ]; then
  :
elif [ -z "${TVM_HOME:-}" ]; then
  echo "!!! TVM_HOME não definido — pulando as medições do TVM"
else
  for model in resnet18 resnet50; do
    for s in "${SHAPES[@]}"; do
      read -r B H W <<<"$s"
      out_json="$OUT/tvm_${model}_${B}x3x${H}x${W}.json"
      if has_requested_repeats "$out_json"; then
        echo "=== [skip] $out_json já contém K=$COMPILE_REPEATS"
        continue
      fi
      run_one "tvm $model ${B}x3x${H}x${W}" \
        env PYTHONPATH="$TVM_HOME/python:$TVM_HOME/build" "$VENV_TVM/bin/python" run_bench.py \
          --no-xla --no-inductor --device cuda --model "$model" --dtype fp32 \
          --batch "$B" --height "$H" --width "$W" --warmup 10 --iters 50 \
          --compile-repeats "$COMPILE_REPEATS" --compile-repeat-timeout-s "$PER_REPEAT_TIMEOUT" \
          --output "$out_json"
    done
  done
fi

echo "=== [$(date +%H:%M:%S)] grade completa em $OUT"
