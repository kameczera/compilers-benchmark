#!/usr/bin/env bash
# Comparacao BERT/GPT-2 entre backends com o protocolo K=5 do artigo
# (processos frios isolados, warmup 10, 50 execucoes cronometradas por processo).
#
#   BERT:  (1,64) (1,128) (8,128)
#   GPT-2: (1,16) (1,128) (8,128) (8,256)
#
# Uso:
#   TVM_HOME=/caminho/para/tvm scripts/run_transformer_grid.sh [dir_de_saida]
#
# Saida: <dir>/<backend>_<modelo>_<batch>x<seq>_k5.json  (padrao: results/transformers)
# Celulas que falham sao registradas em <dir>/coverage.json com o motivo, em vez
# de interromper a grade: o gerador de tabelas as imprime como lacuna explicita.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-results/transformers}"
VENV_XLA="${VENV_XLA:-$ROOT/.venv_xla}"
VENV_TVM="${VENV_TVM:-$ROOT/.venv_tvm}"
COMPILE_REPEATS="${COMPILE_REPEATS:-5}"
PER_RUN_TIMEOUT="${PER_RUN_TIMEOUT:-5400}"
RUN_INDUCTOR="${RUN_INDUCTOR:-1}"
RUN_XLA="${RUN_XLA:-1}"
RUN_TVM="${RUN_TVM:-1}"

cd "$ROOT"
mkdir -p "$OUT"
COVERAGE="$OUT/coverage.json"
[ -f "$COVERAGE" ] || echo '{}' > "$COVERAGE"

BERT_SHAPES=("1 64" "1 128" "8 128")
GPT2_SHAPES=("1 16" "1 128" "8 128" "8 256")

record_failure() { # $1=celula  $2=motivo
  python3 - "$COVERAGE" "$1" "$2" <<'PY'
import json, sys
path, cell, reason = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path, encoding="utf-8") as handle:
    notes = json.load(handle)
notes[cell] = reason
with open(path, "w", encoding="utf-8") as handle:
    json.dump(notes, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}

run_cell() { # $1=backend $2=python $3=modelo $4=batch $5=seq
  local backend="$1" python_bin="$2" model="$3" batch="$4" seq="$5"
  local out_json="$OUT/${backend}_${model}_${batch}x${seq}_k5.json"
  local cell="${backend}/${model}/${batch}x${seq}"

  if [ -f "$out_json" ] && python3 -c "
import json,sys
d=json.load(open('$out_json'))
sys.exit(0 if d.get('meta',{}).get('compile_repeats')==$COMPILE_REPEATS else 1)" 2>/dev/null; then
    echo "=== [skip] $cell ja contem K=$COMPILE_REPEATS"
    return 0
  fi

  echo "=== [$(date +%H:%M:%S)] $cell"
  local log; log="$(mktemp)"
  if timeout "$PER_RUN_TIMEOUT" "$python_bin" scripts/benchmark_transformers.py \
      --backend "$backend" --model "$model" \
      --batch "$batch" --seq-len "$seq" \
      --repeats "$COMPILE_REPEATS" --warmup 10 --iters 50 \
      --output "$out_json" 2>&1 | tee "$log" | tail -3; then
    rm -f "$log"
    return 0
  fi
  local reason; reason="$(grep -aoE '[A-Za-z_.]*(Error|Exception)[^\n]{0,140}' "$log" | tail -1)"
  [ -n "$reason" ] || reason="falhou sem excecao reconhecida (ver log)"
  echo "!!! $cell falhou: $reason"
  record_failure "$cell" "$reason"
  rm -f "$log" "$out_json" 2>/dev/null
  return 0
}

if [ "$RUN_INDUCTOR" -eq 1 ]; then
  for s in "${BERT_SHAPES[@]}"; do read -r B L <<<"$s"; run_cell inductor "$VENV_XLA/bin/python" bert "$B" "$L"; done
  for s in "${GPT2_SHAPES[@]}"; do read -r B L <<<"$s"; run_cell inductor "$VENV_XLA/bin/python" gpt2 "$B" "$L"; done
fi

if [ "$RUN_XLA" -eq 1 ]; then
  for s in "${BERT_SHAPES[@]}"; do read -r B L <<<"$s"; run_cell xla "$VENV_XLA/bin/python" bert "$B" "$L"; done
  for s in "${GPT2_SHAPES[@]}"; do read -r B L <<<"$s"; run_cell xla "$VENV_XLA/bin/python" gpt2 "$B" "$L"; done
fi

if [ "$RUN_TVM" -eq 1 ]; then
  if [ -z "${TVM_HOME:-}" ] && [ ! -x "$VENV_TVM/bin/python" ]; then
    echo "!!! TVM_HOME nao definido — pulando o TVM"
  else
    export PYTHONPATH="${TVM_HOME:-/opt/tvm}/python:${TVM_HOME:-/opt/tvm}/build:${PYTHONPATH:-}"
    for s in "${BERT_SHAPES[@]}"; do read -r B L <<<"$s"; run_cell tvm "$VENV_TVM/bin/python" bert "$B" "$L"; done
    for s in "${GPT2_SHAPES[@]}"; do read -r B L <<<"$s"; run_cell tvm "$VENV_TVM/bin/python" gpt2 "$B" "$L"; done
  fi
fi

echo "=== [$(date +%H:%M:%S)] grade Transformer em $OUT"
