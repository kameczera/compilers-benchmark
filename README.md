# CNN Compilers Benchmark

> Benchmark do *trade-off* entre **tempo de compilação** e **tempo de execução** em compiladores de Machine Learning — **TorchInductor (PyTorch 2)**, **XLA (JAX)** e **TVM (Relax)** — incluindo a variante do TorchInductor com **fold de BatchNorm em Conv** aplicado antes da fusão de operadores. Os resultados alimentam o modelo de custo \(T_b(n) = a_b\,n + b_b\) descrito no artigo (`docs/sblp/main.tex`).

## 1) O que o benchmark coleta

Para cada par *(modelo, entrada, backend)*, o `run_bench.py` gera um JSON com três blocos: `meta` (contexto do experimento), `raw` (medições por backend) e `recommendation` (modelo de custo e melhor backend por regime de uso).

### 1.1 Métricas de tempo (`raw.<backend>.shapes.<NxCxHxW>`)

| Métrica | Campo | Como é medida |
|---|---|---|
| Tempo de compilação | `compile_ms` | **TVM**: medido diretamente — o cronômetro cobre os passes de grafo (Legalize/Fuse*/dlight) **e** o `relax.build`, para isonomia com os proxies. **TorchInductor e XLA**: proxy `first_call_ms − exec_ms` (1ª chamada = compilação + 1ª execução; subtrai-se a latência média em regime estável). |
| *Time to first batch* | `ttfb_ms` | Tempo da primeira chamada compilada (compilação + 1ª execução). |
| Latência por execução | `exec_ms` | Média das amostras por iteração (`--iters`, padrão 50), após warmup. **Cada iteração é cronometrada individualmente com sincronização de GPU** (`torch.cuda.synchronize` / `block_until_ready` / `dev.sync`) — mesma semântica nos três backends. |
| Dispersão | `exec_ms_std`, `exec_ms_ci95`, `exec_samples_ms` | Desvio-padrão amostral (n−1), meia-largura do IC 95% e as amostras brutas por iteração — é daqui que saem os "média ± sd" das tabelas do artigo. |
| 1ª execução isolada | `first_exec_ms` | Só no XLA: segunda chamada do JIT, já compilado. |
| Energia (estimativa) | `energy_j` | `ttfb/1000·P_compile + exec/1000·P_exec·iters` com potências fixas (`--power-compile`, `--power-exec`). **Não é medição real** — é uma estimativa paramétrica. |
| Speedup | `speedup_exec_x` | Razão eager (ou não-fundido) / compilado. |

Variantes por backend:

- **TorchInductor**: `eager` (baseline), `fused_inductor` (`torch.compile` `mode="max-autotune"`) e `fused_fold_inductor` (mesma compilação, mas com **BN dobrado em Conv** via `fold_bn_inplace` antes do `torch.compile`). O fold cobre BasicBlock (`conv1/bn1`, `conv2/bn2`) **e Bottleneck (`conv3/bn3`)** — valide com `make check_fold` (0 BNs restantes + equivalência numérica). O caminho PyTorch roda com **TF32 habilitado** (`set_float32_matmul_precision("high")`) e **layout `channels_last`** em CUDA, espelhando o NHWC nativo do caminho JAX.
- **XLA**: `unfused` (forward sem JIT) e `fused_jit` (`jax.jit`). Usa o **mesmo warmup/iters** dos demais backends.
- **TVM**: `unfused` (legalização + `FuseOps`/`FuseTIR` básicos, sem `FoldConstant`/dlight/planejamento de memória) e `fused` (pipeline completo com `FoldConstant`, `FuseOps`, `FuseTIR`, dlight, DCE, memória estática etc.). Nota: o rótulo `unfused` é histórico — o pipeline mínimo executável ainda inclui a fusão básica.

### 1.2 Número de kernels e tamanho do código (`raw.<backend>.ir_dump`)

Cada backend exporta seu artefato de compilação em `ir_dumps/<backend>/<timestamp>_<modelo>_<shape>/` e conta kernels por **análise estática** do artefato (não é contagem de lançamentos via profiler):

| Backend | Artefato | Contagens em `kernel_count.summary` |
|---|---|---|
| TorchInductor | `fused[_fold]_inductor_output_code.py` — wrapper com `call()` + kernels Triton gerados | `triton_launches_total`, `triton_kernels_unicos`, `extern_kernels_total` (cuDNN/cuBLAS), `extern_funcs_unicas` |
| XLA | `fused_jit_unoptimized.hlo` e `fused_jit_optimized.hlo` | `fusion_total`, `fusion_por_kind` (kLoop/kInput/kCustom…), `custom_total` (custom-calls, ex.: cuDNN), `kernels_totais` |
| TVM | `unfused_tvmscript.py` e `fused_tvmscript.py` | `tvm_call_tir_total`, `tvm_call_tir_kernels_unicos`, `tvm_cls_total` |

Todos os resumos incluem **`code_size_bytes`** e **`code_lines`** (tamanho do código gerado). Em `kernel_count.details` ficam os nomes dos kernels, as contagens por função e, no Inductor, o mapa `fused_ops` (quais ops ATen foram fundidas em cada kernel Triton, extraído dos comentários `Original ATen: [...]`).

Observações:

- O HLO **não-otimizado** do XLA embute os pesos do modelo como literais e pode passar de **90 MB** por dump. Para a análise de kernels use o `fused_jit_optimized.hlo` (dezenas de KB).
- O dump do Inductor concatena o wrapper (que contém as chamadas `triton_*.run(...)` e `extern_kernels.*`) e os módulos dos kernels Triton, com um cabeçalho `# ===== inductor generated file: ...` por arquivo. O campo `capture_method` registra como o código foi capturado (`inductor_cache_files` é o esperado).

### 1.3 Recomendação (`recommendation`)

Modelo linear por backend \(T_b(n) = a_b\,n + b_b\), com `a_exec_ms` = latência média e `b_compile_ms` = compilação:

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

`segments` são as faixas de \(n\) (número de execuções) em que cada backend domina (envelope inferior); `breakpoints` são os pontos de troca; `best_at_runs_exec` é o melhor backend para o \(n\) passado em `--iters`. O gráfico do envelope é salvo em `plots/`.

## 2) Estrutura do repositório

```text
.
├── run_bench.py               # Orquestrador do benchmark
├── collect.py                 # Relatório do ambiente (versões, GPU, driver) → env_reports/
├── backends/
│   ├── pytorch_backend.py     # eager + Inductor + Inductor c/ BN-fold, dump do output_code
│   ├── xla_backend.py         # JAX/XLA, dump de HLO (unopt/opt)
│   ├── tvm_backend.py         # TVM Relax, dump de TVMScript
│   └── common.py              # métricas, modelo T(n)=a·n+b, envelope e plot
├── models/
│   ├── resnet_torch.py        # torchvision ResNet-18/50
│   └── resnet_jax.py          # flaxmodels ResNet-18 (só ResNet-18 no XLA!)
├── envs/
│   ├── requirements_xla.txt   # Ambiente XLA/JAX + PyTorch (venv 1)
│   └── requirements_tvm.txt   # Ambiente TVM (venv 2; TVM vem do fonte, ver §4.2)
├── scripts/
│   ├── run_full_grid.sh       # Grade completa do artigo (todos backends × shapes) → results/
│   ├── make_tables.py         # JSONs → tabelas LaTeX do artigo + summary.md (sem transcrição manual)
│   └── check_fold.py          # Valida o fold BN→Conv (0 BNs restantes + equivalência numérica)
├── Makefile                   # Venvs, smoke tests, check_fold, grid, tables, limpeza de caches
├── ir_dumps/                  # Artefatos de compilação (Triton/HLO/TVMScript)
├── plots/                     # Gráficos do envelope T(n)
├── results/                   # JSONs medidos (uma pasta por campanha de medição)
├── env_reports/               # Relatórios de ambiente e locks
└── docs/sblp/main.tex         # Artigo
```

## 3) Pré-requisitos de sistema

- Linux x86_64 com GPU NVIDIA (referência do artigo: RTX 3050 8 GB, driver 580.x, CUDA 13.0 no driver, nvcc 12.6).
- Python 3.10 para o venv XLA/PyTorch (3.11 também funciona) e Python 3.11 para o venv TVM.
- Para o TVM: checkout compilado do fonte com `USE_CUDA=ON` e `USE_LLVM=ON` (ver §4.2).
- Driver NVIDIA, nvcc e LLVM são dependências de sistema — não são instalados pelos requirements.

## 4) Ambientes virtuais

O benchmark usa **dois venvs separados**, porque as versões de torch exigidas por XLA/Inductor e por TVM conflitam.

### 4.1 venv XLA + PyTorch (`.venv_xla`)

```bash
make xla_env                 # cria .venv_xla a partir de envs/requirements_xla.txt
make check_xla               # confere pip check + GPU visível para torch e jax
```

Se `python3.10` não estiver no PATH, aponte o interpretador: `make xla_env PYTHON_BIN=python3.11`.

### 4.2 venv TVM (`.venv_tvm`)

O TVM do estudo (**0.22.dev0, commit `3b60f1c`**) não é instalável via pip — o pacote `apache-tvm` do PyPI parou no 0.14 e não tem o módulo `relax` usado pelo backend. Compile o TVM do fonte e aponte `TVM_HOME` para o checkout:

```bash
git clone --recursive https://github.com/apache/tvm && cd tvm
git checkout 3b60f1c9b
# build com USE_CUDA=ON e USE_LLVM=ON:
# https://tvm.apache.org/docs/install/from_source.html
```

```bash
make tvm_env                                   # cria .venv_tvm (deps de runtime + torch p/ trace FX)
make smoke_tvm TVM_HOME=/caminho/para/tvm      # valida
```

O Makefile injeta `PYTHONPATH=$TVM_HOME/python:$TVM_HOME/build` ao rodar o TVM.

### 4.3 Variáveis de ambiente (exportadas pelo Makefile)

| Variável | Valor | Por quê |
|---|---|---|
| `XLA_PYTHON_CLIENT_PREALLOCATE` | `false` | JAX não pré-aloca a VRAM toda — evita `Failed to allocate ... bytes` quando PyTorch e JAX dividem a GPU. |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | `0.40` | Limita a fração de VRAM do JAX. |
| `TORCHINDUCTOR_FORCE_DISABLE_CACHES` | `1` | **Necessária para a coleta correta** — ver §7. `run_bench.py` também seta este default sozinho. |

Elas precisam existir **antes do primeiro `import jax`/`import torch`** no processo. Ao rodar manualmente fora do Makefile, exporte-as antes.

## 5) Executando o benchmark

### 5.1 Smoke tests (validação rápida)

```bash
make smoke_torch                               # só TorchInductor (eager + fused + fold)
make smoke_xla                                 # só XLA/JAX
make smoke_xla_torch                           # XLA + Inductor no mesmo processo
make smoke_tvm TVM_HOME=/caminho/para/tvm      # só TVM
```

Cada alvo usa ResNet-18, batch 1, 224×224, warmup 3, 5 iterações, e gera `out_smoke_*.json`.

### 5.2 Medição completa (protocolo do artigo: warmup 10, iters 50)

Rode **cada backend em um processo separado** (evita disputa de VRAM entre runtimes) e **limpe o cache do Inductor antes de cada medição do PyTorch** (§7):

```bash
# --- TorchInductor (eager + fused + fused_fold) ---
make clean_inductor_cache
. .venv_xla/bin/activate
python run_bench.py --no-tvm --no-xla --device cuda --model resnet18 \
  --dtype fp32 --batch 1 --height 224 --width 224 \
  --warmup 10 --iters 50 --output out_inductor_resnet18_1x3x224x224.json
deactivate

# --- XLA ---
. .venv_xla/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.40
python run_bench.py --no-tvm --no-inductor --device cuda --model resnet18 \
  --dtype fp32 --batch 1 --height 224 --width 224 \
  --warmup 10 --iters 50 --output out_xla_resnet18_1x3x224x224.json
deactivate

# --- TVM ---
. .venv_tvm/bin/activate
export TVM_HOME=/caminho/para/tvm
PYTHONPATH=$TVM_HOME/python:$TVM_HOME/build python run_bench.py \
  --no-xla --no-inductor --device cuda --model resnet18 \
  --dtype fp32 --batch 1 --height 224 --width 224 \
  --warmup 10 --iters 50 --output out_tvm_resnet18_1x3x224x224.json
deactivate
```

Repita variando `--batch/--height/--width` (o artigo usa 1/16/64×224² e 1×512², 1×1024²) e `--model resnet50`.

### 5.2b Grade completa e tabelas do artigo (um comando cada)

```bash
make grid TVM_HOME=/caminho/para/tvm     # roda a grade inteira → results/v2/<backend>_<modelo>_<shape>.json
make tables                              # gera results/v2/tables/{table_*.tex, summary.md} a partir dos JSONs
```

O `make tables` produz as tabelas do artigo **diretamente dos JSONs** (compile, exec ± sd, deltas do fold e ponto de equivalência `n_eq`) — nenhuma célula é preenchida à mão. Para regenerar tabelas de outra campanha: `make tables RESULTS_DIR=results/remeasure`.

**Tolerâncias esperadas ao reproduzir** (mesma classe de GPU, RTX 3050):

| Métrica | Tolerância típica | Observação |
|---|---|---|
| `exec_ms` batch>1 / entradas grandes | ±5% | estável entre dias e re-execuções |
| `exec_ms` batch 1 | ±20% | sensível a versão do torch × layout; ver §8 |
| `compile_ms` XLA e TVM | ±10% | |
| `compile_ms` TorchInductor | depende da **versão do torch** | torch 2.9 compila ~1,3–2,6× mais devagar que 2.5.1 nesta GPU; compare apenas dentro da mesma versão (a fixada em `envs/requirements_xla.txt`) |
| Contagens de kernels | exatas | mesma versão de torch/jax/TVM ⇒ mesmos artefatos |

### 5.3 Flags

- `--model resnet18|resnet50` · `--device cuda|cpu` · `--dtype fp32|bf16|fp16`
- `--batch`, `--height`, `--width` — shape NCHW da entrada
- `--warmup`, `--iters` — protocolo de medição
- `--no-xla`, `--no-inductor`, `--no-tvm` — liga/desliga backends
- `--power-compile`, `--power-exec` — potências (W) da estimativa de energia
- `--compile-budget-ms` — penaliza backends acima do orçamento de compilação na recomendação
- `--output arquivo.json`

## 6) Conferindo se a coleta veio completa

Depois de rodar, verifique no JSON:

1. **Sem erros**: não deve haver `inductor_error`, `xla_error`, `tvm_error` em `raw`.
2. **Kernels contados**: em `raw.inductor.ir_dump.fused_inductor.kernel_count.summary`, `triton_launches_total` e `extern_kernels_total` devem ser **> 0** (ResNet-18 com torch 2.9: 18 lançamentos Triton + 21 chamadas extern; ResNet-50: 50 + 54). Se vierem zerados, veja §7.
3. **`capture_method`**: deve ser `inductor_cache_files`. `stdout_fallback` indica captura degradada.
4. **Dumps não-vazios** em `ir_dumps/` (o `*_inductor_output_code.py` de uma ResNet-18 tem centenas de KB).
5. **GPU de verdade**: `make check_xla` confirma `torch.cuda.is_available()` e `jax.devices('gpu')`. Se o JAX cair para CPU, a comparação não vale.

## 7) ⚠️ Cache do TorchInductor: leia antes de medir PyTorch

**Sintoma.** `*_inductor_output_code.py` vazio em `ir_dumps/inductor/`, contagem de kernels zerada (com um campo `warning` no JSON) e `compile_ms` do Inductor muito menor do que o real.

**Causa.** O TorchInductor mantém um cache persistente de compilação (FX graph cache) em `/tmp/torchinductor_$USER` (mais o cache do Triton em `~/.triton/cache`). A partir da **segunda** execução com o mesmo modelo/shape/versões, o Inductor reutiliza o cache: o codegen não roda, nenhum `output_code` novo é gerado — some o artefato para contar kernels — e o tempo medido vira tempo de *cache hit*, não de compilação.

**Como o repositório trata isso.**

1. `run_bench.py` e o `Makefile` setam `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` por padrão (precisa existir antes do `import torch`; por isso está no topo do `run_bench.py`).
2. O backend também força `torch._inductor.config.force_disable_caches = True` em runtime e captura o código gerado diretamente dos arquivos produzidos pelo Inductor (wrapper + kernels Triton), sem depender de `TORCH_LOGS`.
3. Para garantir compilação "fria" (o número reportado como `compile_ms`), **sempre** rode antes de cada medição do PyTorch:

```bash
make clean_inductor_cache
# remove /tmp/torchinductor_$USER, ~/.cache/torch/inductor e ~/.triton/cache
```

**Se mesmo assim vier vazio** (ex.: outra versão de torch com layout de cache diferente), o JSON marca `kernel_count.warning` e `capture_method: stdout_fallback`. Nesse caso: `make clean_inductor_cache`, confirme `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` no ambiente e repita a medição. Este procedimento faz parte do artefato justamente para que avaliadores consigam reproduzir a contagem de kernels do PyTorch. O repositório mantém **uma instância de exemplo por backend** em `ir_dumps/` (gerada com coleta válida — `code_size_bytes > 0` no JSON correspondente); o HLO não-otimizado do XLA (~93 MB) fica fora do controle de versão via `.gitignore`.

*Medição de cache quente (opcional).* Para medir de propósito o tempo com cache reaproveitado, rode com `TORCHINDUCTOR_FORCE_DISABLE_CACHES=0` **sem** limpar o cache — e reporte separadamente, pois não é comparável ao `compile_ms` dos outros backends.

## 8) Limitações conhecidas da coleta

- **XLA roda sempre ResNet-18** (via `flaxmodels`), mesmo com `--model resnet50` — não há ResNet-50 no caminho JAX deste harness.
- **`compile_ms` do Inductor e do XLA é um proxy** (`1ª chamada − latência média`); só o TVM tem compilação cronometrada isoladamente. Consequência: ruído da 1ª execução entra no proxy.
- **A contagem de kernels é estática** (análise do artefato de compilação). Ela não conta lançamentos reais em GPU; para isso use um profiler (ex.: Nsight Systems) por fora.
- **`energy_j` é estimativa paramétrica**, não medição (não usa NVML/RAPL).
- **TVM parte de um trace FX do PyTorch** — o export fragmenta operadores que o PyTorch trata como atômicos, o que infla a contagem de kernels do TVM (discutido no artigo, Seção *Evaluation*).
- **Precisão numérica assimétrica na execução**: torch (cuDNN) e XLA usam TF32 em GPUs Ampere; os kernels CUDA gerados pelo TVM (dlight) fazem FP32 puro — leve vantagem de execução para os dois primeiros.
- **XLA com batch > 1 pode falhar de forma INTERMITENTE** com `XlaRuntimeError: INTERNAL: an unsupported value or parameter was passed to the function` na primeira chamada JIT, dependendo da combinação driver NVIDIA × cuDNN (observado com driver 580.105 + jax 0.6.2 + cuDNN 9.10 na RTX 3050; batch 1 nunca falhou). **Re-executar a mesma medição costuma passar** — o `scripts/run_full_grid.sh` re-tenta até 3× automaticamente. `XLA_FLAGS=--xla_gpu_autotune_level=0` evita o erro, mas degrada a execução (~17×) e não serve para comparação.
- **BERT/GPT-2 do artigo não estão neste harness** — este repositório cobre a parte de CNNs (ResNet-18/50) e o estudo de fold; os transformers foram medidos com scripts próprios.
- **`compile_ms` do TorchInductor depende fortemente da versão do torch** (~2× entre 2.5.1 e 2.9 nesta GPU). O artefato fixa a versão em `envs/requirements_xla.txt`; números de compilação só são comparáveis dentro da mesma versão.
- Os smoke tests (batch 1, poucas iterações) servem para validar o ambiente, **não** para comparar backends — use o protocolo completo (§5.2).

## 9) Reprodutibilidade

```bash
# relatório do ambiente (GPU, driver, versões) → env_reports/environment_report.json
. .venv_xla/bin/activate && python collect.py

# lock do venv XLA → env_reports/requirements_xla.lock.txt
make lock_xla
```

O snapshot completo do ambiente TVM original está em `env_reports/requirements_tvm_snapshot_original.txt`.

## 10) Limpeza

```bash
make clean_inductor_cache   # caches do TorchInductor/Triton (ver §7)
make clean                  # remove .venv_xla, .venv_tvm, out_*.json e locks
```
