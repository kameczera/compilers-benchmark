# CNN Compilers Benchmark

> Benchmark do *trade-off* entre **tempo de compilação** e **tempo de execução** em compiladores de Machine Learning — **TorchInductor (PyTorch 2)**, **XLA (JAX)** e **TVM (Relax)** — incluindo a variante do TorchInductor com **fold de BatchNorm em Conv** aplicado antes da fusão de operadores. Os resultados alimentam o modelo de custo \(T_b(n) = a_b\,n + b_b\) descrito no artigo (`docs/sblp/main.tex`).

## Artifact Status

- **Artigo aceito:** [PDF camera-ready do artigo SBLP 2026](docs/sblp/main.pdf); a [fonte LaTeX](docs/sblp/main.tex) também acompanha o artefato.
- **Versão persistente:** <https://doi.org/10.5281/zenodo.21731237> — DOI específico desta versão do artefato, e não do registro "todas as versões".
- **Repositório de desenvolvimento:** <https://github.com/kameczera/compilers-benchmark>.
- **Licença:** [MIT](LICENSE), permitindo uso, modificação e redistribuição.
- **Selos pretendidos:** *Available* (após depósito com DOI) e *Functional* (instalação reproduzível, verificações automáticas, smoke test dos três backends e reprodução completa).

O guia curto para o comitê está em
[ARTIFACT_EVALUATION.md](ARTIFACT_EVALUATION.md). Antes de criar o arquivo a
ser depositado, execute `make artifact_submission_check`: esse alvo falha se
faltarem DOI, PDF, dados e evidências auxiliares K=5, relatório do ambiente ou
as seis tabelas geradas consumidas pelo artigo.

## Requirements

O caminho recomendado usa Docker para evitar a instalação manual de duas
pilhas Python incompatíveis. Requisitos do host:

- Linux x86_64, Docker Engine e NVIDIA Container Toolkit;
- GPU NVIDIA com pelo menos 8 GiB de VRAM e driver compatível com CUDA 12.8;
- 16 GiB de RAM e pelo menos 60 GiB livres para imagem, cache de build e uma execução curta;
- conexão de rede durante o build para obter o TVM no commit fixado e os pacotes Python.

**Hosts com SELinux (Fedora, RHEL, CentOS).** Com SELinux em `Enforcing`, o
rótulo padrão do contêiner bloqueia `/dev/nvidia*` e o `nvidia-smi` de dentro
da imagem falha com `Failed to initialize NVML: Insufficient Permissions`,
mesmo com `--gpus all`. Os alvos `docker_*` do Makefile detectam isso via
`getenforce` e acrescentam `--security-opt label=disable` sozinhos; ao chamar
`docker run` à mão nesses hosts, adicione a mesma flag.

O preflight e a validação algébrica do fold rodam em CPU. O smoke test e a
reprodução dos tempos exigem GPU CUDA. Reserve 25--40 GiB para a imagem, até
30 minutos para o smoke e várias horas mais pelo menos 20 GiB adicionais para
a grade K=5 completa. Estimativas e limitações estão detalhadas no
[guia de avaliação](ARTIFACT_EVALUATION.md#2-resource-estimate).

Para instalação nativa, use Python 3.10/3.11 e os arquivos versionados em
`envs/`; TVM 0.22.dev0 deve ser compilado do commit
`3b60f1c9b8907dcf5d39a033876020e96e6915b2` com CUDA, cuDNN e LLVM. Driver,
CUDA do host e LLVM não são instalados por `pip`.

## Installation

Na raiz de uma cópia da **versão exata arquivada**:

```bash
# verificações portáveis: estrutura, paper, sintaxe Python e shell
make artifact_check

# caminho recomendado para instalar os três backends
make docker_build
make docker_verify
make docker_smoke
```

O `docker_verify` deve listar a GPU, as versões de Torch/JAX/TVM e terminar
com `[OK] Imagem e GPU validadas`. O `docker_smoke` executa uma ResNet-18
pequena nos três backends e termina com
`[OK] Smoke dos três backends concluído em /artifacts`. As saídas ficam em
`artifacts/results/`, `artifacts/ir_dumps/`, `artifacts/plots/` e
`artifacts/env_reports/`.

Se a máquina ainda não tiver acesso a uma GPU, a verificação CPU que sustenta
a correção do fold pode ser executada em um ambiente com PyTorch/torchvision:

```bash
python scripts/check_fold.py
```

Ela deve informar `BN restantes=0` e `[OK]` para `resnet18` e `resnet50`.
Os comandos acima validam instalação e funcionalidade; eles não substituem a
grade experimental completa descrita na Seção 5.2.

## 1) O que o benchmark coleta

Para cada par *(modelo, entrada, backend)*, o `run_bench.py` gera um JSON com três blocos: `meta` (contexto do experimento), `raw` (medições por backend) e `recommendation` (modelo de custo e melhor backend por regime de uso).

### 1.1 Métricas de tempo (`raw.<backend>.shapes.<NxCxHxW>`)

| Métrica | Campo | Como é medida |
|---|---|---|
| Tempo de compilação | `compile_ms`, `compile_ms_std`, `compile_ms_ci95`, `compile_samples_ms` | Com `--compile-repeats K`, cada amostra vem de um processo frio com caches próprios; `compile_ms` é a média, o desvio é amostral e o IC 95% usa Student-t. **TVM** é medido diretamente (passes + `relax.build`). **TorchInductor e XLA** usam o proxy `first_call_ms − exec_ms`. |
| *Time to first batch* | `ttfb_ms` | Tempo da primeira chamada compilada (compilação + 1ª execução). |
| Latência por execução | `exec_ms` | Média das amostras por iteração (`--iters`, padrão 50), após warmup. **Cada iteração é cronometrada individualmente com sincronização de GPU** (`torch.cuda.synchronize` / `block_until_ready` / `dev.sync`) — mesma semântica nos três backends. |
| Dispersão de execução | `exec_ms_std`, `exec_run_means_ms_std`, `exec_ms_ci95`, `exec_samples_ms`, `exec_run_means_ms` | `exec_ms_std` descreve as amostras individuais; as tabelas usam `exec_run_means_ms_std` e o IC 95% usa Student-t sobre as K médias de processo, evitando pseudorreplicação. |
| 1ª execução isolada | `first_exec_ms` | Só no XLA: segunda chamada do JIT, já compilado. |
| Energia (estimativa) | `energy_j` | `ttfb/1000·P_compile + exec/1000·P_exec·iters` com potências fixas (`--power-compile`, `--power-exec`). **Não é medição real** — é uma estimativa paramétrica. |
| Speedup | `speedup_exec_x` | Razão eager (ou não-fundido) / compilado. |

Variantes por backend:

- **TorchInductor**: `eager` (baseline), `fused_inductor` (`torch.compile` `mode="max-autotune"`) e `fused_fold_inductor` (mesma compilação, mas com **BN dobrado em Conv** via `fold_bn_inplace` antes do `torch.compile`). Base e fold começam do mesmo `state_dict` com seed 0. O fold cobre BasicBlock (`conv1/bn1`, `conv2/bn2`) **e Bottleneck (`conv3/bn3`)** — valide com `make check_fold` (0 BNs restantes + equivalência numérica). O caminho PyTorch roda com **TF32 habilitado** (`set_float32_matmul_precision("high")`) e **layout `channels_last`** em CUDA.
- **XLA**: `unfused` (forward sem JIT) e `fused_jit` (`jax.jit`). Usa a implementação e o layout nativos do `flaxmodels`, mas fixa `pretrained=None`, remove a normalização de entrada embutida e retorna logits, como os caminhos Torch/TVM; ainda assim, a comparação é entre stacks completos, não entre grafos numericamente idênticos.
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
│   └── resnet_jax.py          # implementações JAX de ResNet-18/50
├── envs/
│   ├── requirements_xla.txt   # Ambiente XLA/JAX + PyTorch (venv 1)
│   └── requirements_tvm.txt   # Ambiente TVM (venv 2; TVM vem do fonte, ver §4.2)
├── scripts/
│   ├── run_full_grid.sh       # Grade completa do artigo (todos backends × shapes) → results/
│   ├── make_tables.py         # JSONs → tabelas LaTeX do artigo + summary.md (sem transcrição manual)
│   ├── check_fold.py          # Valida o fold BN→Conv (0 BNs restantes + equivalência numérica)
│   ├── plot_folds_en.py       # JSONs → equivalencia.png e envelope_exemplo.png
│   ├── plot_ir_figs_en.py     # JSONs → fusion_rate.png e as duas figuras de kernels
│   ├── prune_regenerable.py   # Remove o regenerável sem tocar em dado referenciado
│   └── validate_artifact.py   # Preflight portátil e checklist estrito da versão arquivada
├── ARTIFACT_EVALUATION.md     # Roteiro curto para os avaliadores
├── CITATION.cff               # Metadados de citação
├── Makefile                   # Venvs, smoke tests, check_fold, grid, tables, limpeza de caches
├── ir_dumps/                  # Gerado: artefatos de compilação (Triton/HLO/TVMScript)
├── plots/                     # Gerado: gráficos do envelope T(n)
├── results/                   # Versão arquivada: JSONs medidos por campanha
├── env_reports/               # Gerado: relatórios de ambiente e locks
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
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | `0.75` | Limita a fração de VRAM do JAX. |
| `XLA_LIBRARY_PATH` | `/usr/local/lib64` | Faz o plugin CUDA do JAX carregar a cuDNN 9.11.0 do host; a versão efetiva é registrada no JSON. |
| `XLA_FLAGS` | vazio (padrão do XLA) | Permite sobrescrever flags do XLA; a campanha principal não altera o autotuner. |
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

### 5.2 Medição completa (protocolo ResNet do artigo: K=5, warmup 10, iters 50)

`--compile-repeats 5` cria automaticamente cinco processos isolados por
backend e usa diretórios de cache exclusivos. Não é necessário limpar caches
manualmente entre essas repetições. “Frio” aqui se refere ao cache de código
do compilador; pesos do modelo e o page cache do sistema operacional continuam
compartilhados e ficam fora do intervalo cronometrado. No TorchInductor,
original e fold também são compilados em processos separados, evitando
reaproveitamento em memória entre as duas variantes. Nos processos repetidos
de XLA/TVM, somente a variante JIT/fundida publicada é executada; o baseline
unfused não roda antes dela e, portanto, não aquece o estado do compilador:

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
export TVM_HOME=/caminho/para/tvm
PYTHONPATH=$TVM_HOME/python:$TVM_HOME/build python run_bench.py \
  --no-xla --no-inductor --device cuda --model resnet18 \
  --dtype fp32 --batch 1 --height 224 --width 224 \
  --warmup 10 --iters 50 --compile-repeats 5 \
  --output out_tvm_resnet18_1x3x224x224.json
deactivate
```

Repita variando `--batch/--height/--width` (o artigo usa 1/16/64×224² e 1×512², 1×1024²) e `--model resnet50`.

### 5.2a Auditoria BERT: aplicabilidade do fold

O BERT padrão do Hugging Face é pós-LayerNorm: a saída normalizada também
alimenta o caminho residual. Portanto, absorver o affine da LayerNorm apenas
nas projeções lineares mudaria a função do modelo. O script abaixo mede uma
variante-controle semanticamente idêntica, em processos e caches separados,
para auditar a antiga alegação de fold sem aplicar uma transformação ilegal:

O script mede com `local_files_only=True`, ou seja, **não baixa o modelo** — nenhum
acesso de rede ocorre dentro do intervalo cronometrado. Baixe o checkpoint uma
única vez antes da primeira coleta:

```bash
. .venv_xla/bin/activate
python -c "from transformers import BertModel; BertModel.from_pretrained('bert-base-uncased')"
```

Sem esse passo o script falha com
`OSError: We couldn't connect to 'https://huggingface.co' ...`. O cache padrão
fica em `~/.cache/huggingface` (~1,3 GB para o `bert-base-uncased`); use
`HF_HOME` para escolher outro diretório. Depois:

```bash
python scripts/benchmark_bert_fold.py \
  --batch 1 --seq-len 64 \
  --repeats 5 --warmup 10 --iters 50 \
  --output results/bert_fold/bert_1x64_k5.json
deactivate
```

No contêiner, monte o cache do host (`-v "$HOME/.cache/huggingface:/hf" -e HF_HOME=/hf`).
O script usa `torch` e `transformers`, ambos fixados em `envs/requirements_xla.txt`.
Os três JSONs da campanha publicada acompanham o artefato em `results/bert_fold/`.

Repita para `(1,128)` e `(8,128)`. Cada JSON contém as cinco amostras frias de
compilação, 250 amostras de execução por variante, desvios-padrão, ICs de 95%
sobre as médias de processo, metadados de cache e os artefatos gerados.

### 5.2b Grade completa e tabelas do artigo (um comando cada)

```bash
make grid TVM_HOME=/caminho/para/tvm COMPILE_REPEATS=5
make tables                              # gera results/k5/tables/{table_*.tex, summary.md} a partir dos JSONs
make fold_stats                          # Welch + correção de Holm nas médias de processo
make bert_audit                          # auditoria do fold no BERT nas três formas → results/bert_fold/
make bert_table                          # gera docs/sblp/generated/table_bert_ln_fold.tex dos JSONs
make transformers_audit TVM_HOME=/caminho/para/tvm  # 21 células BERT/GPT-2 K=5
make transformers_tables                # gera as duas tabelas Transformer dos JSONs
```

O `make tables` exige a grade ResNet completa de 30 células K=5. O
`make transformers_tables` exige as 21 células Transformer K=5; ele falha em
vez de preencher uma célula ausente com `--`. Juntos, os geradores produzem as
tabelas do artigo **diretamente dos JSONs** em `results/k5/tables/` e
`docs/sblp/generated/`. Nenhuma célula é preenchida à mão.

A grade Transformer usa configurações base BERT/GPT-2 com pesos aleatórios
determinísticos, dispensando downloads. Para GPT-2, os caminhos PyTorch recebem
uma máscara causal 4D fixa. Antes da importação TVM, o alias exportado
`aten._unsafe_view` é normalizado para o `aten.reshape` equivalente e suportado;
formas e valores não mudam. Cada um dos 21 JSONs arquivados contém cinco
compilações frias, cinco médias de processo, 250 amostras sincronizadas e cinco
caminhos de IR.

Para retomar somente alguns backends, use `RUN_INDUCTOR=0|1`,
`RUN_XLA=0|1` e `RUN_TVM=0|1`; JSONs que já contenham o K solicitado são
ignorados. Por exemplo:

```bash
RUN_INDUCTOR=0 RUN_XLA=0 RUN_TVM=1 \
  make grid TVM_HOME=/caminho/para/tvm COMPILE_REPEATS=5
```

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
- `--compile-repeats K` — K compilações frias em processos independentes (`K=5` nas medições ResNet repetidas do artigo)
- `--compile-repeat-attempts` — máximo de tentativas para obter cada repetição completa; falhas são descartadas e registradas no JSON
- `--compile-repeat-timeout-s` — limite por processo isolado
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

A versão arquivada deve incluir também `results/k5/cache_audit.md`, com as
amostras brutas e a verificação dos artefatos gerados em cada processo. O alvo
`make artifact_submission_check` impede que um pacote sem a grade JSON K=5
completa seja tratado como pronto para submissão.

## 7) ⚠️ Cache do TorchInductor: leia antes de medir PyTorch

**Sintoma.** `*_inductor_output_code.py` vazio em `ir_dumps/inductor/`, contagem de kernels zerada (com um campo `warning` no JSON) e `compile_ms` do Inductor muito menor do que o real.

**Causa.** O TorchInductor mantém um cache persistente de compilação (FX graph cache) em `/tmp/torchinductor_$USER` (mais o cache do Triton em `~/.triton/cache`). A partir da **segunda** execução com o mesmo modelo/shape/versões, o Inductor reutiliza o cache: o codegen não roda, nenhum `output_code` novo é gerado — some o artefato para contar kernels — e o tempo medido vira tempo de *cache hit*, não de compilação.

**Como o repositório trata isso.**

1. `run_bench.py` e o `Makefile` setam `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` por padrão (precisa existir antes do `import torch`; por isso está no topo do `run_bench.py`).
2. O backend também força `torch._inductor.config.force_disable_caches = True` em runtime e captura o código gerado diretamente dos arquivos produzidos pelo Inductor (wrapper + kernels Triton), sem depender de `TORCH_LOGS`.
3. Para garantir compilação "fria" (o número reportado como `compile_ms`), **sempre** rode antes de cada medição do PyTorch:

```bash
make clean_inductor_cache
# remove /tmp/torchinductor_$USER; K>1 usa caches privados por processo
```

**Se mesmo assim vier vazio** (ex.: outra versão de torch com layout de cache diferente), o JSON marca `kernel_count.warning` e `capture_method: stdout_fallback`. Nesse caso: `make clean_inductor_cache`, confirme `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` no ambiente e repita a medição. Este procedimento faz parte do artefato justamente para que avaliadores consigam reproduzir a contagem de kernels do PyTorch. A versão depositada carrega cerca de 130 MB de dumps otimizados e código gerado, incluindo os 105 IRs Transformer referenciados, **exceto** o HLO não-otimizado do XLA: ele embute os pesos como literais (~195 MB por dump, ~7 GB na grade) e é regenerável pelo comando desta seção. O `make cache_audit` conhece essa exceção na grade ResNet — verifica e faz o hash dos 200 artefatos arquivados e marca os 50 HLOs não-otimizados declarados como *declared but not archived*, usando o `code_size_bytes` registrado no JSON. Qualquer outro artefato esperado ausente faz a auditoria falhar.

*Medição de cache quente (opcional).* Para medir de propósito o tempo com cache reaproveitado, rode com `TORCHINDUCTOR_FORCE_DISABLE_CACHES=0` **sem** limpar o cache — e reporte separadamente, pois não é comparável ao `compile_ms` dos outros backends.

## 8) Limitações conhecidas da coleta

- **`compile_ms` do Inductor e do XLA é um proxy** (`1ª chamada − latência média`); o TVM cronometra passes + `relax.build` depois do trace/import FX. As janelas não incluem o mesmo trabalho de frontend, logo a comparação é operacional entre stacks nativos, não uma comparação isonômica apenas dos compiladores.
- **A contagem de kernels é estática** (análise do artefato de compilação). Ela não conta lançamentos reais em GPU; para isso use um profiler (ex.: Nsight Systems) por fora.
- **`energy_j` é estimativa paramétrica**, não medição (não usa NVML/RAPL).
- **TVM parte de um trace FX do PyTorch** — o export fragmenta operadores que o PyTorch trata como atômicos, o que infla a contagem de kernels do TVM (discutido no artigo, Seção *Evaluation*).
- **Precisão numérica assimétrica na execução**: torch (cuDNN) e XLA usam TF32 em GPUs Ampere; os kernels CUDA gerados pelo TVM (dlight) fazem FP32 puro — leve vantagem de execução para os dois primeiros.
- **Compatibilidade JAX/cuDNN**: a cuDNN 9.10.2 instalada no ambiente Python produziu `XlaRuntimeError: INTERNAL` intermitente em batch \(>1\) nesta RTX 3050. A campanha principal carrega explicitamente a cuDNN 9.11.0 do host (`XLA_LIBRARY_PATH=/usr/local/lib64`) e registra `cudnn_build_version` e `cudnn_runtime_version`; um controle ResNet-18, batch 64, completou \(K=5\) sem retry usando o autotuner padrão. O coletor ainda registra qualquer tentativa que falhe e nunca a inclui nas estatísticas.
- **Planos XLA em entradas 1024×1024**: durante o autotuning, alguns planos candidatos pedem workspaces maiores que os 8 GiB disponíveis e são rejeitados com avisos não fatais do alocador; as cinco compilações de cada célula ainda concluem e selecionam um plano executável. Isso pode limitar a melhor latência alcançável pelo XLA nesta GPU.
- **Transformers usam pesos aleatórios e formas fixas.** A grade BERT/GPT-2 é completa e repetida em K=5, mas mede as configurações base inicializadas com seed 0, não a acurácia de checkpoints treinados. O GPT-2 usa a máscara causal 4D e a normalização de alias de exportação descritas na Seção 5.2b.
- **`compile_ms` do TorchInductor depende fortemente da versão do torch** (~2× entre 2.5.1 e 2.9 nesta GPU). O artefato fixa a versão em `envs/requirements_xla.txt`; números de compilação só são comparáveis dentro da mesma versão.
- Os smoke tests (batch 1, poucas iterações) servem para validar o ambiente, **não** para comparar backends — use o protocolo completo (§5.2).

## 9) Reprodutibilidade

```bash
# relatório do ambiente (GPU, driver, versões) → env_reports/environment_report.json
. .venv_xla/bin/activate && python collect.py

# lock do venv XLA → env_reports/requirements_xla.lock.txt
make lock_xla
```

As dependências necessárias do ambiente TVM estão fixadas em
`envs/requirements_tvm.txt`; `collect.py` gera um relatório completo das
versões efetivamente carregadas em cada reprodução.

## 10) Limpeza

```bash
make clean_inductor_cache   # caches do TorchInductor/Triton (ver §7)
make clean                  # remove .venv_xla, .venv_tvm, out_*.json e locks
make prune_check            # relata o que é regenerável, sem apagar
make prune                  # remove o regenerável (tipicamente ~7 GB)
```

O `prune` remove só o que a grade recria e o depósito não arquiva: os HLOs
não-otimizados do XLA (§7), diretórios de IR órfãos de tentativas que falharam,
`torch_compile_debug/`, `artifacts/` e `__pycache__/`. Ele **nunca** toca em JSON
de resultado, tabela gerada, relatório de ambiente ou IR referenciado por algum
JSON: antes de apagar ele recalcula o conjunto referenciado e aborta se a
remoção fosse derrubar qualquer arquivo citado — exceto o HLO não-otimizado, que
os JSONs citam mas o pacote declara como *declared but not archived*. Use
`make prune_check` primeiro para ver o que sairia.

## 10.1) Figuras do artigo

As cinco figuras saem dos mesmos JSONs K=5 que alimentam as tabelas:

```bash
make figures    # equivalencia.png, envelope_exemplo.png, fusion_rate.png,
                # kernels_interno.png, kernels_externo.png
```

O alvo roda no venv XLA porque os scripts dependem de matplotlib. Assim como as
tabelas, nenhuma figura do artigo é montada à mão.

## 11) Artefato Docker

A imagem contém os três backends e preserva os dois ambientes Python
separados exigidos pelas versões incompatíveis de PyTorch. O TVM é compilado
no build a partir do commit `3b60f1c9b8907dcf5d39a033876020e96e6915b2`.

Pré-requisitos do host:

- Docker com NVIDIA Container Toolkit;
- driver NVIDIA compatível com CUDA 12.8;
- GPU passada ao contêiner com `--gpus all`.

```bash
make docker_build             # imagem cnnbench:artifact
make docker_verify            # imports, GPU e equivalência do fold
make docker_smoke             # smoke real em Inductor, XLA e TVM
```

Todas as saídas são gravadas no diretório local `artifacts/`, montado como
`/artifacts` no contêiner. Para executar apenas um backend (em host com
SELinux em `Enforcing`, acrescente `--security-opt label=disable`):

```bash
mkdir -p artifacts
docker run --gpus all --rm \
  -v "$PWD/artifacts:/artifacts:Z" \
  cnnbench:artifact inductor \
  --model resnet18 --device cuda --dtype fp32 \
  --batch 1 --height 224 --width 224 \
  --warmup 10 --iters 50 --compile-repeats 5
```

Troque `inductor` por `xla` ou `tvm`. A grade completa fica disponível com:

```bash
make docker_grid              # equivale a `docker run ... cnnbench:artifact grid`
```

Dentro do contêiner não existe a cuDNN 9.11 do host de referência: o grid
detecta a ausência, segue com a cuDNN do próprio ambiente (9.10.2) e registra
`cudnn_runtime_version` em cada JSON. Isso reproduz o protocolo, mas os
números do XLA podem divergir dos publicados — ver §8. Para exigir a
biblioteca de referência e abortar sem ela, use `XLA_STRICT_CUDNN=1`.

### 11.1 Imagem pronta (pula o build)

O build do zero leva 45--120 minutos porque compila o TVM do fonte. Para evitá-lo,
o registro do Zenodo (`https://doi.org/10.5281/zenodo.21731237`) disponibiliza, após a publicação, a imagem já
construída como **arquivo separado**, `cnnbench-artifact.tar.gz`, ao lado do pacote
do artefato — baixe apenas se quiser pular o build:

```bash
# no diretório onde o arquivo foi baixado, junto do .sha256 publicado
sha256sum --check cnnbench-artifact.tar.gz.sha256
docker load < cnnbench-artifact.tar.gz
docker image ls cnnbench:artifact          # confirma a tag
make docker_verify                          # segue o roteiro normal a partir daqui
```

O `docker load` produz exatamente a tag `cnnbench:artifact` usada por todos os
alvos `docker_*`, então o restante do roteiro não muda. Quem preferir construir
localmente pode ignorar esse arquivo e rodar `make docker_build`.

Para regerar o arquivo a partir de uma imagem local (exige ~10 GiB livres):

```bash
make docker_export          # gera dist/cnnbench-artifact.tar.gz + .sha256
```

## 12) Preparação do depósito no Zenodo

Crie primeiro um rascunho no Zenodo e use a opção **Reserve DOI**. Não invente
o número e ainda não publique o rascunho. Com o DOI específico da versão
reservado, finalize os arquivos locais:

```bash
make set_artifact_doi DOI=10.5281/zenodo.NUMERO
make paper_pdf
make artifact_submission_check
make artifact_package
```

O último comando cria e verifica, inclusive após uma extração limpa:

- `dist/cnnbench-source-v1.0.0.tar.gz`;
- `dist/cnnbench-source-v1.0.0.tar.gz.sha256`.

Envie esses dois arquivos ao mesmo registro. Se quiser oferecer a imagem pronta,
execute também `make docker_export` e envie o `.tar.gz` da imagem e seu
checksum como arquivos separados. O pacote-fonte omite somente os HLOs XLA não
otimizados regeneráveis (cerca de 7 GiB); os JSONs, os 105 IRs Transformer
referenciados, os demais IRs, as seis tabelas e o PDF permanecem no arquivo.

No formulário do Zenodo, selecione o tipo **Software**, acesso **Open**, versão
`1.0.0` e licença **MIT**; copie título, autores, descrição e palavras-chave de
`.zenodo.json`. Confira o DOI dentro do PDF, do README e de `CITATION.cff` antes
de usar **Publish**. Depois da publicação, trate esse registro como imutável e
crie uma nova versão no Zenodo para qualquer correção posterior.
