# CNN Compilers Benchmark — README

> Comparação justa de *operator fusion* e latência entre **XLA (JAX)**, **TorchInductor (PyTorch)** e **TVM**, com protocolo comum e seleção do backend ótimo via modelo de custo \(T_b(n)=a\_b\cdot n + b\_b\).

## 1) Visão geral

Este projeto mede o tempo de **compilação** (TTFB), **latência média** por inferência e **granularidade de fusão** de três compiladores de DL:
- **XLA (JAX)**
- **TorchInductor (PyTorch)**
- **TVM** (em ambiente separado, devido a restrições de versão)

O benchmark consolida os resultados em **JSON** e calcula recomendações usando a **envoltória inferior** do custo \(T_b(n)=a\_b n + b\_b\), indicando qual backend é preferível para um dado número de execuções **n**.

## 2) Estrutura do repositório

```
.
├── run_bench.py              # Script principal do benchmark (CLI)
├── resnet_torch.py           # Modelos Torch (ResNet-18/50)
├── resnet_jax.py             # Modelo JAX/Flax (ResNet-18)
├── Makefile                  # Criação de ambientes e comandos utilitários
├── requirements_xla.txt      # Requisitos para ambiente XLA+PyTorch (mesmo venv)
├── requirements_tvm.txt      # Requisitos para ambiente TVM (venv separado)
└── out_*.json                # Saídas do benchmark (geradas em execução)
```

> XLA (JAX) e TorchInductor compartilham **o mesmo venv** para garantir **mesmas versões** de Python/CUDA/cuDNN/cuBLAS (comparação justa). O **TVM** roda **isolado**, pois depende de um conjunto de versões diferente para funcionar corretamente.

## 3) Pré‑requisitos

- GPU NVIDIA com driver recente (CUDA 12.x compatível com os wheels instalados).
- Python 3.11 (padrão) para o ambiente **XLA+Torch**.  
  - Você pode mudar com `PYTHON_BIN=python3.10` (ou outro) ao criar o ambiente.
- Python para o ambiente **TVM** (padrão = mesmo do XLA+Torch).  
  - Se precisar, defina `PYTHON_TVM_BIN=python3.10` ao criar o ambiente TVM.
- `make` e acesso à internet para baixar as dependências.

## 4) Instalação dos ambientes

Crie os dois ambientes virtuais com:

```bash
make xla_env      # XLA + PyTorch no MESMO venv (mesmas libs de sistema)
make tvm_env      # TVM em venv separado
```

O alvo `xla_env`:
- Instala `requirements_xla.txt` (derivado do seu artefato JAX).
- Instala **PyTorch + CUDA** a partir do canal oficial, tentando automaticamente: `cu128 → cu126 → cu124 → cu121`.
- Faz *sanity checks* de GPU para **PyTorch** e **JAX**.

Você pode forçar versões/canais, por exemplo:

```bash
make xla_env TORCH_VERSION=2.9.0 TORCHVISION_VERSION=0.24.0 TORCH_CU_CHOICES="cu126 cu124"
make tvm_env  PYTHON_TVM_BIN=python3.10
```

## 5) Execução rápida

Depois de criar os ambientes:

```bash
# XLA + Torch (mesmo venv)
make test_xla

# TVM (venv separado)
make test_tvm
```

Esses comandos executam **ResNet-18** e **ResNet-50** em GPU e geram saídas `out_*.json`.

## 6) Uso avançado do benchmark (CLI)

Ative o venv apropriado e chame `run_bench.py` diretamente com os *flags* que precisar.

**XLA + Torch no mesmo ambiente**:
```bash
source .venv_xla/bin/activate
python run_bench.py \
  --no-tvm \
  --device cuda \
  --model resnet18 \
  --warmup 10 --iters 100 \
  --batch 32 --height 224 --width 224 \
  --dtype fp32 \
  --output out_xla_torch_resnet18.json
```

**TVM em ambiente separado**:
```bash
source .venv_tvm/bin/activate
python run_bench.py \
  --no-xla --no-inductor \
  --device cuda \
  --model resnet18 \
  --warmup 10 --iters 100 \
  --batch 32 --height 224 --width 224 \
  --dtype fp32 \
  --output out_tvm_resnet18.json
```

### Principais flags suportadas
- `--device {cuda,cpu}`
- `--model {resnet18,resnet50,...}`
- `--batch`, `--height`, `--width` → define o *shape* (NCHW para Torch; NHWC para JAX com adaptação interna)
- `--dtype {fp32,bf16,fp16}`
- `--warmup`, `--iters` → número de execuções para aquecimento e medição
- `--compile-budget-ms` → opcional, orçamento de compilação (limite)
- `--output <arquivo.json>` → caminho do JSON de saída
- `--no-xla`, `--no-inductor`, `--no-tvm` → desabilita backends específicos

> Dica: deixe **todos** habilitados no XLA+Torch quando quiser comparar os três (desde que TVM esteja instalado no mesmo venv, o que **não** é o caso por padrão). Para a comparação **justa** XLA vs Torch, mantenha apenas `--no-tvm` no venv XLA; e rode o TVM separado.

## 7) Formato da saída (JSON)

Cada execução salva um JSON com chaves como:
- `meta`: informações do experimento (device, dtype, warmup, iters, shape, model).
- `raw`: métricas por backend (tempo de compilação, latência média, etc.).
- `recommendation`: resultado da seleção pelo modelo \(T_b(n)=a n + b\); inclui `best` e, quando disponível, intervalos de dominância/cruzamento.
- `errors`: mensagens de erro por backend, se houver.

O script imprime algo como:
```
[OK] Saved: /caminho/out_xxx.json
Best backend: XLA
```
ou um aviso caso não seja possível selecionar o melhor (verifique `errors` no JSON).# compilers-benchmark
