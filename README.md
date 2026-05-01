# CNN Compilers Benchmark — README

> Benchmark de *operator fusion*, tempo de compilação e latência entre **XLA (JAX)**, **TorchInductor (PyTorch)** e, opcionalmente, **TVM**. O resultado é consolidado em JSON e pode ser usado pelo modelo de custo \(T_b(n)=a_b\cdot n+b_b\).

## 1) Visão geral

Este projeto mede:

- **TTFB / primeira chamada**: tempo até a primeira inferência, incluindo compilação quando o backend compila sob demanda.
- **`compile_ms`**: estimativa do custo fixo de compilação.
- **`exec_ms`**: latência média por inferência após warmup.
- **Artefatos de compilação**: HLO no XLA/JAX, código TorchInductor/Triton no PyTorch e, quando usado, TVMScript/TIR no TVM.

Backends principais:

- **XLA (JAX)**
- **TorchInductor (PyTorch)**
- **TVM** em ambiente separado, quando necessário.

## 2) Estrutura esperada do repositório

```text
.
├── run_bench.py              # Script principal do benchmark
├── backends/
│   ├── xla_backend.py
│   ├── pytorch_backend.py
│   └── tvm_backend.py
├── models/
│   ├── resnet_jax.py
│   └── resnet_torch.py
├── Makefile                  # Atalhos para criar envs e rodar smoke tests
├── requirements_xla.txt      # Ambiente XLA + PyTorch
├── requirements_tvm.txt      # Opcional: ambiente TVM separado
└── out_*.json                # Saídas geradas pelo benchmark
```

## 3) Pré-requisitos

Ambiente usado para validar `requirements_xla.txt`:

```text
Python: 3.10.18
Sistema: Linux x86_64
GPU: NVIDIA GeForce RTX 3050 8 GB
Driver NVIDIA: 580.105.08
CUDA reportado pelo nvidia-smi: 13.0
NVCC do sistema: CUDA 12.6 / V12.6.77
LLVM do sistema: 19.1.7
JAX: 0.6.2
jaxlib: 0.6.2
PyTorch: 2.9.0+cu128
TorchVision: 0.24.0+cu128
cuDNN via pacote Python NVIDIA: 9.10.2.21
```

Observações:

- O `requirements_xla.txt` instala as bibliotecas Python necessárias para **XLA/JAX + PyTorch/TorchInductor**.
- O driver NVIDIA, `nvcc` e LLVM são dependências de sistema; eles não são instalados por `pip`.
- O TVM pode exigir outro conjunto de versões, por isso continua recomendado em ambiente separado.

## 4) Por que precisamos dos exports do XLA/JAX?

Ao rodar JAX/XLA em GPU, o JAX costuma reservar uma grande parte da VRAM na primeira operação. Isso é útil para reduzir overhead e fragmentação, mas pode gerar erro de alocação quando:

- PyTorch/TorchInductor rodou antes e deixou cache de CUDA ocupado;
- o ambiente gráfico, VS Code, notebook ou outro processo já usa VRAM;
- XLA precisa materializar constantes/artefatos de compilação e não encontra um bloco livre.

O erro pode parecer estranho, por exemplo:

```text
XlaRuntimeError: INTERNAL: Failed to allocate 9437184 bytes for new constant
```

Mesmo parecendo pouca memória, o problema pode ser reserva prévia, fragmentação ou disputa de memória com outro runtime.

Por isso, os testes rápidos usam:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.40
```

Significado:

- `XLA_PYTHON_CLIENT_PREALLOCATE=false`: desliga a pré-alocação agressiva de VRAM pelo JAX. A memória passa a ser alocada sob demanda.
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.40`: limita a fração de VRAM que o JAX tenta reservar quando a pré-alocação está ativa. Mantemos mesmo assim como proteção adicional e documentação do regime de teste.

Essas variáveis precisam existir **antes do primeiro `import jax`** no processo Python. O `Makefile` já exporta as duas por padrão.

## 5) Caminho rápido com Conda

Crie um ambiente novo a partir do `requirements_xla.txt`:

```bash
make conda_xla_env CONDA_ENV_XLA=teste_xla
```

Cheque se PyTorch e JAX veem a GPU:

```bash
make conda_check_xla CONDA_ENV_XLA=teste_xla
```

Rode o smoke test conjunto, sem TVM:

```bash
make conda_smoke_xla_torch CONDA_ENV_XLA=teste_xla
```

Esse comando executa uma configuração pequena:

```text
ResNet-18, batch=1, 224x224, fp32, warmup=3, iters=5
```

e gera:

```text
out_smoke_xla_torch.json
```

Para testar separadamente:

```bash
make conda_smoke_xla CONDA_ENV_XLA=teste_xla      # apenas XLA/JAX
make conda_smoke_torch CONDA_ENV_XLA=teste_xla    # apenas PyTorch/TorchInductor
```

Se o teste separado funciona e o conjunto falha, o problema provavelmente é disputa de memória entre os runtimes no mesmo processo. Nesse caso, use os testes separados para validar o ambiente e rode benchmarks finais em processos isolados.

## 6) Caminho alternativo com venv

```bash
make xla_env
make check_xla
make smoke_xla_torch
```

Os alvos `smoke_xla`, `smoke_torch` e `smoke_xla_torch` seguem a mesma lógica dos alvos Conda.

## 7) Execução manual

Ative o ambiente e exporte as variáveis antes de rodar:

```bash
conda activate teste_xla

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.40

python run_bench.py \
  --no-tvm \
  --device cuda \
  --model resnet18 \
  --dtype fp32 \
  --batch 1 \
  --height 224 \
  --width 224 \
  --warmup 3 \
  --iters 5 \
  --output out_smoke_xla_torch.json
```

Apenas XLA/JAX:

```bash
python run_bench.py \
  --no-tvm --no-inductor \
  --device cuda \
  --model resnet18 \
  --dtype fp32 \
  --batch 1 \
  --height 224 \
  --width 224 \
  --warmup 3 \
  --iters 5 \
  --output out_smoke_xla.json
```

Apenas PyTorch/TorchInductor:

```bash
python run_bench.py \
  --no-tvm --no-xla \
  --device cuda \
  --model resnet18 \
  --dtype fp32 \
  --batch 1 \
  --height 224 \
  --width 224 \
  --warmup 3 \
  --iters 5 \
  --output out_smoke_torch.json
```

## 8) Principais flags do benchmark

- `--device cuda|cpu`
- `--model resnet18|resnet50`
- `--dtype fp32|bf16|fp16`
- `--batch`, `--height`, `--width`
- `--warmup`, `--iters`
- `--output arquivo.json`
- `--no-xla`, `--no-inductor`, `--no-tvm`

## 9) Como saber se está certo?

Depois do teste rápido, confira:

```bash
python -m pip check
```

E verifique o JSON:

```bash
cat out_smoke_xla_torch.json
```

O ideal é não haver campos como:

```text
xla_error
inductor_error
```

Também é bom confirmar:

```bash
nvidia-smi
```

Se o JAX cair para CPU, o teste não é válido para comparação GPU.

## 10) Limpeza

Remover ambiente Conda de teste:

```bash
make conda_clean_xla CONDA_ENV_XLA=teste_xla
```

Remover venvs e JSONs gerados pelo Makefile:

```bash
make clean
```
