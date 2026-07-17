# Errata: valores da versão submetida × re-medição v2 (código corrigido)

Gerado automaticamente (make_errata). Campanha v2: warmup 10 / iters 50 nos TRÊS backends,
sync por iteração (média ± sd de 50 amostras), fold cobrindo Bottleneck (bn3), janela de
compilação do TVM = passes + build. Ambiente: torch 2.9.0+cu128, jax 0.6.2, TVM 0.22.dev0,
RTX 3050, driver 580.105.08. Compile do Inductor NÃO é comparável ao da versão submetida
(medida em torch 2.5.1) — efeito de versão, ver README §5.2b/§8.

## Tabela 1 (ResNets)

| backend | modelo | shape | comp. paper | comp. v2 | exec paper | exec v2 (±sd) | status |
|---|---|---|---|---|---|---|---|
| inductor | resnet18 | 1x3x224x224 | 8550 | 13263 | 1.76 ± 0.03 | 1.94 ± 0.14 | compile: outra versão de torch |
| inductor | resnet18 | 16x3x224x224 | 6233 | 17367 | 11.78 ± 0.18 | 11.58 ± 0.34 | compile: outra versão de torch |
| inductor | resnet18 | 64x3x224x224 | 6348 | 13688 | 40.51 ± 0.42 | 41.17 ± 0.45 | compile: outra versão de torch |
| inductor | resnet18 | 1x3x512x512 | 6168 | 14378 | 4.70 ± 0.05 | 4.88 ± 0.27 | compile: outra versão de torch |
| inductor | resnet18 | 1x3x1024x1024 | 5929 | 15838 | 15.07 ± 0.16 | 14.44 ± 0.34 | compile: outra versão de torch |
| inductor | resnet50 | 1x3x224x224 | 13902 | 22596 | 3.55 ± 0.04 | 3.73 ± 0.10 | compile: outra versão de torch |
| inductor | resnet50 | 16x3x224x224 | 13125 | 24907 | 29.20 ± 0.35 | 29.94 ± 0.11 | compile: outra versão de torch |
| inductor | resnet50 | 64x3x224x224 | 11952 | 22098 | 107.14 ± 1.29 | 108.62 ± 0.19 | compile: outra versão de torch |
| inductor | resnet50 | 1x3x512x512 | 12373 | 24127 | 10.73 ± 0.13 | 11.34 ± 0.27 | compile: outra versão de torch |
| inductor | resnet50 | 1x3x1024x1024 | 11767 | 22418 | 37.16 ± 0.45 | 38.69 ± 0.10 | compile: outra versão de torch |
| tvm | resnet18 | 1x3x224x224 | 10273 | 7155 | 6.94 ± 0.07 | 8.02 ± 0.29 | exec do paper subestimado (bug de sync) |
| tvm | resnet18 | 16x3x224x224 | 10004 | 7420 | 82.53 ± 0.62 | 94.11 ± 0.10 | exec do paper subestimado (bug de sync) |
| tvm | resnet18 | 64x3x224x224 | 9899 | 7390 | 321.09 ± 1.90 | 369.61 ± 1.25 | exec do paper subestimado (bug de sync) |
| tvm | resnet18 | 1x3x512x512 | 4992 | 6807 | 27.00 ± 0.31 | 31.58 ± 0.36 | exec do paper subestimado (bug de sync) |
| tvm | resnet18 | 1x3x1024x1024 | 9256 | 6881 | 103.97 ± 0.84 | 122.57 ± 0.22 | exec do paper subestimado (bug de sync) |
| tvm | resnet50 | 1x3x224x224 | 10273 | 13044 | 19.47 ± 0.18 | 19.37 ± 0.02 | compile do paper = coluna duplicada do r18 |
| tvm | resnet50 | 16x3x224x224 | 10004 | 13362 | 247.63 ± 2.23 | 261.87 ± 0.10 | exec do paper subestimado (bug de sync); compile do paper = coluna duplicada do r18 |
| tvm | resnet50 | 64x3x224x224 | 9896 | 13856 | 986.00 ± 8.87 | 1053.01 ± 9.15 | exec do paper subestimado (bug de sync); compile do paper = coluna duplicada do r18 |
| tvm | resnet50 | 1x3x512x512 | 9648 | 13112 | 91.01 ± 0.82 | 95.81 ± 0.36 | exec do paper subestimado (bug de sync); compile do paper = coluna duplicada do r18 |
| tvm | resnet50 | 1x3x1024x1024 | 9256 | 13068 | 347.04 ± 3.12 | 377.32 ± 3.38 | exec do paper subestimado (bug de sync); compile do paper = coluna duplicada do r18 |
| xla | resnet18 | 1x3x224x224 | 2457 | 2253 | 1.99 ± 0.03 | 2.09 ± 0.14 | ok |
| xla | resnet18 | 16x3x224x224 | 2917 | 2794 | 11.39 ± 0.15 | 11.27 ± 0.11 | ok |
| xla | resnet18 | 64x3x224x224 | 4640 | 5214 | 38.79 ± 0.39 | 38.37 ± 0.53 | ok |
| xla | resnet18 | 1x3x512x512 | 3050 | 3056 | 4.17 ± 0.05 | 4.14 ± 0.12 | ok |
| xla | resnet18 | 1x3x1024x1024 | 3297 | 2953 | 12.75 ± 0.12 | 12.59 ± 0.14 | ok |
| xla | resnet50 | 1x3x224x224 | 4748 | — | 4.21 ± 0.04 | — | não medível: harness só roda ResNet-18 no XLA |
| xla | resnet50 | 16x3x224x224 | 5936 | — | 29.71 ± 0.30 | — | não medível: harness só roda ResNet-18 no XLA |
| xla | resnet50 | 64x3x224x224 | 9581 | — | 101.00 ± 1.01 | — | não medível: harness só roda ResNet-18 no XLA |
| xla | resnet50 | 1x3x512x512 | 4671 | — | 10.70 ± 0.11 | — | não medível: harness só roda ResNet-18 no XLA |
| xla | resnet50 | 1x3x1024x1024 | 6028 | — | 33.77 ± 0.34 | — | não medível: harness só roda ResNet-18 no XLA |

## Tabelas de fold (TorchInductor)

Na versão submetida o fold do ResNet-50 era **incompleto** (bn3 dos Bottlenecks não era
dobrado — 96 menções a batch_norm restavam no dump). Na v2 o fold é completo (0 menções).

| modelo | shape | Δcomp paper | Δcomp v2 | Δexec paper | Δexec v2 | n_eq v2 |
|---|---|---|---|---|---|---|
| resnet18 | 1x3x224x224 | -49.6% | -60.6% | +11.9% | -5.4% | 0 |
| resnet18 | 16x3x224x224 | -58.8% | -59.4% | -0.7% | -0.0% | 0 |
| resnet18 | 64x3x224x224 | -56.0% | -46.6% | +5.4% | +0.3% | 60545 |
| resnet18 | 1x3x512x512 | -52.8% | -37.7% | +2.1% | -2.1% | 0 |
| resnet18 | 1x3x1024x1024 | -56.4% | -49.4% | -0.1% | -1.0% | 0 |
| resnet50 | 1x3x224x224 | -17.5% | -31.3% | -2.0% | -0.3% | 0 |
| resnet50 | 16x3x224x224 | -43.6% | -29.3% | -1.3% | -1.6% | 0 |
| resnet50 | 64x3x224x224 | -28.2% | -10.7% | -1.1% | -1.8% | 0 |
| resnet50 | 1x3x512x512 | -37.3% | -32.7% | -1.2% | -4.0% | 0 |
| resnet50 | 1x3x1024x1024 | -35.3% | -13.9% | -0.2% | -3.3% | 0 |

