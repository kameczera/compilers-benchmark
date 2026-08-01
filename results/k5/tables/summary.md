# Resumo gerado por scripts/make_tables.py

## Compile / Exec (média ± sd, ms)

| backend | modelo | shape | compile_ms ± sd | compile IC95 (±) | exec_ms ± sd | exec IC95 (±) | status |
|---|---|---|---|---|---|---|---|
| inductor | resnet18 | 1x3x224x224 | 11765 ± 1036 | 1285.7 | 1.88 ± 0.03 | 0.041 |  |
| inductor | resnet18 | 16x3x224x224 | 14958 ± 907 | 1125.7 | 11.86 ± 0.16 | 0.196 |  |
| inductor | resnet18 | 64x3x224x224 | 13116 ± 791 | 981.5 | 42.43 ± 0.13 | 0.163 |  |
| inductor | resnet18 | 1x3x512x512 | 15544 ± 1364 | 1693.2 | 4.85 ± 0.08 | 0.104 |  |
| inductor | resnet18 | 1x3x1024x1024 | 16502 ± 1110 | 1378.1 | 15.03 ± 0.17 | 0.217 |  |
| inductor | resnet50 | 1x3x224x224 | 24424 ± 1539 | 1910.2 | 3.78 ± 0.08 | 0.101 |  |
| inductor | resnet50 | 16x3x224x224 | 22519 ± 1094 | 1358.6 | 31.09 ± 0.11 | 0.133 |  |
| inductor | resnet50 | 64x3x224x224 | 21689 ± 586 | 728.0 | 112.51 ± 0.24 | 0.302 |  |
| inductor | resnet50 | 1x3x512x512 | 23221 ± 1411 | 1751.3 | 11.85 ± 0.09 | 0.112 |  |
| inductor | resnet50 | 1x3x1024x1024 | 23834 ± 548 | 679.9 | 40.19 ± 0.14 | 0.168 |  |
| tvm | resnet18 | 1x3x224x224 | 8104 ± 79 | 98.5 | 8.26 ± 0.07 | 0.091 |  |
| tvm | resnet18 | 16x3x224x224 | 8333 ± 42 | 52.3 | 97.56 ± 0.06 | 0.079 |  |
| tvm | resnet18 | 64x3x224x224 | 8216 ± 36 | 44.6 | 382.88 ± 1.19 | 1.482 |  |
| tvm | resnet18 | 1x3x512x512 | 7734 ± 34 | 42.4 | 33.01 ± 0.13 | 0.164 |  |
| tvm | resnet18 | 1x3x1024x1024 | 7727 ± 44 | 54.4 | 128.08 ± 0.31 | 0.387 |  |
| tvm | resnet50 | 1x3x224x224 | 13978 ± 28 | 35.2 | 20.01 ± 0.07 | 0.085 |  |
| tvm | resnet50 | 16x3x224x224 | 14318 ± 130 | 161.6 | 270.32 ± 0.54 | 0.665 |  |
| tvm | resnet50 | 64x3x224x224 | 14184 ± 47 | 58.9 | 1063.30 ± 1.27 | 1.578 |  |
| tvm | resnet50 | 1x3x512x512 | 13564 ± 74 | 92.1 | 97.22 ± 0.22 | 0.272 |  |
| tvm | resnet50 | 1x3x1024x1024 | 13577 ± 107 | 133.0 | 383.86 ± 0.37 | 0.465 |  |
| xla | resnet18 | 1x3x224x224 | 1829 ± 12 | 15.0 | 2.13 ± 0.04 | 0.044 |  |
| xla | resnet18 | 16x3x224x224 | 2408 ± 27 | 33.2 | 11.61 ± 0.05 | 0.060 |  |
| xla | resnet18 | 64x3x224x224 | 4560 ± 100 | 123.6 | 38.71 ± 0.12 | 0.153 |  |
| xla | resnet18 | 1x3x512x512 | 2599 ± 54 | 67.0 | 4.23 ± 0.02 | 0.030 |  |
| xla | resnet18 | 1x3x1024x1024 | 2663 ± 76 | 94.4 | 12.94 ± 0.01 | 0.018 |  |
| xla | resnet50 | 1x3x224x224 | 3503 ± 85 | 105.4 | 4.19 ± 0.05 | 0.062 |  |
| xla | resnet50 | 16x3x224x224 | 4800 ± 33 | 40.8 | 29.29 ± 0.13 | 0.158 |  |
| xla | resnet50 | 64x3x224x224 | 9352 ± 121 | 150.4 | 105.22 ± 0.26 | 0.325 |  |
| xla | resnet50 | 1x3x512x512 | 4013 ± 27 | 33.5 | 10.38 ± 0.04 | 0.053 |  |
| xla | resnet50 | 1x3x1024x1024 | 5011 ± 73 | 90.7 | 34.54 ± 0.02 | 0.024 |  |

## Fold (TorchInductor): deltas e crossover de custo total

| modelo | shape | Δcompile | Δexec | n_cross (0 = fold domina) |
|---|---|---|---|---|
| resnet18 | 1x3x224x224 | -41.1% | -0.9% | 0 |
| resnet18 | 16x3x224x224 | -39.7% | -0.6% | 0 |
| resnet18 | 64x3x224x224 | -29.5% | -0.4% | 0 |
| resnet18 | 1x3x512x512 | -34.4% | -1.0% | 0 |
| resnet18 | 1x3x1024x1024 | -38.3% | -0.9% | 0 |
| resnet50 | 1x3x224x224 | -28.6% | -1.8% | 0 |
| resnet50 | 16x3x224x224 | -12.5% | -1.5% | 0 |
| resnet50 | 64x3x224x224 | -1.9% | -2.1% | 0 |
| resnet50 | 1x3x512x512 | -16.3% | -3.9% | 0 |
| resnet50 | 1x3x1024x1024 | -11.8% | -3.2% | 0 |

## Kernels (análise estática dos artefatos)

| backend | modelo | shape | contagem |
|---|---|---|---|
| inductor | resnet18 | fused_inductor | {'triton_launches_total': 18, 'triton_kernels_unicos': 13, 'extern_kernels_total': 21, 'extern_funcs_unicas': 2, 'direct_aten_calls_total': 0} |
| inductor | resnet18 | fused_fold_inductor | {'triton_launches_total': 18, 'triton_kernels_unicos': 13, 'extern_kernels_total': 21, 'extern_funcs_unicas': 2, 'direct_aten_calls_total': 0} |
| inductor | resnet50 | fused_inductor | {'triton_launches_total': 50, 'triton_kernels_unicos': 21, 'extern_kernels_total': 54, 'extern_funcs_unicas': 3, 'direct_aten_calls_total': 0} |
| inductor | resnet50 | fused_fold_inductor | {'triton_launches_total': 65, 'triton_kernels_unicos': 20, 'extern_kernels_total': 54, 'extern_funcs_unicas': 3, 'direct_aten_calls_total': 0} |
| tvm | resnet18 | fused | {'tvm_call_tir_total': 0, 'tvm_call_tir_kernels_unicos': 0, 'tvm_cls_total': 60, 'tvm_cls_kernels_unicos': 28} |
| tvm | resnet50 | fused | {'tvm_call_tir_total': 0, 'tvm_call_tir_kernels_unicos': 0, 'tvm_cls_total': 158, 'tvm_cls_kernels_unicos': 50} |
| xla | resnet18 | fused_jit_unoptimized | {'kernels_totais': 0, 'fusion_total': 0, 'fusion_unicos': 0, 'custom_total': 0, 'custom_targets_unicos': 0} |
| xla | resnet18 | fused_jit_optimized | {'kernels_totais': 28, 'fusion_total': 8, 'fusion_unicos': 8, 'custom_total': 20, 'custom_targets_unicos': 1} |
| xla | resnet50 | fused_jit_unoptimized | {'kernels_totais': 0, 'fusion_total': 0, 'fusion_unicos': 0, 'custom_total': 0, 'custom_targets_unicos': 0} |
| xla | resnet50 | fused_jit_optimized | {'kernels_totais': 64, 'fusion_total': 11, 'fusion_unicos': 11, 'custom_total': 53, 'custom_targets_unicos': 1} |
