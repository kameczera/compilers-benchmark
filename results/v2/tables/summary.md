# Resumo gerado por scripts/make_tables.py

## Compile / Exec (média ± sd, ms)

| backend | modelo | shape | compile_ms | exec_ms ± sd | IC95 (±) | status |
|---|---|---|---|---|---|---|
| inductor | resnet18 | 1x3x224x224 | 13263 | 1.94 ± 0.14 | 0.040 |  |
| inductor | resnet18 | 16x3x224x224 | 17367 | 11.58 ± 0.34 | 0.095 |  |
| inductor | resnet18 | 64x3x224x224 | 13688 | 41.17 ± 0.45 | 0.125 |  |
| inductor | resnet18 | 1x3x512x512 | 14378 | 4.88 ± 0.27 | 0.074 |  |
| inductor | resnet18 | 1x3x1024x1024 | 15838 | 14.44 ± 0.34 | 0.094 |  |
| inductor | resnet50 | 1x3x224x224 | 22596 | 3.73 ± 0.10 | 0.029 |  |
| inductor | resnet50 | 16x3x224x224 | 24907 | 29.94 ± 0.11 | 0.029 |  |
| inductor | resnet50 | 64x3x224x224 | 22098 | 108.62 ± 0.19 | 0.054 |  |
| inductor | resnet50 | 1x3x512x512 | 24127 | 11.34 ± 0.27 | 0.076 |  |
| inductor | resnet50 | 1x3x1024x1024 | 22418 | 38.69 ± 0.10 | 0.027 |  |
| tvm | resnet18 | 1x3x224x224 | 7155 | 8.02 ± 0.29 | 0.082 |  |
| tvm | resnet18 | 16x3x224x224 | 7420 | 94.11 ± 0.10 | 0.029 |  |
| tvm | resnet18 | 64x3x224x224 | 7390 | 369.61 ± 1.25 | 0.347 |  |
| tvm | resnet18 | 1x3x512x512 | 6807 | 31.58 ± 0.36 | 0.099 |  |
| tvm | resnet18 | 1x3x1024x1024 | 6881 | 122.57 ± 0.22 | 0.062 |  |
| tvm | resnet50 | 1x3x224x224 | 13044 | 19.37 ± 0.02 | 0.006 |  |
| tvm | resnet50 | 16x3x224x224 | 13362 | 261.87 ± 0.10 | 0.028 |  |
| tvm | resnet50 | 64x3x224x224 | 13856 | 1053.01 ± 9.15 | 2.537 |  |
| tvm | resnet50 | 1x3x512x512 | 13112 | 95.81 ± 0.36 | 0.100 |  |
| tvm | resnet50 | 1x3x1024x1024 | 13068 | 377.32 ± 3.38 | 0.937 |  |
| xla | resnet18 | 1x3x224x224 | 2253 | 2.09 ± 0.14 | 0.039 |  |
| xla | resnet18 | 16x3x224x224 | 2794 | 11.27 ± 0.11 | 0.031 |  |
| xla | resnet18 | 64x3x224x224 | 5214 | 38.37 ± 0.53 | 0.146 |  |
| xla | resnet18 | 1x3x512x512 | 3056 | 4.14 ± 0.12 | 0.032 |  |
| xla | resnet18 | 1x3x1024x1024 | 2953 | 12.59 ± 0.14 | 0.038 |  |

## Fold (TorchInductor): deltas e ponto de equivalência n_eq

| modelo | shape | Δcompile | Δexec | n_eq |
|---|---|---|---|---|
| resnet18 | 1x3x224x224 | -60.6% | -5.4% | 0 |
| resnet18 | 16x3x224x224 | -59.4% | -0.0% | 0 |
| resnet18 | 64x3x224x224 | -46.6% | +0.3% | 60545 |
| resnet18 | 1x3x512x512 | -37.7% | -2.1% | 0 |
| resnet18 | 1x3x1024x1024 | -49.4% | -1.0% | 0 |
| resnet50 | 1x3x224x224 | -31.3% | -0.3% | 0 |
| resnet50 | 16x3x224x224 | -29.3% | -1.6% | 0 |
| resnet50 | 64x3x224x224 | -10.7% | -1.8% | 0 |
| resnet50 | 1x3x512x512 | -32.7% | -4.0% | 0 |
| resnet50 | 1x3x1024x1024 | -13.9% | -3.3% | 0 |

## Kernels (análise estática dos artefatos)

| backend | modelo | shape | contagem |
|---|---|---|---|
| inductor | resnet18 | fused_inductor | {'triton_launches_total': 18, 'triton_kernels_unicos': 13, 'extern_kernels_total': 21, 'extern_funcs_unicas': 2} |
| inductor | resnet18 | fused_fold_inductor | {'triton_launches_total': 18, 'triton_kernels_unicos': 13, 'extern_kernels_total': 21, 'extern_funcs_unicas': 2} |
| inductor | resnet50 | fused_inductor | {'triton_launches_total': 50, 'triton_kernels_unicos': 21, 'extern_kernels_total': 54, 'extern_funcs_unicas': 3} |
| inductor | resnet50 | fused_fold_inductor | {'triton_launches_total': 65, 'triton_kernels_unicos': 20, 'extern_kernels_total': 54, 'extern_funcs_unicas': 3} |
| tvm | resnet18 | unfused | {'tvm_call_tir_total': 62, 'tvm_call_tir_kernels_unicos': 30, 'tvm_cls_total': 0, 'tvm_cls_kernels_unicos': 0} |
| tvm | resnet18 | fused | {'tvm_call_tir_total': 0, 'tvm_call_tir_kernels_unicos': 0, 'tvm_cls_total': 60, 'tvm_cls_kernels_unicos': 28} |
| tvm | resnet50 | unfused | {'tvm_call_tir_total': 160, 'tvm_call_tir_kernels_unicos': 52, 'tvm_cls_total': 0, 'tvm_cls_kernels_unicos': 0} |
| tvm | resnet50 | fused | {'tvm_call_tir_total': 0, 'tvm_call_tir_kernels_unicos': 0, 'tvm_cls_total': 158, 'tvm_cls_kernels_unicos': 50} |
| xla | resnet18 | fused_jit_unoptimized | {'kernels_totais': 0, 'fusion_total': 0, 'fusion_unicos': 0, 'custom_total': 0, 'custom_targets_unicos': 0} |
| xla | resnet18 | fused_jit_optimized | {'kernels_totais': 25, 'fusion_total': 5, 'fusion_unicos': 5, 'custom_total': 20, 'custom_targets_unicos': 1} |
