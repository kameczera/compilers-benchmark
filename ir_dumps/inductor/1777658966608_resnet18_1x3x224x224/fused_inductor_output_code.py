/home/kamei/miniconda3/envs/teste_xla/lib/python3.10/site-packages/torch/backends/cuda/__init__.py:131: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  return torch._C._get_cublas_allow_tf32()
/home/kamei/miniconda3/envs/teste_xla/lib/python3.10/site-packages/torch/_inductor/compile_fx.py:312: UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. Consider setting `torch.set_float32_matmul_precision('high')` for better performance.
  warnings.warn(
Autotune Choices Stats:
{"num_choices": 2, "num_triton_choices": 0, "best_kernel": "bias_addmm", "best_time": 0.016383999958634377}
AUTOTUNE addmm(1x1000, 1x512, 512x1000)
strides: [0, 1], [0, 1], [1, 512]
dtypes: torch.float32, torch.float32, torch.float32
  bias_addmm 0.0164 ms 100.0% 
  addmm 0.0481 ms 34.0% 
SingleProcess AUTOTUNE benchmarking takes 0.0290 seconds and 0.0002 seconds precompiling for 2 choices
