# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/3j/c3jk6mg2lc2xfg7zi77fwaaesmkg537cslmzrw7slhi3r5mg5mjc.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'kernel_num_gb': 0.002408448, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x0), tmp4, None)


def get_args():
    arg_0 = rand_strided((1, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 200704,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=40)
    num_gb = 0.002408448
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/5v/c5vqntiv7mo6rv4baieqwlqdhur64wus4dwszqhd2xjvjpzmye3l.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'kernel_num_gb': 0.003211264, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, None)


def get_args():
    arg_0 = rand_strided((1, 128, 56, 56), (401408, 1, 7168, 128), device='cuda:0', dtype=torch.float32)
    return arg_0, 401408,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=40)
    num_gb = 0.003211264
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/a3/ca3zsfljkpr57223swqssivrqjcogpvnbsv4pfgqkiffvu4fle3h.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'kernel_num_gb': 0.000802816, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)


def get_args():
    arg_0 = rand_strided((1, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    return arg_0, 100352,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=40)
    num_gb = 0.000802816
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/cg/ccgz3bdgi4oyhkzx3qcuddbnb4c4prrjmg3y62nsoiteydpm4dqd.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'kernel_num_gb': 0.004816896, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x0), tmp4, None)


def get_args():
    arg_0 = rand_strided((1, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 401408,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=40)
    num_gb = 0.004816896
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/cl/cclzs3uy6unisgpis4keo4rzxhagnqs333n647hulgqjcc5zymgo.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'kernel_num_gb': 0.009633792, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x0), tmp4, None)


def get_args():
    arg_0 = rand_strided((1, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 802816,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=40)
    num_gb = 0.009633792
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/g6/cg6rqtokorapatn7454sgz3hee4ezyof4uwporhzdyfpbctc6eh5.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'kernel_num_gb': 0.001212416, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 2048)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x2), xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(in_out_ptr0 + (x2), tmp6, xmask)


def get_args():
    arg_0 = rand_strided((1, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 100352,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=40)
    num_gb = 0.001212416
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/jh/cjhcgqeodnfqruuatfzrwcl5q33rltbjk4qhe6lbmqgt2ezm75l3.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'kernel_num_gb': 0.000200704, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)


def get_args():
    arg_0 = rand_strided((1, 512, 7, 7), (25088, 1, 3584, 512), device='cuda:0', dtype=torch.float32)
    return arg_0, 25088,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=40)
    num_gb = 0.000200704
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/ju/cjudnhsoewxvpxohjbdwe2n7pmcwocyvfgdsmlvudi3ynzxblxyb.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'kernel_num_gb': 0.001204224, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)


def get_args():
    arg_0 = rand_strided((1, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1, 2048, 7, 7), (100352, 1, 14336, 2048), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 100352,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=40)
    num_gb = 0.001204224
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/kp/ckpamrpfo7canqx6sdgzrnzggxkxwcuaar7whlctjw4qglid3gny.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'kernel_num_gb': 0.001605632, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, None)


def get_args():
    arg_0 = rand_strided((1, 256, 28, 28), (200704, 1, 7168, 256), device='cuda:0', dtype=torch.float32)
    return arg_0, 200704,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=40)
    num_gb = 0.001605632
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/lq/clqq32b3i36c6qg6eppktk6hnolalql563ejhqsi4ptkb6znri37.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'kernel_num_gb': 0.002412544, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(in_out_ptr0 + (x2), tmp6, None)


def get_args():
    arg_0 = rand_strided((1, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1, 1024, 14, 14), (200704, 1, 14336, 1024), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 200704,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=40)
    num_gb = 0.002412544
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/ql/cql5ivdlspiofraoc4zjqvjnaddodmdd7gg2pyspkqcfopbii2qx.py =====
# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p
from torch._C import _cuda_getCurrentRawStream as get_raw_stream



# kernel path: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/rd/crddz7i2hfomvnbtzww3u6cal3bvtcyfbrbkq7oacywi5b2a25qa.py
# Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   x => convolution
#   x_1 => relu
# Graph fragment:
#   %buf0 : Tensor "f32[1, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = PlaceHolder[target=buf0]
#   %arg1_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %convolution : Tensor "f32[1, 64, 112, 112][802816, 1, 7168, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [2, 2], [3, 3], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 64, 112, 112][802816, 1, 7168, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   return %relu
triton_poi_fused_convolution_relu_0 = async_compile.triton('triton_poi_fused_convolution_relu_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 9634048}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/ki/ckizf6rhalxt7vthcmd4t6yoeubdum6bhnd2an3cwreg4ppagt6c.py
# Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x => convolution
#   x_1 => relu
#   x_2 => _low_memory_max_pool_with_offsets
# Graph fragment:
#   %relu : Tensor "f32[1, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = PlaceHolder[target=relu]
#   %convolution : Tensor "f32[1, 64, 112, 112][802816, 1, 7168, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [2, 2], [3, 3], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : Tensor "f32[1, 64, 112, 112][802816, 1, 7168, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu, [3, 3], [2, 2], [1, 1], [1, 1], False), kwargs = {})
#   return %getitem
triton_poi_fused_convolution_max_pool2d_with_indices_relu_1 = async_compile.triton('triton_poi_fused_convolution_max_pool2d_with_indices_relu_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 8830976}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex // 3584
    x1 = ((xindex // 64) % 56)
    x0 = (xindex % 64)
    x4 = xindex
    tmp0 = (-1) + 2*x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-7232) + x0 + 128*x1 + 14336*x2), tmp10, other=float("-inf"))
    tmp12 = 2*x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-7168) + x0 + 128*x1 + 14336*x2), tmp16, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp11, tmp17)
    tmp19 = 1 + 2*x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-7104) + x0 + 128*x1 + 14336*x2), tmp23, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp18, tmp24)
    tmp26 = 2*x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-64) + x0 + 128*x1 + 14336*x2), tmp30, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp25, tmp31)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x0 + 128*x1 + 14336*x2), tmp33, other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp32, tmp34)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (64 + x0 + 128*x1 + 14336*x2), tmp36, other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp35, tmp37)
    tmp39 = 1 + 2*x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (7104 + x0 + 128*x1 + 14336*x2), tmp43, other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp38, tmp44)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (7168 + x0 + 128*x1 + 14336*x2), tmp46, other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp45, tmp47)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (7232 + x0 + 128*x1 + 14336*x2), tmp49, other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp48, tmp50)
    tl.store(out_ptr0 + (x4), tmp51, None)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/5n/c5n3lz3sm6mk4iwxzttmlca6ctuevvoxvsxunidgnoba7f2xb4qx.py
# Topologically Sorted Source Nodes: [out, out_1], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   out => convolution_1
#   out_1 => relu_1
# Graph fragment:
#   %buf3 : Tensor "f32[3136, 64][64, 1]cuda:0" = PlaceHolder[target=buf3]
#   %convolution_1 : Tensor "f32[1, 64, 56, 56][200704, 1, 3584, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg3_1, %arg4_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 64, 56, 56][200704, 1, 3584, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   return %relu_1
triton_poi_fused_convolution_relu_2 = async_compile.triton('triton_poi_fused_convolution_relu_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 2408448}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_2(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/3i/c3ifg4cngegy6vw4icwowlrwtk4d66bywfsnf6ilxxov66socsad.py
# Topologically Sorted Source Nodes: [out, out_1, out_2, out_3, out_4], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   out => convolution_1
#   out_1 => relu_1
#   out_2 => convolution_2
#   out_3 => relu_2
#   out_4 => convolution_3
# Graph fragment:
#   %buf5 : Tensor "f32[1, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = PlaceHolder[target=buf5]
#   %arg6_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %convolution_1 : Tensor "f32[1, 64, 56, 56][200704, 1, 3584, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg3_1, %arg4_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 64, 56, 56][200704, 1, 3584, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %convolution_2 : Tensor "f32[1, 64, 56, 56][200704, 1, 3584, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_1, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[1, 64, 56, 56][200704, 1, 3584, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : Tensor "f32[1, 256, 56, 56][802816, 1, 14336, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf6
triton_poi_fused_convolution_relu_3 = async_compile.triton('triton_poi_fused_convolution_relu_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 2408704}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/x7/cx76xekonbbq6erlyvkttofd7dllhgm2eyje6rphdcabzstpu7se.py
# Topologically Sorted Source Nodes: [out, out_1, out_2, out_3, out_4, input_1, out_5, out_6], Original ATen: [aten.convolution, aten.relu, aten.add]
# Source node to ATen node mapping:
#   input_1 => convolution_4
#   out => convolution_1
#   out_1 => relu_1
#   out_2 => convolution_2
#   out_3 => relu_2
#   out_4 => convolution_3
#   out_5 => add
#   out_6 => relu_3
# Graph fragment:
#   %buf7 : Tensor "f32[3136, 256][256, 1]cuda:0" = PlaceHolder[target=buf7]
#   %buf8 : Tensor "f32[3136, 256][256, 1]cuda:0" = PlaceHolder[target=buf8]
#   %convolution_1 : Tensor "f32[1, 64, 56, 56][200704, 1, 3584, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg3_1, %arg4_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : Tensor "f32[1, 64, 56, 56][200704, 1, 3584, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %convolution_2 : Tensor "f32[1, 64, 56, 56][200704, 1, 3584, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_1, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : Tensor "f32[1, 64, 56, 56][200704, 1, 3584, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : Tensor "f32[1, 256, 56, 56][802816, 1, 14336, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, %arg8_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_4 : Tensor "f32[1, 256, 56, 56][802816, 1, 14336, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg9_1, %arg10_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add : Tensor "f32[1, 256, 56, 56][802816, 1, 14336, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_3, %convolution_4), kwargs = {})
#   %relu_3 : Tensor "f32[1, 256, 56, 56][802816, 1, 14336, 256]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add,), kwargs = {})
#   return %relu_3
triton_poi_fused_add_convolution_relu_4 = async_compile.triton('triton_poi_fused_add_convolution_relu_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_relu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 12845056}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_relu_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/5i/c5igahntqbkcfwtmk4yppdbyv7vf5avezm4ctx7rj3x463s7ivbs.py
# Topologically Sorted Source Nodes: [out_21, out_22], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   out_21 => convolution_11
#   out_22 => relu_10
# Graph fragment:
#   %buf22 : Tensor "f32[3136, 128][128, 1]cuda:0" = PlaceHolder[target=buf22]
#   %convolution_11 : Tensor "f32[1, 128, 56, 56][401408, 1, 7168, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_9, %arg23_1, %arg24_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_10 : Tensor "f32[1, 128, 56, 56][401408, 1, 7168, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_11,), kwargs = {})
#   return %relu_10
triton_poi_fused_convolution_relu_5 = async_compile.triton('triton_poi_fused_convolution_relu_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 4816896}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_5(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/nq/cnq4xyl4ryjxseh7jwmkoacl2dcvzqszfuubnvf4sfom7ozh3zww.py
# Topologically Sorted Source Nodes: [out_21, out_22, out_23, out_24, out_25], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   out_21 => convolution_11
#   out_22 => relu_10
#   out_23 => convolution_12
#   out_24 => relu_11
#   out_25 => convolution_13
# Graph fragment:
#   %buf24 : Tensor "f32[1, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = PlaceHolder[target=buf24]
#   %arg26_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg26_1]
#   %convolution_11 : Tensor "f32[1, 128, 56, 56][401408, 1, 7168, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_9, %arg23_1, %arg24_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_10 : Tensor "f32[1, 128, 56, 56][401408, 1, 7168, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_11,), kwargs = {})
#   %convolution_12 : Tensor "f32[1, 128, 28, 28][100352, 1, 3584, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_10, %arg25_1, %arg26_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_11 : Tensor "f32[1, 128, 28, 28][100352, 1, 3584, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_12,), kwargs = {})
#   %convolution_13 : Tensor "f32[1, 512, 28, 28][401408, 1, 14336, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_11, %arg27_1, %arg28_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf25
triton_poi_fused_convolution_relu_6 = async_compile.triton('triton_poi_fused_convolution_relu_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 1204736}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_6(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/ei/ceiln4p2y4mn7emrv2h3fhxdcbmyyhvppirmxgnrqgumg45rjrlh.py
# Topologically Sorted Source Nodes: [out_21, out_22, out_23, out_24, out_25, input_2, out_26, out_27], Original ATen: [aten.convolution, aten.relu, aten.add]
# Source node to ATen node mapping:
#   input_2 => convolution_14
#   out_21 => convolution_11
#   out_22 => relu_10
#   out_23 => convolution_12
#   out_24 => relu_11
#   out_25 => convolution_13
#   out_26 => add_3
#   out_27 => relu_12
# Graph fragment:
#   %buf26 : Tensor "f32[784, 512][512, 1]cuda:0" = PlaceHolder[target=buf26]
#   %buf27 : Tensor "f32[1, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = PlaceHolder[target=buf27]
#   %arg30_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg30_1]
#   %convolution_11 : Tensor "f32[1, 128, 56, 56][401408, 1, 7168, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_9, %arg23_1, %arg24_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_10 : Tensor "f32[1, 128, 56, 56][401408, 1, 7168, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_11,), kwargs = {})
#   %convolution_12 : Tensor "f32[1, 128, 28, 28][100352, 1, 3584, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_10, %arg25_1, %arg26_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_11 : Tensor "f32[1, 128, 28, 28][100352, 1, 3584, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_12,), kwargs = {})
#   %convolution_13 : Tensor "f32[1, 512, 28, 28][401408, 1, 14336, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_11, %arg27_1, %arg28_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_14 : Tensor "f32[1, 512, 28, 28][401408, 1, 14336, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_9, %arg29_1, %arg30_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_3 : Tensor "f32[1, 512, 28, 28][401408, 1, 14336, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_13, %convolution_14), kwargs = {})
#   %relu_12 : Tensor "f32[1, 512, 28, 28][401408, 1, 14336, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
#   return %relu_12
triton_poi_fused_add_convolution_relu_7 = async_compile.triton('triton_poi_fused_add_convolution_relu_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_relu_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 6424576}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_relu_7(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(in_out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/ln/clnwtkgg23trlygu3g42ei5wpl7alsehe3x726ilmik4faj72mvf.py
# Topologically Sorted Source Nodes: [out_28, out_29], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   out_28 => convolution_15
#   out_29 => relu_13
# Graph fragment:
#   %buf29 : Tensor "f32[784, 128][128, 1]cuda:0" = PlaceHolder[target=buf29]
#   %convolution_15 : Tensor "f32[1, 128, 28, 28][100352, 1, 3584, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_12, %arg31_1, %arg32_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_13 : Tensor "f32[1, 128, 28, 28][100352, 1, 3584, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_15,), kwargs = {})
#   return %relu_13
triton_poi_fused_convolution_relu_8 = async_compile.triton('triton_poi_fused_convolution_relu_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 1204224}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_8(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/dl/cdl4oz55t2etuindw7put3xwg3zxvcoznnpk3xcfm6nludzmaa4a.py
# Topologically Sorted Source Nodes: [out_28, out_29, out_30, out_31, out_32, out_33, out_34], Original ATen: [aten.convolution, aten.relu, aten.add]
# Source node to ATen node mapping:
#   out_28 => convolution_15
#   out_29 => relu_13
#   out_30 => convolution_16
#   out_31 => relu_14
#   out_32 => convolution_17
#   out_33 => add_4
#   out_34 => relu_15
# Graph fragment:
#   %buf33 : Tensor "f32[784, 512][512, 1]cuda:0" = PlaceHolder[target=buf33]
#   %relu_12 : Tensor "f32[1, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = PlaceHolder[target=relu_12]
#   %convolution_15 : Tensor "f32[1, 128, 28, 28][100352, 1, 3584, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_12, %arg31_1, %arg32_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_13 : Tensor "f32[1, 128, 28, 28][100352, 1, 3584, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_15,), kwargs = {})
#   %convolution_16 : Tensor "f32[1, 128, 28, 28][100352, 1, 3584, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_13, %arg33_1, %arg34_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_14 : Tensor "f32[1, 128, 28, 28][100352, 1, 3584, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_16,), kwargs = {})
#   %convolution_17 : Tensor "f32[1, 512, 28, 28][401408, 1, 14336, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_14, %arg35_1, %arg36_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_4 : Tensor "f32[1, 512, 28, 28][401408, 1, 14336, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_17, %relu_12), kwargs = {})
#   %relu_15 : Tensor "f32[1, 512, 28, 28][401408, 1, 14336, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_4,), kwargs = {})
#   return %relu_15
triton_poi_fused_add_convolution_relu_9 = async_compile.triton('triton_poi_fused_add_convolution_relu_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_relu_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 6422528}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_relu_9(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/wz/cwzofax6cyonsvhtnuxdr2q7ge6iaqpoimg5zzxkjtvxvhe5t6vt.py
# Topologically Sorted Source Nodes: [out_49, out_50, out_51, out_52, out_53], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   out_49 => convolution_24
#   out_50 => relu_22
#   out_51 => convolution_25
#   out_52 => relu_23
#   out_53 => convolution_26
# Graph fragment:
#   %buf49 : Tensor "f32[1, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = PlaceHolder[target=buf49]
#   %arg52_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg52_1]
#   %convolution_24 : Tensor "f32[1, 256, 28, 28][200704, 1, 7168, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_21, %arg49_1, %arg50_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_22 : Tensor "f32[1, 256, 28, 28][200704, 1, 7168, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_24,), kwargs = {})
#   %convolution_25 : Tensor "f32[1, 256, 14, 14][50176, 1, 3584, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_22, %arg51_1, %arg52_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_23 : Tensor "f32[1, 256, 14, 14][50176, 1, 3584, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_25,), kwargs = {})
#   %convolution_26 : Tensor "f32[1, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_23, %arg53_1, %arg54_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf50
triton_poi_fused_convolution_relu_10 = async_compile.triton('triton_poi_fused_convolution_relu_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 603136}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_10(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/yb/cyb3xg5ofpcedgj7sjvckbfdbqieqijbjwjdmuhlsn7y4eqksui3.py
# Topologically Sorted Source Nodes: [out_49, out_50, out_51, out_52, out_53, input_3, out_54, out_55], Original ATen: [aten.convolution, aten.relu, aten.add]
# Source node to ATen node mapping:
#   input_3 => convolution_27
#   out_49 => convolution_24
#   out_50 => relu_22
#   out_51 => convolution_25
#   out_52 => relu_23
#   out_53 => convolution_26
#   out_54 => add_7
#   out_55 => relu_24
# Graph fragment:
#   %buf51 : Tensor "f32[196, 1024][1024, 1]cuda:0" = PlaceHolder[target=buf51]
#   %buf52 : Tensor "f32[1, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = PlaceHolder[target=buf52]
#   %arg56_1 : Tensor "f32[1024][1]cuda:0" = PlaceHolder[target=arg56_1]
#   %convolution_24 : Tensor "f32[1, 256, 28, 28][200704, 1, 7168, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_21, %arg49_1, %arg50_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_22 : Tensor "f32[1, 256, 28, 28][200704, 1, 7168, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_24,), kwargs = {})
#   %convolution_25 : Tensor "f32[1, 256, 14, 14][50176, 1, 3584, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_22, %arg51_1, %arg52_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_23 : Tensor "f32[1, 256, 14, 14][50176, 1, 3584, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_25,), kwargs = {})
#   %convolution_26 : Tensor "f32[1, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_23, %arg53_1, %arg54_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_27 : Tensor "f32[1, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_21, %arg55_1, %arg56_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_7 : Tensor "f32[1, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_26, %convolution_27), kwargs = {})
#   %relu_24 : Tensor "f32[1, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_7,), kwargs = {})
#   return %relu_24
triton_poi_fused_add_convolution_relu_11 = async_compile.triton('triton_poi_fused_add_convolution_relu_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_relu_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 3215360}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_relu_11(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(in_out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/gh/cgho4kuo72khce7s5gngtec5f6ldhlwdwudkqkjldiubt7oeafw5.py
# Topologically Sorted Source Nodes: [out_56, out_57], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   out_56 => convolution_28
#   out_57 => relu_25
# Graph fragment:
#   %buf54 : Tensor "f32[196, 256][256, 1]cuda:0" = PlaceHolder[target=buf54]
#   %convolution_28 : Tensor "f32[1, 256, 14, 14][50176, 1, 3584, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_24, %arg57_1, %arg58_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_25 : Tensor "f32[1, 256, 14, 14][50176, 1, 3584, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_28,), kwargs = {})
#   return %relu_25
triton_poi_fused_convolution_relu_12 = async_compile.triton('triton_poi_fused_convolution_relu_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 602112}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_12(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/fd/cfdswb6lasxrjus6y4cbn6mvots7lthjdtbfeu5uspw4v6qnhoch.py
# Topologically Sorted Source Nodes: [out_56, out_57, out_58, out_59, out_60, out_61, out_62], Original ATen: [aten.convolution, aten.relu, aten.add]
# Source node to ATen node mapping:
#   out_56 => convolution_28
#   out_57 => relu_25
#   out_58 => convolution_29
#   out_59 => relu_26
#   out_60 => convolution_30
#   out_61 => add_8
#   out_62 => relu_27
# Graph fragment:
#   %buf58 : Tensor "f32[196, 1024][1024, 1]cuda:0" = PlaceHolder[target=buf58]
#   %relu_24 : Tensor "f32[1, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = PlaceHolder[target=relu_24]
#   %convolution_28 : Tensor "f32[1, 256, 14, 14][50176, 1, 3584, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_24, %arg57_1, %arg58_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_25 : Tensor "f32[1, 256, 14, 14][50176, 1, 3584, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_28,), kwargs = {})
#   %convolution_29 : Tensor "f32[1, 256, 14, 14][50176, 1, 3584, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_25, %arg59_1, %arg60_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_26 : Tensor "f32[1, 256, 14, 14][50176, 1, 3584, 256]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_29,), kwargs = {})
#   %convolution_30 : Tensor "f32[1, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_26, %arg61_1, %arg62_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_8 : Tensor "f32[1, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_30, %relu_24), kwargs = {})
#   %relu_27 : Tensor "f32[1, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_8,), kwargs = {})
#   return %relu_27
triton_poi_fused_add_convolution_relu_13 = async_compile.triton('triton_poi_fused_add_convolution_relu_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_relu_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 3211264}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_relu_13(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/ja/cjau7whsl6zxyv7uxjfnq33r3r2vzajf45mftk57jkfngxqc3asw.py
# Topologically Sorted Source Nodes: [out_91, out_92, out_93, out_94, out_95], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   out_91 => convolution_43
#   out_92 => relu_40
#   out_93 => convolution_44
#   out_94 => relu_41
#   out_95 => convolution_45
# Graph fragment:
#   %buf86 : Tensor "f32[1, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = PlaceHolder[target=buf86]
#   %arg90_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg90_1]
#   %convolution_43 : Tensor "f32[1, 512, 14, 14][100352, 1, 7168, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_39, %arg87_1, %arg88_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_40 : Tensor "f32[1, 512, 14, 14][100352, 1, 7168, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_43,), kwargs = {})
#   %convolution_44 : Tensor "f32[1, 512, 7, 7][25088, 1, 3584, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_40, %arg89_1, %arg90_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_41 : Tensor "f32[1, 512, 7, 7][25088, 1, 3584, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_44,), kwargs = {})
#   %convolution_45 : Tensor "f32[1, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_41, %arg91_1, %arg92_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf87
triton_poi_fused_convolution_relu_14 = async_compile.triton('triton_poi_fused_convolution_relu_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 303104}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_14(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/7d/c7daw56u74eemp5ctegs7tusihpvrmi2j3xx3zrwwj3fdncjrtwf.py
# Topologically Sorted Source Nodes: [out_91, out_92, out_93, out_94, out_95, input_4, out_96, out_97], Original ATen: [aten.convolution, aten.relu, aten.add]
# Source node to ATen node mapping:
#   input_4 => convolution_46
#   out_91 => convolution_43
#   out_92 => relu_40
#   out_93 => convolution_44
#   out_94 => relu_41
#   out_95 => convolution_45
#   out_96 => add_13
#   out_97 => relu_42
# Graph fragment:
#   %buf88 : Tensor "f32[49, 2048][2048, 1]cuda:0" = PlaceHolder[target=buf88]
#   %buf89 : Tensor "f32[1, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = PlaceHolder[target=buf89]
#   %arg94_1 : Tensor "f32[2048][1]cuda:0" = PlaceHolder[target=arg94_1]
#   %convolution_43 : Tensor "f32[1, 512, 14, 14][100352, 1, 7168, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_39, %arg87_1, %arg88_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_40 : Tensor "f32[1, 512, 14, 14][100352, 1, 7168, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_43,), kwargs = {})
#   %convolution_44 : Tensor "f32[1, 512, 7, 7][25088, 1, 3584, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_40, %arg89_1, %arg90_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_41 : Tensor "f32[1, 512, 7, 7][25088, 1, 3584, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_44,), kwargs = {})
#   %convolution_45 : Tensor "f32[1, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_41, %arg91_1, %arg92_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_46 : Tensor "f32[1, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_39, %arg93_1, %arg94_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_13 : Tensor "f32[1, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_45, %convolution_46), kwargs = {})
#   %relu_42 : Tensor "f32[1, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_13,), kwargs = {})
#   return %relu_42
triton_poi_fused_add_convolution_relu_15 = async_compile.triton('triton_poi_fused_add_convolution_relu_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_relu_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 1613824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_relu_15(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 2048)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x2), xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(in_out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/qm/cqmtgq5ei4wc2st4sa3ewq5nih3bqi2ksz72kv237f2sea7hl63d.py
# Topologically Sorted Source Nodes: [out_98, out_99], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   out_98 => convolution_47
#   out_99 => relu_43
# Graph fragment:
#   %buf91 : Tensor "f32[49, 512][512, 1]cuda:0" = PlaceHolder[target=buf91]
#   %convolution_47 : Tensor "f32[1, 512, 7, 7][25088, 1, 3584, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_42, %arg95_1, %arg96_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_43 : Tensor "f32[1, 512, 7, 7][25088, 1, 3584, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_47,), kwargs = {})
#   return %relu_43
triton_poi_fused_convolution_relu_16 = async_compile.triton('triton_poi_fused_convolution_relu_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 301056}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_16(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/ov/cov6rjx6csbblg6qg5pqss7nzflk537wnspoepvvgerx5lcwq5dn.py
# Topologically Sorted Source Nodes: [out_98, out_99, out_100, out_101, out_102, out_103, out_104], Original ATen: [aten.convolution, aten.relu, aten.add]
# Source node to ATen node mapping:
#   out_100 => convolution_48
#   out_101 => relu_44
#   out_102 => convolution_49
#   out_103 => add_14
#   out_104 => relu_45
#   out_98 => convolution_47
#   out_99 => relu_43
# Graph fragment:
#   %buf95 : Tensor "f32[49, 2048][2048, 1]cuda:0" = PlaceHolder[target=buf95]
#   %relu_42 : Tensor "f32[1, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = PlaceHolder[target=relu_42]
#   %convolution_47 : Tensor "f32[1, 512, 7, 7][25088, 1, 3584, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_42, %arg95_1, %arg96_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_43 : Tensor "f32[1, 512, 7, 7][25088, 1, 3584, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_47,), kwargs = {})
#   %convolution_48 : Tensor "f32[1, 512, 7, 7][25088, 1, 3584, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_43, %arg97_1, %arg98_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_44 : Tensor "f32[1, 512, 7, 7][25088, 1, 3584, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_48,), kwargs = {})
#   %convolution_49 : Tensor "f32[1, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_44, %arg99_1, %arg100_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_14 : Tensor "f32[1, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_49, %relu_42), kwargs = {})
#   %relu_45 : Tensor "f32[1, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_14,), kwargs = {})
#   return %relu_45
triton_poi_fused_add_convolution_relu_17 = async_compile.triton('triton_poi_fused_add_convolution_relu_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_relu_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 1605632}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_relu_17(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/qg/cqgrb2omir2y7vfqcxc7y3zsddid5vjo6zqoesadzhyon7epd64p.py
# Topologically Sorted Source Nodes: [out_105, out_106, out_107, out_108, out_109, out_110, out_111, x_3], Original ATen: [aten.convolution, aten.relu, aten.add, aten.mean]
# Source node to ATen node mapping:
#   out_105 => convolution_50
#   out_106 => relu_46
#   out_107 => convolution_51
#   out_108 => relu_47
#   out_109 => convolution_52
#   out_110 => add_15
#   out_111 => relu_48
#   x_3 => mean
# Graph fragment:
#   %buf101 : Tensor "f32[49, 2048][2048, 1]cuda:0" = PlaceHolder[target=buf101]
#   %relu_45 : Tensor "f32[1, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = PlaceHolder[target=relu_45]
#   %buf102 : Tensor "f32[1, 2048, 1, 1][2048, 1, 2048, 2048]cuda:0" = PlaceHolder[target=buf102]
#   %convolution_50 : Tensor "f32[1, 512, 7, 7][25088, 1, 3584, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_45, %arg101_1, %arg102_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_46 : Tensor "f32[1, 512, 7, 7][25088, 1, 3584, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_50,), kwargs = {})
#   %convolution_51 : Tensor "f32[1, 512, 7, 7][25088, 1, 3584, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_46, %arg103_1, %arg104_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_47 : Tensor "f32[1, 512, 7, 7][25088, 1, 3584, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_51,), kwargs = {})
#   %convolution_52 : Tensor "f32[1, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_47, %arg105_1, %arg106_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_15 : Tensor "f32[1, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_52, %relu_45), kwargs = {})
#   %relu_48 : Tensor "f32[1, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_15,), kwargs = {})
#   %mean : Tensor "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_48, [-1, -2], True), kwargs = {})
#   return %buf102,%mean
triton_per_fused_add_convolution_mean_relu_18 = async_compile.triton('triton_per_fused_add_convolution_mean_relu_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r0_': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_mean_relu_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 819200, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_add_convolution_mean_relu_18(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2048
    r0_numel = 49
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 2048*r0_1), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 2048*r0_1), r0_mask & xmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None].to(tl.float32)
    tmp9 = 49.0
    tmp10 = (tmp8 / tmp9)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, xmask)
''', device_str='cuda')

def partition_0(args):
    arg2_1, arg0_1, arg1_1, arg4_1, arg3_1, arg5_1, arg6_1, arg8_1, arg7_1, arg10_1, arg9_1, arg12_1, arg11_1, arg13_1, arg14_1, arg16_1, arg15_1, arg18_1, arg17_1, arg19_1, arg20_1, arg22_1, arg21_1, arg24_1, arg23_1, arg25_1, arg26_1, arg28_1, arg27_1, arg29_1, arg30_1, arg32_1, arg31_1, arg33_1, arg34_1, arg36_1, arg35_1, arg38_1, arg37_1, arg39_1, arg40_1, arg42_1, arg41_1, arg44_1, arg43_1, arg45_1, arg46_1, arg48_1, arg47_1, arg50_1, arg49_1, arg51_1, arg52_1, arg54_1, arg53_1, arg55_1, arg56_1, arg58_1, arg57_1, arg59_1, arg60_1, arg62_1, arg61_1, arg64_1, arg63_1, arg65_1, arg66_1, arg68_1, arg67_1, arg70_1, arg69_1, arg71_1, arg72_1, arg74_1, arg73_1, arg76_1, arg75_1, arg77_1, arg78_1, arg80_1, arg79_1, arg82_1, arg81_1, arg83_1, arg84_1, arg86_1, arg85_1, arg88_1, arg87_1, arg89_1, arg90_1, arg92_1, arg91_1, arg93_1, arg94_1, arg96_1, arg95_1, arg97_1, arg98_1, arg100_1, arg99_1, arg102_1, arg101_1, arg103_1, arg104_1, arg106_1, arg105_1, arg108_1, arg107_1 = args
    args.clear()
    assert_size_stride(arg2_1, (1, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(arg0_1, (64, 3, 7, 7), (147, 1, 21, 3))
    assert_size_stride(arg1_1, (64, ), (1, ))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (64, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(arg5_1, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(arg6_1, (64, ), (1, ))
    assert_size_stride(arg8_1, (256, ), (1, ))
    assert_size_stride(arg7_1, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(arg10_1, (256, ), (1, ))
    assert_size_stride(arg9_1, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(arg12_1, (64, ), (1, ))
    assert_size_stride(arg11_1, (64, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg13_1, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(arg14_1, (64, ), (1, ))
    assert_size_stride(arg16_1, (256, ), (1, ))
    assert_size_stride(arg15_1, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(arg18_1, (64, ), (1, ))
    assert_size_stride(arg17_1, (64, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg19_1, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(arg20_1, (64, ), (1, ))
    assert_size_stride(arg22_1, (256, ), (1, ))
    assert_size_stride(arg21_1, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(arg24_1, (128, ), (1, ))
    assert_size_stride(arg23_1, (128, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg25_1, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(arg26_1, (128, ), (1, ))
    assert_size_stride(arg28_1, (512, ), (1, ))
    assert_size_stride(arg27_1, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(arg29_1, (512, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg30_1, (512, ), (1, ))
    assert_size_stride(arg32_1, (128, ), (1, ))
    assert_size_stride(arg31_1, (128, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg33_1, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(arg34_1, (128, ), (1, ))
    assert_size_stride(arg36_1, (512, ), (1, ))
    assert_size_stride(arg35_1, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(arg38_1, (128, ), (1, ))
    assert_size_stride(arg37_1, (128, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg39_1, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(arg40_1, (128, ), (1, ))
    assert_size_stride(arg42_1, (512, ), (1, ))
    assert_size_stride(arg41_1, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(arg44_1, (128, ), (1, ))
    assert_size_stride(arg43_1, (128, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg45_1, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(arg46_1, (128, ), (1, ))
    assert_size_stride(arg48_1, (512, ), (1, ))
    assert_size_stride(arg47_1, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(arg50_1, (256, ), (1, ))
    assert_size_stride(arg49_1, (256, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg51_1, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(arg52_1, (256, ), (1, ))
    assert_size_stride(arg54_1, (1024, ), (1, ))
    assert_size_stride(arg53_1, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg55_1, (1024, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg56_1, (1024, ), (1, ))
    assert_size_stride(arg58_1, (256, ), (1, ))
    assert_size_stride(arg57_1, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg59_1, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(arg60_1, (256, ), (1, ))
    assert_size_stride(arg62_1, (1024, ), (1, ))
    assert_size_stride(arg61_1, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg64_1, (256, ), (1, ))
    assert_size_stride(arg63_1, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg65_1, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(arg66_1, (256, ), (1, ))
    assert_size_stride(arg68_1, (1024, ), (1, ))
    assert_size_stride(arg67_1, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg70_1, (256, ), (1, ))
    assert_size_stride(arg69_1, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg71_1, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(arg72_1, (256, ), (1, ))
    assert_size_stride(arg74_1, (1024, ), (1, ))
    assert_size_stride(arg73_1, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg76_1, (256, ), (1, ))
    assert_size_stride(arg75_1, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg77_1, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(arg78_1, (256, ), (1, ))
    assert_size_stride(arg80_1, (1024, ), (1, ))
    assert_size_stride(arg79_1, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg82_1, (256, ), (1, ))
    assert_size_stride(arg81_1, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg83_1, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(arg84_1, (256, ), (1, ))
    assert_size_stride(arg86_1, (1024, ), (1, ))
    assert_size_stride(arg85_1, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg88_1, (512, ), (1, ))
    assert_size_stride(arg87_1, (512, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg89_1, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(arg90_1, (512, ), (1, ))
    assert_size_stride(arg92_1, (2048, ), (1, ))
    assert_size_stride(arg91_1, (2048, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg93_1, (2048, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg94_1, (2048, ), (1, ))
    assert_size_stride(arg96_1, (512, ), (1, ))
    assert_size_stride(arg95_1, (512, 2048, 1, 1), (2048, 1, 2048, 2048))
    assert_size_stride(arg97_1, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(arg98_1, (512, ), (1, ))
    assert_size_stride(arg100_1, (2048, ), (1, ))
    assert_size_stride(arg99_1, (2048, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg102_1, (512, ), (1, ))
    assert_size_stride(arg101_1, (512, 2048, 1, 1), (2048, 1, 2048, 2048))
    assert_size_stride(arg103_1, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(arg104_1, (512, ), (1, ))
    assert_size_stride(arg106_1, (2048, ), (1, ))
    assert_size_stride(arg105_1, (2048, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg108_1, (1000, ), (1, ))
    assert_size_stride(arg107_1, (1000, 2048), (2048, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg2_1, arg0_1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (1, 64, 112, 112), (802816, 1, 7168, 64), 'torch.ops.aten.convolution.default')
        del arg0_1
        del arg2_1
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_0.run(buf1, arg1_1, 802816, stream=stream0)
        del arg1_1
        buf2 = empty_strided_cuda((1, 64, 56, 56), (200704, 1, 3584, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_1.run(buf1, buf2, 200704, stream=stream0)
        buf3 = empty_strided_cuda((3136, 64), (64, 1), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg4_1, (3136, 64), (0, 1), 0), reinterpret_tensor(buf2, (3136, 64), (64, 1), 0), reinterpret_tensor(arg3_1, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf3)
        del arg3_1
        del arg4_1
        buf4 = reinterpret_tensor(buf3, (1, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [out, out_1], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_2.run(buf4, 200704, stream=stream0)
        # Topologically Sorted Source Nodes: [out, out_1, out_2], Original ATen: [aten.convolution, aten.relu]
        buf5 = extern_kernels.convolution(buf4, arg5_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (1, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution.default')
        del arg5_1
        del buf4
        buf6 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [out, out_1, out_2, out_3, out_4], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_3.run(buf6, arg6_1, 200704, stream=stream0)
        del arg6_1
        buf7 = reinterpret_tensor(buf1, (3136, 256), (256, 1), 0); del buf1  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg8_1, (3136, 256), (0, 1), 0), reinterpret_tensor(buf6, (3136, 64), (64, 1), 0), reinterpret_tensor(arg7_1, (64, 256), (1, 64), 0), alpha=1, beta=1, out=buf7)
        del arg7_1
        del arg8_1
        del buf6
        buf8 = empty_strided_cuda((3136, 256), (256, 1), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg10_1, (3136, 256), (0, 1), 0), reinterpret_tensor(buf2, (3136, 64), (64, 1), 0), reinterpret_tensor(arg9_1, (64, 256), (1, 64), 0), alpha=1, beta=1, out=buf8)
        del arg10_1
        del arg9_1
        del buf2
        buf9 = reinterpret_tensor(buf7, (1, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [out, out_1, out_2, out_3, out_4, input_1, out_5, out_6], Original ATen: [aten.convolution, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_relu_4.run(buf9, buf8, 802816, stream=stream0)
        buf10 = empty_strided_cuda((3136, 64), (64, 1), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg12_1, (3136, 64), (0, 1), 0), reinterpret_tensor(buf9, (3136, 256), (256, 1), 0), reinterpret_tensor(arg11_1, (256, 64), (1, 256), 0), alpha=1, beta=1, out=buf10)
        del arg11_1
        del arg12_1
        buf11 = reinterpret_tensor(buf10, (1, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf10  # reuse
        # Topologically Sorted Source Nodes: [out_7, out_8], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_2.run(buf11, 200704, stream=stream0)
        # Topologically Sorted Source Nodes: [out_7, out_8, out_9], Original ATen: [aten.convolution, aten.relu]
        buf12 = extern_kernels.convolution(buf11, arg13_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (1, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution.default')
        del arg13_1
        del buf11
        buf13 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [out_7, out_8, out_9, out_10, out_11], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_3.run(buf13, arg14_1, 200704, stream=stream0)
        del arg14_1
        buf14 = buf8; del buf8  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg16_1, (3136, 256), (0, 1), 0), reinterpret_tensor(buf13, (3136, 64), (64, 1), 0), reinterpret_tensor(arg15_1, (64, 256), (1, 64), 0), alpha=1, beta=1, out=buf14)
        del arg15_1
        del arg16_1
        del buf13
        buf15 = reinterpret_tensor(buf14, (1, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [out_7, out_8, out_9, out_10, out_11, out_12, out_13], Original ATen: [aten.convolution, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_relu_4.run(buf15, buf9, 802816, stream=stream0)
        buf16 = empty_strided_cuda((3136, 64), (64, 1), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg18_1, (3136, 64), (0, 1), 0), reinterpret_tensor(buf15, (3136, 256), (256, 1), 0), reinterpret_tensor(arg17_1, (256, 64), (1, 256), 0), alpha=1, beta=1, out=buf16)
        del arg17_1
        del arg18_1
        buf17 = reinterpret_tensor(buf16, (1, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [out_14, out_15], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_2.run(buf17, 200704, stream=stream0)
        # Topologically Sorted Source Nodes: [out_14, out_15, out_16], Original ATen: [aten.convolution, aten.relu]
        buf18 = extern_kernels.convolution(buf17, arg19_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (1, 64, 56, 56), (200704, 1, 3584, 64), 'torch.ops.aten.convolution.default')
        del arg19_1
        del buf17
        buf19 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [out_14, out_15, out_16, out_17, out_18], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_3.run(buf19, arg20_1, 200704, stream=stream0)
        del arg20_1
        buf20 = reinterpret_tensor(buf9, (3136, 256), (256, 1), 0); del buf9  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg22_1, (3136, 256), (0, 1), 0), reinterpret_tensor(buf19, (3136, 64), (64, 1), 0), reinterpret_tensor(arg21_1, (64, 256), (1, 64), 0), alpha=1, beta=1, out=buf20)
        del arg21_1
        del arg22_1
        del buf19
        buf21 = reinterpret_tensor(buf20, (1, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [out_14, out_15, out_16, out_17, out_18, out_19, out_20], Original ATen: [aten.convolution, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_relu_4.run(buf21, buf15, 802816, stream=stream0)
        del buf15
        buf22 = empty_strided_cuda((3136, 128), (128, 1), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg24_1, (3136, 128), (0, 1), 0), reinterpret_tensor(buf21, (3136, 256), (256, 1), 0), reinterpret_tensor(arg23_1, (256, 128), (1, 256), 0), alpha=1, beta=1, out=buf22)
        del arg23_1
        del arg24_1
        buf23 = reinterpret_tensor(buf22, (1, 128, 56, 56), (401408, 1, 7168, 128), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [out_21, out_22], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_5.run(buf23, 401408, stream=stream0)
        # Topologically Sorted Source Nodes: [out_21, out_22, out_23], Original ATen: [aten.convolution, aten.relu]
        buf24 = extern_kernels.convolution(buf23, arg25_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (1, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution.default')
        del arg25_1
        buf25 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [out_21, out_22, out_23, out_24, out_25], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_6.run(buf25, arg26_1, 100352, stream=stream0)
        del arg26_1
        buf26 = reinterpret_tensor(buf23, (784, 512), (512, 1), 0); del buf23  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg28_1, (784, 512), (0, 1), 0), reinterpret_tensor(buf25, (784, 128), (128, 1), 0), reinterpret_tensor(arg27_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf26)
        del arg27_1
        del arg28_1
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf21, arg29_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (1, 512, 28, 28), (401408, 1, 14336, 512), 'torch.ops.aten.convolution.default')
        del arg29_1
        del buf21
        buf28 = reinterpret_tensor(buf26, (1, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [out_21, out_22, out_23, out_24, out_25, input_2, out_26, out_27], Original ATen: [aten.convolution, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_relu_7.run(buf28, buf27, arg30_1, 401408, stream=stream0)
        del arg30_1
        buf29 = reinterpret_tensor(buf25, (784, 128), (128, 1), 0); del buf25  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg32_1, (784, 128), (0, 1), 0), reinterpret_tensor(buf28, (784, 512), (512, 1), 0), reinterpret_tensor(arg31_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf29)
        del arg31_1
        del arg32_1
        buf30 = reinterpret_tensor(buf29, (1, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [out_28, out_29], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_8.run(buf30, 100352, stream=stream0)
        # Topologically Sorted Source Nodes: [out_28, out_29, out_30], Original ATen: [aten.convolution, aten.relu]
        buf31 = extern_kernels.convolution(buf30, arg33_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (1, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution.default')
        del arg33_1
        del buf30
        buf32 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [out_28, out_29, out_30, out_31, out_32], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_6.run(buf32, arg34_1, 100352, stream=stream0)
        del arg34_1
        buf33 = reinterpret_tensor(buf27, (784, 512), (512, 1), 0); del buf27  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg36_1, (784, 512), (0, 1), 0), reinterpret_tensor(buf32, (784, 128), (128, 1), 0), reinterpret_tensor(arg35_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf33)
        del arg35_1
        del arg36_1
        buf34 = reinterpret_tensor(buf33, (1, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [out_28, out_29, out_30, out_31, out_32, out_33, out_34], Original ATen: [aten.convolution, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_relu_9.run(buf34, buf28, 401408, stream=stream0)
        buf35 = reinterpret_tensor(buf32, (784, 128), (128, 1), 0); del buf32  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg38_1, (784, 128), (0, 1), 0), reinterpret_tensor(buf34, (784, 512), (512, 1), 0), reinterpret_tensor(arg37_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf35)
        del arg37_1
        del arg38_1
        buf36 = reinterpret_tensor(buf35, (1, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf35  # reuse
        # Topologically Sorted Source Nodes: [out_35, out_36], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_8.run(buf36, 100352, stream=stream0)
        # Topologically Sorted Source Nodes: [out_35, out_36, out_37], Original ATen: [aten.convolution, aten.relu]
        buf37 = extern_kernels.convolution(buf36, arg39_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (1, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution.default')
        del arg39_1
        del buf36
        buf38 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [out_35, out_36, out_37, out_38, out_39], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_6.run(buf38, arg40_1, 100352, stream=stream0)
        del arg40_1
        buf39 = reinterpret_tensor(buf28, (784, 512), (512, 1), 0); del buf28  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg42_1, (784, 512), (0, 1), 0), reinterpret_tensor(buf38, (784, 128), (128, 1), 0), reinterpret_tensor(arg41_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf39)
        del arg41_1
        del arg42_1
        buf40 = reinterpret_tensor(buf39, (1, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf39  # reuse
        # Topologically Sorted Source Nodes: [out_35, out_36, out_37, out_38, out_39, out_40, out_41], Original ATen: [aten.convolution, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_relu_9.run(buf40, buf34, 401408, stream=stream0)
        buf41 = reinterpret_tensor(buf38, (784, 128), (128, 1), 0); del buf38  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg44_1, (784, 128), (0, 1), 0), reinterpret_tensor(buf40, (784, 512), (512, 1), 0), reinterpret_tensor(arg43_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf41)
        del arg43_1
        del arg44_1
        buf42 = reinterpret_tensor(buf41, (1, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [out_42, out_43], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_8.run(buf42, 100352, stream=stream0)
        # Topologically Sorted Source Nodes: [out_42, out_43, out_44], Original ATen: [aten.convolution, aten.relu]
        buf43 = extern_kernels.convolution(buf42, arg45_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (1, 128, 28, 28), (100352, 1, 3584, 128), 'torch.ops.aten.convolution.default')
        del arg45_1
        del buf42
        buf44 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [out_42, out_43, out_44, out_45, out_46], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_6.run(buf44, arg46_1, 100352, stream=stream0)
        del arg46_1
        buf45 = reinterpret_tensor(buf34, (784, 512), (512, 1), 0); del buf34  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg48_1, (784, 512), (0, 1), 0), reinterpret_tensor(buf44, (784, 128), (128, 1), 0), reinterpret_tensor(arg47_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf45)
        del arg47_1
        del arg48_1
        buf46 = reinterpret_tensor(buf45, (1, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf45  # reuse
        # Topologically Sorted Source Nodes: [out_42, out_43, out_44, out_45, out_46, out_47, out_48], Original ATen: [aten.convolution, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_relu_9.run(buf46, buf40, 401408, stream=stream0)
        del buf40
        buf47 = empty_strided_cuda((784, 256), (256, 1), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg50_1, (784, 256), (0, 1), 0), reinterpret_tensor(buf46, (784, 512), (512, 1), 0), reinterpret_tensor(arg49_1, (512, 256), (1, 512), 0), alpha=1, beta=1, out=buf47)
        del arg49_1
        del arg50_1
        buf48 = reinterpret_tensor(buf47, (1, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf47  # reuse
        # Topologically Sorted Source Nodes: [out_49, out_50], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_2.run(buf48, 200704, stream=stream0)
        # Topologically Sorted Source Nodes: [out_49, out_50, out_51], Original ATen: [aten.convolution, aten.relu]
        buf49 = extern_kernels.convolution(buf48, arg51_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (1, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        del arg51_1
        buf50 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [out_49, out_50, out_51, out_52, out_53], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_10.run(buf50, arg52_1, 50176, stream=stream0)
        del arg52_1
        buf51 = reinterpret_tensor(buf48, (196, 1024), (1024, 1), 0); del buf48  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg54_1, (196, 1024), (0, 1), 0), reinterpret_tensor(buf50, (196, 256), (256, 1), 0), reinterpret_tensor(arg53_1, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf51)
        del arg53_1
        del arg54_1
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf46, arg55_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (1, 1024, 14, 14), (200704, 1, 14336, 1024), 'torch.ops.aten.convolution.default')
        del arg55_1
        del buf46
        buf53 = reinterpret_tensor(buf51, (1, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [out_49, out_50, out_51, out_52, out_53, input_3, out_54, out_55], Original ATen: [aten.convolution, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_relu_11.run(buf53, buf52, arg56_1, 200704, stream=stream0)
        del arg56_1
        buf54 = reinterpret_tensor(buf50, (196, 256), (256, 1), 0); del buf50  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.addmm(reinterpret_tensor(arg58_1, (196, 256), (0, 1), 0), reinterpret_tensor(buf53, (196, 1024), (1024, 1), 0), reinterpret_tensor(arg57_1, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf54)
        del arg57_1
        del arg58_1
        buf55 = reinterpret_tensor(buf54, (1, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [out_56, out_57], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_12.run(buf55, 50176, stream=stream0)
        # Topologically Sorted Source Nodes: [out_56, out_57, out_58], Original ATen: [aten.convolution, aten.relu]
        buf56 = extern_kernels.convolution(buf55, arg59_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (1, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        del arg59_1
        del buf55
        buf57 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [out_56, out_57, out_58, out_59, out_60], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_10.run(buf57, arg60_1, 50176, stream=stream0)
        del arg60_1
        buf58 = reinterpret_tensor(buf52, (196, 1024), (1024, 1), 0); del buf52  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg62_1, (196, 1024), (0, 1), 0), reinterpret_tensor(buf57, (196, 256), (256, 1), 0), reinterpret_tensor(arg61_1, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf58)
        del arg61_1
        del arg62_1
        buf59 = reinterpret_tensor(buf58, (1, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf58  # reuse
        # Topologically Sorted Source Nodes: [out_56, out_57, out_58, out_59, out_60, out_61, out_62], Original ATen: [aten.convolution, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_relu_13.run(buf59, buf53, 200704, stream=stream0)
        buf60 = reinterpret_tensor(buf57, (196, 256), (256, 1), 0); del buf57  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.addmm(reinterpret_tensor(arg64_1, (196, 256), (0, 1), 0), reinterpret_tensor(buf59, (196, 1024), (1024, 1), 0), reinterpret_tensor(arg63_1, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf60)
        del arg63_1
        del arg64_1
        buf61 = reinterpret_tensor(buf60, (1, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [out_63, out_64], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_12.run(buf61, 50176, stream=stream0)
        # Topologically Sorted Source Nodes: [out_63, out_64, out_65], Original ATen: [aten.convolution, aten.relu]
        buf62 = extern_kernels.convolution(buf61, arg65_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (1, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        del arg65_1
        del buf61
        buf63 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [out_63, out_64, out_65, out_66, out_67], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_10.run(buf63, arg66_1, 50176, stream=stream0)
        del arg66_1
        buf64 = reinterpret_tensor(buf53, (196, 1024), (1024, 1), 0); del buf53  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg68_1, (196, 1024), (0, 1), 0), reinterpret_tensor(buf63, (196, 256), (256, 1), 0), reinterpret_tensor(arg67_1, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf64)
        del arg67_1
        del arg68_1
        buf65 = reinterpret_tensor(buf64, (1, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf64  # reuse
        # Topologically Sorted Source Nodes: [out_63, out_64, out_65, out_66, out_67, out_68, out_69], Original ATen: [aten.convolution, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_relu_13.run(buf65, buf59, 200704, stream=stream0)
        buf66 = reinterpret_tensor(buf63, (196, 256), (256, 1), 0); del buf63  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.addmm(reinterpret_tensor(arg70_1, (196, 256), (0, 1), 0), reinterpret_tensor(buf65, (196, 1024), (1024, 1), 0), reinterpret_tensor(arg69_1, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf66)
        del arg69_1
        del arg70_1
        buf67 = reinterpret_tensor(buf66, (1, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [out_70, out_71], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_12.run(buf67, 50176, stream=stream0)
        # Topologically Sorted Source Nodes: [out_70, out_71, out_72], Original ATen: [aten.convolution, aten.relu]
        buf68 = extern_kernels.convolution(buf67, arg71_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (1, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        del arg71_1
        del buf67
        buf69 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [out_70, out_71, out_72, out_73, out_74], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_10.run(buf69, arg72_1, 50176, stream=stream0)
        del arg72_1
        buf70 = reinterpret_tensor(buf59, (196, 1024), (1024, 1), 0); del buf59  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg74_1, (196, 1024), (0, 1), 0), reinterpret_tensor(buf69, (196, 256), (256, 1), 0), reinterpret_tensor(arg73_1, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf70)
        del arg73_1
        del arg74_1
        buf71 = reinterpret_tensor(buf70, (1, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf70  # reuse
        # Topologically Sorted Source Nodes: [out_70, out_71, out_72, out_73, out_74, out_75, out_76], Original ATen: [aten.convolution, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_relu_13.run(buf71, buf65, 200704, stream=stream0)
        buf72 = reinterpret_tensor(buf69, (196, 256), (256, 1), 0); del buf69  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.addmm(reinterpret_tensor(arg76_1, (196, 256), (0, 1), 0), reinterpret_tensor(buf71, (196, 1024), (1024, 1), 0), reinterpret_tensor(arg75_1, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf72)
        del arg75_1
        del arg76_1
        buf73 = reinterpret_tensor(buf72, (1, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf72  # reuse
        # Topologically Sorted Source Nodes: [out_77, out_78], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_12.run(buf73, 50176, stream=stream0)
        # Topologically Sorted Source Nodes: [out_77, out_78, out_79], Original ATen: [aten.convolution, aten.relu]
        buf74 = extern_kernels.convolution(buf73, arg77_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (1, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        del arg77_1
        del buf73
        buf75 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [out_77, out_78, out_79, out_80, out_81], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_10.run(buf75, arg78_1, 50176, stream=stream0)
        del arg78_1
        buf76 = reinterpret_tensor(buf65, (196, 1024), (1024, 1), 0); del buf65  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg80_1, (196, 1024), (0, 1), 0), reinterpret_tensor(buf75, (196, 256), (256, 1), 0), reinterpret_tensor(arg79_1, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf76)
        del arg79_1
        del arg80_1
        buf77 = reinterpret_tensor(buf76, (1, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf76  # reuse
        # Topologically Sorted Source Nodes: [out_77, out_78, out_79, out_80, out_81, out_82, out_83], Original ATen: [aten.convolution, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_relu_13.run(buf77, buf71, 200704, stream=stream0)
        buf78 = reinterpret_tensor(buf75, (196, 256), (256, 1), 0); del buf75  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.addmm(reinterpret_tensor(arg82_1, (196, 256), (0, 1), 0), reinterpret_tensor(buf77, (196, 1024), (1024, 1), 0), reinterpret_tensor(arg81_1, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf78)
        del arg81_1
        del arg82_1
        buf79 = reinterpret_tensor(buf78, (1, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [out_84, out_85], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_12.run(buf79, 50176, stream=stream0)
        # Topologically Sorted Source Nodes: [out_84, out_85, out_86], Original ATen: [aten.convolution, aten.relu]
        buf80 = extern_kernels.convolution(buf79, arg83_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (1, 256, 14, 14), (50176, 1, 3584, 256), 'torch.ops.aten.convolution.default')
        del arg83_1
        del buf79
        buf81 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [out_84, out_85, out_86, out_87, out_88], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_10.run(buf81, arg84_1, 50176, stream=stream0)
        del arg84_1
        buf82 = reinterpret_tensor(buf71, (196, 1024), (1024, 1), 0); del buf71  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg86_1, (196, 1024), (0, 1), 0), reinterpret_tensor(buf81, (196, 256), (256, 1), 0), reinterpret_tensor(arg85_1, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf82)
        del arg85_1
        del arg86_1
        del buf81
        buf83 = reinterpret_tensor(buf82, (1, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf82  # reuse
        # Topologically Sorted Source Nodes: [out_84, out_85, out_86, out_87, out_88, out_89, out_90], Original ATen: [aten.convolution, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_relu_13.run(buf83, buf77, 200704, stream=stream0)
        del buf77
        buf84 = reinterpret_tensor(buf44, (196, 512), (512, 1), 0); del buf44  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.addmm(reinterpret_tensor(arg88_1, (196, 512), (0, 1), 0), reinterpret_tensor(buf83, (196, 1024), (1024, 1), 0), reinterpret_tensor(arg87_1, (1024, 512), (1, 1024), 0), alpha=1, beta=1, out=buf84)
        del arg87_1
        del arg88_1
        buf85 = reinterpret_tensor(buf84, (1, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf84  # reuse
        # Topologically Sorted Source Nodes: [out_91, out_92], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_8.run(buf85, 100352, stream=stream0)
        # Topologically Sorted Source Nodes: [out_91, out_92, out_93], Original ATen: [aten.convolution, aten.relu]
        buf86 = extern_kernels.convolution(buf85, arg89_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (1, 512, 7, 7), (25088, 1, 3584, 512), 'torch.ops.aten.convolution.default')
        del arg89_1
        buf87 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [out_91, out_92, out_93, out_94, out_95], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_14.run(buf87, arg90_1, 25088, stream=stream0)
        del arg90_1
        buf88 = reinterpret_tensor(buf85, (49, 2048), (2048, 1), 0); del buf85  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg92_1, (49, 2048), (0, 1), 0), reinterpret_tensor(buf87, (49, 512), (512, 1), 0), reinterpret_tensor(arg91_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf88)
        del arg91_1
        del arg92_1
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf83, arg93_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (1, 2048, 7, 7), (100352, 1, 14336, 2048), 'torch.ops.aten.convolution.default')
        del arg93_1
        del buf83
        buf90 = reinterpret_tensor(buf88, (1, 2048, 7, 7), (100352, 1, 14336, 2048), 0); del buf88  # reuse
        # Topologically Sorted Source Nodes: [out_91, out_92, out_93, out_94, out_95, input_4, out_96, out_97], Original ATen: [aten.convolution, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_relu_15.run(buf90, buf89, arg94_1, 100352, stream=stream0)
        del arg94_1
        buf91 = reinterpret_tensor(buf87, (49, 512), (512, 1), 0); del buf87  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg96_1, (49, 512), (0, 1), 0), reinterpret_tensor(buf90, (49, 2048), (2048, 1), 0), reinterpret_tensor(arg95_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf91)
        del arg95_1
        del arg96_1
        buf92 = reinterpret_tensor(buf91, (1, 512, 7, 7), (25088, 1, 3584, 512), 0); del buf91  # reuse
        # Topologically Sorted Source Nodes: [out_98, out_99], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_16.run(buf92, 25088, stream=stream0)
        # Topologically Sorted Source Nodes: [out_98, out_99, out_100], Original ATen: [aten.convolution, aten.relu]
        buf93 = extern_kernels.convolution(buf92, arg97_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (1, 512, 7, 7), (25088, 1, 3584, 512), 'torch.ops.aten.convolution.default')
        del arg97_1
        del buf92
        buf94 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [out_98, out_99, out_100, out_101, out_102], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_14.run(buf94, arg98_1, 25088, stream=stream0)
        del arg98_1
        buf95 = reinterpret_tensor(buf89, (49, 2048), (2048, 1), 0); del buf89  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg100_1, (49, 2048), (0, 1), 0), reinterpret_tensor(buf94, (49, 512), (512, 1), 0), reinterpret_tensor(arg99_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf95)
        del arg100_1
        del arg99_1
        buf96 = reinterpret_tensor(buf95, (1, 2048, 7, 7), (100352, 1, 14336, 2048), 0); del buf95  # reuse
        # Topologically Sorted Source Nodes: [out_98, out_99, out_100, out_101, out_102, out_103, out_104], Original ATen: [aten.convolution, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_relu_17.run(buf96, buf90, 100352, stream=stream0)
        buf97 = reinterpret_tensor(buf94, (49, 512), (512, 1), 0); del buf94  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg102_1, (49, 512), (0, 1), 0), reinterpret_tensor(buf96, (49, 2048), (2048, 1), 0), reinterpret_tensor(arg101_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf97)
        del arg101_1
        del arg102_1
        buf98 = reinterpret_tensor(buf97, (1, 512, 7, 7), (25088, 1, 3584, 512), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [out_105, out_106], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_16.run(buf98, 25088, stream=stream0)
        # Topologically Sorted Source Nodes: [out_105, out_106, out_107], Original ATen: [aten.convolution, aten.relu]
        buf99 = extern_kernels.convolution(buf98, arg103_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf99, (1, 512, 7, 7), (25088, 1, 3584, 512), 'torch.ops.aten.convolution.default')
        del arg103_1
        del buf98
        buf100 = buf99; del buf99  # reuse
        # Topologically Sorted Source Nodes: [out_105, out_106, out_107, out_108, out_109], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_14.run(buf100, arg104_1, 25088, stream=stream0)
        del arg104_1
        buf101 = reinterpret_tensor(buf90, (49, 2048), (2048, 1), 0); del buf90  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg106_1, (49, 2048), (0, 1), 0), reinterpret_tensor(buf100, (49, 512), (512, 1), 0), reinterpret_tensor(arg105_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf101)
        del arg105_1
        del arg106_1
        del buf100
        buf102 = empty_strided_cuda((1, 2048, 1, 1), (2048, 1, 2048, 2048), torch.float32)
        buf103 = reinterpret_tensor(buf102, (1, 2048, 1, 1), (2048, 1, 1, 1), 0); del buf102  # reuse
        # Topologically Sorted Source Nodes: [out_105, out_106, out_107, out_108, out_109, out_110, out_111, x_3], Original ATen: [aten.convolution, aten.relu, aten.add, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_convolution_mean_relu_18.run(buf103, buf101, buf96, 2048, 49, stream=stream0)
        del buf101
        del buf96
        buf104 = empty_strided_cuda((1, 1000), (1000, 1), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.addmm(reinterpret_tensor(arg108_1, (1, 1000), (0, 1), 0), reinterpret_tensor(buf103, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg107_1, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf104)
        del arg107_1
        del arg108_1
        del buf103
    return (buf104, )


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1 = args
        args.clear()
        partition0_args = [arg2_1, arg0_1, arg1_1, arg4_1, arg3_1, arg5_1, arg6_1, arg8_1, arg7_1, arg10_1, arg9_1, arg12_1, arg11_1, arg13_1, arg14_1, arg16_1, arg15_1, arg18_1, arg17_1, arg19_1, arg20_1, arg22_1, arg21_1, arg24_1, arg23_1, arg25_1, arg26_1, arg28_1, arg27_1, arg29_1, arg30_1, arg32_1, arg31_1, arg33_1, arg34_1, arg36_1, arg35_1, arg38_1, arg37_1, arg39_1, arg40_1, arg42_1, arg41_1, arg44_1, arg43_1, arg45_1, arg46_1, arg48_1, arg47_1, arg50_1, arg49_1, arg51_1, arg52_1, arg54_1, arg53_1, arg55_1, arg56_1, arg58_1, arg57_1, arg59_1, arg60_1, arg62_1, arg61_1, arg64_1, arg63_1, arg65_1, arg66_1, arg68_1, arg67_1, arg70_1, arg69_1, arg71_1, arg72_1, arg74_1, arg73_1, arg76_1, arg75_1, arg77_1, arg78_1, arg80_1, arg79_1, arg82_1, arg81_1, arg83_1, arg84_1, arg86_1, arg85_1, arg88_1, arg87_1, arg89_1, arg90_1, arg92_1, arg91_1, arg93_1, arg94_1, arg96_1, arg95_1, arg97_1, arg98_1, arg100_1, arg99_1, arg102_1, arg101_1, arg103_1, arg104_1, arg106_1, arg105_1, arg108_1, arg107_1]
        del arg2_1, arg0_1, arg1_1, arg4_1, arg3_1, arg5_1, arg6_1, arg8_1, arg7_1, arg10_1, arg9_1, arg12_1, arg11_1, arg13_1, arg14_1, arg16_1, arg15_1, arg18_1, arg17_1, arg19_1, arg20_1, arg22_1, arg21_1, arg24_1, arg23_1, arg25_1, arg26_1, arg28_1, arg27_1, arg29_1, arg30_1, arg32_1, arg31_1, arg33_1, arg34_1, arg36_1, arg35_1, arg38_1, arg37_1, arg39_1, arg40_1, arg42_1, arg41_1, arg44_1, arg43_1, arg45_1, arg46_1, arg48_1, arg47_1, arg50_1, arg49_1, arg51_1, arg52_1, arg54_1, arg53_1, arg55_1, arg56_1, arg58_1, arg57_1, arg59_1, arg60_1, arg62_1, arg61_1, arg64_1, arg63_1, arg65_1, arg66_1, arg68_1, arg67_1, arg70_1, arg69_1, arg71_1, arg72_1, arg74_1, arg73_1, arg76_1, arg75_1, arg77_1, arg78_1, arg80_1, arg79_1, arg82_1, arg81_1, arg83_1, arg84_1, arg86_1, arg85_1, arg88_1, arg87_1, arg89_1, arg90_1, arg92_1, arg91_1, arg93_1, arg94_1, arg96_1, arg95_1, arg97_1, arg98_1, arg100_1, arg99_1, arg102_1, arg101_1, arg103_1, arg104_1, arg106_1, arg105_1, arg108_1, arg107_1
        (buf104,) = self.partitions[0](partition0_args)
        del partition0_args
        return (buf104, )

runner = Runner(partitions=[partition_0,])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 7, 7), (147, 1, 21, 3), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((64, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((64, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((128, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((512, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((128, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((128, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((128, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((256, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((1024, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((2048, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((2048, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((2048, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/vw/cvwqffa5hfd344xdzwrr4fhzdoknoumwvs3ezyxkoh7jdqu5qrbf.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'kernel_num_gb': 0.004818944, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(in_out_ptr0 + (x2), tmp6, None)


def get_args():
    arg_0 = rand_strided((1, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1, 512, 28, 28), (401408, 1, 14336, 512), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 401408,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=40)
    num_gb = 0.004818944
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/ws/cwsjmy265pbspqnlizkfyy3ysldzjqjbb5ofmbaqmedwfqhhlo7y.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'kernel_num_gb': 0.000401408, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)


def get_args():
    arg_0 = rand_strided((1, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    return arg_0, 50176,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=40)
    num_gb = 0.000401408
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/xr/cxrpx5heaptqqfbc2ub3ri4afczj6lrozvgbxn5iy4a7y62jrjjs.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'kernel_num_gb': 0.000802816, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)


def get_args():
    arg_0 = rand_strided((1, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    return arg_0, 100352,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=40)
    num_gb = 0.000802816
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/xs/cxsvcdzefhb6m4rv7elorafznw6ngkxtw6cq2ipoqbjphpnloloz.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'kernel_num_gb': 0.009633792, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x0), tmp4, None)


def get_args():
    arg_0 = rand_strided((1, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((3136, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, 802816,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=40)
    num_gb = 0.009633792
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/zd/czdfw64ladpmrvkjzrgfon6hucmr7xfkbkocx6dpsywb4ptqx3fi.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'kernel_num_gb': 0.001605632, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, None)


def get_args():
    arg_0 = rand_strided((1, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    return arg_0, 200704,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=40)
    num_gb = 0.001605632
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/3i/c3ifg4cngegy6vw4icwowlrwtk4d66bywfsnf6ilxxov66socsad.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 2408704}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/5i/c5igahntqbkcfwtmk4yppdbyv7vf5avezm4ctx7rj3x463s7ivbs.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 4816896}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_5(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, None)


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/5n/c5n3lz3sm6mk4iwxzttmlca6ctuevvoxvsxunidgnoba7f2xb4qx.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 2408448}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_2(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, None)


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/7d/c7daw56u74eemp5ctegs7tusihpvrmi2j3xx3zrwwj3fdncjrtwf.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_relu_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 1613824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_relu_15(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 2048)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x2), xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(in_out_ptr0 + (x2), tmp6, xmask)


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/dl/cdl4oz55t2etuindw7put3xwg3zxvcoznnpk3xcfm6nludzmaa4a.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_relu_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 6422528}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_relu_9(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x0), tmp4, None)


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/ei/ceiln4p2y4mn7emrv2h3fhxdcbmyyhvppirmxgnrqgumg45rjrlh.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_relu_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 6424576}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_relu_7(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(in_out_ptr0 + (x2), tmp6, None)


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/fd/cfdswb6lasxrjus6y4cbn6mvots7lthjdtbfeu5uspw4v6qnhoch.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_relu_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 3211264}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_relu_13(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x0), tmp4, None)


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/gh/cgho4kuo72khce7s5gngtec5f6ldhlwdwudkqkjldiubt7oeafw5.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 602112}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_12(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/ja/cjau7whsl6zxyv7uxjfnq33r3r2vzajf45mftk57jkfngxqc3asw.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 303104}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_14(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/ki/ckizf6rhalxt7vthcmd4t6yoeubdum6bhnd2an3cwreg4ppagt6c.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 8830976}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex // 3584
    x1 = ((xindex // 64) % 56)
    x0 = (xindex % 64)
    x4 = xindex
    tmp0 = (-1) + 2*x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-7232) + x0 + 128*x1 + 14336*x2), tmp10, other=float("-inf"))
    tmp12 = 2*x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-7168) + x0 + 128*x1 + 14336*x2), tmp16, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp11, tmp17)
    tmp19 = 1 + 2*x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-7104) + x0 + 128*x1 + 14336*x2), tmp23, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp18, tmp24)
    tmp26 = 2*x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-64) + x0 + 128*x1 + 14336*x2), tmp30, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp25, tmp31)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x0 + 128*x1 + 14336*x2), tmp33, other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp32, tmp34)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (64 + x0 + 128*x1 + 14336*x2), tmp36, other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp35, tmp37)
    tmp39 = 1 + 2*x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (7104 + x0 + 128*x1 + 14336*x2), tmp43, other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp38, tmp44)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (7168 + x0 + 128*x1 + 14336*x2), tmp46, other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp45, tmp47)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (7232 + x0 + 128*x1 + 14336*x2), tmp49, other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp48, tmp50)
    tl.store(out_ptr0 + (x4), tmp51, None)


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/ln/clnwtkgg23trlygu3g42ei5wpl7alsehe3x726ilmik4faj72mvf.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 1204224}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_8(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/nq/cnq4xyl4ryjxseh7jwmkoacl2dcvzqszfuubnvf4sfom7ozh3zww.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 1204736}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_6(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/ov/cov6rjx6csbblg6qg5pqss7nzflk537wnspoepvvgerx5lcwq5dn.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_relu_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 1605632}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_relu_17(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/qg/cqgrb2omir2y7vfqcxc7y3zsddid5vjo6zqoesadzhyon7epd64p.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r0_': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_mean_relu_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 819200, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_add_convolution_mean_relu_18(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2048
    r0_numel = 49
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 2048*r0_1), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 2048*r0_1), r0_mask & xmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None].to(tl.float32)
    tmp9 = 49.0
    tmp10 = (tmp8 / tmp9)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, xmask)


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/qm/cqmtgq5ei4wc2st4sa3ewq5nih3bqi2ksz72kv237f2sea7hl63d.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 301056}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_16(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/rd/crddz7i2hfomvnbtzww3u6cal3bvtcyfbrbkq7oacywi5b2a25qa.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 9634048}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/3i/c3ifg4cngegy6vw4icwowlrwtk4d66bywfsnf6ilxxov66socsad.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 2408704}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/5i/c5igahntqbkcfwtmk4yppdbyv7vf5avezm4ctx7rj3x463s7ivbs.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 4816896}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_5(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, None)


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/5n/c5n3lz3sm6mk4iwxzttmlca6ctuevvoxvsxunidgnoba7f2xb4qx.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 2408448}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_2(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, None)


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/7d/c7daw56u74eemp5ctegs7tusihpvrmi2j3xx3zrwwj3fdncjrtwf.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_relu_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 1613824}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_relu_15(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 2048)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x2), xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(in_out_ptr0 + (x2), tmp6, xmask)


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/dl/cdl4oz55t2etuindw7put3xwg3zxvcoznnpk3xcfm6nludzmaa4a.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_relu_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 6422528}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_relu_9(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x0), tmp4, None)


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/ei/ceiln4p2y4mn7emrv2h3fhxdcbmyyhvppirmxgnrqgumg45rjrlh.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_relu_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 6424576}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_relu_7(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(in_out_ptr0 + (x2), tmp6, None)


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/fd/cfdswb6lasxrjus6y4cbn6mvots7lthjdtbfeu5uspw4v6qnhoch.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_relu_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 3211264}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_relu_13(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x0), tmp4, None)


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/gh/cgho4kuo72khce7s5gngtec5f6ldhlwdwudkqkjldiubt7oeafw5.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 602112}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_12(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/ja/cjau7whsl6zxyv7uxjfnq33r3r2vzajf45mftk57jkfngxqc3asw.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 303104}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_14(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/ki/ckizf6rhalxt7vthcmd4t6yoeubdum6bhnd2an3cwreg4ppagt6c.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 8830976}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex // 3584
    x1 = ((xindex // 64) % 56)
    x0 = (xindex % 64)
    x4 = xindex
    tmp0 = (-1) + 2*x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-7232) + x0 + 128*x1 + 14336*x2), tmp10, other=float("-inf"))
    tmp12 = 2*x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-7168) + x0 + 128*x1 + 14336*x2), tmp16, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp11, tmp17)
    tmp19 = 1 + 2*x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-7104) + x0 + 128*x1 + 14336*x2), tmp23, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp18, tmp24)
    tmp26 = 2*x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-64) + x0 + 128*x1 + 14336*x2), tmp30, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp25, tmp31)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x0 + 128*x1 + 14336*x2), tmp33, other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp32, tmp34)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (64 + x0 + 128*x1 + 14336*x2), tmp36, other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp35, tmp37)
    tmp39 = 1 + 2*x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (7104 + x0 + 128*x1 + 14336*x2), tmp43, other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp38, tmp44)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (7168 + x0 + 128*x1 + 14336*x2), tmp46, other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp45, tmp47)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (7232 + x0 + 128*x1 + 14336*x2), tmp49, other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp48, tmp50)
    tl.store(out_ptr0 + (x4), tmp51, None)


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/ln/clnwtkgg23trlygu3g42ei5wpl7alsehe3x726ilmik4faj72mvf.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 1204224}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_8(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/nq/cnq4xyl4ryjxseh7jwmkoacl2dcvzqszfuubnvf4sfom7ozh3zww.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 1204736}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_6(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/ov/cov6rjx6csbblg6qg5pqss7nzflk537wnspoepvvgerx5lcwq5dn.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_relu_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 1605632}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_relu_17(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/qg/cqgrb2omir2y7vfqcxc7y3zsddid5vjo6zqoesadzhyon7epd64p.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r0_': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_mean_relu_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 819200, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_add_convolution_mean_relu_18(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2048
    r0_numel = 49
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 2048*r0_1), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + 2048*r0_1), r0_mask & xmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None].to(tl.float32)
    tmp9 = 49.0
    tmp10 = (tmp8 / tmp9)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, xmask)


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/qm/cqmtgq5ei4wc2st4sa3ewq5nih3bqi2ksz72kv237f2sea7hl63d.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 301056}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_16(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/rd/crddz7i2hfomvnbtzww3u6cal3bvtcyfbrbkq7oacywi5b2a25qa.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 9634048}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/wz/cwzofax6cyonsvhtnuxdr2q7ge6iaqpoimg5zzxkjtvxvhe5t6vt.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 603136}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_10(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/x7/cx76xekonbbq6erlyvkttofd7dllhgm2eyje6rphdcabzstpu7se.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_relu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 12845056}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_relu_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x0), tmp4, None)


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/tmpgde7ylor/yb/cyb3xg5ofpcedgj7sjvckbfdbqieqijbjwjdmuhlsn7y4eqksui3.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_relu_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 3215360}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_relu_11(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(in_out_ptr0 + (x2), tmp6, None)


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/wz/cwzofax6cyonsvhtnuxdr2q7ge6iaqpoimg5zzxkjtvxvhe5t6vt.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 603136}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_10(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/x7/cx76xekonbbq6erlyvkttofd7dllhgm2eyje6rphdcabzstpu7se.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_relu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 12845056}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_relu_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x0), tmp4, None)


# ===== inductor generated file: /tmp/cnnbench-compile-repeats-1rdlj89b/inductor/repeat_04/fold/attempt_01/torchinductor/yb/cyb3xg5ofpcedgj7sjvckbfdbqieqijbjwjdmuhlsn7y4eqksui3.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_relu_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '5C4E406C711B3861DF9C100323E0EC398E2F633BD8802E2E564CD4776AA7ED44', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False, 'tiling_scores': {'x': 3215360}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_relu_11(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(in_out_ptr0 + (x2), tmp6, None)
