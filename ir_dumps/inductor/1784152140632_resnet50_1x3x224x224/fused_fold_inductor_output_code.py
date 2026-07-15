# ===== inductor generated file: /tmp/torchinductor_kamei/tmpb3zpfqrg/22/c22ojebo7ehbcjs5jbmmsq2v3aenqz5avymrdjnrww63gl2twiwq.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'kernel_num_gb': 0.0},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, None)


def get_args():
    arg_0 = rand_strided((1, 128, 28, 28), (100352, 1, 3584, 128), device='cuda:0', dtype=torch.float32)
    return arg_0,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_.run(*args, 100352, grid=grid(100352), stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_.benchmark_all_configs(*args, 100352, grid=grid(100352))


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=40, fast_flush=True)
    num_gb = 0.0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")


# ===== inductor generated file: /tmp/torchinductor_kamei/tmpb3zpfqrg/c6/cc6qve6fu2vs7yzjon7d6irdxn5yqcqrwx36ndx4rzcqmvgkfpqq.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'kernel_num_gb': 0.0},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, None)


def get_args():
    arg_0 = rand_strided((1, 256, 56, 56), (802816, 1, 14336, 256), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((3136, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_.run(*args, 802816, grid=grid(802816), stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_.benchmark_all_configs(*args, 802816, grid=grid(802816))


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=40, fast_flush=True)
    num_gb = 0.0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")


# ===== inductor generated file: /tmp/torchinductor_kamei/tmpb3zpfqrg/dh/cdhz7iivxj5bzli6o6ayj6dooavvsdexktpezr2fzdlrwik7pgcf.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid

@triton_heuristics.pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'kernel_num_gb': 0.0},
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
    return arg_0,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_.run(*args, 401408, grid=grid(401408), stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_.benchmark_all_configs(*args, 401408, grid=grid(401408))


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=40, fast_flush=True)
    num_gb = 0.0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")


# ===== inductor generated file: /tmp/torchinductor_kamei/tmpb3zpfqrg/e2/ce2m5imyy6wewfiyqom2dps3de6r5weovndavoasj5i5a3xtndup.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'kernel_num_gb': 0.0},
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
    return arg_0,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_.run(*args, 25088, grid=grid(25088), stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_.benchmark_all_configs(*args, 25088, grid=grid(25088))


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=40, fast_flush=True)
    num_gb = 0.0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")


# ===== inductor generated file: /tmp/torchinductor_kamei/tmpb3zpfqrg/g2/cg2e6b3ibaja2sn7rnyszns73vpxr7cvzwqmzouggqqzef5hixb6.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'kernel_num_gb': 0.0},
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
    return arg_0,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_.run(*args, 200704, grid=grid(200704), stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_.benchmark_all_configs(*args, 200704, grid=grid(200704))


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=40, fast_flush=True)
    num_gb = 0.0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")


# ===== inductor generated file: /tmp/torchinductor_kamei/tmpb3zpfqrg/h6/ch6ticxv6kfadxoecyznnotvgy2uusvyi7v3efgy6mrtsik4z5he.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'kernel_num_gb': 0.0},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, None)


def get_args():
    arg_0 = rand_strided((1, 512, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
    return arg_0,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_.run(*args, 100352, grid=grid(100352), stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_.benchmark_all_configs(*args, 100352, grid=grid(100352))


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=40, fast_flush=True)
    num_gb = 0.0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")


# ===== inductor generated file: /tmp/torchinductor_kamei/tmpb3zpfqrg/jq/cjqzhcnlbwc7r32ubhochvk77a2urpjhrftw5htp47ji2tlnnfb6.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'kernel_num_gb': 0.0},
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
    return arg_0,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_.run(*args, 50176, grid=grid(50176), stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_.benchmark_all_configs(*args, 50176, grid=grid(50176))


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=40, fast_flush=True)
    num_gb = 0.0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")


# ===== inductor generated file: /tmp/torchinductor_kamei/tmpb3zpfqrg/kp/ckpeldrcdeggmgubbqzrxm2nvou2ouewjg3xlftw2gcdijyeyqjf.py =====
# AOT ID: ['1_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, grid_combo_kernels, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_kamei/tmpb3zpfqrg/lh/clh32qktusc3xqs3tfxnxdo65xnwwwpr3juep4f4mboepnl2gpr4.py
# Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   x => convolution
#   x_1 => relu
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [2, 2], [3, 3], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
triton_poi_fused_convolution_relu_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_0', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/tmpb3zpfqrg/e5/ce5rpohrwatzefhv4ufejoz5ukoh4gmceub452hrywf46anq4n4s.py
# Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x => convolution
#   x_1 => relu
#   x_2 => _low_memory_max_pool2d_with_offsets
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg0_1, %arg1_1, [2, 2], [3, 3], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu, [3, 3], [2, 2], [1, 1], [1, 1], False), kwargs = {})
triton_poi_fused_convolution_max_pool2d_with_indices_relu_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 3584)
    x1 = (xindex // 64) % 56
    x0 = xindex % 64
    x4 = xindex
    tmp0 = (-1) + (2*x2)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x1)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-7232) + x0 + (128*x1) + (14336*x2)), tmp10, other=float("-inf"))
    tmp12 = 2*x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-7168) + x0 + (128*x1) + (14336*x2)), tmp16, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + (2*x1)
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-7104) + x0 + (128*x1) + (14336*x2)), tmp23, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 2*x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-64) + x0 + (128*x1) + (14336*x2)), tmp30, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x0 + (128*x1) + (14336*x2)), tmp33, other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (64 + x0 + (128*x1) + (14336*x2)), tmp36, other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = 1 + (2*x2)
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (7104 + x0 + (128*x1) + (14336*x2)), tmp43, other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp38)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (7168 + x0 + (128*x1) + (14336*x2)), tmp46, other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (7232 + x0 + (128*x1) + (14336*x2)), tmp49, other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp50, tmp48)
    tl.store(out_ptr0 + (x4), tmp51, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/tmpb3zpfqrg/pm/cpmgdrjk5xvet4vvgopklynqv3y6ditbbwm2sb2fdhpcoequm6nn.py
# Topologically Sorted Source Nodes: [out_1], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   out_1 => relu_1
# Graph fragment:
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
triton_poi_fused_relu_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/tmpb3zpfqrg/if/ciffvox6i3orrhkfuosohkobkfltpvnbzv4r4bjdd4sfbh32kuwy.py
# Topologically Sorted Source Nodes: [out_1, out_2, out_3, out_4], Original ATen: [aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   out_1 => relu_1
#   out_2 => convolution_2
#   out_3 => relu_2
#   out_4 => convolution_3
# Graph fragment:
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_1, %arg5_1, %arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
#   %convolution_3 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %arg7_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_relu_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_3', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/tmpb3zpfqrg/2l/c2lml6wpxrcqw4lq6fjpth2cfz5xf2evatr7onigd2xde2is7fkk.py
# Topologically Sorted Source Nodes: [out_5, out_6, out_7], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_5 => add_1, mul_1, mul_2, sub
#   out_6 => add_2
#   out_7 => relu_3
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %convolution_4), kwargs = {})
#   %relu_3 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_2,), kwargs = {})
triton_poi_fused_add_cudnn_batch_norm_relu_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cudnn_batch_norm_relu_4', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/tmpb3zpfqrg/25/c2565gcz6gg3zyvudxe4p3mrlzf424rlgczglgrt47lsrpw5nrf7.py
# Topologically Sorted Source Nodes: [out_21, out_22, out_23], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_21 => add_7, mul_7, mul_8, sub_2
#   out_22 => add_8
#   out_23 => relu_9
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_10, %unsqueeze_17), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %relu_6), kwargs = {})
#   %relu_9 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_8,), kwargs = {})
triton_poi_fused_add_cudnn_batch_norm_relu_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cudnn_batch_norm_relu_5', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/tmpb3zpfqrg/xa/cxafh6hqtknez4tydrkdfs5l6fqa6hfphkv2zo6szm5oxomte6r3.py
# Topologically Sorted Source Nodes: [out_25], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   out_25 => relu_10
# Graph fragment:
#   %relu_10 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_11,), kwargs = {})
triton_poi_fused_relu_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_6', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/tmpb3zpfqrg/e3/ce3bnuz4kdqfskmocijr3j2ntxskgmoympjjeuc5uck23r3qg2q6.py
# Topologically Sorted Source Nodes: [out_25, out_26, out_27, out_28], Original ATen: [aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   out_25 => relu_10
#   out_26 => convolution_12
#   out_27 => relu_11
#   out_28 => convolution_13
# Graph fragment:
#   %relu_10 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_11,), kwargs = {})
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_10, %arg34_1, %arg35_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_11 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_12,), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_11, %arg36_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_relu_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_7', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/tmpb3zpfqrg/oc/cochuo4yfxnqhqfzcnbi7ckjubo7pphqfpacmn4bynqyuucrvxmi.py
# Topologically Sorted Source Nodes: [out_29, input_2, out_30, out_31], Original ATen: [aten.cudnn_batch_norm, aten.convolution, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_2 => convolution_14
#   out_29 => add_10, mul_10, mul_11, sub_3
#   out_30 => add_11
#   out_31 => relu_12
# Graph fragment:
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_13, %unsqueeze_25), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_9, %arg41_1, %arg42_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %convolution_14), kwargs = {})
#   %relu_12 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_11,), kwargs = {})
triton_poi_fused_add_convolution_cudnn_batch_norm_relu_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_cudnn_batch_norm_relu_8', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), None)
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp18 = tmp16 + tmp17
    tmp19 = tmp15 + tmp18
    tmp20 = tl.full([1], 0, tl.int32)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/tmpb3zpfqrg/dw/cdwnvuz5vuvkee5j4fwwz3hu37y3qpyicwxrzcqkwt434jatmspd.py
# Topologically Sorted Source Nodes: [out_33], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   out_33 => relu_13
# Graph fragment:
#   %relu_13 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_15,), kwargs = {})
triton_poi_fused_relu_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_9', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/tmpb3zpfqrg/6t/c6t7n4xry6q2zxdoq4hniwurqcffvby2a7vzzi5gqwvqm5c64liz.py
# Topologically Sorted Source Nodes: [out_37, out_38, out_39], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_37 => add_13, mul_13, mul_14, sub_4
#   out_38 => add_14
#   out_39 => relu_15
# Graph fragment:
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_17, %unsqueeze_33), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_37), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_39), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_13, %relu_12), kwargs = {})
#   %relu_15 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_14,), kwargs = {})
triton_poi_fused_add_cudnn_batch_norm_relu_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cudnn_batch_norm_relu_10', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/tmpb3zpfqrg/h3/ch3akm6ueyhturnekgx5jj4s2ngeuopdv5cbqrgzs7qq5w6rnpkn.py
# Topologically Sorted Source Nodes: [out_57, out_58, out_59, out_60], Original ATen: [aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   out_57 => relu_22
#   out_58 => convolution_25
#   out_59 => relu_23
#   out_60 => convolution_26
# Graph fragment:
#   %relu_22 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_24,), kwargs = {})
#   %convolution_25 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_22, %arg72_1, %arg73_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_23 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_25,), kwargs = {})
#   %convolution_26 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_23, %arg74_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_relu_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_11', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/tmpb3zpfqrg/oq/coqt5cajb762s5omwvztapnahmioa4pd6fe5oj6qgkymgxfnb7og.py
# Topologically Sorted Source Nodes: [out_61, input_3, out_62, out_63], Original ATen: [aten.cudnn_batch_norm, aten.convolution, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_3 => convolution_27
#   out_61 => add_22, mul_22, mul_23, sub_7
#   out_62 => add_23
#   out_63 => relu_24
# Graph fragment:
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_26, %unsqueeze_57), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_59), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %unsqueeze_61), kwargs = {})
#   %add_22 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %unsqueeze_63), kwargs = {})
#   %convolution_27 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_21, %arg79_1, %arg80_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_22, %convolution_27), kwargs = {})
#   %relu_24 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_23,), kwargs = {})
triton_poi_fused_add_convolution_cudnn_batch_norm_relu_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_cudnn_batch_norm_relu_12', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), None)
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp18 = tmp16 + tmp17
    tmp19 = tmp15 + tmp18
    tmp20 = tl.full([1], 0, tl.int32)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/tmpb3zpfqrg/at/cat77m4xzjz7njffubxv2nvshduhw5o3vytdc4jla3czc2koyli3.py
# Topologically Sorted Source Nodes: [out_65], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   out_65 => relu_25
# Graph fragment:
#   %relu_25 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_28,), kwargs = {})
triton_poi_fused_relu_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_13', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/tmpb3zpfqrg/nv/cnvt5e6ma7disnpv4heudu224wnia2uxhr7kk3yeghqjhtigr4mt.py
# Topologically Sorted Source Nodes: [out_69, out_70, out_71], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_69 => add_25, mul_25, mul_26, sub_8
#   out_70 => add_26
#   out_71 => relu_27
# Graph fragment:
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_30, %unsqueeze_65), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_67), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %unsqueeze_69), kwargs = {})
#   %add_25 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %unsqueeze_71), kwargs = {})
#   %add_26 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_25, %relu_24), kwargs = {})
#   %relu_27 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_26,), kwargs = {})
triton_poi_fused_add_cudnn_batch_norm_relu_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cudnn_batch_norm_relu_14', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/tmpb3zpfqrg/md/cmdnkzbdprtaryj2mccow56arwl75fhexmly5pv26eif3lmrcp5n.py
# Topologically Sorted Source Nodes: [out_105, out_106, out_107, out_108], Original ATen: [aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   out_105 => relu_40
#   out_106 => convolution_44
#   out_107 => relu_41
#   out_108 => convolution_45
# Graph fragment:
#   %relu_40 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_43,), kwargs = {})
#   %convolution_44 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_40, %arg128_1, %arg129_1, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_41 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_44,), kwargs = {})
#   %convolution_45 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_41, %arg130_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_relu_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_15', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/tmpb3zpfqrg/lh/clhz5exshgpbl33dzuudnv4lvsndrivw3gnetwoph7jxjcg6tvpl.py
# Topologically Sorted Source Nodes: [out_109, input_4, out_110, out_111], Original ATen: [aten.cudnn_batch_norm, aten.convolution, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_4 => convolution_46
#   out_109 => add_40, mul_40, mul_41, sub_13
#   out_110 => add_41
#   out_111 => relu_42
# Graph fragment:
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_45, %unsqueeze_105), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_107), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_40, %unsqueeze_109), kwargs = {})
#   %add_40 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_41, %unsqueeze_111), kwargs = {})
#   %convolution_46 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_39, %arg135_1, %arg136_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_41 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_40, %convolution_46), kwargs = {})
#   %relu_42 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_41,), kwargs = {})
triton_poi_fused_add_convolution_cudnn_batch_norm_relu_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_cudnn_batch_norm_relu_16', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 2048
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), None)
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp18 = tmp16 + tmp17
    tmp19 = tmp15 + tmp18
    tmp20 = tl.full([1], 0, tl.int32)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/tmpb3zpfqrg/zy/czyl5lufhqeib6stzy4lac5xwoq5whap4ilhyshwzgldkoox572o.py
# Topologically Sorted Source Nodes: [out_113], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   out_113 => relu_43
# Graph fragment:
#   %relu_43 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_47,), kwargs = {})
triton_poi_fused_relu_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_17', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/tmpb3zpfqrg/yn/cyno2fmy3faurpdqs7m3swsliirivqiyvzxz24uuazvx3kr4clc7.py
# Topologically Sorted Source Nodes: [out_117, out_118, out_119], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_117 => add_43, mul_43, mul_44, sub_14
#   out_118 => add_44
#   out_119 => relu_45
# Graph fragment:
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_49, %unsqueeze_113), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %unsqueeze_115), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, %unsqueeze_117), kwargs = {})
#   %add_43 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_44, %unsqueeze_119), kwargs = {})
#   %add_44 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_43, %relu_42), kwargs = {})
#   %relu_45 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_44,), kwargs = {})
triton_poi_fused_add_cudnn_batch_norm_relu_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cudnn_batch_norm_relu_18', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 2048
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/tmpb3zpfqrg/ob/cobjt6a5qna5znq6sgxi5cu2ri44on5jbk6omajfjxex5hljlktw.py
# Topologically Sorted Source Nodes: [out_125, out_126, out_127, x_3], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu, aten.mean]
# Source node to ATen node mapping:
#   out_125 => add_46, mul_46, mul_47, sub_15
#   out_126 => add_47
#   out_127 => relu_48
#   x_3 => mean
# Graph fragment:
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_52, %unsqueeze_121), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %unsqueeze_123), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_46, %unsqueeze_125), kwargs = {})
#   %add_46 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %unsqueeze_127), kwargs = {})
#   %add_47 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_46, %relu_45), kwargs = {})
#   %relu_48 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_47,), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_48, [-1, -2], True), kwargs = {})
triton_per_fused_add_cudnn_batch_norm_mean_relu_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cudnn_batch_norm_mean_relu_19', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (2048*r1)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0 + (2048*r1)), rmask, other=0.0)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1, 1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp22 = tl.where(rmask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None]
    tmp24 = 49.0
    tmp25 = tmp23 / tmp24
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp25, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 3, 7, 7), (147, 1, 21, 3))
    assert_size_stride(arg1_1, (64, ), (1, ))
    assert_size_stride(arg2_1, (1, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(arg3_1, (64, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg5_1, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(arg6_1, (64, ), (1, ))
    assert_size_stride(arg7_1, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(arg8_1, (256, ), (1, ))
    assert_size_stride(arg9_1, (256, ), (1, ))
    assert_size_stride(arg10_1, (256, ), (1, ))
    assert_size_stride(arg11_1, (256, ), (1, ))
    assert_size_stride(arg12_1, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(arg13_1, (256, ), (1, ))
    assert_size_stride(arg14_1, (64, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg15_1, (64, ), (1, ))
    assert_size_stride(arg16_1, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(arg17_1, (64, ), (1, ))
    assert_size_stride(arg18_1, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(arg19_1, (256, ), (1, ))
    assert_size_stride(arg20_1, (256, ), (1, ))
    assert_size_stride(arg21_1, (256, ), (1, ))
    assert_size_stride(arg22_1, (256, ), (1, ))
    assert_size_stride(arg23_1, (64, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg24_1, (64, ), (1, ))
    assert_size_stride(arg25_1, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(arg26_1, (64, ), (1, ))
    assert_size_stride(arg27_1, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(arg28_1, (256, ), (1, ))
    assert_size_stride(arg29_1, (256, ), (1, ))
    assert_size_stride(arg30_1, (256, ), (1, ))
    assert_size_stride(arg31_1, (256, ), (1, ))
    assert_size_stride(arg32_1, (128, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg33_1, (128, ), (1, ))
    assert_size_stride(arg34_1, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(arg35_1, (128, ), (1, ))
    assert_size_stride(arg36_1, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(arg37_1, (512, ), (1, ))
    assert_size_stride(arg38_1, (512, ), (1, ))
    assert_size_stride(arg39_1, (512, ), (1, ))
    assert_size_stride(arg40_1, (512, ), (1, ))
    assert_size_stride(arg41_1, (512, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg42_1, (512, ), (1, ))
    assert_size_stride(arg43_1, (128, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg44_1, (128, ), (1, ))
    assert_size_stride(arg45_1, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(arg46_1, (128, ), (1, ))
    assert_size_stride(arg47_1, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(arg48_1, (512, ), (1, ))
    assert_size_stride(arg49_1, (512, ), (1, ))
    assert_size_stride(arg50_1, (512, ), (1, ))
    assert_size_stride(arg51_1, (512, ), (1, ))
    assert_size_stride(arg52_1, (128, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg53_1, (128, ), (1, ))
    assert_size_stride(arg54_1, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(arg55_1, (128, ), (1, ))
    assert_size_stride(arg56_1, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(arg57_1, (512, ), (1, ))
    assert_size_stride(arg58_1, (512, ), (1, ))
    assert_size_stride(arg59_1, (512, ), (1, ))
    assert_size_stride(arg60_1, (512, ), (1, ))
    assert_size_stride(arg61_1, (128, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg62_1, (128, ), (1, ))
    assert_size_stride(arg63_1, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(arg64_1, (128, ), (1, ))
    assert_size_stride(arg65_1, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(arg66_1, (512, ), (1, ))
    assert_size_stride(arg67_1, (512, ), (1, ))
    assert_size_stride(arg68_1, (512, ), (1, ))
    assert_size_stride(arg69_1, (512, ), (1, ))
    assert_size_stride(arg70_1, (256, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg71_1, (256, ), (1, ))
    assert_size_stride(arg72_1, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(arg73_1, (256, ), (1, ))
    assert_size_stride(arg74_1, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg75_1, (1024, ), (1, ))
    assert_size_stride(arg76_1, (1024, ), (1, ))
    assert_size_stride(arg77_1, (1024, ), (1, ))
    assert_size_stride(arg78_1, (1024, ), (1, ))
    assert_size_stride(arg79_1, (1024, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg80_1, (1024, ), (1, ))
    assert_size_stride(arg81_1, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg82_1, (256, ), (1, ))
    assert_size_stride(arg83_1, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(arg84_1, (256, ), (1, ))
    assert_size_stride(arg85_1, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg86_1, (1024, ), (1, ))
    assert_size_stride(arg87_1, (1024, ), (1, ))
    assert_size_stride(arg88_1, (1024, ), (1, ))
    assert_size_stride(arg89_1, (1024, ), (1, ))
    assert_size_stride(arg90_1, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg91_1, (256, ), (1, ))
    assert_size_stride(arg92_1, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(arg93_1, (256, ), (1, ))
    assert_size_stride(arg94_1, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg95_1, (1024, ), (1, ))
    assert_size_stride(arg96_1, (1024, ), (1, ))
    assert_size_stride(arg97_1, (1024, ), (1, ))
    assert_size_stride(arg98_1, (1024, ), (1, ))
    assert_size_stride(arg99_1, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg100_1, (256, ), (1, ))
    assert_size_stride(arg101_1, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(arg102_1, (256, ), (1, ))
    assert_size_stride(arg103_1, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg104_1, (1024, ), (1, ))
    assert_size_stride(arg105_1, (1024, ), (1, ))
    assert_size_stride(arg106_1, (1024, ), (1, ))
    assert_size_stride(arg107_1, (1024, ), (1, ))
    assert_size_stride(arg108_1, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg109_1, (256, ), (1, ))
    assert_size_stride(arg110_1, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(arg111_1, (256, ), (1, ))
    assert_size_stride(arg112_1, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg113_1, (1024, ), (1, ))
    assert_size_stride(arg114_1, (1024, ), (1, ))
    assert_size_stride(arg115_1, (1024, ), (1, ))
    assert_size_stride(arg116_1, (1024, ), (1, ))
    assert_size_stride(arg117_1, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg118_1, (256, ), (1, ))
    assert_size_stride(arg119_1, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(arg120_1, (256, ), (1, ))
    assert_size_stride(arg121_1, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg122_1, (1024, ), (1, ))
    assert_size_stride(arg123_1, (1024, ), (1, ))
    assert_size_stride(arg124_1, (1024, ), (1, ))
    assert_size_stride(arg125_1, (1024, ), (1, ))
    assert_size_stride(arg126_1, (512, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg127_1, (512, ), (1, ))
    assert_size_stride(arg128_1, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(arg129_1, (512, ), (1, ))
    assert_size_stride(arg130_1, (2048, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg131_1, (2048, ), (1, ))
    assert_size_stride(arg132_1, (2048, ), (1, ))
    assert_size_stride(arg133_1, (2048, ), (1, ))
    assert_size_stride(arg134_1, (2048, ), (1, ))
    assert_size_stride(arg135_1, (2048, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg136_1, (2048, ), (1, ))
    assert_size_stride(arg137_1, (512, 2048, 1, 1), (2048, 1, 2048, 2048))
    assert_size_stride(arg138_1, (512, ), (1, ))
    assert_size_stride(arg139_1, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(arg140_1, (512, ), (1, ))
    assert_size_stride(arg141_1, (2048, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg142_1, (2048, ), (1, ))
    assert_size_stride(arg143_1, (2048, ), (1, ))
    assert_size_stride(arg144_1, (2048, ), (1, ))
    assert_size_stride(arg145_1, (2048, ), (1, ))
    assert_size_stride(arg146_1, (512, 2048, 1, 1), (2048, 1, 2048, 2048))
    assert_size_stride(arg147_1, (512, ), (1, ))
    assert_size_stride(arg148_1, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(arg149_1, (512, ), (1, ))
    assert_size_stride(arg150_1, (2048, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg151_1, (2048, ), (1, ))
    assert_size_stride(arg152_1, (2048, ), (1, ))
    assert_size_stride(arg153_1, (2048, ), (1, ))
    assert_size_stride(arg154_1, (2048, ), (1, ))
    assert_size_stride(arg155_1, (1000, 2048), (2048, 1))
    assert_size_stride(arg156_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg2_1, arg0_1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (1, 64, 112, 112), (802816, 1, 7168, 64))
        del arg0_1
        del arg2_1
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [x, x_1], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_0.run(buf1, arg1_1, 802816, grid=grid(802816), stream=stream0)
        del arg1_1
        buf2 = empty_strided_cuda((1, 64, 56, 56), (200704, 1, 3584, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.convolution, aten.relu, aten.max_pool2d_with_indices]
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_1.run(buf1, buf2, 200704, grid=grid(200704), stream=stream0)
        buf3 = empty_strided_cuda((3136, 64), (64, 1), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg4_1, (3136, 64), (0, 1), 0), reinterpret_tensor(buf2, (3136, 64), (64, 1), 0), reinterpret_tensor(arg3_1, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf3)
        del arg3_1
        del arg4_1
        buf4 = reinterpret_tensor(buf3, (1, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [out_1], Original ATen: [aten.relu]
        triton_poi_fused_relu_2.run(buf4, 200704, grid=grid(200704), stream=stream0)
        # Topologically Sorted Source Nodes: [out_1, out_2], Original ATen: [aten.relu, aten.convolution]
        buf5 = extern_kernels.convolution(buf4, arg5_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (1, 64, 56, 56), (200704, 1, 3584, 64))
        del arg5_1
        del buf4
        buf6 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [out_1, out_2, out_3, out_4], Original ATen: [aten.relu, aten.convolution]
        triton_poi_fused_convolution_relu_3.run(buf6, arg6_1, 200704, grid=grid(200704), stream=stream0)
        del arg6_1
        buf7 = reinterpret_tensor(buf1, (3136, 256), (256, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [out_1, out_2, out_3, out_4], Original ATen: [aten.relu, aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf6, (3136, 64), (64, 1), 0), reinterpret_tensor(arg7_1, (64, 256), (1, 64), 0), out=buf7)
        del arg7_1
        del buf6
        buf8 = empty_strided_cuda((3136, 256), (256, 1), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg13_1, (3136, 256), (0, 1), 0), reinterpret_tensor(buf2, (3136, 64), (64, 1), 0), reinterpret_tensor(arg12_1, (64, 256), (1, 64), 0), alpha=1, beta=1, out=buf8)
        del arg12_1
        del arg13_1
        buf9 = reinterpret_tensor(buf7, (1, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [out_5, out_6, out_7], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu]
        triton_poi_fused_add_cudnn_batch_norm_relu_4.run(buf9, arg8_1, arg9_1, arg10_1, arg11_1, buf8, 802816, grid=grid(802816), stream=stream0)
        del arg10_1
        del arg11_1
        del arg8_1
        del arg9_1
        buf10 = reinterpret_tensor(buf2, (3136, 64), (64, 1), 0); del buf2  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg15_1, (3136, 64), (0, 1), 0), reinterpret_tensor(buf9, (3136, 256), (256, 1), 0), reinterpret_tensor(arg14_1, (256, 64), (1, 256), 0), alpha=1, beta=1, out=buf10)
        del arg14_1
        del arg15_1
        buf11 = reinterpret_tensor(buf10, (1, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf10  # reuse
        # Topologically Sorted Source Nodes: [out_9], Original ATen: [aten.relu]
        triton_poi_fused_relu_2.run(buf11, 200704, grid=grid(200704), stream=stream0)
        # Topologically Sorted Source Nodes: [out_9, out_10], Original ATen: [aten.relu, aten.convolution]
        buf12 = extern_kernels.convolution(buf11, arg16_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (1, 64, 56, 56), (200704, 1, 3584, 64))
        del arg16_1
        del buf11
        buf13 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [out_9, out_10, out_11, out_12], Original ATen: [aten.relu, aten.convolution]
        triton_poi_fused_convolution_relu_3.run(buf13, arg17_1, 200704, grid=grid(200704), stream=stream0)
        del arg17_1
        buf14 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [out_9, out_10, out_11, out_12], Original ATen: [aten.relu, aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf13, (3136, 64), (64, 1), 0), reinterpret_tensor(arg18_1, (64, 256), (1, 64), 0), out=buf14)
        del arg18_1
        buf15 = reinterpret_tensor(buf14, (1, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [out_13, out_14, out_15], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu]
        triton_poi_fused_add_cudnn_batch_norm_relu_4.run(buf15, arg19_1, arg20_1, arg21_1, arg22_1, buf9, 802816, grid=grid(802816), stream=stream0)
        del arg19_1
        del arg20_1
        del arg21_1
        del arg22_1
        buf16 = reinterpret_tensor(buf13, (3136, 64), (64, 1), 0); del buf13  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg24_1, (3136, 64), (0, 1), 0), reinterpret_tensor(buf15, (3136, 256), (256, 1), 0), reinterpret_tensor(arg23_1, (256, 64), (1, 256), 0), alpha=1, beta=1, out=buf16)
        del arg23_1
        del arg24_1
        buf17 = reinterpret_tensor(buf16, (1, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [out_17], Original ATen: [aten.relu]
        triton_poi_fused_relu_2.run(buf17, 200704, grid=grid(200704), stream=stream0)
        # Topologically Sorted Source Nodes: [out_17, out_18], Original ATen: [aten.relu, aten.convolution]
        buf18 = extern_kernels.convolution(buf17, arg25_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (1, 64, 56, 56), (200704, 1, 3584, 64))
        del arg25_1
        del buf17
        buf19 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [out_17, out_18, out_19, out_20], Original ATen: [aten.relu, aten.convolution]
        triton_poi_fused_convolution_relu_3.run(buf19, arg26_1, 200704, grid=grid(200704), stream=stream0)
        del arg26_1
        buf20 = reinterpret_tensor(buf9, (3136, 256), (256, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [out_17, out_18, out_19, out_20], Original ATen: [aten.relu, aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf19, (3136, 64), (64, 1), 0), reinterpret_tensor(arg27_1, (64, 256), (1, 64), 0), out=buf20)
        del arg27_1
        buf21 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [out_21, out_22, out_23], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu]
        triton_poi_fused_add_cudnn_batch_norm_relu_5.run(buf21, buf20, arg28_1, arg29_1, arg30_1, arg31_1, 802816, grid=grid(802816), stream=stream0)
        del arg28_1
        del arg29_1
        del arg30_1
        del arg31_1
        del buf20
        buf22 = empty_strided_cuda((3136, 128), (128, 1), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg33_1, (3136, 128), (0, 1), 0), reinterpret_tensor(buf21, (3136, 256), (256, 1), 0), reinterpret_tensor(arg32_1, (256, 128), (1, 256), 0), alpha=1, beta=1, out=buf22)
        del arg32_1
        del arg33_1
        buf23 = reinterpret_tensor(buf22, (1, 128, 56, 56), (401408, 1, 7168, 128), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [out_25], Original ATen: [aten.relu]
        triton_poi_fused_relu_6.run(buf23, 401408, grid=grid(401408), stream=stream0)
        # Topologically Sorted Source Nodes: [out_25, out_26], Original ATen: [aten.relu, aten.convolution]
        buf24 = extern_kernels.convolution(buf23, arg34_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (1, 128, 28, 28), (100352, 1, 3584, 128))
        del arg34_1
        buf25 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [out_25, out_26, out_27, out_28], Original ATen: [aten.relu, aten.convolution]
        triton_poi_fused_convolution_relu_7.run(buf25, arg35_1, 100352, grid=grid(100352), stream=stream0)
        del arg35_1
        buf26 = reinterpret_tensor(buf23, (784, 512), (512, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [out_25, out_26, out_27, out_28], Original ATen: [aten.relu, aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf25, (784, 128), (128, 1), 0), reinterpret_tensor(arg36_1, (128, 512), (1, 128), 0), out=buf26)
        del arg36_1
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf21, arg41_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (1, 512, 28, 28), (401408, 1, 14336, 512))
        del arg41_1
        del buf21
        buf28 = reinterpret_tensor(buf26, (1, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [out_29, input_2, out_30, out_31], Original ATen: [aten.cudnn_batch_norm, aten.convolution, aten.add, aten.relu]
        triton_poi_fused_add_convolution_cudnn_batch_norm_relu_8.run(buf28, arg37_1, arg38_1, arg39_1, arg40_1, buf27, arg42_1, 401408, grid=grid(401408), stream=stream0)
        del arg37_1
        del arg38_1
        del arg39_1
        del arg40_1
        del arg42_1
        buf29 = reinterpret_tensor(buf25, (784, 128), (128, 1), 0); del buf25  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg44_1, (784, 128), (0, 1), 0), reinterpret_tensor(buf28, (784, 512), (512, 1), 0), reinterpret_tensor(arg43_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf29)
        del arg43_1
        del arg44_1
        buf30 = reinterpret_tensor(buf29, (1, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [out_33], Original ATen: [aten.relu]
        triton_poi_fused_relu_9.run(buf30, 100352, grid=grid(100352), stream=stream0)
        # Topologically Sorted Source Nodes: [out_33, out_34], Original ATen: [aten.relu, aten.convolution]
        buf31 = extern_kernels.convolution(buf30, arg45_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (1, 128, 28, 28), (100352, 1, 3584, 128))
        del arg45_1
        del buf30
        buf32 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [out_33, out_34, out_35, out_36], Original ATen: [aten.relu, aten.convolution]
        triton_poi_fused_convolution_relu_7.run(buf32, arg46_1, 100352, grid=grid(100352), stream=stream0)
        del arg46_1
        buf33 = reinterpret_tensor(buf27, (784, 512), (512, 1), 0); del buf27  # reuse
        # Topologically Sorted Source Nodes: [out_33, out_34, out_35, out_36], Original ATen: [aten.relu, aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf32, (784, 128), (128, 1), 0), reinterpret_tensor(arg47_1, (128, 512), (1, 128), 0), out=buf33)
        del arg47_1
        buf34 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [out_37, out_38, out_39], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu]
        triton_poi_fused_add_cudnn_batch_norm_relu_10.run(buf34, buf33, arg48_1, arg49_1, arg50_1, arg51_1, 401408, grid=grid(401408), stream=stream0)
        del arg48_1
        del arg49_1
        del arg50_1
        del arg51_1
        buf35 = reinterpret_tensor(buf32, (784, 128), (128, 1), 0); del buf32  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg53_1, (784, 128), (0, 1), 0), reinterpret_tensor(buf34, (784, 512), (512, 1), 0), reinterpret_tensor(arg52_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf35)
        del arg52_1
        del arg53_1
        buf36 = reinterpret_tensor(buf35, (1, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf35  # reuse
        # Topologically Sorted Source Nodes: [out_41], Original ATen: [aten.relu]
        triton_poi_fused_relu_9.run(buf36, 100352, grid=grid(100352), stream=stream0)
        # Topologically Sorted Source Nodes: [out_41, out_42], Original ATen: [aten.relu, aten.convolution]
        buf37 = extern_kernels.convolution(buf36, arg54_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (1, 128, 28, 28), (100352, 1, 3584, 128))
        del arg54_1
        del buf36
        buf38 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [out_41, out_42, out_43, out_44], Original ATen: [aten.relu, aten.convolution]
        triton_poi_fused_convolution_relu_7.run(buf38, arg55_1, 100352, grid=grid(100352), stream=stream0)
        del arg55_1
        buf39 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [out_41, out_42, out_43, out_44], Original ATen: [aten.relu, aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf38, (784, 128), (128, 1), 0), reinterpret_tensor(arg56_1, (128, 512), (1, 128), 0), out=buf39)
        del arg56_1
        buf40 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [out_45, out_46, out_47], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu]
        triton_poi_fused_add_cudnn_batch_norm_relu_10.run(buf40, buf39, arg57_1, arg58_1, arg59_1, arg60_1, 401408, grid=grid(401408), stream=stream0)
        del arg57_1
        del arg58_1
        del arg59_1
        del arg60_1
        buf41 = reinterpret_tensor(buf38, (784, 128), (128, 1), 0); del buf38  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg62_1, (784, 128), (0, 1), 0), reinterpret_tensor(buf40, (784, 512), (512, 1), 0), reinterpret_tensor(arg61_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf41)
        del arg61_1
        del arg62_1
        buf42 = reinterpret_tensor(buf41, (1, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [out_49], Original ATen: [aten.relu]
        triton_poi_fused_relu_9.run(buf42, 100352, grid=grid(100352), stream=stream0)
        # Topologically Sorted Source Nodes: [out_49, out_50], Original ATen: [aten.relu, aten.convolution]
        buf43 = extern_kernels.convolution(buf42, arg63_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (1, 128, 28, 28), (100352, 1, 3584, 128))
        del arg63_1
        del buf42
        buf44 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [out_49, out_50, out_51, out_52], Original ATen: [aten.relu, aten.convolution]
        triton_poi_fused_convolution_relu_7.run(buf44, arg64_1, 100352, grid=grid(100352), stream=stream0)
        del arg64_1
        buf45 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [out_49, out_50, out_51, out_52], Original ATen: [aten.relu, aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf44, (784, 128), (128, 1), 0), reinterpret_tensor(arg65_1, (128, 512), (1, 128), 0), out=buf45)
        del arg65_1
        buf46 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [out_53, out_54, out_55], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu]
        triton_poi_fused_add_cudnn_batch_norm_relu_10.run(buf46, buf45, arg66_1, arg67_1, arg68_1, arg69_1, 401408, grid=grid(401408), stream=stream0)
        del arg66_1
        del arg67_1
        del arg68_1
        del arg69_1
        del buf45
        buf47 = reinterpret_tensor(buf19, (784, 256), (256, 1), 0); del buf19  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg71_1, (784, 256), (0, 1), 0), reinterpret_tensor(buf46, (784, 512), (512, 1), 0), reinterpret_tensor(arg70_1, (512, 256), (1, 512), 0), alpha=1, beta=1, out=buf47)
        del arg70_1
        del arg71_1
        buf48 = reinterpret_tensor(buf47, (1, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf47  # reuse
        # Topologically Sorted Source Nodes: [out_57], Original ATen: [aten.relu]
        triton_poi_fused_relu_2.run(buf48, 200704, grid=grid(200704), stream=stream0)
        # Topologically Sorted Source Nodes: [out_57, out_58], Original ATen: [aten.relu, aten.convolution]
        buf49 = extern_kernels.convolution(buf48, arg72_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (1, 256, 14, 14), (50176, 1, 3584, 256))
        del arg72_1
        buf50 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [out_57, out_58, out_59, out_60], Original ATen: [aten.relu, aten.convolution]
        triton_poi_fused_convolution_relu_11.run(buf50, arg73_1, 50176, grid=grid(50176), stream=stream0)
        del arg73_1
        buf51 = reinterpret_tensor(buf48, (196, 1024), (1024, 1), 0); del buf48  # reuse
        # Topologically Sorted Source Nodes: [out_57, out_58, out_59, out_60], Original ATen: [aten.relu, aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf50, (196, 256), (256, 1), 0), reinterpret_tensor(arg74_1, (256, 1024), (1, 256), 0), out=buf51)
        del arg74_1
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf46, arg79_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (1, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg79_1
        del buf46
        buf53 = reinterpret_tensor(buf51, (1, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [out_61, input_3, out_62, out_63], Original ATen: [aten.cudnn_batch_norm, aten.convolution, aten.add, aten.relu]
        triton_poi_fused_add_convolution_cudnn_batch_norm_relu_12.run(buf53, arg75_1, arg76_1, arg77_1, arg78_1, buf52, arg80_1, 200704, grid=grid(200704), stream=stream0)
        del arg75_1
        del arg76_1
        del arg77_1
        del arg78_1
        del arg80_1
        buf54 = reinterpret_tensor(buf50, (196, 256), (256, 1), 0); del buf50  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.addmm(reinterpret_tensor(arg82_1, (196, 256), (0, 1), 0), reinterpret_tensor(buf53, (196, 1024), (1024, 1), 0), reinterpret_tensor(arg81_1, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf54)
        del arg81_1
        del arg82_1
        buf55 = reinterpret_tensor(buf54, (1, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [out_65], Original ATen: [aten.relu]
        triton_poi_fused_relu_13.run(buf55, 50176, grid=grid(50176), stream=stream0)
        # Topologically Sorted Source Nodes: [out_65, out_66], Original ATen: [aten.relu, aten.convolution]
        buf56 = extern_kernels.convolution(buf55, arg83_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (1, 256, 14, 14), (50176, 1, 3584, 256))
        del arg83_1
        del buf55
        buf57 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [out_65, out_66, out_67, out_68], Original ATen: [aten.relu, aten.convolution]
        triton_poi_fused_convolution_relu_11.run(buf57, arg84_1, 50176, grid=grid(50176), stream=stream0)
        del arg84_1
        buf58 = reinterpret_tensor(buf52, (196, 1024), (1024, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [out_65, out_66, out_67, out_68], Original ATen: [aten.relu, aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf57, (196, 256), (256, 1), 0), reinterpret_tensor(arg85_1, (256, 1024), (1, 256), 0), out=buf58)
        del arg85_1
        buf59 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [out_69, out_70, out_71], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu]
        triton_poi_fused_add_cudnn_batch_norm_relu_14.run(buf59, buf58, arg86_1, arg87_1, arg88_1, arg89_1, 200704, grid=grid(200704), stream=stream0)
        del arg86_1
        del arg87_1
        del arg88_1
        del arg89_1
        buf60 = reinterpret_tensor(buf57, (196, 256), (256, 1), 0); del buf57  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.addmm(reinterpret_tensor(arg91_1, (196, 256), (0, 1), 0), reinterpret_tensor(buf59, (196, 1024), (1024, 1), 0), reinterpret_tensor(arg90_1, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf60)
        del arg90_1
        del arg91_1
        buf61 = reinterpret_tensor(buf60, (1, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [out_73], Original ATen: [aten.relu]
        triton_poi_fused_relu_13.run(buf61, 50176, grid=grid(50176), stream=stream0)
        # Topologically Sorted Source Nodes: [out_73, out_74], Original ATen: [aten.relu, aten.convolution]
        buf62 = extern_kernels.convolution(buf61, arg92_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (1, 256, 14, 14), (50176, 1, 3584, 256))
        del arg92_1
        del buf61
        buf63 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [out_73, out_74, out_75, out_76], Original ATen: [aten.relu, aten.convolution]
        triton_poi_fused_convolution_relu_11.run(buf63, arg93_1, 50176, grid=grid(50176), stream=stream0)
        del arg93_1
        buf64 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [out_73, out_74, out_75, out_76], Original ATen: [aten.relu, aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf63, (196, 256), (256, 1), 0), reinterpret_tensor(arg94_1, (256, 1024), (1, 256), 0), out=buf64)
        del arg94_1
        buf65 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [out_77, out_78, out_79], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu]
        triton_poi_fused_add_cudnn_batch_norm_relu_14.run(buf65, buf64, arg95_1, arg96_1, arg97_1, arg98_1, 200704, grid=grid(200704), stream=stream0)
        del arg95_1
        del arg96_1
        del arg97_1
        del arg98_1
        buf66 = reinterpret_tensor(buf63, (196, 256), (256, 1), 0); del buf63  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.addmm(reinterpret_tensor(arg100_1, (196, 256), (0, 1), 0), reinterpret_tensor(buf65, (196, 1024), (1024, 1), 0), reinterpret_tensor(arg99_1, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf66)
        del arg100_1
        del arg99_1
        buf67 = reinterpret_tensor(buf66, (1, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [out_81], Original ATen: [aten.relu]
        triton_poi_fused_relu_13.run(buf67, 50176, grid=grid(50176), stream=stream0)
        # Topologically Sorted Source Nodes: [out_81, out_82], Original ATen: [aten.relu, aten.convolution]
        buf68 = extern_kernels.convolution(buf67, arg101_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (1, 256, 14, 14), (50176, 1, 3584, 256))
        del arg101_1
        del buf67
        buf69 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [out_81, out_82, out_83, out_84], Original ATen: [aten.relu, aten.convolution]
        triton_poi_fused_convolution_relu_11.run(buf69, arg102_1, 50176, grid=grid(50176), stream=stream0)
        del arg102_1
        buf70 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [out_81, out_82, out_83, out_84], Original ATen: [aten.relu, aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf69, (196, 256), (256, 1), 0), reinterpret_tensor(arg103_1, (256, 1024), (1, 256), 0), out=buf70)
        del arg103_1
        buf71 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [out_85, out_86, out_87], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu]
        triton_poi_fused_add_cudnn_batch_norm_relu_14.run(buf71, buf70, arg104_1, arg105_1, arg106_1, arg107_1, 200704, grid=grid(200704), stream=stream0)
        del arg104_1
        del arg105_1
        del arg106_1
        del arg107_1
        buf72 = reinterpret_tensor(buf69, (196, 256), (256, 1), 0); del buf69  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.addmm(reinterpret_tensor(arg109_1, (196, 256), (0, 1), 0), reinterpret_tensor(buf71, (196, 1024), (1024, 1), 0), reinterpret_tensor(arg108_1, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf72)
        del arg108_1
        del arg109_1
        buf73 = reinterpret_tensor(buf72, (1, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf72  # reuse
        # Topologically Sorted Source Nodes: [out_89], Original ATen: [aten.relu]
        triton_poi_fused_relu_13.run(buf73, 50176, grid=grid(50176), stream=stream0)
        # Topologically Sorted Source Nodes: [out_89, out_90], Original ATen: [aten.relu, aten.convolution]
        buf74 = extern_kernels.convolution(buf73, arg110_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (1, 256, 14, 14), (50176, 1, 3584, 256))
        del arg110_1
        del buf73
        buf75 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [out_89, out_90, out_91, out_92], Original ATen: [aten.relu, aten.convolution]
        triton_poi_fused_convolution_relu_11.run(buf75, arg111_1, 50176, grid=grid(50176), stream=stream0)
        del arg111_1
        buf76 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [out_89, out_90, out_91, out_92], Original ATen: [aten.relu, aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf75, (196, 256), (256, 1), 0), reinterpret_tensor(arg112_1, (256, 1024), (1, 256), 0), out=buf76)
        del arg112_1
        buf77 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [out_93, out_94, out_95], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu]
        triton_poi_fused_add_cudnn_batch_norm_relu_14.run(buf77, buf76, arg113_1, arg114_1, arg115_1, arg116_1, 200704, grid=grid(200704), stream=stream0)
        del arg113_1
        del arg114_1
        del arg115_1
        del arg116_1
        buf78 = reinterpret_tensor(buf75, (196, 256), (256, 1), 0); del buf75  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.addmm(reinterpret_tensor(arg118_1, (196, 256), (0, 1), 0), reinterpret_tensor(buf77, (196, 1024), (1024, 1), 0), reinterpret_tensor(arg117_1, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf78)
        del arg117_1
        del arg118_1
        buf79 = reinterpret_tensor(buf78, (1, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [out_97], Original ATen: [aten.relu]
        triton_poi_fused_relu_13.run(buf79, 50176, grid=grid(50176), stream=stream0)
        # Topologically Sorted Source Nodes: [out_97, out_98], Original ATen: [aten.relu, aten.convolution]
        buf80 = extern_kernels.convolution(buf79, arg119_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (1, 256, 14, 14), (50176, 1, 3584, 256))
        del arg119_1
        del buf79
        buf81 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [out_97, out_98, out_99, out_100], Original ATen: [aten.relu, aten.convolution]
        triton_poi_fused_convolution_relu_11.run(buf81, arg120_1, 50176, grid=grid(50176), stream=stream0)
        del arg120_1
        buf82 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [out_97, out_98, out_99, out_100], Original ATen: [aten.relu, aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf81, (196, 256), (256, 1), 0), reinterpret_tensor(arg121_1, (256, 1024), (1, 256), 0), out=buf82)
        del arg121_1
        del buf81
        buf83 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [out_101, out_102, out_103], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu]
        triton_poi_fused_add_cudnn_batch_norm_relu_14.run(buf83, buf82, arg122_1, arg123_1, arg124_1, arg125_1, 200704, grid=grid(200704), stream=stream0)
        del arg122_1
        del arg123_1
        del arg124_1
        del arg125_1
        del buf82
        buf84 = reinterpret_tensor(buf44, (196, 512), (512, 1), 0); del buf44  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg127_1, (196, 512), (0, 1), 0), reinterpret_tensor(buf83, (196, 1024), (1024, 1), 0), reinterpret_tensor(arg126_1, (1024, 512), (1, 1024), 0), alpha=1, beta=1, out=buf84)
        del arg126_1
        del arg127_1
        buf85 = reinterpret_tensor(buf84, (1, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf84  # reuse
        # Topologically Sorted Source Nodes: [out_105], Original ATen: [aten.relu]
        triton_poi_fused_relu_9.run(buf85, 100352, grid=grid(100352), stream=stream0)
        # Topologically Sorted Source Nodes: [out_105, out_106], Original ATen: [aten.relu, aten.convolution]
        buf86 = extern_kernels.convolution(buf85, arg128_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (1, 512, 7, 7), (25088, 1, 3584, 512))
        del arg128_1
        buf87 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [out_105, out_106, out_107, out_108], Original ATen: [aten.relu, aten.convolution]
        triton_poi_fused_convolution_relu_15.run(buf87, arg129_1, 25088, grid=grid(25088), stream=stream0)
        del arg129_1
        buf88 = reinterpret_tensor(buf85, (49, 2048), (2048, 1), 0); del buf85  # reuse
        # Topologically Sorted Source Nodes: [out_105, out_106, out_107, out_108], Original ATen: [aten.relu, aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf87, (49, 512), (512, 1), 0), reinterpret_tensor(arg130_1, (512, 2048), (1, 512), 0), out=buf88)
        del arg130_1
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf83, arg135_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (1, 2048, 7, 7), (100352, 1, 14336, 2048))
        del arg135_1
        del buf83
        buf90 = reinterpret_tensor(buf88, (1, 2048, 7, 7), (100352, 1, 14336, 2048), 0); del buf88  # reuse
        # Topologically Sorted Source Nodes: [out_109, input_4, out_110, out_111], Original ATen: [aten.cudnn_batch_norm, aten.convolution, aten.add, aten.relu]
        triton_poi_fused_add_convolution_cudnn_batch_norm_relu_16.run(buf90, arg131_1, arg132_1, arg133_1, arg134_1, buf89, arg136_1, 100352, grid=grid(100352), stream=stream0)
        del arg131_1
        del arg132_1
        del arg133_1
        del arg134_1
        del arg136_1
        buf91 = reinterpret_tensor(buf87, (49, 512), (512, 1), 0); del buf87  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg138_1, (49, 512), (0, 1), 0), reinterpret_tensor(buf90, (49, 2048), (2048, 1), 0), reinterpret_tensor(arg137_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf91)
        del arg137_1
        del arg138_1
        buf92 = reinterpret_tensor(buf91, (1, 512, 7, 7), (25088, 1, 3584, 512), 0); del buf91  # reuse
        # Topologically Sorted Source Nodes: [out_113], Original ATen: [aten.relu]
        triton_poi_fused_relu_17.run(buf92, 25088, grid=grid(25088), stream=stream0)
        # Topologically Sorted Source Nodes: [out_113, out_114], Original ATen: [aten.relu, aten.convolution]
        buf93 = extern_kernels.convolution(buf92, arg139_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (1, 512, 7, 7), (25088, 1, 3584, 512))
        del arg139_1
        del buf92
        buf94 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [out_113, out_114, out_115, out_116], Original ATen: [aten.relu, aten.convolution]
        triton_poi_fused_convolution_relu_15.run(buf94, arg140_1, 25088, grid=grid(25088), stream=stream0)
        del arg140_1
        buf95 = reinterpret_tensor(buf89, (49, 2048), (2048, 1), 0); del buf89  # reuse
        # Topologically Sorted Source Nodes: [out_113, out_114, out_115, out_116], Original ATen: [aten.relu, aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf94, (49, 512), (512, 1), 0), reinterpret_tensor(arg141_1, (512, 2048), (1, 512), 0), out=buf95)
        del arg141_1
        buf96 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [out_117, out_118, out_119], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu]
        triton_poi_fused_add_cudnn_batch_norm_relu_18.run(buf96, buf95, arg142_1, arg143_1, arg144_1, arg145_1, 100352, grid=grid(100352), stream=stream0)
        del arg142_1
        del arg143_1
        del arg144_1
        del arg145_1
        buf97 = reinterpret_tensor(buf94, (49, 512), (512, 1), 0); del buf94  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg147_1, (49, 512), (0, 1), 0), reinterpret_tensor(buf96, (49, 2048), (2048, 1), 0), reinterpret_tensor(arg146_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf97)
        del arg146_1
        del arg147_1
        buf98 = reinterpret_tensor(buf97, (1, 512, 7, 7), (25088, 1, 3584, 512), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [out_121], Original ATen: [aten.relu]
        triton_poi_fused_relu_17.run(buf98, 25088, grid=grid(25088), stream=stream0)
        # Topologically Sorted Source Nodes: [out_121, out_122], Original ATen: [aten.relu, aten.convolution]
        buf99 = extern_kernels.convolution(buf98, arg148_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf99, (1, 512, 7, 7), (25088, 1, 3584, 512))
        del arg148_1
        del buf98
        buf100 = buf99; del buf99  # reuse
        # Topologically Sorted Source Nodes: [out_121, out_122, out_123, out_124], Original ATen: [aten.relu, aten.convolution]
        triton_poi_fused_convolution_relu_15.run(buf100, arg149_1, 25088, grid=grid(25088), stream=stream0)
        del arg149_1
        buf101 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [out_121, out_122, out_123, out_124], Original ATen: [aten.relu, aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf100, (49, 512), (512, 1), 0), reinterpret_tensor(arg150_1, (512, 2048), (1, 512), 0), out=buf101)
        del arg150_1
        del buf100
        buf102 = empty_strided_cuda((1, 2048, 1, 1), (2048, 1, 2048, 2048), torch.float32)
        buf103 = reinterpret_tensor(buf102, (1, 2048, 1, 1), (2048, 1, 1, 1), 0); del buf102  # reuse
        # Topologically Sorted Source Nodes: [out_125, out_126, out_127, x_3], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu, aten.mean]
        triton_per_fused_add_cudnn_batch_norm_mean_relu_19.run(buf103, buf101, arg151_1, arg152_1, arg153_1, arg154_1, buf96, 2048, 49, grid=grid(2048), stream=stream0)
        del arg151_1
        del arg152_1
        del arg153_1
        del arg154_1
        del buf101
        del buf96
        buf104 = empty_strided_cuda((1, 1000), (1000, 1), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg156_1, (1, 1000), (0, 1), 0), reinterpret_tensor(buf103, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg155_1, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf104)
        del arg155_1
        del arg156_1
        del buf103
    return (buf104, )


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
    arg9_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((64, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((64, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((128, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((512, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((128, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((128, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((128, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((256, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((1024, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((2048, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((2048, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((2048, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)


# ===== inductor generated file: /tmp/torchinductor_kamei/tmpb3zpfqrg/p2/cp2iavfn4o3rdgacvgplshni2doewjblt5yxfqhi6g2ttyaq6fcn.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'kernel_num_gb': 0.0},
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
    return arg_0,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_.run(*args, 200704, grid=grid(200704), stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_.benchmark_all_configs(*args, 200704, grid=grid(200704))


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=40, fast_flush=True)
    num_gb = 0.0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")


# ===== inductor generated file: /tmp/torchinductor_kamei/tmpb3zpfqrg/25/c2565gcz6gg3zyvudxe4p3mrlzf424rlgczglgrt47lsrpw5nrf7.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cudnn_batch_norm_relu_5', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, None)


# ===== inductor generated file: /tmp/torchinductor_kamei/tmpb3zpfqrg/2l/c2lml6wpxrcqw4lq6fjpth2cfz5xf2evatr7onigd2xde2is7fkk.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cudnn_batch_norm_relu_4', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, None)


# ===== inductor generated file: /tmp/torchinductor_kamei/tmpb3zpfqrg/6t/c6t7n4xry6q2zxdoq4hniwurqcffvby2a7vzzi5gqwvqm5c64liz.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cudnn_batch_norm_relu_10', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, None)


# ===== inductor generated file: /tmp/torchinductor_kamei/tmpb3zpfqrg/at/cat77m4xzjz7njffubxv2nvshduhw5o3vytdc4jla3czc2koyli3.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_13', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


# ===== inductor generated file: /tmp/torchinductor_kamei/tmpb3zpfqrg/dw/cdwnvuz5vuvkee5j4fwwz3hu37y3qpyicwxrzcqkwt434jatmspd.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_9', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, None)


# ===== inductor generated file: /tmp/torchinductor_kamei/tmpb3zpfqrg/e3/ce3bnuz4kdqfskmocijr3j2ntxskgmoympjjeuc5uck23r3qg2q6.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_7', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)


# ===== inductor generated file: /tmp/torchinductor_kamei/tmpb3zpfqrg/e5/ce5rpohrwatzefhv4ufejoz5ukoh4gmceub452hrywf46anq4n4s.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 3584)
    x1 = (xindex // 64) % 56
    x0 = xindex % 64
    x4 = xindex
    tmp0 = (-1) + (2*x2)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x1)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-7232) + x0 + (128*x1) + (14336*x2)), tmp10, other=float("-inf"))
    tmp12 = 2*x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-7168) + x0 + (128*x1) + (14336*x2)), tmp16, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + (2*x1)
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-7104) + x0 + (128*x1) + (14336*x2)), tmp23, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 2*x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-64) + x0 + (128*x1) + (14336*x2)), tmp30, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x0 + (128*x1) + (14336*x2)), tmp33, other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (64 + x0 + (128*x1) + (14336*x2)), tmp36, other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = 1 + (2*x2)
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (7104 + x0 + (128*x1) + (14336*x2)), tmp43, other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp38)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (7168 + x0 + (128*x1) + (14336*x2)), tmp46, other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (7232 + x0 + (128*x1) + (14336*x2)), tmp49, other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp50, tmp48)
    tl.store(out_ptr0 + (x4), tmp51, None)


# ===== inductor generated file: /tmp/torchinductor_kamei/tmpb3zpfqrg/h3/ch3akm6ueyhturnekgx5jj4s2ngeuopdv5cbqrgzs7qq5w6rnpkn.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_11', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)


# ===== inductor generated file: /tmp/torchinductor_kamei/tmpb3zpfqrg/if/ciffvox6i3orrhkfuosohkobkfltpvnbzv4r4bjdd4sfbh32kuwy.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_3', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)


# ===== inductor generated file: /tmp/torchinductor_kamei/tmpb3zpfqrg/lh/clh32qktusc3xqs3tfxnxdo65xnwwwpr3juep4f4mboepnl2gpr4.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_0', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)


# ===== inductor generated file: /tmp/torchinductor_kamei/tmpb3zpfqrg/lh/clhz5exshgpbl33dzuudnv4lvsndrivw3gnetwoph7jxjcg6tvpl.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_cudnn_batch_norm_relu_16', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 2048
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), None)
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp18 = tmp16 + tmp17
    tmp19 = tmp15 + tmp18
    tmp20 = tl.full([1], 0, tl.int32)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp21, None)


# ===== inductor generated file: /tmp/torchinductor_kamei/tmpb3zpfqrg/md/cmdnkzbdprtaryj2mccow56arwl75fhexmly5pv26eif3lmrcp5n.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_15', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)


# ===== inductor generated file: /tmp/torchinductor_kamei/tmpb3zpfqrg/nv/cnvt5e6ma7disnpv4heudu224wnia2uxhr7kk3yeghqjhtigr4mt.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cudnn_batch_norm_relu_14', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, None)


# ===== inductor generated file: /tmp/torchinductor_kamei/tmpb3zpfqrg/ob/cobjt6a5qna5znq6sgxi5cu2ri44on5jbk6omajfjxex5hljlktw.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cudnn_batch_norm_mean_relu_19', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (2048*r1)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0 + (2048*r1)), rmask, other=0.0)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1, 1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp22 = tl.where(rmask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None]
    tmp24 = 49.0
    tmp25 = tmp23 / tmp24
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp25, None)


# ===== inductor generated file: /tmp/torchinductor_kamei/tmpb3zpfqrg/oc/cochuo4yfxnqhqfzcnbi7ckjubo7pphqfpacmn4bynqyuucrvxmi.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_cudnn_batch_norm_relu_8', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), None)
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp18 = tmp16 + tmp17
    tmp19 = tmp15 + tmp18
    tmp20 = tl.full([1], 0, tl.int32)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp21, None)


# ===== inductor generated file: /tmp/torchinductor_kamei/tmpb3zpfqrg/oq/coqt5cajb762s5omwvztapnahmioa4pd6fe5oj6qgkymgxfnb7og.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_cudnn_batch_norm_relu_12', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), None)
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp18 = tmp16 + tmp17
    tmp19 = tmp15 + tmp18
    tmp20 = tl.full([1], 0, tl.int32)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp21, None)


# ===== inductor generated file: /tmp/torchinductor_kamei/tmpb3zpfqrg/pm/cpmgdrjk5xvet4vvgopklynqv3y6ditbbwm2sb2fdhpcoequm6nn.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


# ===== inductor generated file: /tmp/torchinductor_kamei/tmpb3zpfqrg/xa/cxafh6hqtknez4tydrkdfs5l6fqa6hfphkv2zo6szm5oxomte6r3.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_6', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


# ===== inductor generated file: /tmp/torchinductor_kamei/tmpb3zpfqrg/yn/cyno2fmy3faurpdqs7m3swsliirivqiyvzxz24uuazvx3kr4clc7.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cudnn_batch_norm_relu_18', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 2048
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, None)


# ===== inductor generated file: /tmp/torchinductor_kamei/tmpb3zpfqrg/zy/czyl5lufhqeib6stzy4lac5xwoq5whap4ilhyshwzgldkoox572o.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_17', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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
