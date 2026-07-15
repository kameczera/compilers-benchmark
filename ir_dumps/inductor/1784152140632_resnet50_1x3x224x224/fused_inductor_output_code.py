# ===== inductor generated file: /tmp/torchinductor_kamei/tmp7zaxe2pd/w2/cw2ypq3dp32gw4zx2wp7kqf6rbsaxga2qmgc37hbqwwupdbl7uuy.py =====
# AOT ID: ['0_inference']
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


# kernel path: /tmp/torchinductor_kamei/tmp7zaxe2pd/6t/c6tlrbztbpbcjqh5ewgtq3bcgijdjzyu5h2ljhvo4mojsdeiafvf.py
# Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten.cudnn_batch_norm, aten.relu]
# Source node to ATen node mapping:
#   x_1 => add_1, mul_1, mul_2, sub
#   x_2 => relu
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
triton_poi_fused_cudnn_batch_norm_relu_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cudnn_batch_norm_relu_0', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/tmp7zaxe2pd/rv/crv3uburz5u5q564q2ndo5wikyctx2txmqafwntcm5ctigggy6g4.py
# Topologically Sorted Source Nodes: [x_1, x_2, x_3], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_1 => add_1, mul_1, mul_2, sub
#   x_2 => relu
#   x_3 => _low_memory_max_pool2d_with_offsets
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu, [3, 3], [2, 2], [1, 1], [1, 1], False), kwargs = {})
triton_poi_fused_cudnn_batch_norm_max_pool2d_with_indices_relu_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cudnn_batch_norm_max_pool2d_with_indices_relu_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


# kernel path: /tmp/torchinductor_kamei/tmp7zaxe2pd/tm/ctmkd7g5pune4vpksti47iyhkzczrkbrk5736djap5df447jy5dd.py
# Topologically Sorted Source Nodes: [out_1, out_2], Original ATen: [aten.cudnn_batch_norm, aten.relu]
# Source node to ATen node mapping:
#   out_1 => add_3, mul_4, mul_5, sub_1
#   out_2 => relu_1
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
triton_poi_fused_cudnn_batch_norm_relu_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cudnn_batch_norm_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/tmp7zaxe2pd/x2/cx2jrpae4xobnunfmgh2gouqssunhj3hbdzzliemk455ftveg7f5.py
# Topologically Sorted Source Nodes: [out_7, input_2, out_8, out_9, out_10], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   input_2 => add_9, mul_13, mul_14, sub_4
#   out_10 => convolution_5
#   out_7 => add_7, mul_10, mul_11, sub_3
#   out_8 => add_10
#   out_9 => relu_3
# Graph fragment:
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_37), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_39), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %add_9), kwargs = {})
#   %relu_3 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_10,), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_3, %arg26_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_add_convolution_cudnn_batch_norm_relu_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_cudnn_batch_norm_relu_3', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
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
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x2), tmp29, None)
    tl.store(out_ptr0 + (x2), tmp31, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/tmp7zaxe2pd/a4/ca4llt5wfkqdyjisdxvmmny2pbxwylwye7weeayrvawz6r3ad275.py
# Topologically Sorted Source Nodes: [out_9, out_17, out_18, out_19], Original ATen: [aten.relu, aten.cudnn_batch_norm, aten.add]
# Source node to ATen node mapping:
#   out_17 => add_16, mul_22, mul_23, sub_7
#   out_18 => add_17
#   out_19 => relu_6
#   out_9 => relu_3
# Graph fragment:
#   %relu_3 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_10,), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %unsqueeze_57), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_59), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %unsqueeze_61), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %unsqueeze_63), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_16, %relu_3), kwargs = {})
#   %relu_6 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_17,), kwargs = {})
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
    tmp17 = tl.full([1], 0, tl.int32)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tmp19 = tmp15 + tmp18
    tmp20 = triton_helpers.maximum(tmp17, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/tmp7zaxe2pd/25/c2565gcz6gg3zyvudxe4p3mrlzf424rlgczglgrt47lsrpw5nrf7.py
# Topologically Sorted Source Nodes: [out_27, out_28, out_29], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_27 => add_23, mul_31, mul_32, sub_10
#   out_28 => add_24
#   out_29 => relu_9
# Graph fragment:
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_10, %unsqueeze_81), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %unsqueeze_83), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_31, %unsqueeze_85), kwargs = {})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_32, %unsqueeze_87), kwargs = {})
#   %add_24 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_23, %relu_6), kwargs = {})
#   %relu_9 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_24,), kwargs = {})
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


# kernel path: /tmp/torchinductor_kamei/tmp7zaxe2pd/ss/cssaqwcpd72z3jcnwd3t7s5yns5mpsupuqusicbrqbms3sssl232.py
# Topologically Sorted Source Nodes: [out_31, out_32], Original ATen: [aten.cudnn_batch_norm, aten.relu]
# Source node to ATen node mapping:
#   out_31 => add_26, mul_34, mul_35, sub_11
#   out_32 => relu_10
# Graph fragment:
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_11, %unsqueeze_89), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_91), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %unsqueeze_93), kwargs = {})
#   %add_26 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %unsqueeze_95), kwargs = {})
#   %relu_10 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_26,), kwargs = {})
triton_poi_fused_cudnn_batch_norm_relu_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cudnn_batch_norm_relu_6', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/tmp7zaxe2pd/m2/cm2rl7sliyqjx5o7fllsok6r6nc2o7isjxfa23qh2fdjuqsvychl.py
# Topologically Sorted Source Nodes: [out_34, out_35, out_36], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   out_34 => add_28, mul_37, mul_38, sub_12
#   out_35 => relu_11
#   out_36 => convolution_13
# Graph fragment:
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_12, %unsqueeze_97), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_99), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %unsqueeze_101), kwargs = {})
#   %add_28 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, %unsqueeze_103), kwargs = {})
#   %relu_11 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_28,), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_11, %arg66_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_cudnn_batch_norm_relu_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_cudnn_batch_norm_relu_7', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/tmp7zaxe2pd/36/c36xq3mkl263lwbeqe655hilcx6snpob4sfq5lv5ii6wpjk742u5.py
# Topologically Sorted Source Nodes: [out_37, input_4, out_38, out_39, out_40], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   input_4 => add_32, mul_43, mul_44, sub_14
#   out_37 => add_30, mul_40, mul_41, sub_13
#   out_38 => add_33
#   out_39 => relu_12
#   out_40 => convolution_15
# Graph fragment:
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_13, %unsqueeze_105), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_107), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_40, %unsqueeze_109), kwargs = {})
#   %add_30 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_41, %unsqueeze_111), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_14, %unsqueeze_113), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %unsqueeze_115), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, %unsqueeze_117), kwargs = {})
#   %add_32 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_44, %unsqueeze_119), kwargs = {})
#   %add_33 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_30, %add_32), kwargs = {})
#   %relu_12 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_33,), kwargs = {})
#   %convolution_15 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_12, %arg76_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_cudnn_batch_norm_relu_8', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
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
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x2), tmp29, None)
    tl.store(out_ptr0 + (x2), tmp31, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/tmp7zaxe2pd/4v/c4vrmqvevn6bagqj5teiftukd72pq4zdo22ykaar4mu4jsvew7ds.py
# Topologically Sorted Source Nodes: [out_39, out_47, out_48, out_49], Original ATen: [aten.relu, aten.cudnn_batch_norm, aten.add]
# Source node to ATen node mapping:
#   out_39 => relu_12
#   out_47 => add_39, mul_52, mul_53, sub_17
#   out_48 => add_40
#   out_49 => relu_15
# Graph fragment:
#   %relu_12 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_33,), kwargs = {})
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_17, %unsqueeze_137), kwargs = {})
#   %mul_52 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %unsqueeze_139), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_52, %unsqueeze_141), kwargs = {})
#   %add_39 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_53, %unsqueeze_143), kwargs = {})
#   %add_40 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_39, %relu_12), kwargs = {})
#   %relu_15 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_40,), kwargs = {})
triton_poi_fused_add_cudnn_batch_norm_relu_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cudnn_batch_norm_relu_9', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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
    tmp17 = tl.full([1], 0, tl.int32)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tmp19 = tmp15 + tmp18
    tmp20 = triton_helpers.maximum(tmp17, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/tmp7zaxe2pd/6t/c6t7n4xry6q2zxdoq4hniwurqcffvby2a7vzzi5gqwvqm5c64liz.py
# Topologically Sorted Source Nodes: [out_57, out_58, out_59], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_57 => add_46, mul_61, mul_62, sub_20
#   out_58 => add_47
#   out_59 => relu_18
# Graph fragment:
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_20, %unsqueeze_161), kwargs = {})
#   %mul_61 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %unsqueeze_163), kwargs = {})
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_61, %unsqueeze_165), kwargs = {})
#   %add_46 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_62, %unsqueeze_167), kwargs = {})
#   %add_47 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_46, %relu_15), kwargs = {})
#   %relu_18 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_47,), kwargs = {})
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


# kernel path: /tmp/torchinductor_kamei/tmp7zaxe2pd/zi/czigjrwvlh5bxngc3kdt6dfhofenbk5xoexqmerkloo4aexnhvnp.py
# Topologically Sorted Source Nodes: [out_71, out_72], Original ATen: [aten.cudnn_batch_norm, aten.relu]
# Source node to ATen node mapping:
#   out_71 => add_56, mul_73, mul_74, sub_24
#   out_72 => relu_22
# Graph fragment:
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_24, %unsqueeze_193), kwargs = {})
#   %mul_73 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %unsqueeze_195), kwargs = {})
#   %mul_74 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_73, %unsqueeze_197), kwargs = {})
#   %add_56 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_74, %unsqueeze_199), kwargs = {})
#   %relu_22 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_56,), kwargs = {})
triton_poi_fused_cudnn_batch_norm_relu_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cudnn_batch_norm_relu_11', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/tmp7zaxe2pd/hn/chn57svt7o3ync3nccptfjk4xsno2vn4tm6uakihrkwtxqut6t4l.py
# Topologically Sorted Source Nodes: [out_74, out_75, out_76], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   out_74 => add_58, mul_76, mul_77, sub_25
#   out_75 => relu_23
#   out_76 => convolution_26
# Graph fragment:
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_25, %unsqueeze_201), kwargs = {})
#   %mul_76 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %unsqueeze_203), kwargs = {})
#   %mul_77 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_76, %unsqueeze_205), kwargs = {})
#   %add_58 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_77, %unsqueeze_207), kwargs = {})
#   %relu_23 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_58,), kwargs = {})
#   %convolution_26 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_23, %arg131_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_cudnn_batch_norm_relu_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_cudnn_batch_norm_relu_12', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/tmp7zaxe2pd/zn/cznqosyljq7ipxvwtmsegrbbdaxpkkmbme3g6tmbhwngkyv7lzdw.py
# Topologically Sorted Source Nodes: [out_77, input_6, out_78, out_79, out_80], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   input_6 => add_62, mul_82, mul_83, sub_27
#   out_77 => add_60, mul_79, mul_80, sub_26
#   out_78 => add_63
#   out_79 => relu_24
#   out_80 => convolution_28
# Graph fragment:
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_26, %unsqueeze_209), kwargs = {})
#   %mul_79 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %unsqueeze_211), kwargs = {})
#   %mul_80 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_79, %unsqueeze_213), kwargs = {})
#   %add_60 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_80, %unsqueeze_215), kwargs = {})
#   %sub_27 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_27, %unsqueeze_217), kwargs = {})
#   %mul_82 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %unsqueeze_219), kwargs = {})
#   %mul_83 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_82, %unsqueeze_221), kwargs = {})
#   %add_62 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_83, %unsqueeze_223), kwargs = {})
#   %add_63 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_60, %add_62), kwargs = {})
#   %relu_24 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_63,), kwargs = {})
#   %convolution_28 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_24, %arg141_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_add_convolution_cudnn_batch_norm_relu_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_cudnn_batch_norm_relu_13', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
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
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x2), tmp29, None)
    tl.store(out_ptr0 + (x2), tmp31, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/tmp7zaxe2pd/bo/cbog35jsbzryke5fflddtmivyc3n435in4jonoj6onobre2qncct.py
# Topologically Sorted Source Nodes: [out_79, out_87, out_88, out_89], Original ATen: [aten.relu, aten.cudnn_batch_norm, aten.add]
# Source node to ATen node mapping:
#   out_79 => relu_24
#   out_87 => add_69, mul_91, mul_92, sub_30
#   out_88 => add_70
#   out_89 => relu_27
# Graph fragment:
#   %relu_24 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_63,), kwargs = {})
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_30, %unsqueeze_241), kwargs = {})
#   %mul_91 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_30, %unsqueeze_243), kwargs = {})
#   %mul_92 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_91, %unsqueeze_245), kwargs = {})
#   %add_69 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_92, %unsqueeze_247), kwargs = {})
#   %add_70 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_69, %relu_24), kwargs = {})
#   %relu_27 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_70,), kwargs = {})
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
    tmp17 = tl.full([1], 0, tl.int32)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tmp19 = tmp15 + tmp18
    tmp20 = triton_helpers.maximum(tmp17, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/tmp7zaxe2pd/c3/cc3yeklurc5cpt4hetkja2uc7u5zzye7nbzogfvzyda5mzi2uuvl.py
# Topologically Sorted Source Nodes: [out_97, out_98, out_99], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_97 => add_76, mul_100, mul_101, sub_33
#   out_98 => add_77
#   out_99 => relu_30
# Graph fragment:
#   %sub_33 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_33, %unsqueeze_265), kwargs = {})
#   %mul_100 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_33, %unsqueeze_267), kwargs = {})
#   %mul_101 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_100, %unsqueeze_269), kwargs = {})
#   %add_76 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_101, %unsqueeze_271), kwargs = {})
#   %add_77 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_76, %relu_27), kwargs = {})
#   %relu_30 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_77,), kwargs = {})
triton_poi_fused_add_cudnn_batch_norm_relu_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cudnn_batch_norm_relu_15', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


# kernel path: /tmp/torchinductor_kamei/tmp7zaxe2pd/4s/c4seuq74px4b6onmrfqfealyovhwnusllvikclz53ld56od2oiwi.py
# Topologically Sorted Source Nodes: [out_131, out_132], Original ATen: [aten.cudnn_batch_norm, aten.relu]
# Source node to ATen node mapping:
#   out_131 => add_100, mul_130, mul_131, sub_43
#   out_132 => relu_40
# Graph fragment:
#   %sub_43 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_43, %unsqueeze_345), kwargs = {})
#   %mul_130 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_43, %unsqueeze_347), kwargs = {})
#   %mul_131 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_130, %unsqueeze_349), kwargs = {})
#   %add_100 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_131, %unsqueeze_351), kwargs = {})
#   %relu_40 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_100,), kwargs = {})
triton_poi_fused_cudnn_batch_norm_relu_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cudnn_batch_norm_relu_16', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/tmp7zaxe2pd/jj/cjjb7d6rfh74sekobeww2mwt6qxbbz6vrylwa3i2ubgqgszlvirp.py
# Topologically Sorted Source Nodes: [out_134, out_135, out_136], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   out_134 => add_102, mul_133, mul_134, sub_44
#   out_135 => relu_41
#   out_136 => convolution_45
# Graph fragment:
#   %sub_44 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_44, %unsqueeze_353), kwargs = {})
#   %mul_133 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_44, %unsqueeze_355), kwargs = {})
#   %mul_134 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_133, %unsqueeze_357), kwargs = {})
#   %add_102 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_134, %unsqueeze_359), kwargs = {})
#   %relu_41 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_102,), kwargs = {})
#   %convolution_45 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_41, %arg226_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_cudnn_batch_norm_relu_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_cudnn_batch_norm_relu_17', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/tmp7zaxe2pd/nw/cnwvho6smkwe336lad27gvxyqqkzzfawo2omm2uto22ze7h2xhuz.py
# Topologically Sorted Source Nodes: [out_137, input_8, out_138, out_139, out_140], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   input_8 => add_106, mul_139, mul_140, sub_46
#   out_137 => add_104, mul_136, mul_137, sub_45
#   out_138 => add_107
#   out_139 => relu_42
#   out_140 => convolution_47
# Graph fragment:
#   %sub_45 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_45, %unsqueeze_361), kwargs = {})
#   %mul_136 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_45, %unsqueeze_363), kwargs = {})
#   %mul_137 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_136, %unsqueeze_365), kwargs = {})
#   %add_104 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_137, %unsqueeze_367), kwargs = {})
#   %sub_46 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_46, %unsqueeze_369), kwargs = {})
#   %mul_139 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_46, %unsqueeze_371), kwargs = {})
#   %mul_140 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_139, %unsqueeze_373), kwargs = {})
#   %add_106 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_140, %unsqueeze_375), kwargs = {})
#   %add_107 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_104, %add_106), kwargs = {})
#   %relu_42 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_107,), kwargs = {})
#   %convolution_47 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_42, %arg236_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_add_convolution_cudnn_batch_norm_relu_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_cudnn_batch_norm_relu_18', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
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
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x2), tmp29, None)
    tl.store(out_ptr0 + (x2), tmp31, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/tmp7zaxe2pd/il/cilepl63nzqv5c6sari7t5xtpkhv4gfgh6aju4isfedht2q2lxrm.py
# Topologically Sorted Source Nodes: [out_139, out_147, out_148, out_149], Original ATen: [aten.relu, aten.cudnn_batch_norm, aten.add]
# Source node to ATen node mapping:
#   out_139 => relu_42
#   out_147 => add_113, mul_148, mul_149, sub_49
#   out_148 => add_114
#   out_149 => relu_45
# Graph fragment:
#   %relu_42 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_107,), kwargs = {})
#   %sub_49 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_49, %unsqueeze_393), kwargs = {})
#   %mul_148 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_49, %unsqueeze_395), kwargs = {})
#   %mul_149 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_148, %unsqueeze_397), kwargs = {})
#   %add_113 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_149, %unsqueeze_399), kwargs = {})
#   %add_114 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_113, %relu_42), kwargs = {})
#   %relu_45 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_114,), kwargs = {})
triton_poi_fused_add_cudnn_batch_norm_relu_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cudnn_batch_norm_relu_19', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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
    tmp17 = tl.full([1], 0, tl.int32)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tmp19 = tmp15 + tmp18
    tmp20 = triton_helpers.maximum(tmp17, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_kamei/tmp7zaxe2pd/3p/c3pqzkruzffy3jwvhl7giku5sun7y5rth3tyijp7q7tyoeo4qzc5.py
# Topologically Sorted Source Nodes: [out_157, out_158, out_159, x_4], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu, aten.mean]
# Source node to ATen node mapping:
#   out_157 => add_120, mul_157, mul_158, sub_52
#   out_158 => add_121
#   out_159 => relu_48
#   x_4 => mean
# Graph fragment:
#   %sub_52 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_52, %unsqueeze_417), kwargs = {})
#   %mul_157 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_52, %unsqueeze_419), kwargs = {})
#   %mul_158 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_157, %unsqueeze_421), kwargs = {})
#   %add_120 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_158, %unsqueeze_423), kwargs = {})
#   %add_121 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_120, %relu_45), kwargs = {})
#   %relu_48 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_121,), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_48, [-1, -2], True), kwargs = {})
triton_per_fused_add_cudnn_batch_norm_mean_relu_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cudnn_batch_norm_mean_relu_20', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 3, 7, 7), (147, 1, 21, 3))
    assert_size_stride(arg1_1, (1, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(arg2_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (64, ), (1, ))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (64, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(arg7_1, (64, ), (1, ))
    assert_size_stride(arg8_1, (64, ), (1, ))
    assert_size_stride(arg9_1, (64, ), (1, ))
    assert_size_stride(arg10_1, (64, ), (1, ))
    assert_size_stride(arg11_1, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(arg12_1, (64, ), (1, ))
    assert_size_stride(arg13_1, (64, ), (1, ))
    assert_size_stride(arg14_1, (64, ), (1, ))
    assert_size_stride(arg15_1, (64, ), (1, ))
    assert_size_stride(arg16_1, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(arg17_1, (256, ), (1, ))
    assert_size_stride(arg18_1, (256, ), (1, ))
    assert_size_stride(arg19_1, (256, ), (1, ))
    assert_size_stride(arg20_1, (256, ), (1, ))
    assert_size_stride(arg21_1, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(arg22_1, (256, ), (1, ))
    assert_size_stride(arg23_1, (256, ), (1, ))
    assert_size_stride(arg24_1, (256, ), (1, ))
    assert_size_stride(arg25_1, (256, ), (1, ))
    assert_size_stride(arg26_1, (64, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg27_1, (64, ), (1, ))
    assert_size_stride(arg28_1, (64, ), (1, ))
    assert_size_stride(arg29_1, (64, ), (1, ))
    assert_size_stride(arg30_1, (64, ), (1, ))
    assert_size_stride(arg31_1, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(arg32_1, (64, ), (1, ))
    assert_size_stride(arg33_1, (64, ), (1, ))
    assert_size_stride(arg34_1, (64, ), (1, ))
    assert_size_stride(arg35_1, (64, ), (1, ))
    assert_size_stride(arg36_1, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(arg37_1, (256, ), (1, ))
    assert_size_stride(arg38_1, (256, ), (1, ))
    assert_size_stride(arg39_1, (256, ), (1, ))
    assert_size_stride(arg40_1, (256, ), (1, ))
    assert_size_stride(arg41_1, (64, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg42_1, (64, ), (1, ))
    assert_size_stride(arg43_1, (64, ), (1, ))
    assert_size_stride(arg44_1, (64, ), (1, ))
    assert_size_stride(arg45_1, (64, ), (1, ))
    assert_size_stride(arg46_1, (64, 64, 3, 3), (576, 1, 192, 64))
    assert_size_stride(arg47_1, (64, ), (1, ))
    assert_size_stride(arg48_1, (64, ), (1, ))
    assert_size_stride(arg49_1, (64, ), (1, ))
    assert_size_stride(arg50_1, (64, ), (1, ))
    assert_size_stride(arg51_1, (256, 64, 1, 1), (64, 1, 64, 64))
    assert_size_stride(arg52_1, (256, ), (1, ))
    assert_size_stride(arg53_1, (256, ), (1, ))
    assert_size_stride(arg54_1, (256, ), (1, ))
    assert_size_stride(arg55_1, (256, ), (1, ))
    assert_size_stride(arg56_1, (128, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg57_1, (128, ), (1, ))
    assert_size_stride(arg58_1, (128, ), (1, ))
    assert_size_stride(arg59_1, (128, ), (1, ))
    assert_size_stride(arg60_1, (128, ), (1, ))
    assert_size_stride(arg61_1, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(arg62_1, (128, ), (1, ))
    assert_size_stride(arg63_1, (128, ), (1, ))
    assert_size_stride(arg64_1, (128, ), (1, ))
    assert_size_stride(arg65_1, (128, ), (1, ))
    assert_size_stride(arg66_1, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(arg67_1, (512, ), (1, ))
    assert_size_stride(arg68_1, (512, ), (1, ))
    assert_size_stride(arg69_1, (512, ), (1, ))
    assert_size_stride(arg70_1, (512, ), (1, ))
    assert_size_stride(arg71_1, (512, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg72_1, (512, ), (1, ))
    assert_size_stride(arg73_1, (512, ), (1, ))
    assert_size_stride(arg74_1, (512, ), (1, ))
    assert_size_stride(arg75_1, (512, ), (1, ))
    assert_size_stride(arg76_1, (128, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg77_1, (128, ), (1, ))
    assert_size_stride(arg78_1, (128, ), (1, ))
    assert_size_stride(arg79_1, (128, ), (1, ))
    assert_size_stride(arg80_1, (128, ), (1, ))
    assert_size_stride(arg81_1, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(arg82_1, (128, ), (1, ))
    assert_size_stride(arg83_1, (128, ), (1, ))
    assert_size_stride(arg84_1, (128, ), (1, ))
    assert_size_stride(arg85_1, (128, ), (1, ))
    assert_size_stride(arg86_1, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(arg87_1, (512, ), (1, ))
    assert_size_stride(arg88_1, (512, ), (1, ))
    assert_size_stride(arg89_1, (512, ), (1, ))
    assert_size_stride(arg90_1, (512, ), (1, ))
    assert_size_stride(arg91_1, (128, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg92_1, (128, ), (1, ))
    assert_size_stride(arg93_1, (128, ), (1, ))
    assert_size_stride(arg94_1, (128, ), (1, ))
    assert_size_stride(arg95_1, (128, ), (1, ))
    assert_size_stride(arg96_1, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(arg97_1, (128, ), (1, ))
    assert_size_stride(arg98_1, (128, ), (1, ))
    assert_size_stride(arg99_1, (128, ), (1, ))
    assert_size_stride(arg100_1, (128, ), (1, ))
    assert_size_stride(arg101_1, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(arg102_1, (512, ), (1, ))
    assert_size_stride(arg103_1, (512, ), (1, ))
    assert_size_stride(arg104_1, (512, ), (1, ))
    assert_size_stride(arg105_1, (512, ), (1, ))
    assert_size_stride(arg106_1, (128, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg107_1, (128, ), (1, ))
    assert_size_stride(arg108_1, (128, ), (1, ))
    assert_size_stride(arg109_1, (128, ), (1, ))
    assert_size_stride(arg110_1, (128, ), (1, ))
    assert_size_stride(arg111_1, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(arg112_1, (128, ), (1, ))
    assert_size_stride(arg113_1, (128, ), (1, ))
    assert_size_stride(arg114_1, (128, ), (1, ))
    assert_size_stride(arg115_1, (128, ), (1, ))
    assert_size_stride(arg116_1, (512, 128, 1, 1), (128, 1, 128, 128))
    assert_size_stride(arg117_1, (512, ), (1, ))
    assert_size_stride(arg118_1, (512, ), (1, ))
    assert_size_stride(arg119_1, (512, ), (1, ))
    assert_size_stride(arg120_1, (512, ), (1, ))
    assert_size_stride(arg121_1, (256, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg122_1, (256, ), (1, ))
    assert_size_stride(arg123_1, (256, ), (1, ))
    assert_size_stride(arg124_1, (256, ), (1, ))
    assert_size_stride(arg125_1, (256, ), (1, ))
    assert_size_stride(arg126_1, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(arg127_1, (256, ), (1, ))
    assert_size_stride(arg128_1, (256, ), (1, ))
    assert_size_stride(arg129_1, (256, ), (1, ))
    assert_size_stride(arg130_1, (256, ), (1, ))
    assert_size_stride(arg131_1, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg132_1, (1024, ), (1, ))
    assert_size_stride(arg133_1, (1024, ), (1, ))
    assert_size_stride(arg134_1, (1024, ), (1, ))
    assert_size_stride(arg135_1, (1024, ), (1, ))
    assert_size_stride(arg136_1, (1024, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg137_1, (1024, ), (1, ))
    assert_size_stride(arg138_1, (1024, ), (1, ))
    assert_size_stride(arg139_1, (1024, ), (1, ))
    assert_size_stride(arg140_1, (1024, ), (1, ))
    assert_size_stride(arg141_1, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg142_1, (256, ), (1, ))
    assert_size_stride(arg143_1, (256, ), (1, ))
    assert_size_stride(arg144_1, (256, ), (1, ))
    assert_size_stride(arg145_1, (256, ), (1, ))
    assert_size_stride(arg146_1, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(arg147_1, (256, ), (1, ))
    assert_size_stride(arg148_1, (256, ), (1, ))
    assert_size_stride(arg149_1, (256, ), (1, ))
    assert_size_stride(arg150_1, (256, ), (1, ))
    assert_size_stride(arg151_1, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg152_1, (1024, ), (1, ))
    assert_size_stride(arg153_1, (1024, ), (1, ))
    assert_size_stride(arg154_1, (1024, ), (1, ))
    assert_size_stride(arg155_1, (1024, ), (1, ))
    assert_size_stride(arg156_1, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg157_1, (256, ), (1, ))
    assert_size_stride(arg158_1, (256, ), (1, ))
    assert_size_stride(arg159_1, (256, ), (1, ))
    assert_size_stride(arg160_1, (256, ), (1, ))
    assert_size_stride(arg161_1, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(arg162_1, (256, ), (1, ))
    assert_size_stride(arg163_1, (256, ), (1, ))
    assert_size_stride(arg164_1, (256, ), (1, ))
    assert_size_stride(arg165_1, (256, ), (1, ))
    assert_size_stride(arg166_1, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg167_1, (1024, ), (1, ))
    assert_size_stride(arg168_1, (1024, ), (1, ))
    assert_size_stride(arg169_1, (1024, ), (1, ))
    assert_size_stride(arg170_1, (1024, ), (1, ))
    assert_size_stride(arg171_1, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg172_1, (256, ), (1, ))
    assert_size_stride(arg173_1, (256, ), (1, ))
    assert_size_stride(arg174_1, (256, ), (1, ))
    assert_size_stride(arg175_1, (256, ), (1, ))
    assert_size_stride(arg176_1, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(arg177_1, (256, ), (1, ))
    assert_size_stride(arg178_1, (256, ), (1, ))
    assert_size_stride(arg179_1, (256, ), (1, ))
    assert_size_stride(arg180_1, (256, ), (1, ))
    assert_size_stride(arg181_1, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg182_1, (1024, ), (1, ))
    assert_size_stride(arg183_1, (1024, ), (1, ))
    assert_size_stride(arg184_1, (1024, ), (1, ))
    assert_size_stride(arg185_1, (1024, ), (1, ))
    assert_size_stride(arg186_1, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg187_1, (256, ), (1, ))
    assert_size_stride(arg188_1, (256, ), (1, ))
    assert_size_stride(arg189_1, (256, ), (1, ))
    assert_size_stride(arg190_1, (256, ), (1, ))
    assert_size_stride(arg191_1, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(arg192_1, (256, ), (1, ))
    assert_size_stride(arg193_1, (256, ), (1, ))
    assert_size_stride(arg194_1, (256, ), (1, ))
    assert_size_stride(arg195_1, (256, ), (1, ))
    assert_size_stride(arg196_1, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg197_1, (1024, ), (1, ))
    assert_size_stride(arg198_1, (1024, ), (1, ))
    assert_size_stride(arg199_1, (1024, ), (1, ))
    assert_size_stride(arg200_1, (1024, ), (1, ))
    assert_size_stride(arg201_1, (256, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg202_1, (256, ), (1, ))
    assert_size_stride(arg203_1, (256, ), (1, ))
    assert_size_stride(arg204_1, (256, ), (1, ))
    assert_size_stride(arg205_1, (256, ), (1, ))
    assert_size_stride(arg206_1, (256, 256, 3, 3), (2304, 1, 768, 256))
    assert_size_stride(arg207_1, (256, ), (1, ))
    assert_size_stride(arg208_1, (256, ), (1, ))
    assert_size_stride(arg209_1, (256, ), (1, ))
    assert_size_stride(arg210_1, (256, ), (1, ))
    assert_size_stride(arg211_1, (1024, 256, 1, 1), (256, 1, 256, 256))
    assert_size_stride(arg212_1, (1024, ), (1, ))
    assert_size_stride(arg213_1, (1024, ), (1, ))
    assert_size_stride(arg214_1, (1024, ), (1, ))
    assert_size_stride(arg215_1, (1024, ), (1, ))
    assert_size_stride(arg216_1, (512, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg217_1, (512, ), (1, ))
    assert_size_stride(arg218_1, (512, ), (1, ))
    assert_size_stride(arg219_1, (512, ), (1, ))
    assert_size_stride(arg220_1, (512, ), (1, ))
    assert_size_stride(arg221_1, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(arg222_1, (512, ), (1, ))
    assert_size_stride(arg223_1, (512, ), (1, ))
    assert_size_stride(arg224_1, (512, ), (1, ))
    assert_size_stride(arg225_1, (512, ), (1, ))
    assert_size_stride(arg226_1, (2048, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg227_1, (2048, ), (1, ))
    assert_size_stride(arg228_1, (2048, ), (1, ))
    assert_size_stride(arg229_1, (2048, ), (1, ))
    assert_size_stride(arg230_1, (2048, ), (1, ))
    assert_size_stride(arg231_1, (2048, 1024, 1, 1), (1024, 1, 1024, 1024))
    assert_size_stride(arg232_1, (2048, ), (1, ))
    assert_size_stride(arg233_1, (2048, ), (1, ))
    assert_size_stride(arg234_1, (2048, ), (1, ))
    assert_size_stride(arg235_1, (2048, ), (1, ))
    assert_size_stride(arg236_1, (512, 2048, 1, 1), (2048, 1, 2048, 2048))
    assert_size_stride(arg237_1, (512, ), (1, ))
    assert_size_stride(arg238_1, (512, ), (1, ))
    assert_size_stride(arg239_1, (512, ), (1, ))
    assert_size_stride(arg240_1, (512, ), (1, ))
    assert_size_stride(arg241_1, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(arg242_1, (512, ), (1, ))
    assert_size_stride(arg243_1, (512, ), (1, ))
    assert_size_stride(arg244_1, (512, ), (1, ))
    assert_size_stride(arg245_1, (512, ), (1, ))
    assert_size_stride(arg246_1, (2048, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg247_1, (2048, ), (1, ))
    assert_size_stride(arg248_1, (2048, ), (1, ))
    assert_size_stride(arg249_1, (2048, ), (1, ))
    assert_size_stride(arg250_1, (2048, ), (1, ))
    assert_size_stride(arg251_1, (512, 2048, 1, 1), (2048, 1, 2048, 2048))
    assert_size_stride(arg252_1, (512, ), (1, ))
    assert_size_stride(arg253_1, (512, ), (1, ))
    assert_size_stride(arg254_1, (512, ), (1, ))
    assert_size_stride(arg255_1, (512, ), (1, ))
    assert_size_stride(arg256_1, (512, 512, 3, 3), (4608, 1, 1536, 512))
    assert_size_stride(arg257_1, (512, ), (1, ))
    assert_size_stride(arg258_1, (512, ), (1, ))
    assert_size_stride(arg259_1, (512, ), (1, ))
    assert_size_stride(arg260_1, (512, ), (1, ))
    assert_size_stride(arg261_1, (2048, 512, 1, 1), (512, 1, 512, 512))
    assert_size_stride(arg262_1, (2048, ), (1, ))
    assert_size_stride(arg263_1, (2048, ), (1, ))
    assert_size_stride(arg264_1, (2048, ), (1, ))
    assert_size_stride(arg265_1, (2048, ), (1, ))
    assert_size_stride(arg266_1, (1000, 2048), (2048, 1))
    assert_size_stride(arg267_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg1_1, arg0_1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (1, 64, 112, 112), (802816, 1, 7168, 64))
        del arg0_1
        del arg1_1
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten.cudnn_batch_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cudnn_batch_norm_relu_0.run(buf1, arg2_1, arg3_1, arg4_1, arg5_1, 802816, grid=grid(802816), stream=stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        buf2 = empty_strided_cuda((1, 64, 56, 56), (200704, 1, 3584, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_1, x_2, x_3], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.max_pool2d_with_indices]
        triton_poi_fused_cudnn_batch_norm_max_pool2d_with_indices_relu_1.run(buf1, buf2, 200704, grid=grid(200704), stream=stream0)
        buf3 = empty_strided_cuda((3136, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf2, (3136, 64), (64, 1), 0), reinterpret_tensor(arg6_1, (64, 64), (1, 64), 0), out=buf3)
        del arg6_1
        buf4 = reinterpret_tensor(buf3, (1, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [out_1, out_2], Original ATen: [aten.cudnn_batch_norm, aten.relu]
        triton_poi_fused_cudnn_batch_norm_relu_2.run(buf4, arg7_1, arg8_1, arg9_1, arg10_1, 200704, grid=grid(200704), stream=stream0)
        del arg10_1
        del arg7_1
        del arg8_1
        del arg9_1
        # Topologically Sorted Source Nodes: [out_1, out_2, out_3], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        buf5 = extern_kernels.convolution(buf4, arg11_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (1, 64, 56, 56), (200704, 1, 3584, 64))
        del arg11_1
        del buf4
        buf6 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [out_4, out_5, out_6], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        triton_poi_fused_cudnn_batch_norm_relu_2.run(buf6, arg12_1, arg13_1, arg14_1, arg15_1, 200704, grid=grid(200704), stream=stream0)
        del arg12_1
        del arg13_1
        del arg14_1
        del arg15_1
        buf7 = reinterpret_tensor(buf1, (3136, 256), (256, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [out_4, out_5, out_6], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf6, (3136, 64), (64, 1), 0), reinterpret_tensor(arg16_1, (64, 256), (1, 64), 0), out=buf7)
        del arg16_1
        del buf6
        buf8 = empty_strided_cuda((3136, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf2, (3136, 64), (64, 1), 0), reinterpret_tensor(arg21_1, (64, 256), (1, 64), 0), out=buf8)
        del arg21_1
        buf9 = reinterpret_tensor(buf7, (1, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf7  # reuse
        buf10 = empty_strided_cuda((1, 256, 56, 56), (802816, 1, 14336, 256), torch.float32)
        # Topologically Sorted Source Nodes: [out_7, input_2, out_8, out_9, out_10], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu, aten.convolution]
        triton_poi_fused_add_convolution_cudnn_batch_norm_relu_3.run(buf9, arg17_1, arg18_1, arg19_1, arg20_1, buf8, arg22_1, arg23_1, arg24_1, arg25_1, buf10, 802816, grid=grid(802816), stream=stream0)
        del arg17_1
        del arg18_1
        del arg19_1
        del arg20_1
        del arg22_1
        del arg23_1
        del arg24_1
        del arg25_1
        del buf8
        buf11 = reinterpret_tensor(buf2, (3136, 64), (64, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [out_9, out_10], Original ATen: [aten.relu, aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf10, (3136, 256), (256, 1), 0), reinterpret_tensor(arg26_1, (256, 64), (1, 256), 0), out=buf11)
        del arg26_1
        buf12 = reinterpret_tensor(buf11, (1, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [out_11, out_12], Original ATen: [aten.cudnn_batch_norm, aten.relu]
        triton_poi_fused_cudnn_batch_norm_relu_2.run(buf12, arg27_1, arg28_1, arg29_1, arg30_1, 200704, grid=grid(200704), stream=stream0)
        del arg27_1
        del arg28_1
        del arg29_1
        del arg30_1
        # Topologically Sorted Source Nodes: [out_11, out_12, out_13], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        buf13 = extern_kernels.convolution(buf12, arg31_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (1, 64, 56, 56), (200704, 1, 3584, 64))
        del arg31_1
        del buf12
        buf14 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [out_14, out_15, out_16], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        triton_poi_fused_cudnn_batch_norm_relu_2.run(buf14, arg32_1, arg33_1, arg34_1, arg35_1, 200704, grid=grid(200704), stream=stream0)
        del arg32_1
        del arg33_1
        del arg34_1
        del arg35_1
        buf15 = reinterpret_tensor(buf10, (3136, 256), (256, 1), 0); del buf10  # reuse
        # Topologically Sorted Source Nodes: [out_14, out_15, out_16], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf14, (3136, 64), (64, 1), 0), reinterpret_tensor(arg36_1, (64, 256), (1, 64), 0), out=buf15)
        del arg36_1
        buf16 = reinterpret_tensor(buf15, (1, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf15  # reuse
        # Topologically Sorted Source Nodes: [out_9, out_17, out_18, out_19], Original ATen: [aten.relu, aten.cudnn_batch_norm, aten.add]
        triton_poi_fused_add_cudnn_batch_norm_relu_4.run(buf16, arg37_1, arg38_1, arg39_1, arg40_1, buf9, 802816, grid=grid(802816), stream=stream0)
        del arg37_1
        del arg38_1
        del arg39_1
        del arg40_1
        buf17 = reinterpret_tensor(buf14, (3136, 64), (64, 1), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [out_20], Original ATen: [aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf16, (3136, 256), (256, 1), 0), reinterpret_tensor(arg41_1, (256, 64), (1, 256), 0), out=buf17)
        del arg41_1
        buf18 = reinterpret_tensor(buf17, (1, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [out_21, out_22], Original ATen: [aten.cudnn_batch_norm, aten.relu]
        triton_poi_fused_cudnn_batch_norm_relu_2.run(buf18, arg42_1, arg43_1, arg44_1, arg45_1, 200704, grid=grid(200704), stream=stream0)
        del arg42_1
        del arg43_1
        del arg44_1
        del arg45_1
        # Topologically Sorted Source Nodes: [out_21, out_22, out_23], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        buf19 = extern_kernels.convolution(buf18, arg46_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (1, 64, 56, 56), (200704, 1, 3584, 64))
        del arg46_1
        buf20 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [out_24, out_25, out_26], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        triton_poi_fused_cudnn_batch_norm_relu_2.run(buf20, arg47_1, arg48_1, arg49_1, arg50_1, 200704, grid=grid(200704), stream=stream0)
        del arg47_1
        del arg48_1
        del arg49_1
        del arg50_1
        buf21 = reinterpret_tensor(buf9, (3136, 256), (256, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [out_24, out_25, out_26], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf20, (3136, 64), (64, 1), 0), reinterpret_tensor(arg51_1, (64, 256), (1, 64), 0), out=buf21)
        del arg51_1
        buf22 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [out_27, out_28, out_29], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu]
        triton_poi_fused_add_cudnn_batch_norm_relu_5.run(buf22, buf21, arg52_1, arg53_1, arg54_1, arg55_1, 802816, grid=grid(802816), stream=stream0)
        del arg52_1
        del arg53_1
        del arg54_1
        del arg55_1
        del buf21
        buf23 = empty_strided_cuda((3136, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_30], Original ATen: [aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf22, (3136, 256), (256, 1), 0), reinterpret_tensor(arg56_1, (256, 128), (1, 256), 0), out=buf23)
        del arg56_1
        buf24 = reinterpret_tensor(buf23, (1, 128, 56, 56), (401408, 1, 7168, 128), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [out_31, out_32], Original ATen: [aten.cudnn_batch_norm, aten.relu]
        triton_poi_fused_cudnn_batch_norm_relu_6.run(buf24, arg57_1, arg58_1, arg59_1, arg60_1, 401408, grid=grid(401408), stream=stream0)
        del arg57_1
        del arg58_1
        del arg59_1
        del arg60_1
        # Topologically Sorted Source Nodes: [out_31, out_32, out_33], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        buf25 = extern_kernels.convolution(buf24, arg61_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (1, 128, 28, 28), (100352, 1, 3584, 128))
        del arg61_1
        buf26 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [out_34, out_35, out_36], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        triton_poi_fused_convolution_cudnn_batch_norm_relu_7.run(buf26, arg62_1, arg63_1, arg64_1, arg65_1, 100352, grid=grid(100352), stream=stream0)
        del arg62_1
        del arg63_1
        del arg64_1
        del arg65_1
        buf27 = reinterpret_tensor(buf24, (784, 512), (512, 1), 0); del buf24  # reuse
        # Topologically Sorted Source Nodes: [out_34, out_35, out_36], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf26, (784, 128), (128, 1), 0), reinterpret_tensor(arg66_1, (128, 512), (1, 128), 0), out=buf27)
        del arg66_1
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf22, arg71_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (1, 512, 28, 28), (401408, 1, 14336, 512))
        del arg71_1
        del buf22
        buf29 = reinterpret_tensor(buf27, (1, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf27  # reuse
        buf30 = empty_strided_cuda((1, 512, 28, 28), (401408, 1, 14336, 512), torch.float32)
        # Topologically Sorted Source Nodes: [out_37, input_4, out_38, out_39, out_40], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu, aten.convolution]
        triton_poi_fused_add_convolution_cudnn_batch_norm_relu_8.run(buf29, arg67_1, arg68_1, arg69_1, arg70_1, buf28, arg72_1, arg73_1, arg74_1, arg75_1, buf30, 401408, grid=grid(401408), stream=stream0)
        del arg67_1
        del arg68_1
        del arg69_1
        del arg70_1
        del arg72_1
        del arg73_1
        del arg74_1
        del arg75_1
        del buf28
        buf31 = reinterpret_tensor(buf26, (784, 128), (128, 1), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [out_39, out_40], Original ATen: [aten.relu, aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf30, (784, 512), (512, 1), 0), reinterpret_tensor(arg76_1, (512, 128), (1, 512), 0), out=buf31)
        del arg76_1
        buf32 = reinterpret_tensor(buf31, (1, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf31  # reuse
        # Topologically Sorted Source Nodes: [out_41, out_42], Original ATen: [aten.cudnn_batch_norm, aten.relu]
        triton_poi_fused_convolution_cudnn_batch_norm_relu_7.run(buf32, arg77_1, arg78_1, arg79_1, arg80_1, 100352, grid=grid(100352), stream=stream0)
        del arg77_1
        del arg78_1
        del arg79_1
        del arg80_1
        # Topologically Sorted Source Nodes: [out_41, out_42, out_43], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        buf33 = extern_kernels.convolution(buf32, arg81_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (1, 128, 28, 28), (100352, 1, 3584, 128))
        del arg81_1
        del buf32
        buf34 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [out_44, out_45, out_46], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        triton_poi_fused_convolution_cudnn_batch_norm_relu_7.run(buf34, arg82_1, arg83_1, arg84_1, arg85_1, 100352, grid=grid(100352), stream=stream0)
        del arg82_1
        del arg83_1
        del arg84_1
        del arg85_1
        buf35 = reinterpret_tensor(buf30, (784, 512), (512, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [out_44, out_45, out_46], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf34, (784, 128), (128, 1), 0), reinterpret_tensor(arg86_1, (128, 512), (1, 128), 0), out=buf35)
        del arg86_1
        buf36 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [out_39, out_47, out_48, out_49], Original ATen: [aten.relu, aten.cudnn_batch_norm, aten.add]
        triton_poi_fused_add_cudnn_batch_norm_relu_9.run(buf36, buf35, arg87_1, arg88_1, arg89_1, arg90_1, 401408, grid=grid(401408), stream=stream0)
        del arg87_1
        del arg88_1
        del arg89_1
        del arg90_1
        buf37 = reinterpret_tensor(buf34, (784, 128), (128, 1), 0); del buf34  # reuse
        # Topologically Sorted Source Nodes: [out_50], Original ATen: [aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf36, (784, 512), (512, 1), 0), reinterpret_tensor(arg91_1, (512, 128), (1, 512), 0), out=buf37)
        del arg91_1
        buf38 = reinterpret_tensor(buf37, (1, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [out_51, out_52], Original ATen: [aten.cudnn_batch_norm, aten.relu]
        triton_poi_fused_convolution_cudnn_batch_norm_relu_7.run(buf38, arg92_1, arg93_1, arg94_1, arg95_1, 100352, grid=grid(100352), stream=stream0)
        del arg92_1
        del arg93_1
        del arg94_1
        del arg95_1
        # Topologically Sorted Source Nodes: [out_51, out_52, out_53], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        buf39 = extern_kernels.convolution(buf38, arg96_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (1, 128, 28, 28), (100352, 1, 3584, 128))
        del arg96_1
        del buf38
        buf40 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [out_54, out_55, out_56], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        triton_poi_fused_convolution_cudnn_batch_norm_relu_7.run(buf40, arg97_1, arg98_1, arg99_1, arg100_1, 100352, grid=grid(100352), stream=stream0)
        del arg100_1
        del arg97_1
        del arg98_1
        del arg99_1
        buf41 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [out_54, out_55, out_56], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf40, (784, 128), (128, 1), 0), reinterpret_tensor(arg101_1, (128, 512), (1, 128), 0), out=buf41)
        del arg101_1
        buf42 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [out_57, out_58, out_59], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu]
        triton_poi_fused_add_cudnn_batch_norm_relu_10.run(buf42, buf41, arg102_1, arg103_1, arg104_1, arg105_1, 401408, grid=grid(401408), stream=stream0)
        del arg102_1
        del arg103_1
        del arg104_1
        del arg105_1
        buf43 = reinterpret_tensor(buf40, (784, 128), (128, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [out_60], Original ATen: [aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf42, (784, 512), (512, 1), 0), reinterpret_tensor(arg106_1, (512, 128), (1, 512), 0), out=buf43)
        del arg106_1
        buf44 = reinterpret_tensor(buf43, (1, 128, 28, 28), (100352, 1, 3584, 128), 0); del buf43  # reuse
        # Topologically Sorted Source Nodes: [out_61, out_62], Original ATen: [aten.cudnn_batch_norm, aten.relu]
        triton_poi_fused_convolution_cudnn_batch_norm_relu_7.run(buf44, arg107_1, arg108_1, arg109_1, arg110_1, 100352, grid=grid(100352), stream=stream0)
        del arg107_1
        del arg108_1
        del arg109_1
        del arg110_1
        # Topologically Sorted Source Nodes: [out_61, out_62, out_63], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        buf45 = extern_kernels.convolution(buf44, arg111_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (1, 128, 28, 28), (100352, 1, 3584, 128))
        del arg111_1
        buf46 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [out_64, out_65, out_66], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        triton_poi_fused_convolution_cudnn_batch_norm_relu_7.run(buf46, arg112_1, arg113_1, arg114_1, arg115_1, 100352, grid=grid(100352), stream=stream0)
        del arg112_1
        del arg113_1
        del arg114_1
        del arg115_1
        buf47 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [out_64, out_65, out_66], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf46, (784, 128), (128, 1), 0), reinterpret_tensor(arg116_1, (128, 512), (1, 128), 0), out=buf47)
        del arg116_1
        buf48 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [out_67, out_68, out_69], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu]
        triton_poi_fused_add_cudnn_batch_norm_relu_10.run(buf48, buf47, arg117_1, arg118_1, arg119_1, arg120_1, 401408, grid=grid(401408), stream=stream0)
        del arg117_1
        del arg118_1
        del arg119_1
        del arg120_1
        del buf47
        buf49 = reinterpret_tensor(buf20, (784, 256), (256, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [out_70], Original ATen: [aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf48, (784, 512), (512, 1), 0), reinterpret_tensor(arg121_1, (512, 256), (1, 512), 0), out=buf49)
        del arg121_1
        buf50 = reinterpret_tensor(buf49, (1, 256, 28, 28), (200704, 1, 7168, 256), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [out_71, out_72], Original ATen: [aten.cudnn_batch_norm, aten.relu]
        triton_poi_fused_cudnn_batch_norm_relu_11.run(buf50, arg122_1, arg123_1, arg124_1, arg125_1, 200704, grid=grid(200704), stream=stream0)
        del arg122_1
        del arg123_1
        del arg124_1
        del arg125_1
        # Topologically Sorted Source Nodes: [out_71, out_72, out_73], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        buf51 = extern_kernels.convolution(buf50, arg126_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (1, 256, 14, 14), (50176, 1, 3584, 256))
        del arg126_1
        buf52 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [out_74, out_75, out_76], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        triton_poi_fused_convolution_cudnn_batch_norm_relu_12.run(buf52, arg127_1, arg128_1, arg129_1, arg130_1, 50176, grid=grid(50176), stream=stream0)
        del arg127_1
        del arg128_1
        del arg129_1
        del arg130_1
        buf53 = reinterpret_tensor(buf50, (196, 1024), (1024, 1), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [out_74, out_75, out_76], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf52, (196, 256), (256, 1), 0), reinterpret_tensor(arg131_1, (256, 1024), (1, 256), 0), out=buf53)
        del arg131_1
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf48, arg136_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (1, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg136_1
        del buf48
        buf55 = reinterpret_tensor(buf53, (1, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf53  # reuse
        buf56 = reinterpret_tensor(buf18, (1, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [out_77, input_6, out_78, out_79, out_80], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu, aten.convolution]
        triton_poi_fused_add_convolution_cudnn_batch_norm_relu_13.run(buf55, arg132_1, arg133_1, arg134_1, arg135_1, buf54, arg137_1, arg138_1, arg139_1, arg140_1, buf56, 200704, grid=grid(200704), stream=stream0)
        del arg132_1
        del arg133_1
        del arg134_1
        del arg135_1
        del arg137_1
        del arg138_1
        del arg139_1
        del arg140_1
        del buf54
        buf57 = reinterpret_tensor(buf52, (196, 256), (256, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [out_79, out_80], Original ATen: [aten.relu, aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf56, (196, 1024), (1024, 1), 0), reinterpret_tensor(arg141_1, (1024, 256), (1, 1024), 0), out=buf57)
        del arg141_1
        buf58 = reinterpret_tensor(buf57, (1, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [out_81, out_82], Original ATen: [aten.cudnn_batch_norm, aten.relu]
        triton_poi_fused_convolution_cudnn_batch_norm_relu_12.run(buf58, arg142_1, arg143_1, arg144_1, arg145_1, 50176, grid=grid(50176), stream=stream0)
        del arg142_1
        del arg143_1
        del arg144_1
        del arg145_1
        # Topologically Sorted Source Nodes: [out_81, out_82, out_83], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        buf59 = extern_kernels.convolution(buf58, arg146_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (1, 256, 14, 14), (50176, 1, 3584, 256))
        del arg146_1
        del buf58
        buf60 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [out_84, out_85, out_86], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        triton_poi_fused_convolution_cudnn_batch_norm_relu_12.run(buf60, arg147_1, arg148_1, arg149_1, arg150_1, 50176, grid=grid(50176), stream=stream0)
        del arg147_1
        del arg148_1
        del arg149_1
        del arg150_1
        buf61 = reinterpret_tensor(buf56, (196, 1024), (1024, 1), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [out_84, out_85, out_86], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf60, (196, 256), (256, 1), 0), reinterpret_tensor(arg151_1, (256, 1024), (1, 256), 0), out=buf61)
        del arg151_1
        buf62 = buf55; del buf55  # reuse
        # Topologically Sorted Source Nodes: [out_79, out_87, out_88, out_89], Original ATen: [aten.relu, aten.cudnn_batch_norm, aten.add]
        triton_poi_fused_add_cudnn_batch_norm_relu_14.run(buf62, buf61, arg152_1, arg153_1, arg154_1, arg155_1, 200704, grid=grid(200704), stream=stream0)
        del arg152_1
        del arg153_1
        del arg154_1
        del arg155_1
        buf63 = reinterpret_tensor(buf60, (196, 256), (256, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [out_90], Original ATen: [aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf62, (196, 1024), (1024, 1), 0), reinterpret_tensor(arg156_1, (1024, 256), (1, 1024), 0), out=buf63)
        del arg156_1
        buf64 = reinterpret_tensor(buf63, (1, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [out_91, out_92], Original ATen: [aten.cudnn_batch_norm, aten.relu]
        triton_poi_fused_convolution_cudnn_batch_norm_relu_12.run(buf64, arg157_1, arg158_1, arg159_1, arg160_1, 50176, grid=grid(50176), stream=stream0)
        del arg157_1
        del arg158_1
        del arg159_1
        del arg160_1
        # Topologically Sorted Source Nodes: [out_91, out_92, out_93], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        buf65 = extern_kernels.convolution(buf64, arg161_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (1, 256, 14, 14), (50176, 1, 3584, 256))
        del arg161_1
        del buf64
        buf66 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [out_94, out_95, out_96], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        triton_poi_fused_convolution_cudnn_batch_norm_relu_12.run(buf66, arg162_1, arg163_1, arg164_1, arg165_1, 50176, grid=grid(50176), stream=stream0)
        del arg162_1
        del arg163_1
        del arg164_1
        del arg165_1
        buf67 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [out_94, out_95, out_96], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf66, (196, 256), (256, 1), 0), reinterpret_tensor(arg166_1, (256, 1024), (1, 256), 0), out=buf67)
        del arg166_1
        buf68 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [out_97, out_98, out_99], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu]
        triton_poi_fused_add_cudnn_batch_norm_relu_15.run(buf68, buf67, arg167_1, arg168_1, arg169_1, arg170_1, 200704, grid=grid(200704), stream=stream0)
        del arg167_1
        del arg168_1
        del arg169_1
        del arg170_1
        buf69 = reinterpret_tensor(buf66, (196, 256), (256, 1), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [out_100], Original ATen: [aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf68, (196, 1024), (1024, 1), 0), reinterpret_tensor(arg171_1, (1024, 256), (1, 1024), 0), out=buf69)
        del arg171_1
        buf70 = reinterpret_tensor(buf69, (1, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [out_101, out_102], Original ATen: [aten.cudnn_batch_norm, aten.relu]
        triton_poi_fused_convolution_cudnn_batch_norm_relu_12.run(buf70, arg172_1, arg173_1, arg174_1, arg175_1, 50176, grid=grid(50176), stream=stream0)
        del arg172_1
        del arg173_1
        del arg174_1
        del arg175_1
        # Topologically Sorted Source Nodes: [out_101, out_102, out_103], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        buf71 = extern_kernels.convolution(buf70, arg176_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (1, 256, 14, 14), (50176, 1, 3584, 256))
        del arg176_1
        del buf70
        buf72 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [out_104, out_105, out_106], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        triton_poi_fused_convolution_cudnn_batch_norm_relu_12.run(buf72, arg177_1, arg178_1, arg179_1, arg180_1, 50176, grid=grid(50176), stream=stream0)
        del arg177_1
        del arg178_1
        del arg179_1
        del arg180_1
        buf73 = buf67; del buf67  # reuse
        # Topologically Sorted Source Nodes: [out_104, out_105, out_106], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf72, (196, 256), (256, 1), 0), reinterpret_tensor(arg181_1, (256, 1024), (1, 256), 0), out=buf73)
        del arg181_1
        buf74 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [out_107, out_108, out_109], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu]
        triton_poi_fused_add_cudnn_batch_norm_relu_15.run(buf74, buf73, arg182_1, arg183_1, arg184_1, arg185_1, 200704, grid=grid(200704), stream=stream0)
        del arg182_1
        del arg183_1
        del arg184_1
        del arg185_1
        buf75 = reinterpret_tensor(buf72, (196, 256), (256, 1), 0); del buf72  # reuse
        # Topologically Sorted Source Nodes: [out_110], Original ATen: [aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf74, (196, 1024), (1024, 1), 0), reinterpret_tensor(arg186_1, (1024, 256), (1, 1024), 0), out=buf75)
        del arg186_1
        buf76 = reinterpret_tensor(buf75, (1, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf75  # reuse
        # Topologically Sorted Source Nodes: [out_111, out_112], Original ATen: [aten.cudnn_batch_norm, aten.relu]
        triton_poi_fused_convolution_cudnn_batch_norm_relu_12.run(buf76, arg187_1, arg188_1, arg189_1, arg190_1, 50176, grid=grid(50176), stream=stream0)
        del arg187_1
        del arg188_1
        del arg189_1
        del arg190_1
        # Topologically Sorted Source Nodes: [out_111, out_112, out_113], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        buf77 = extern_kernels.convolution(buf76, arg191_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (1, 256, 14, 14), (50176, 1, 3584, 256))
        del arg191_1
        del buf76
        buf78 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [out_114, out_115, out_116], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        triton_poi_fused_convolution_cudnn_batch_norm_relu_12.run(buf78, arg192_1, arg193_1, arg194_1, arg195_1, 50176, grid=grid(50176), stream=stream0)
        del arg192_1
        del arg193_1
        del arg194_1
        del arg195_1
        buf79 = buf73; del buf73  # reuse
        # Topologically Sorted Source Nodes: [out_114, out_115, out_116], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf78, (196, 256), (256, 1), 0), reinterpret_tensor(arg196_1, (256, 1024), (1, 256), 0), out=buf79)
        del arg196_1
        buf80 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [out_117, out_118, out_119], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu]
        triton_poi_fused_add_cudnn_batch_norm_relu_15.run(buf80, buf79, arg197_1, arg198_1, arg199_1, arg200_1, 200704, grid=grid(200704), stream=stream0)
        del arg197_1
        del arg198_1
        del arg199_1
        del arg200_1
        buf81 = reinterpret_tensor(buf78, (196, 256), (256, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [out_120], Original ATen: [aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf80, (196, 1024), (1024, 1), 0), reinterpret_tensor(arg201_1, (1024, 256), (1, 1024), 0), out=buf81)
        del arg201_1
        buf82 = reinterpret_tensor(buf81, (1, 256, 14, 14), (50176, 1, 3584, 256), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [out_121, out_122], Original ATen: [aten.cudnn_batch_norm, aten.relu]
        triton_poi_fused_convolution_cudnn_batch_norm_relu_12.run(buf82, arg202_1, arg203_1, arg204_1, arg205_1, 50176, grid=grid(50176), stream=stream0)
        del arg202_1
        del arg203_1
        del arg204_1
        del arg205_1
        # Topologically Sorted Source Nodes: [out_121, out_122, out_123], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        buf83 = extern_kernels.convolution(buf82, arg206_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf83, (1, 256, 14, 14), (50176, 1, 3584, 256))
        del arg206_1
        del buf82
        buf84 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [out_124, out_125, out_126], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        triton_poi_fused_convolution_cudnn_batch_norm_relu_12.run(buf84, arg207_1, arg208_1, arg209_1, arg210_1, 50176, grid=grid(50176), stream=stream0)
        del arg207_1
        del arg208_1
        del arg209_1
        del arg210_1
        buf85 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [out_124, out_125, out_126], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf84, (196, 256), (256, 1), 0), reinterpret_tensor(arg211_1, (256, 1024), (1, 256), 0), out=buf85)
        del arg211_1
        del buf84
        buf86 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [out_127, out_128, out_129], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu]
        triton_poi_fused_add_cudnn_batch_norm_relu_15.run(buf86, buf85, arg212_1, arg213_1, arg214_1, arg215_1, 200704, grid=grid(200704), stream=stream0)
        del arg212_1
        del arg213_1
        del arg214_1
        del arg215_1
        del buf85
        buf87 = reinterpret_tensor(buf46, (196, 512), (512, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [out_130], Original ATen: [aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf86, (196, 1024), (1024, 1), 0), reinterpret_tensor(arg216_1, (1024, 512), (1, 1024), 0), out=buf87)
        del arg216_1
        buf88 = reinterpret_tensor(buf87, (1, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf87  # reuse
        # Topologically Sorted Source Nodes: [out_131, out_132], Original ATen: [aten.cudnn_batch_norm, aten.relu]
        triton_poi_fused_cudnn_batch_norm_relu_16.run(buf88, arg217_1, arg218_1, arg219_1, arg220_1, 100352, grid=grid(100352), stream=stream0)
        del arg217_1
        del arg218_1
        del arg219_1
        del arg220_1
        # Topologically Sorted Source Nodes: [out_131, out_132, out_133], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        buf89 = extern_kernels.convolution(buf88, arg221_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (1, 512, 7, 7), (25088, 1, 3584, 512))
        del arg221_1
        buf90 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [out_134, out_135, out_136], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        triton_poi_fused_convolution_cudnn_batch_norm_relu_17.run(buf90, arg222_1, arg223_1, arg224_1, arg225_1, 25088, grid=grid(25088), stream=stream0)
        del arg222_1
        del arg223_1
        del arg224_1
        del arg225_1
        buf91 = reinterpret_tensor(buf88, (49, 2048), (2048, 1), 0); del buf88  # reuse
        # Topologically Sorted Source Nodes: [out_134, out_135, out_136], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf90, (49, 512), (512, 1), 0), reinterpret_tensor(arg226_1, (512, 2048), (1, 512), 0), out=buf91)
        del arg226_1
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf86, arg231_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (1, 2048, 7, 7), (100352, 1, 14336, 2048))
        del arg231_1
        del buf86
        buf93 = reinterpret_tensor(buf91, (1, 2048, 7, 7), (100352, 1, 14336, 2048), 0); del buf91  # reuse
        buf94 = reinterpret_tensor(buf44, (1, 2048, 7, 7), (100352, 1, 14336, 2048), 0); del buf44  # reuse
        # Topologically Sorted Source Nodes: [out_137, input_8, out_138, out_139, out_140], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu, aten.convolution]
        triton_poi_fused_add_convolution_cudnn_batch_norm_relu_18.run(buf93, arg227_1, arg228_1, arg229_1, arg230_1, buf92, arg232_1, arg233_1, arg234_1, arg235_1, buf94, 100352, grid=grid(100352), stream=stream0)
        del arg227_1
        del arg228_1
        del arg229_1
        del arg230_1
        del arg232_1
        del arg233_1
        del arg234_1
        del arg235_1
        del buf92
        buf95 = reinterpret_tensor(buf90, (49, 512), (512, 1), 0); del buf90  # reuse
        # Topologically Sorted Source Nodes: [out_139, out_140], Original ATen: [aten.relu, aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf94, (49, 2048), (2048, 1), 0), reinterpret_tensor(arg236_1, (2048, 512), (1, 2048), 0), out=buf95)
        del arg236_1
        buf96 = reinterpret_tensor(buf95, (1, 512, 7, 7), (25088, 1, 3584, 512), 0); del buf95  # reuse
        # Topologically Sorted Source Nodes: [out_141, out_142], Original ATen: [aten.cudnn_batch_norm, aten.relu]
        triton_poi_fused_convolution_cudnn_batch_norm_relu_17.run(buf96, arg237_1, arg238_1, arg239_1, arg240_1, 25088, grid=grid(25088), stream=stream0)
        del arg237_1
        del arg238_1
        del arg239_1
        del arg240_1
        # Topologically Sorted Source Nodes: [out_141, out_142, out_143], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        buf97 = extern_kernels.convolution(buf96, arg241_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (1, 512, 7, 7), (25088, 1, 3584, 512))
        del arg241_1
        del buf96
        buf98 = buf97; del buf97  # reuse
        # Topologically Sorted Source Nodes: [out_144, out_145, out_146], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        triton_poi_fused_convolution_cudnn_batch_norm_relu_17.run(buf98, arg242_1, arg243_1, arg244_1, arg245_1, 25088, grid=grid(25088), stream=stream0)
        del arg242_1
        del arg243_1
        del arg244_1
        del arg245_1
        buf99 = reinterpret_tensor(buf94, (49, 2048), (2048, 1), 0); del buf94  # reuse
        # Topologically Sorted Source Nodes: [out_144, out_145, out_146], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf98, (49, 512), (512, 1), 0), reinterpret_tensor(arg246_1, (512, 2048), (1, 512), 0), out=buf99)
        del arg246_1
        buf100 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [out_139, out_147, out_148, out_149], Original ATen: [aten.relu, aten.cudnn_batch_norm, aten.add]
        triton_poi_fused_add_cudnn_batch_norm_relu_19.run(buf100, buf99, arg247_1, arg248_1, arg249_1, arg250_1, 100352, grid=grid(100352), stream=stream0)
        del arg247_1
        del arg248_1
        del arg249_1
        del arg250_1
        buf101 = reinterpret_tensor(buf98, (49, 512), (512, 1), 0); del buf98  # reuse
        # Topologically Sorted Source Nodes: [out_150], Original ATen: [aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf100, (49, 2048), (2048, 1), 0), reinterpret_tensor(arg251_1, (2048, 512), (1, 2048), 0), out=buf101)
        del arg251_1
        buf102 = reinterpret_tensor(buf101, (1, 512, 7, 7), (25088, 1, 3584, 512), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [out_151, out_152], Original ATen: [aten.cudnn_batch_norm, aten.relu]
        triton_poi_fused_convolution_cudnn_batch_norm_relu_17.run(buf102, arg252_1, arg253_1, arg254_1, arg255_1, 25088, grid=grid(25088), stream=stream0)
        del arg252_1
        del arg253_1
        del arg254_1
        del arg255_1
        # Topologically Sorted Source Nodes: [out_151, out_152, out_153], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        buf103 = extern_kernels.convolution(buf102, arg256_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (1, 512, 7, 7), (25088, 1, 3584, 512))
        del arg256_1
        del buf102
        buf104 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [out_154, out_155, out_156], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        triton_poi_fused_convolution_cudnn_batch_norm_relu_17.run(buf104, arg257_1, arg258_1, arg259_1, arg260_1, 25088, grid=grid(25088), stream=stream0)
        del arg257_1
        del arg258_1
        del arg259_1
        del arg260_1
        buf105 = buf99; del buf99  # reuse
        # Topologically Sorted Source Nodes: [out_154, out_155, out_156], Original ATen: [aten.cudnn_batch_norm, aten.relu, aten.convolution]
        extern_kernels.mm(reinterpret_tensor(buf104, (49, 512), (512, 1), 0), reinterpret_tensor(arg261_1, (512, 2048), (1, 512), 0), out=buf105)
        del arg261_1
        del buf104
        buf106 = empty_strided_cuda((1, 2048, 1, 1), (2048, 1, 2048, 2048), torch.float32)
        buf107 = reinterpret_tensor(buf106, (1, 2048, 1, 1), (2048, 1, 1, 1), 0); del buf106  # reuse
        # Topologically Sorted Source Nodes: [out_157, out_158, out_159, x_4], Original ATen: [aten.cudnn_batch_norm, aten.add, aten.relu, aten.mean]
        triton_per_fused_add_cudnn_batch_norm_mean_relu_20.run(buf107, buf105, arg262_1, arg263_1, arg264_1, arg265_1, buf100, 2048, 49, grid=grid(2048), stream=stream0)
        del arg262_1
        del arg263_1
        del arg264_1
        del arg265_1
        del buf100
        del buf105
        buf108 = empty_strided_cuda((1, 1000), (1000, 1), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg267_1, (1, 1000), (0, 1), 0), reinterpret_tensor(buf107, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg266_1, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf108)
        del arg266_1
        del arg267_1
        del buf107
    return (buf108, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 7, 7), (147, 1, 21, 3), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((64, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((64, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((256, 64, 1, 1), (64, 1, 64, 64), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((128, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((512, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((128, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((128, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((128, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((512, 128, 1, 1), (128, 1, 128, 128), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((256, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((1024, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((1024, 256, 1, 1), (256, 1, 256, 256), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((2048, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((2048, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((2048, 512, 1, 1), (512, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)


# ===== inductor generated file: /tmp/torchinductor_kamei/tmp7zaxe2pd/25/c2565gcz6gg3zyvudxe4p3mrlzf424rlgczglgrt47lsrpw5nrf7.py =====

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


# ===== inductor generated file: /tmp/torchinductor_kamei/tmp7zaxe2pd/36/c36xq3mkl263lwbeqe655hilcx6snpob4sfq5lv5ii6wpjk742u5.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_cudnn_batch_norm_relu_8', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
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
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x2), tmp29, None)
    tl.store(out_ptr0 + (x2), tmp31, None)


# ===== inductor generated file: /tmp/torchinductor_kamei/tmp7zaxe2pd/3p/c3pqzkruzffy3jwvhl7giku5sun7y5rth3tyijp7q7tyoeo4qzc5.py =====

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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_cudnn_batch_norm_mean_relu_20', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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


# ===== inductor generated file: /tmp/torchinductor_kamei/tmp7zaxe2pd/4s/c4seuq74px4b6onmrfqfealyovhwnusllvikclz53ld56od2oiwi.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cudnn_batch_norm_relu_16', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, None)


# ===== inductor generated file: /tmp/torchinductor_kamei/tmp7zaxe2pd/4v/c4vrmqvevn6bagqj5teiftukd72pq4zdo22ykaar4mu4jsvew7ds.py =====

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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cudnn_batch_norm_relu_9', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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
    tmp17 = tl.full([1], 0, tl.int32)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tmp19 = tmp15 + tmp18
    tmp20 = triton_helpers.maximum(tmp17, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)


# ===== inductor generated file: /tmp/torchinductor_kamei/tmp7zaxe2pd/6t/c6t7n4xry6q2zxdoq4hniwurqcffvby2a7vzzi5gqwvqm5c64liz.py =====

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


# ===== inductor generated file: /tmp/torchinductor_kamei/tmp7zaxe2pd/6t/c6tlrbztbpbcjqh5ewgtq3bcgijdjzyu5h2ljhvo4mojsdeiafvf.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cudnn_batch_norm_relu_0', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, None)


# ===== inductor generated file: /tmp/torchinductor_kamei/tmp7zaxe2pd/a4/ca4llt5wfkqdyjisdxvmmny2pbxwylwye7weeayrvawz6r3ad275.py =====

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
    tmp17 = tl.full([1], 0, tl.int32)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tmp19 = tmp15 + tmp18
    tmp20 = triton_helpers.maximum(tmp17, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)


# ===== inductor generated file: /tmp/torchinductor_kamei/tmp7zaxe2pd/bo/cbog35jsbzryke5fflddtmivyc3n435in4jonoj6onobre2qncct.py =====

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
    tmp17 = tl.full([1], 0, tl.int32)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tmp19 = tmp15 + tmp18
    tmp20 = triton_helpers.maximum(tmp17, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)


# ===== inductor generated file: /tmp/torchinductor_kamei/tmp7zaxe2pd/c3/cc3yeklurc5cpt4hetkja2uc7u5zzye7nbzogfvzyda5mzi2uuvl.py =====

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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cudnn_batch_norm_relu_15', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


# ===== inductor generated file: /tmp/torchinductor_kamei/tmp7zaxe2pd/hn/chn57svt7o3ync3nccptfjk4xsno2vn4tm6uakihrkwtxqut6t4l.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_cudnn_batch_norm_relu_12', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)


# ===== inductor generated file: /tmp/torchinductor_kamei/tmp7zaxe2pd/il/cilepl63nzqv5c6sari7t5xtpkhv4gfgh6aju4isfedht2q2lxrm.py =====

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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cudnn_batch_norm_relu_19', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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
    tmp17 = tl.full([1], 0, tl.int32)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tmp19 = tmp15 + tmp18
    tmp20 = triton_helpers.maximum(tmp17, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)


# ===== inductor generated file: /tmp/torchinductor_kamei/tmp7zaxe2pd/jj/cjjb7d6rfh74sekobeww2mwt6qxbbz6vrylwa3i2ubgqgszlvirp.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_cudnn_batch_norm_relu_17', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)


# ===== inductor generated file: /tmp/torchinductor_kamei/tmp7zaxe2pd/m2/cm2rl7sliyqjx5o7fllsok6r6nc2o7isjxfa23qh2fdjuqsvychl.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_cudnn_batch_norm_relu_7', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, None)


# ===== inductor generated file: /tmp/torchinductor_kamei/tmp7zaxe2pd/nw/cnwvho6smkwe336lad27gvxyqqkzzfawo2omm2uto22ze7h2xhuz.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_cudnn_batch_norm_relu_18', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
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
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x2), tmp29, None)
    tl.store(out_ptr0 + (x2), tmp31, None)


# ===== inductor generated file: /tmp/torchinductor_kamei/tmp7zaxe2pd/rv/crv3uburz5u5q564q2ndo5wikyctx2txmqafwntcm5ctigggy6g4.py =====

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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cudnn_batch_norm_max_pool2d_with_indices_relu_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


# ===== inductor generated file: /tmp/torchinductor_kamei/tmp7zaxe2pd/ss/cssaqwcpd72z3jcnwd3t7s5yns5mpsupuqusicbrqbms3sssl232.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cudnn_batch_norm_relu_6', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, None)


# ===== inductor generated file: /tmp/torchinductor_kamei/tmp7zaxe2pd/tm/ctmkd7g5pune4vpksti47iyhkzczrkbrk5736djap5df447jy5dd.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cudnn_batch_norm_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, None)


# ===== inductor generated file: /tmp/torchinductor_kamei/tmp7zaxe2pd/x2/cx2jrpae4xobnunfmgh2gouqssunhj3hbdzzliemk455ftveg7f5.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_cudnn_batch_norm_relu_3', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
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
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x2), tmp29, None)
    tl.store(out_ptr0 + (x2), tmp31, None)


# ===== inductor generated file: /tmp/torchinductor_kamei/tmp7zaxe2pd/zi/czigjrwvlh5bxngc3kdt6dfhofenbk5xoexqmerkloo4aexnhvnp.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cudnn_batch_norm_relu_11', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x2), tmp17, None)


# ===== inductor generated file: /tmp/torchinductor_kamei/tmp7zaxe2pd/zn/cznqosyljq7ipxvwtmsegrbbdaxpkkmbme3g6tmbhwngkyv7lzdw.py =====

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, multi_processor_count=20), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_cudnn_batch_norm_relu_13', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'F0BD7791CFB36BEB63EC01B18B3FD72985501D0F79E89ED576CCD798F9404442', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
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
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x2), tmp29, None)
    tl.store(out_ptr0 + (x2), tmp31, None)
