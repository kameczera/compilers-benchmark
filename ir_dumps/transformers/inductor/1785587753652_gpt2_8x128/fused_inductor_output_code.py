# ===== inductor generated file: /tmp/cnnbench-transformers-p50cz5fm/repeat_02/a1/torchinductor/tmpxdyb3vm9/wq/cwqiouzl5rqqzecyxn3st4tw5voktz55qizk6dw2eebgohjbtoi5.debug/output_code.py =====
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


# kernel path: /tmp/cnnbench-transformers-p50cz5fm/repeat_02/a1/torchinductor/tmpxdyb3vm9/jq/cjqwss5atwrya52vg3gpmhy5m6itpshtoqaeg2ei6lm2otgmdhf5.py
# Topologically Sorted Source Nodes: [inputs_embeds, cache_position, position_ids, position_embeds, hidden_states, hidden_states_2], Original ATen: [aten.embedding, aten.arange, aten.unsqueeze, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   cache_position => iota
#   hidden_states => add
#   hidden_states_2 => add_1, add_2, mul, mul_1, rsqrt, sub, var_mean
#   inputs_embeds => embedding
#   position_embeds => embedding_1
#   position_ids => unsqueeze
# Graph fragment:
#   %arg1_1 : Tensor "i64[8, 128][128, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "f32[50257, 768][768, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg3_1 : Tensor "f32[1024, 768][768, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %getitem_1 : Tensor "f32[8, 128, 1][128, 1, 1024]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf1 : Tensor "f32[8, 128, 1][128, 1, 1024]cuda:0" = PlaceHolder[target=buf1]
#   %arg4_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %embedding : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %arg1_1), kwargs = {})
#   %iota : Tensor "i64[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze : Tensor "i64[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, 0), kwargs = {})
#   %embedding_1 : Tensor "f32[1, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %unsqueeze), kwargs = {})
#   %add : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %getitem_1), kwargs = {})
#   %add_1 : Tensor "f32[8, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[8, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg4_1), kwargs = {})
#   %add_2 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %arg5_1), kwargs = {})
#   return %getitem_1,%buf1,%add_2
triton_red_fused_add_arange_embedding_native_layer_norm_unsqueeze_0 = async_compile.triton('triton_red_fused_add_arange_embedding_native_layer_norm_unsqueeze_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_arange_embedding_native_layer_norm_unsqueeze_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_arange_embedding_native_layer_norm_unsqueeze_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 768
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    x0 = (xindex % 128)
    tmp10_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp10_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp10_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp7 = tl.load(in_ptr2 + (r0_2 + 768*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.full([XBLOCK, R0_BLOCK], 50257, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert(((0 <= tmp4) & (tmp4 < 50257)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 50257")
        tmp6 = tl.load(in_ptr1 + (r0_2 + 768*tmp4), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp10_mean_next, tmp10_m2_next, tmp10_weight_next = triton_helpers.welford_reduce(
            tmp9, tmp10_mean, tmp10_m2, tmp10_weight, roffset == 0
        )
        tmp10_mean = tl.where(r0_mask & xmask, tmp10_mean_next, tmp10_mean)
        tmp10_m2 = tl.where(r0_mask & xmask, tmp10_m2_next, tmp10_m2)
        tmp10_weight = tl.where(r0_mask & xmask, tmp10_weight_next, tmp10_weight)
    tmp11, tmp12, tmp13 = triton_helpers.welford(tmp10_mean, tmp10_m2, tmp10_weight, 1)
    tmp10 = tmp11[:, None]
    tmp14 = tmp12[:, None]
    tmp15 = tmp13[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp22 = tl.load(in_ptr2 + (r0_2 + 768*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp31 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp33 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.full([XBLOCK, R0_BLOCK], 50257, tl.int32)
        tmp17 = tmp0 + tmp16
        tmp18 = tmp0 < 0
        tmp19 = tl.where(tmp18, tmp17, tmp0)
        tl.device_assert(((0 <= tmp19) & (tmp19 < 50257)) | ~(xmask), "index out of bounds: 0 <= tmp19 < 50257")
        tmp21 = tl.load(in_ptr1 + (r0_2 + 768*tmp19), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp23 = tmp21 + tmp22
        tmp24 = tmp23 - tmp10
        tmp25 = 768.0
        tmp26 = (tmp14 / tmp25)
        tmp27 = 1e-05
        tmp28 = tmp26 + tmp27
        tmp29 = libdevice.rsqrt(tmp28)
        tmp30 = tmp24 * tmp29
        tmp32 = tmp30 * tmp31
        tmp34 = tmp32 + tmp33
        tl.store(out_ptr2 + (r0_2 + 768*x3), tmp34, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-transformers-p50cz5fm/repeat_02/a1/torchinductor/tmpxdyb3vm9/4h/c4hbd3bs4g2qthy7zrebrqivychrqd2oziojglrfg2tuj2xcnsvr.py
# Topologically Sorted Source Nodes: [inputs_embeds, cache_position, position_ids, position_embeds, hidden_states, x_2, x_3, hidden_states_3, hidden_states_4], Original ATen: [aten.embedding, aten.arange, aten.unsqueeze, aten.add, aten.addmm, aten.view, aten.native_layer_norm]
# Source node to ATen node mapping:
#   cache_position => iota
#   hidden_states => add
#   hidden_states_3 => add_3
#   hidden_states_4 => add_4, add_5, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
#   inputs_embeds => embedding
#   position_embeds => embedding_1
#   position_ids => unsqueeze
#   x_2 => add_tensor_35
#   x_3 => view_8
# Graph fragment:
#   %mm_default_35 : Tensor "f32[1024, 768][768, 1]cuda:0" = PlaceHolder[target=mm_default_35]
#   %arg8_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %arg1_1 : Tensor "i64[8, 128][128, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "f32[50257, 768][768, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg3_1 : Tensor "f32[1024, 768][768, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %add_3 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0" = PlaceHolder[target=add_3]
#   %getitem_10 : Tensor "f32[8, 128, 1][128, 1, 1024]cuda:0" = PlaceHolder[target=getitem_10]
#   %buf13 : Tensor "f32[8, 128, 1][128, 1, 1024]cuda:0" = PlaceHolder[target=buf13]
#   %arg10_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg10_1]
#   %arg11_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg11_1]
#   %embedding : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %arg1_1), kwargs = {})
#   %iota : Tensor "i64[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze : Tensor "i64[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, 0), kwargs = {})
#   %embedding_1 : Tensor "f32[1, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %unsqueeze), kwargs = {})
#   %add : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %add_tensor_35 : Tensor "f32[1024, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_35, %arg8_1), kwargs = {})
#   %view_8 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_35, [8, 128, 768]), kwargs = {})
#   %add_3 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_8, %add), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_3, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %getitem_10), kwargs = {})
#   %add_4 : Tensor "f32[8, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_9, 1e-05), kwargs = {})
#   %rsqrt_1 : Tensor "f32[8, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %mul_2 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_3 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %arg10_1), kwargs = {})
#   %add_5 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %arg11_1), kwargs = {})
#   return %add_3,%getitem_10,%buf13,%add_5
triton_per_fused_add_addmm_arange_embedding_native_layer_norm_unsqueeze_view_1 = async_compile.triton('triton_per_fused_add_addmm_arange_embedding_native_layer_norm_unsqueeze_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_arange_embedding_native_layer_norm_unsqueeze_view_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 6, 'num_reduction': 4, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_addmm_arange_embedding_native_layer_norm_unsqueeze_view_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
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
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (r0_2 + 768*x3), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (r0_2 + 768*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr5 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tl.full([XBLOCK, R0_BLOCK], 50257, tl.int32)
    tmp5 = tmp3 + tmp4
    tmp6 = tmp3 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp3)
    tl.device_assert(((0 <= tmp7) & (tmp7 < 50257)) | ~(xmask), "index out of bounds: 0 <= tmp7 < 50257")
    tmp9 = tl.load(in_ptr2 + (r0_2 + 768*tmp7), r0_mask & xmask, other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tmp2 + tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(r0_mask & xmask, tmp13, 0)
    tmp16 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp18 = tl.where(r0_mask & xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None].to(tl.float32)
    tmp20 = tl.full([XBLOCK, 1], 768, tl.int32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = (tmp19 / tmp21)
    tmp23 = tmp13 - tmp22
    tmp24 = tmp23 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
    tmp27 = tl.where(r0_mask & xmask, tmp25, 0)
    tmp28 = tl.sum(tmp27, 1)[:, None].to(tl.float32)
    tmp29 = tmp12 - tmp22
    tmp30 = 768.0
    tmp31 = (tmp28 / tmp30)
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.rsqrt(tmp33)
    tmp35 = tmp29 * tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp37 + tmp38
    tl.store(in_out_ptr0 + (r0_2 + 768*x3), tmp12, r0_mask & xmask)
    tl.store(out_ptr2 + (r0_2 + 768*x3), tmp39, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-transformers-p50cz5fm/repeat_02/a1/torchinductor/tmpxdyb3vm9/ea/ceaoz3uu6u24l5sygramjdrcfjvvi5huw4pndtejhhfpig2qrkiv.py
# Topologically Sorted Source Nodes: [x_4, x_5, mul, pow_1, mul_1, add_2, mul_2, tanh, add_3, hidden_states_5], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
# Source node to ATen node mapping:
#   add_2 => add_6
#   add_3 => add_7
#   hidden_states_5 => mul_7
#   mul => mul_4
#   mul_1 => mul_5
#   mul_2 => mul_6
#   pow_1 => pow_1
#   tanh => tanh
#   x_4 => add_tensor_34
#   x_5 => view_10
# Graph fragment:
#   %mm_default_34 : Tensor "f32[1024, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_default_34]
#   %arg12_1 : Tensor "f32[3072][1]cuda:0" = PlaceHolder[target=arg12_1]
#   %add_tensor_34 : Tensor "f32[1024, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_34, %arg12_1), kwargs = {})
#   %view_10 : Tensor "f32[8, 128, 3072][393216, 3072, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_34, [8, 128, 3072]), kwargs = {})
#   %mul_4 : Tensor "f32[8, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_10, 0.5), kwargs = {})
#   %pow_1 : Tensor "f32[8, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view_10, 3.0), kwargs = {})
#   %mul_5 : Tensor "f32[8, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_1, 0.044715), kwargs = {})
#   %add_6 : Tensor "f32[8, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_10, %mul_5), kwargs = {})
#   %mul_6 : Tensor "f32[8, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, 0.7978845608028654), kwargs = {})
#   %tanh : Tensor "f32[8, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_6,), kwargs = {})
#   %add_7 : Tensor "f32[8, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tanh, 1.0), kwargs = {})
#   %mul_7 : Tensor "f32[8, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %add_7), kwargs = {})
#   return %mul_7
triton_poi_fused_add_addmm_mul_pow_tanh_view_2 = async_compile.triton('triton_poi_fused_add_addmm_mul_pow_tanh_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_mul_pow_tanh_view_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 37761024}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_mul_pow_tanh_view_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 3072)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = tmp2 * tmp2
    tmp6 = tmp5 * tmp2
    tmp7 = 0.044715
    tmp8 = tmp6 * tmp7
    tmp9 = tmp2 + tmp8
    tmp10 = 0.7978845608028654
    tmp11 = tmp9 * tmp10
    tmp12 = libdevice.tanh(tmp11)
    tmp13 = 1.0
    tmp14 = tmp12 + tmp13
    tmp15 = tmp4 * tmp14
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-transformers-p50cz5fm/repeat_02/a1/torchinductor/tmpxdyb3vm9/wz/cwznbdrhxienrtupkya6knusvpneohiudyrpaybqvypvjpmv4hgq.py
# Topologically Sorted Source Nodes: [x_6, x_7, hidden_states_7, hidden_states_8], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_7 => add_8
#   hidden_states_8 => add_10, add_9, mul_8, mul_9, rsqrt_2, sub_2, var_mean_2
#   x_6 => add_tensor_33
#   x_7 => view_12
# Graph fragment:
#   %add_3 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0" = PlaceHolder[target=add_3]
#   %mm_default_33 : Tensor "f32[1024, 768][768, 1]cuda:0" = PlaceHolder[target=mm_default_33]
#   %arg14_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg14_1]
#   %getitem_12 : Tensor "f32[8, 128, 1][128, 1, 1024]cuda:0" = PlaceHolder[target=getitem_12]
#   %buf20 : Tensor "f32[8, 128, 1][128, 1, 1024]cuda:0" = PlaceHolder[target=buf20]
#   %arg16_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg16_1]
#   %arg17_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg17_1]
#   %add_tensor_33 : Tensor "f32[1024, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_33, %arg14_1), kwargs = {})
#   %view_12 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_33, [8, 128, 768]), kwargs = {})
#   %add_8 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %view_12), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_8, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_8, %getitem_12), kwargs = {})
#   %add_9 : Tensor "f32[8, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_11, 1e-05), kwargs = {})
#   %rsqrt_2 : Tensor "f32[8, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_8 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_2), kwargs = {})
#   %mul_9 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %arg16_1), kwargs = {})
#   %add_10 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %arg17_1), kwargs = {})
#   return %getitem_12,%buf20,%add_10
triton_per_fused_add_addmm_native_layer_norm_view_3 = async_compile.triton('triton_per_fused_add_addmm_native_layer_norm_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_native_layer_norm_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 12592128}}
)
@triton.jit
def triton_per_fused_add_addmm_native_layer_norm_view_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp12 = tl.full([XBLOCK, 1], 768, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = (tmp11 / tmp13)
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(r0_mask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None].to(tl.float32)
    tmp21 = tmp4 - tmp14
    tmp22 = 768.0
    tmp23 = (tmp20 / tmp22)
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp31, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-transformers-p50cz5fm/repeat_02/a1/torchinductor/tmpxdyb3vm9/d6/cd6ybkiu75diiuhq75l2mwfs6w2h4zixpdedql34g3irkcc6jcqc.py
# Topologically Sorted Source Nodes: [x_6, x_7, hidden_states_7, x_10, x_11, hidden_states_9, hidden_states_10], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_10 => add_12, add_13, mul_10, mul_11, rsqrt_3, sub_3, var_mean_3
#   hidden_states_7 => add_8
#   hidden_states_9 => add_11
#   x_10 => add_tensor_32
#   x_11 => view_20
#   x_6 => add_tensor_33
#   x_7 => view_12
# Graph fragment:
#   %mm_default_32 : Tensor "f32[1024, 768][768, 1]cuda:0" = PlaceHolder[target=mm_default_32]
#   %arg20_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg20_1]
#   %add_3 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0" = PlaceHolder[target=add_3]
#   %mm_default_33 : Tensor "f32[1024, 768][768, 1]cuda:0" = PlaceHolder[target=mm_default_33]
#   %arg14_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg14_1]
#   %add_11 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0" = PlaceHolder[target=add_11]
#   %getitem_21 : Tensor "f32[8, 128, 1][128, 1, 1024]cuda:0" = PlaceHolder[target=getitem_21]
#   %buf32 : Tensor "f32[8, 128, 1][128, 1, 1024]cuda:0" = PlaceHolder[target=buf32]
#   %arg22_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg22_1]
#   %arg23_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg23_1]
#   %add_tensor_33 : Tensor "f32[1024, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_33, %arg14_1), kwargs = {})
#   %view_12 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_33, [8, 128, 768]), kwargs = {})
#   %add_8 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %view_12), kwargs = {})
#   %add_tensor_32 : Tensor "f32[1024, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_32, %arg20_1), kwargs = {})
#   %view_20 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_32, [8, 128, 768]), kwargs = {})
#   %add_11 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_20, %add_8), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_11, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_11, %getitem_21), kwargs = {})
#   %add_12 : Tensor "f32[8, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_20, 1e-05), kwargs = {})
#   %rsqrt_3 : Tensor "f32[8, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_12,), kwargs = {})
#   %mul_10 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_3), kwargs = {})
#   %mul_11 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %arg22_1), kwargs = {})
#   %add_13 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %arg23_1), kwargs = {})
#   return %add_11,%getitem_21,%buf32,%add_13
triton_per_fused_add_addmm_native_layer_norm_view_4 = async_compile.triton('triton_per_fused_add_addmm_native_layer_norm_view_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_native_layer_norm_view_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 22032384}}
)
@triton.jit
def triton_per_fused_add_addmm_native_layer_norm_view_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
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
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp6 = tmp4 + tmp5
    tmp7 = tmp3 + tmp6
    tmp8 = tmp2 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.where(r0_mask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp14 = tl.where(r0_mask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None].to(tl.float32)
    tmp16 = tl.full([XBLOCK, 1], 768, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = (tmp15 / tmp17)
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
    tmp23 = tl.where(r0_mask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None].to(tl.float32)
    tmp25 = tmp8 - tmp18
    tmp26 = 768.0
    tmp27 = (tmp24 / tmp26)
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp8, r0_mask & xmask)
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp35, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-transformers-p50cz5fm/repeat_02/a1/torchinductor/tmpxdyb3vm9/bo/cbogixtlgmsshy55xfrz275tizvetdt26ih37appoc6jgr7ajyk3.py
# Topologically Sorted Source Nodes: [x_94, x_95, hidden_states_73, hidden_states_74], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_73 => add_96
#   hidden_states_74 => add_97, add_98, mul_96, mul_97, rsqrt_24, sub_24, var_mean_24
#   x_94 => add_tensor
#   x_95 => view_144
# Graph fragment:
#   %add_91 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0" = PlaceHolder[target=add_91]
#   %mm_default : Tensor "f32[1024, 768][768, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %arg146_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg146_1]
#   %getitem_133 : Tensor "f32[8, 128, 1][128, 1, 1024]cuda:0" = PlaceHolder[target=getitem_133]
#   %buf229 : Tensor "f32[8, 128, 1][128, 1, 1024]cuda:0" = PlaceHolder[target=buf229]
#   %arg148_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg148_1]
#   %arg149_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg149_1]
#   %add_tensor : Tensor "f32[1024, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %arg146_1), kwargs = {})
#   %view_144 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor, [8, 128, 768]), kwargs = {})
#   %add_96 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_91, %view_144), kwargs = {})
#   %var_mean_24 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_96, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_24 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_96, %getitem_133), kwargs = {})
#   %add_97 : Tensor "f32[8, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_132, 1e-05), kwargs = {})
#   %rsqrt_24 : Tensor "f32[8, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_97,), kwargs = {})
#   %mul_96 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %rsqrt_24), kwargs = {})
#   %mul_97 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_96, %arg148_1), kwargs = {})
#   %add_98 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_97, %arg149_1), kwargs = {})
#   return %getitem_133,%buf229,%add_98
triton_per_fused_add_addmm_native_layer_norm_view_5 = async_compile.triton('triton_per_fused_add_addmm_native_layer_norm_view_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_native_layer_norm_view_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 12592128}}
)
@triton.jit
def triton_per_fused_add_addmm_native_layer_norm_view_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
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
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp12 = tl.full([XBLOCK, 1], 768, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = (tmp11 / tmp13)
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(r0_mask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None].to(tl.float32)
    tmp21 = tmp4 - tmp14
    tmp22 = 768.0
    tmp23 = (tmp20 / tmp22)
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp31, r0_mask & xmask)
''', device_str='cuda')


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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1 = args
        args.clear()
        assert_size_stride(arg0_1, (1, 1, 128, 128), (16384, 16384, 128, 1))
        assert_size_stride(arg1_1, (8, 128), (128, 1))
        assert_size_stride(arg2_1, (50257, 768), (768, 1))
        assert_size_stride(arg3_1, (1024, 768), (768, 1))
        assert_size_stride(arg4_1, (768, ), (1, ))
        assert_size_stride(arg5_1, (768, ), (1, ))
        assert_size_stride(arg6_1, (2304, ), (1, ))
        assert_size_stride(arg7_1, (768, 2304), (2304, 1))
        assert_size_stride(arg8_1, (768, ), (1, ))
        assert_size_stride(arg9_1, (768, 768), (768, 1))
        assert_size_stride(arg10_1, (768, ), (1, ))
        assert_size_stride(arg11_1, (768, ), (1, ))
        assert_size_stride(arg12_1, (3072, ), (1, ))
        assert_size_stride(arg13_1, (768, 3072), (3072, 1))
        assert_size_stride(arg14_1, (768, ), (1, ))
        assert_size_stride(arg15_1, (3072, 768), (768, 1))
        assert_size_stride(arg16_1, (768, ), (1, ))
        assert_size_stride(arg17_1, (768, ), (1, ))
        assert_size_stride(arg18_1, (2304, ), (1, ))
        assert_size_stride(arg19_1, (768, 2304), (2304, 1))
        assert_size_stride(arg20_1, (768, ), (1, ))
        assert_size_stride(arg21_1, (768, 768), (768, 1))
        assert_size_stride(arg22_1, (768, ), (1, ))
        assert_size_stride(arg23_1, (768, ), (1, ))
        assert_size_stride(arg24_1, (3072, ), (1, ))
        assert_size_stride(arg25_1, (768, 3072), (3072, 1))
        assert_size_stride(arg26_1, (768, ), (1, ))
        assert_size_stride(arg27_1, (3072, 768), (768, 1))
        assert_size_stride(arg28_1, (768, ), (1, ))
        assert_size_stride(arg29_1, (768, ), (1, ))
        assert_size_stride(arg30_1, (2304, ), (1, ))
        assert_size_stride(arg31_1, (768, 2304), (2304, 1))
        assert_size_stride(arg32_1, (768, ), (1, ))
        assert_size_stride(arg33_1, (768, 768), (768, 1))
        assert_size_stride(arg34_1, (768, ), (1, ))
        assert_size_stride(arg35_1, (768, ), (1, ))
        assert_size_stride(arg36_1, (3072, ), (1, ))
        assert_size_stride(arg37_1, (768, 3072), (3072, 1))
        assert_size_stride(arg38_1, (768, ), (1, ))
        assert_size_stride(arg39_1, (3072, 768), (768, 1))
        assert_size_stride(arg40_1, (768, ), (1, ))
        assert_size_stride(arg41_1, (768, ), (1, ))
        assert_size_stride(arg42_1, (2304, ), (1, ))
        assert_size_stride(arg43_1, (768, 2304), (2304, 1))
        assert_size_stride(arg44_1, (768, ), (1, ))
        assert_size_stride(arg45_1, (768, 768), (768, 1))
        assert_size_stride(arg46_1, (768, ), (1, ))
        assert_size_stride(arg47_1, (768, ), (1, ))
        assert_size_stride(arg48_1, (3072, ), (1, ))
        assert_size_stride(arg49_1, (768, 3072), (3072, 1))
        assert_size_stride(arg50_1, (768, ), (1, ))
        assert_size_stride(arg51_1, (3072, 768), (768, 1))
        assert_size_stride(arg52_1, (768, ), (1, ))
        assert_size_stride(arg53_1, (768, ), (1, ))
        assert_size_stride(arg54_1, (2304, ), (1, ))
        assert_size_stride(arg55_1, (768, 2304), (2304, 1))
        assert_size_stride(arg56_1, (768, ), (1, ))
        assert_size_stride(arg57_1, (768, 768), (768, 1))
        assert_size_stride(arg58_1, (768, ), (1, ))
        assert_size_stride(arg59_1, (768, ), (1, ))
        assert_size_stride(arg60_1, (3072, ), (1, ))
        assert_size_stride(arg61_1, (768, 3072), (3072, 1))
        assert_size_stride(arg62_1, (768, ), (1, ))
        assert_size_stride(arg63_1, (3072, 768), (768, 1))
        assert_size_stride(arg64_1, (768, ), (1, ))
        assert_size_stride(arg65_1, (768, ), (1, ))
        assert_size_stride(arg66_1, (2304, ), (1, ))
        assert_size_stride(arg67_1, (768, 2304), (2304, 1))
        assert_size_stride(arg68_1, (768, ), (1, ))
        assert_size_stride(arg69_1, (768, 768), (768, 1))
        assert_size_stride(arg70_1, (768, ), (1, ))
        assert_size_stride(arg71_1, (768, ), (1, ))
        assert_size_stride(arg72_1, (3072, ), (1, ))
        assert_size_stride(arg73_1, (768, 3072), (3072, 1))
        assert_size_stride(arg74_1, (768, ), (1, ))
        assert_size_stride(arg75_1, (3072, 768), (768, 1))
        assert_size_stride(arg76_1, (768, ), (1, ))
        assert_size_stride(arg77_1, (768, ), (1, ))
        assert_size_stride(arg78_1, (2304, ), (1, ))
        assert_size_stride(arg79_1, (768, 2304), (2304, 1))
        assert_size_stride(arg80_1, (768, ), (1, ))
        assert_size_stride(arg81_1, (768, 768), (768, 1))
        assert_size_stride(arg82_1, (768, ), (1, ))
        assert_size_stride(arg83_1, (768, ), (1, ))
        assert_size_stride(arg84_1, (3072, ), (1, ))
        assert_size_stride(arg85_1, (768, 3072), (3072, 1))
        assert_size_stride(arg86_1, (768, ), (1, ))
        assert_size_stride(arg87_1, (3072, 768), (768, 1))
        assert_size_stride(arg88_1, (768, ), (1, ))
        assert_size_stride(arg89_1, (768, ), (1, ))
        assert_size_stride(arg90_1, (2304, ), (1, ))
        assert_size_stride(arg91_1, (768, 2304), (2304, 1))
        assert_size_stride(arg92_1, (768, ), (1, ))
        assert_size_stride(arg93_1, (768, 768), (768, 1))
        assert_size_stride(arg94_1, (768, ), (1, ))
        assert_size_stride(arg95_1, (768, ), (1, ))
        assert_size_stride(arg96_1, (3072, ), (1, ))
        assert_size_stride(arg97_1, (768, 3072), (3072, 1))
        assert_size_stride(arg98_1, (768, ), (1, ))
        assert_size_stride(arg99_1, (3072, 768), (768, 1))
        assert_size_stride(arg100_1, (768, ), (1, ))
        assert_size_stride(arg101_1, (768, ), (1, ))
        assert_size_stride(arg102_1, (2304, ), (1, ))
        assert_size_stride(arg103_1, (768, 2304), (2304, 1))
        assert_size_stride(arg104_1, (768, ), (1, ))
        assert_size_stride(arg105_1, (768, 768), (768, 1))
        assert_size_stride(arg106_1, (768, ), (1, ))
        assert_size_stride(arg107_1, (768, ), (1, ))
        assert_size_stride(arg108_1, (3072, ), (1, ))
        assert_size_stride(arg109_1, (768, 3072), (3072, 1))
        assert_size_stride(arg110_1, (768, ), (1, ))
        assert_size_stride(arg111_1, (3072, 768), (768, 1))
        assert_size_stride(arg112_1, (768, ), (1, ))
        assert_size_stride(arg113_1, (768, ), (1, ))
        assert_size_stride(arg114_1, (2304, ), (1, ))
        assert_size_stride(arg115_1, (768, 2304), (2304, 1))
        assert_size_stride(arg116_1, (768, ), (1, ))
        assert_size_stride(arg117_1, (768, 768), (768, 1))
        assert_size_stride(arg118_1, (768, ), (1, ))
        assert_size_stride(arg119_1, (768, ), (1, ))
        assert_size_stride(arg120_1, (3072, ), (1, ))
        assert_size_stride(arg121_1, (768, 3072), (3072, 1))
        assert_size_stride(arg122_1, (768, ), (1, ))
        assert_size_stride(arg123_1, (3072, 768), (768, 1))
        assert_size_stride(arg124_1, (768, ), (1, ))
        assert_size_stride(arg125_1, (768, ), (1, ))
        assert_size_stride(arg126_1, (2304, ), (1, ))
        assert_size_stride(arg127_1, (768, 2304), (2304, 1))
        assert_size_stride(arg128_1, (768, ), (1, ))
        assert_size_stride(arg129_1, (768, 768), (768, 1))
        assert_size_stride(arg130_1, (768, ), (1, ))
        assert_size_stride(arg131_1, (768, ), (1, ))
        assert_size_stride(arg132_1, (3072, ), (1, ))
        assert_size_stride(arg133_1, (768, 3072), (3072, 1))
        assert_size_stride(arg134_1, (768, ), (1, ))
        assert_size_stride(arg135_1, (3072, 768), (768, 1))
        assert_size_stride(arg136_1, (768, ), (1, ))
        assert_size_stride(arg137_1, (768, ), (1, ))
        assert_size_stride(arg138_1, (2304, ), (1, ))
        assert_size_stride(arg139_1, (768, 2304), (2304, 1))
        assert_size_stride(arg140_1, (768, ), (1, ))
        assert_size_stride(arg141_1, (768, 768), (768, 1))
        assert_size_stride(arg142_1, (768, ), (1, ))
        assert_size_stride(arg143_1, (768, ), (1, ))
        assert_size_stride(arg144_1, (3072, ), (1, ))
        assert_size_stride(arg145_1, (768, 3072), (3072, 1))
        assert_size_stride(arg146_1, (768, ), (1, ))
        assert_size_stride(arg147_1, (3072, 768), (768, 1))
        assert_size_stride(arg148_1, (768, ), (1, ))
        assert_size_stride(arg149_1, (768, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf3 = empty_strided_cuda((8, 128, 768), (98304, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [inputs_embeds, cache_position, position_ids, position_embeds, hidden_states, hidden_states_2], Original ATen: [aten.embedding, aten.arange, aten.unsqueeze, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_arange_embedding_native_layer_norm_unsqueeze_0.run(arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, buf3, 1024, 768, stream=stream0)
            del arg4_1
            del arg5_1
            buf4 = empty_strided_cuda((1024, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [inputs_embeds, cache_position, position_ids, position_embeds, hidden_states, hidden_states_2, view_1, x], Original ATen: [aten.embedding, aten.arange, aten.unsqueeze, aten.add, aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.addmm(arg6_1, reinterpret_tensor(buf3, (1024, 768), (768, 1), 0), arg7_1, alpha=1, beta=1, out=buf4)
            del arg6_1
            del arg7_1
            # Topologically Sorted Source Nodes: [x_1, split, view_5, query_states_1, view_3, key_states_1, view_4, value_states_1, attn_output], Original ATen: [aten.view, aten.split, aten.transpose, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf4, (8, 12, 128, 64), (294912, 64, 2304, 1), 0), reinterpret_tensor(buf4, (8, 12, 128, 64), (294912, 64, 2304, 1), 768), reinterpret_tensor(buf4, (8, 12, 128, 64), (294912, 64, 2304, 1), 1536), reinterpret_tensor(arg0_1, (8, 12, 128, 128), (0, 0, 128, 1), 0), False)
            del buf4
            buf6 = buf5[0]
            assert_size_stride(buf6, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf6, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf5
            buf10 = reinterpret_tensor(buf3, (1024, 768), (768, 1), 0); del buf3  # reuse
            # Topologically Sorted Source Nodes: [transpose_3, reshape, view_6, x_2], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf6, (1024, 768), (768, 1), 0), arg9_1, out=buf10)
            del arg9_1
            buf11 = reinterpret_tensor(buf10, (8, 128, 768), (98304, 768, 1), 0); del buf10  # reuse
            buf15 = reinterpret_tensor(buf6, (8, 128, 768), (98304, 768, 1), 0); del buf6  # reuse
            # Topologically Sorted Source Nodes: [inputs_embeds, cache_position, position_ids, position_embeds, hidden_states, x_2, x_3, hidden_states_3, hidden_states_4], Original ATen: [aten.embedding, aten.arange, aten.unsqueeze, aten.add, aten.addmm, aten.view, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_arange_embedding_native_layer_norm_unsqueeze_view_1.run(buf11, arg8_1, arg1_1, arg2_1, arg3_1, arg10_1, arg11_1, buf15, 1024, 768, stream=stream0)
            del arg10_1
            del arg11_1
            del arg1_1
            del arg2_1
            del arg3_1
            del arg8_1
            buf16 = empty_strided_cuda((1024, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_4, view_8, x_4], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf15, (1024, 768), (768, 1), 0), arg13_1, out=buf16)
            del arg13_1
            del buf15
            buf17 = reinterpret_tensor(buf16, (8, 128, 3072), (393216, 3072, 1), 0); del buf16  # reuse
            # Topologically Sorted Source Nodes: [x_4, x_5, mul, pow_1, mul_1, add_2, mul_2, tanh, add_3, hidden_states_5], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_2.run(buf17, arg12_1, 3145728, stream=stream0)
            del arg12_1
            buf18 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_4, x_5, mul, pow_1, mul_1, add_2, mul_2, tanh, add_3, hidden_states_5, view_10, x_6], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf17, (1024, 3072), (3072, 1), 0), arg15_1, out=buf18)
            del arg15_1
            del buf17
            buf22 = empty_strided_cuda((8, 128, 768), (98304, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_6, x_7, hidden_states_7, hidden_states_8], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_3.run(buf11, buf18, arg14_1, arg16_1, arg17_1, buf22, 1024, 768, stream=stream0)
            del arg16_1
            del arg17_1
            buf23 = empty_strided_cuda((1024, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_6, x_7, hidden_states_7, hidden_states_8, view_12, x_8], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg18_1, reinterpret_tensor(buf22, (1024, 768), (768, 1), 0), arg19_1, alpha=1, beta=1, out=buf23)
            del arg18_1
            del arg19_1
            # Topologically Sorted Source Nodes: [x_9, split_1, view_16, query_states_3, view_14, key_states_3, view_15, value_states_3, attn_output_4], Original ATen: [aten.view, aten.split, aten.transpose, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf24 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf23, (8, 12, 128, 64), (294912, 64, 2304, 1), 0), reinterpret_tensor(buf23, (8, 12, 128, 64), (294912, 64, 2304, 1), 768), reinterpret_tensor(buf23, (8, 12, 128, 64), (294912, 64, 2304, 1), 1536), reinterpret_tensor(arg0_1, (8, 12, 128, 128), (0, 0, 128, 1), 0), False)
            del buf23
            buf25 = buf24[0]
            assert_size_stride(buf25, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf25, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf24
            buf29 = reinterpret_tensor(buf22, (1024, 768), (768, 1), 0); del buf22  # reuse
            # Topologically Sorted Source Nodes: [transpose_7, reshape_1, view_17, x_10], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf25, (1024, 768), (768, 1), 0), arg21_1, out=buf29)
            del arg21_1
            buf30 = reinterpret_tensor(buf29, (8, 128, 768), (98304, 768, 1), 0); del buf29  # reuse
            buf34 = reinterpret_tensor(buf25, (8, 128, 768), (98304, 768, 1), 0); del buf25  # reuse
            # Topologically Sorted Source Nodes: [x_6, x_7, hidden_states_7, x_10, x_11, hidden_states_9, hidden_states_10], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf30, arg20_1, buf11, buf18, arg14_1, arg22_1, arg23_1, buf34, 1024, 768, stream=stream0)
            del arg14_1
            del arg20_1
            del arg22_1
            del arg23_1
            del buf11
            del buf18
            buf35 = empty_strided_cuda((1024, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_10, view_19, x_12], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf34, (1024, 768), (768, 1), 0), arg25_1, out=buf35)
            del arg25_1
            del buf34
            buf36 = reinterpret_tensor(buf35, (8, 128, 3072), (393216, 3072, 1), 0); del buf35  # reuse
            # Topologically Sorted Source Nodes: [x_12, x_13, mul_4, pow_2, mul_5, add_6, mul_6, tanh_1, add_7, hidden_states_11], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_2.run(buf36, arg24_1, 3145728, stream=stream0)
            del arg24_1
            buf37 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_12, x_13, mul_4, pow_2, mul_5, add_6, mul_6, tanh_1, add_7, hidden_states_11, view_21, x_14], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf36, (1024, 3072), (3072, 1), 0), arg27_1, out=buf37)
            del arg27_1
            del buf36
            buf41 = empty_strided_cuda((8, 128, 768), (98304, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_14, x_15, hidden_states_13, hidden_states_14], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_3.run(buf30, buf37, arg26_1, arg28_1, arg29_1, buf41, 1024, 768, stream=stream0)
            del arg28_1
            del arg29_1
            buf42 = empty_strided_cuda((1024, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_14, x_15, hidden_states_13, hidden_states_14, view_23, x_16], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg30_1, reinterpret_tensor(buf41, (1024, 768), (768, 1), 0), arg31_1, alpha=1, beta=1, out=buf42)
            del arg30_1
            del arg31_1
            # Topologically Sorted Source Nodes: [x_17, split_2, view_27, query_states_5, view_25, key_states_5, view_26, value_states_5, attn_output_8], Original ATen: [aten.view, aten.split, aten.transpose, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf43 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf42, (8, 12, 128, 64), (294912, 64, 2304, 1), 0), reinterpret_tensor(buf42, (8, 12, 128, 64), (294912, 64, 2304, 1), 768), reinterpret_tensor(buf42, (8, 12, 128, 64), (294912, 64, 2304, 1), 1536), reinterpret_tensor(arg0_1, (8, 12, 128, 128), (0, 0, 128, 1), 0), False)
            del buf42
            buf44 = buf43[0]
            assert_size_stride(buf44, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf44, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf43
            buf48 = reinterpret_tensor(buf41, (1024, 768), (768, 1), 0); del buf41  # reuse
            # Topologically Sorted Source Nodes: [transpose_11, reshape_2, view_28, x_18], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf44, (1024, 768), (768, 1), 0), arg33_1, out=buf48)
            del arg33_1
            buf49 = reinterpret_tensor(buf48, (8, 128, 768), (98304, 768, 1), 0); del buf48  # reuse
            buf53 = reinterpret_tensor(buf44, (8, 128, 768), (98304, 768, 1), 0); del buf44  # reuse
            # Topologically Sorted Source Nodes: [x_14, x_15, hidden_states_13, x_18, x_19, hidden_states_15, hidden_states_16], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf49, arg32_1, buf30, buf37, arg26_1, arg34_1, arg35_1, buf53, 1024, 768, stream=stream0)
            del arg26_1
            del arg32_1
            del arg34_1
            del arg35_1
            del buf30
            del buf37
            buf54 = empty_strided_cuda((1024, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_16, view_30, x_20], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf53, (1024, 768), (768, 1), 0), arg37_1, out=buf54)
            del arg37_1
            del buf53
            buf55 = reinterpret_tensor(buf54, (8, 128, 3072), (393216, 3072, 1), 0); del buf54  # reuse
            # Topologically Sorted Source Nodes: [x_20, x_21, mul_8, pow_3, mul_9, add_10, mul_10, tanh_2, add_11, hidden_states_17], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_2.run(buf55, arg36_1, 3145728, stream=stream0)
            del arg36_1
            buf56 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_20, x_21, mul_8, pow_3, mul_9, add_10, mul_10, tanh_2, add_11, hidden_states_17, view_32, x_22], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf55, (1024, 3072), (3072, 1), 0), arg39_1, out=buf56)
            del arg39_1
            del buf55
            buf60 = empty_strided_cuda((8, 128, 768), (98304, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_22, x_23, hidden_states_19, hidden_states_20], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_3.run(buf49, buf56, arg38_1, arg40_1, arg41_1, buf60, 1024, 768, stream=stream0)
            del arg40_1
            del arg41_1
            buf61 = empty_strided_cuda((1024, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_22, x_23, hidden_states_19, hidden_states_20, view_34, x_24], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg42_1, reinterpret_tensor(buf60, (1024, 768), (768, 1), 0), arg43_1, alpha=1, beta=1, out=buf61)
            del arg42_1
            del arg43_1
            # Topologically Sorted Source Nodes: [x_25, split_3, view_38, query_states_7, view_36, key_states_7, view_37, value_states_7, attn_output_12], Original ATen: [aten.view, aten.split, aten.transpose, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf62 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf61, (8, 12, 128, 64), (294912, 64, 2304, 1), 0), reinterpret_tensor(buf61, (8, 12, 128, 64), (294912, 64, 2304, 1), 768), reinterpret_tensor(buf61, (8, 12, 128, 64), (294912, 64, 2304, 1), 1536), reinterpret_tensor(arg0_1, (8, 12, 128, 128), (0, 0, 128, 1), 0), False)
            del buf61
            buf63 = buf62[0]
            assert_size_stride(buf63, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf63, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf62
            buf67 = reinterpret_tensor(buf60, (1024, 768), (768, 1), 0); del buf60  # reuse
            # Topologically Sorted Source Nodes: [transpose_15, reshape_3, view_39, x_26], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf63, (1024, 768), (768, 1), 0), arg45_1, out=buf67)
            del arg45_1
            buf68 = reinterpret_tensor(buf67, (8, 128, 768), (98304, 768, 1), 0); del buf67  # reuse
            buf72 = reinterpret_tensor(buf63, (8, 128, 768), (98304, 768, 1), 0); del buf63  # reuse
            # Topologically Sorted Source Nodes: [x_22, x_23, hidden_states_19, x_26, x_27, hidden_states_21, hidden_states_22], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf68, arg44_1, buf49, buf56, arg38_1, arg46_1, arg47_1, buf72, 1024, 768, stream=stream0)
            del arg38_1
            del arg44_1
            del arg46_1
            del arg47_1
            del buf49
            del buf56
            buf73 = empty_strided_cuda((1024, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_22, view_41, x_28], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf72, (1024, 768), (768, 1), 0), arg49_1, out=buf73)
            del arg49_1
            del buf72
            buf74 = reinterpret_tensor(buf73, (8, 128, 3072), (393216, 3072, 1), 0); del buf73  # reuse
            # Topologically Sorted Source Nodes: [x_28, x_29, mul_12, pow_4, mul_13, add_14, mul_14, tanh_3, add_15, hidden_states_23], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_2.run(buf74, arg48_1, 3145728, stream=stream0)
            del arg48_1
            buf75 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_28, x_29, mul_12, pow_4, mul_13, add_14, mul_14, tanh_3, add_15, hidden_states_23, view_43, x_30], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf74, (1024, 3072), (3072, 1), 0), arg51_1, out=buf75)
            del arg51_1
            del buf74
            buf79 = empty_strided_cuda((8, 128, 768), (98304, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_30, x_31, hidden_states_25, hidden_states_26], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_3.run(buf68, buf75, arg50_1, arg52_1, arg53_1, buf79, 1024, 768, stream=stream0)
            del arg52_1
            del arg53_1
            buf80 = empty_strided_cuda((1024, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_30, x_31, hidden_states_25, hidden_states_26, view_45, x_32], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg54_1, reinterpret_tensor(buf79, (1024, 768), (768, 1), 0), arg55_1, alpha=1, beta=1, out=buf80)
            del arg54_1
            del arg55_1
            # Topologically Sorted Source Nodes: [x_33, split_4, view_49, query_states_9, view_47, key_states_9, view_48, value_states_9, attn_output_16], Original ATen: [aten.view, aten.split, aten.transpose, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf81 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf80, (8, 12, 128, 64), (294912, 64, 2304, 1), 0), reinterpret_tensor(buf80, (8, 12, 128, 64), (294912, 64, 2304, 1), 768), reinterpret_tensor(buf80, (8, 12, 128, 64), (294912, 64, 2304, 1), 1536), reinterpret_tensor(arg0_1, (8, 12, 128, 128), (0, 0, 128, 1), 0), False)
            del buf80
            buf82 = buf81[0]
            assert_size_stride(buf82, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf82, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf81
            buf86 = reinterpret_tensor(buf79, (1024, 768), (768, 1), 0); del buf79  # reuse
            # Topologically Sorted Source Nodes: [transpose_19, reshape_4, view_50, x_34], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf82, (1024, 768), (768, 1), 0), arg57_1, out=buf86)
            del arg57_1
            buf87 = reinterpret_tensor(buf86, (8, 128, 768), (98304, 768, 1), 0); del buf86  # reuse
            buf91 = reinterpret_tensor(buf82, (8, 128, 768), (98304, 768, 1), 0); del buf82  # reuse
            # Topologically Sorted Source Nodes: [x_30, x_31, hidden_states_25, x_34, x_35, hidden_states_27, hidden_states_28], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf87, arg56_1, buf68, buf75, arg50_1, arg58_1, arg59_1, buf91, 1024, 768, stream=stream0)
            del arg50_1
            del arg56_1
            del arg58_1
            del arg59_1
            del buf68
            del buf75
            buf92 = empty_strided_cuda((1024, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_28, view_52, x_36], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf91, (1024, 768), (768, 1), 0), arg61_1, out=buf92)
            del arg61_1
            del buf91
            buf93 = reinterpret_tensor(buf92, (8, 128, 3072), (393216, 3072, 1), 0); del buf92  # reuse
            # Topologically Sorted Source Nodes: [x_36, x_37, mul_16, pow_5, mul_17, add_18, mul_18, tanh_4, add_19, hidden_states_29], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_2.run(buf93, arg60_1, 3145728, stream=stream0)
            del arg60_1
            buf94 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_36, x_37, mul_16, pow_5, mul_17, add_18, mul_18, tanh_4, add_19, hidden_states_29, view_54, x_38], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf93, (1024, 3072), (3072, 1), 0), arg63_1, out=buf94)
            del arg63_1
            del buf93
            buf98 = empty_strided_cuda((8, 128, 768), (98304, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_38, x_39, hidden_states_31, hidden_states_32], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_3.run(buf87, buf94, arg62_1, arg64_1, arg65_1, buf98, 1024, 768, stream=stream0)
            del arg64_1
            del arg65_1
            buf99 = empty_strided_cuda((1024, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_38, x_39, hidden_states_31, hidden_states_32, view_56, x_40], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg66_1, reinterpret_tensor(buf98, (1024, 768), (768, 1), 0), arg67_1, alpha=1, beta=1, out=buf99)
            del arg66_1
            del arg67_1
            # Topologically Sorted Source Nodes: [x_41, split_5, view_60, query_states_11, view_58, key_states_11, view_59, value_states_11, attn_output_20], Original ATen: [aten.view, aten.split, aten.transpose, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf100 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf99, (8, 12, 128, 64), (294912, 64, 2304, 1), 0), reinterpret_tensor(buf99, (8, 12, 128, 64), (294912, 64, 2304, 1), 768), reinterpret_tensor(buf99, (8, 12, 128, 64), (294912, 64, 2304, 1), 1536), reinterpret_tensor(arg0_1, (8, 12, 128, 128), (0, 0, 128, 1), 0), False)
            del buf99
            buf101 = buf100[0]
            assert_size_stride(buf101, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf101, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf100
            buf105 = reinterpret_tensor(buf98, (1024, 768), (768, 1), 0); del buf98  # reuse
            # Topologically Sorted Source Nodes: [transpose_23, reshape_5, view_61, x_42], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf101, (1024, 768), (768, 1), 0), arg69_1, out=buf105)
            del arg69_1
            buf106 = reinterpret_tensor(buf105, (8, 128, 768), (98304, 768, 1), 0); del buf105  # reuse
            buf110 = reinterpret_tensor(buf101, (8, 128, 768), (98304, 768, 1), 0); del buf101  # reuse
            # Topologically Sorted Source Nodes: [x_38, x_39, hidden_states_31, x_42, x_43, hidden_states_33, hidden_states_34], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf106, arg68_1, buf87, buf94, arg62_1, arg70_1, arg71_1, buf110, 1024, 768, stream=stream0)
            del arg62_1
            del arg68_1
            del arg70_1
            del arg71_1
            del buf87
            del buf94
            buf111 = empty_strided_cuda((1024, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_34, view_63, x_44], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf110, (1024, 768), (768, 1), 0), arg73_1, out=buf111)
            del arg73_1
            del buf110
            buf112 = reinterpret_tensor(buf111, (8, 128, 3072), (393216, 3072, 1), 0); del buf111  # reuse
            # Topologically Sorted Source Nodes: [x_44, x_45, mul_20, pow_6, mul_21, add_22, mul_22, tanh_5, add_23, hidden_states_35], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_2.run(buf112, arg72_1, 3145728, stream=stream0)
            del arg72_1
            buf113 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_44, x_45, mul_20, pow_6, mul_21, add_22, mul_22, tanh_5, add_23, hidden_states_35, view_65, x_46], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf112, (1024, 3072), (3072, 1), 0), arg75_1, out=buf113)
            del arg75_1
            del buf112
            buf117 = empty_strided_cuda((8, 128, 768), (98304, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_46, x_47, hidden_states_37, hidden_states_38], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_3.run(buf106, buf113, arg74_1, arg76_1, arg77_1, buf117, 1024, 768, stream=stream0)
            del arg76_1
            del arg77_1
            buf118 = empty_strided_cuda((1024, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_46, x_47, hidden_states_37, hidden_states_38, view_67, x_48], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg78_1, reinterpret_tensor(buf117, (1024, 768), (768, 1), 0), arg79_1, alpha=1, beta=1, out=buf118)
            del arg78_1
            del arg79_1
            # Topologically Sorted Source Nodes: [x_49, split_6, view_71, query_states_13, view_69, key_states_13, view_70, value_states_13, attn_output_24], Original ATen: [aten.view, aten.split, aten.transpose, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf119 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf118, (8, 12, 128, 64), (294912, 64, 2304, 1), 0), reinterpret_tensor(buf118, (8, 12, 128, 64), (294912, 64, 2304, 1), 768), reinterpret_tensor(buf118, (8, 12, 128, 64), (294912, 64, 2304, 1), 1536), reinterpret_tensor(arg0_1, (8, 12, 128, 128), (0, 0, 128, 1), 0), False)
            del buf118
            buf120 = buf119[0]
            assert_size_stride(buf120, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf120, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf119
            buf124 = reinterpret_tensor(buf117, (1024, 768), (768, 1), 0); del buf117  # reuse
            # Topologically Sorted Source Nodes: [transpose_27, reshape_6, view_72, x_50], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf120, (1024, 768), (768, 1), 0), arg81_1, out=buf124)
            del arg81_1
            buf125 = reinterpret_tensor(buf124, (8, 128, 768), (98304, 768, 1), 0); del buf124  # reuse
            buf129 = reinterpret_tensor(buf120, (8, 128, 768), (98304, 768, 1), 0); del buf120  # reuse
            # Topologically Sorted Source Nodes: [x_46, x_47, hidden_states_37, x_50, x_51, hidden_states_39, hidden_states_40], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf125, arg80_1, buf106, buf113, arg74_1, arg82_1, arg83_1, buf129, 1024, 768, stream=stream0)
            del arg74_1
            del arg80_1
            del arg82_1
            del arg83_1
            del buf106
            del buf113
            buf130 = empty_strided_cuda((1024, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_40, view_74, x_52], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf129, (1024, 768), (768, 1), 0), arg85_1, out=buf130)
            del arg85_1
            del buf129
            buf131 = reinterpret_tensor(buf130, (8, 128, 3072), (393216, 3072, 1), 0); del buf130  # reuse
            # Topologically Sorted Source Nodes: [x_52, x_53, mul_24, pow_7, mul_25, add_26, mul_26, tanh_6, add_27, hidden_states_41], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_2.run(buf131, arg84_1, 3145728, stream=stream0)
            del arg84_1
            buf132 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_52, x_53, mul_24, pow_7, mul_25, add_26, mul_26, tanh_6, add_27, hidden_states_41, view_76, x_54], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf131, (1024, 3072), (3072, 1), 0), arg87_1, out=buf132)
            del arg87_1
            del buf131
            buf136 = empty_strided_cuda((8, 128, 768), (98304, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_54, x_55, hidden_states_43, hidden_states_44], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_3.run(buf125, buf132, arg86_1, arg88_1, arg89_1, buf136, 1024, 768, stream=stream0)
            del arg88_1
            del arg89_1
            buf137 = empty_strided_cuda((1024, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_54, x_55, hidden_states_43, hidden_states_44, view_78, x_56], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg90_1, reinterpret_tensor(buf136, (1024, 768), (768, 1), 0), arg91_1, alpha=1, beta=1, out=buf137)
            del arg90_1
            del arg91_1
            # Topologically Sorted Source Nodes: [x_57, split_7, view_82, query_states_15, view_80, key_states_15, view_81, value_states_15, attn_output_28], Original ATen: [aten.view, aten.split, aten.transpose, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf138 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf137, (8, 12, 128, 64), (294912, 64, 2304, 1), 0), reinterpret_tensor(buf137, (8, 12, 128, 64), (294912, 64, 2304, 1), 768), reinterpret_tensor(buf137, (8, 12, 128, 64), (294912, 64, 2304, 1), 1536), reinterpret_tensor(arg0_1, (8, 12, 128, 128), (0, 0, 128, 1), 0), False)
            del buf137
            buf139 = buf138[0]
            assert_size_stride(buf139, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf139, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf138
            buf143 = reinterpret_tensor(buf136, (1024, 768), (768, 1), 0); del buf136  # reuse
            # Topologically Sorted Source Nodes: [transpose_31, reshape_7, view_83, x_58], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf139, (1024, 768), (768, 1), 0), arg93_1, out=buf143)
            del arg93_1
            buf144 = reinterpret_tensor(buf143, (8, 128, 768), (98304, 768, 1), 0); del buf143  # reuse
            buf148 = reinterpret_tensor(buf139, (8, 128, 768), (98304, 768, 1), 0); del buf139  # reuse
            # Topologically Sorted Source Nodes: [x_54, x_55, hidden_states_43, x_58, x_59, hidden_states_45, hidden_states_46], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf144, arg92_1, buf125, buf132, arg86_1, arg94_1, arg95_1, buf148, 1024, 768, stream=stream0)
            del arg86_1
            del arg92_1
            del arg94_1
            del arg95_1
            del buf125
            del buf132
            buf149 = empty_strided_cuda((1024, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_46, view_85, x_60], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf148, (1024, 768), (768, 1), 0), arg97_1, out=buf149)
            del arg97_1
            del buf148
            buf150 = reinterpret_tensor(buf149, (8, 128, 3072), (393216, 3072, 1), 0); del buf149  # reuse
            # Topologically Sorted Source Nodes: [x_60, x_61, mul_28, pow_8, mul_29, add_30, mul_30, tanh_7, add_31, hidden_states_47], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_2.run(buf150, arg96_1, 3145728, stream=stream0)
            del arg96_1
            buf151 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_60, x_61, mul_28, pow_8, mul_29, add_30, mul_30, tanh_7, add_31, hidden_states_47, view_87, x_62], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf150, (1024, 3072), (3072, 1), 0), arg99_1, out=buf151)
            del arg99_1
            del buf150
            buf155 = empty_strided_cuda((8, 128, 768), (98304, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_62, x_63, hidden_states_49, hidden_states_50], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_3.run(buf144, buf151, arg98_1, arg100_1, arg101_1, buf155, 1024, 768, stream=stream0)
            del arg100_1
            del arg101_1
            buf156 = empty_strided_cuda((1024, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_62, x_63, hidden_states_49, hidden_states_50, view_89, x_64], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg102_1, reinterpret_tensor(buf155, (1024, 768), (768, 1), 0), arg103_1, alpha=1, beta=1, out=buf156)
            del arg102_1
            del arg103_1
            # Topologically Sorted Source Nodes: [x_65, split_8, view_93, query_states_17, view_91, key_states_17, view_92, value_states_17, attn_output_32], Original ATen: [aten.view, aten.split, aten.transpose, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf157 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf156, (8, 12, 128, 64), (294912, 64, 2304, 1), 0), reinterpret_tensor(buf156, (8, 12, 128, 64), (294912, 64, 2304, 1), 768), reinterpret_tensor(buf156, (8, 12, 128, 64), (294912, 64, 2304, 1), 1536), reinterpret_tensor(arg0_1, (8, 12, 128, 128), (0, 0, 128, 1), 0), False)
            del buf156
            buf158 = buf157[0]
            assert_size_stride(buf158, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf158, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf157
            buf162 = reinterpret_tensor(buf155, (1024, 768), (768, 1), 0); del buf155  # reuse
            # Topologically Sorted Source Nodes: [transpose_35, reshape_8, view_94, x_66], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf158, (1024, 768), (768, 1), 0), arg105_1, out=buf162)
            del arg105_1
            buf163 = reinterpret_tensor(buf162, (8, 128, 768), (98304, 768, 1), 0); del buf162  # reuse
            buf167 = reinterpret_tensor(buf158, (8, 128, 768), (98304, 768, 1), 0); del buf158  # reuse
            # Topologically Sorted Source Nodes: [x_62, x_63, hidden_states_49, x_66, x_67, hidden_states_51, hidden_states_52], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf163, arg104_1, buf144, buf151, arg98_1, arg106_1, arg107_1, buf167, 1024, 768, stream=stream0)
            del arg104_1
            del arg106_1
            del arg107_1
            del arg98_1
            del buf144
            del buf151
            buf168 = empty_strided_cuda((1024, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_52, view_96, x_68], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf167, (1024, 768), (768, 1), 0), arg109_1, out=buf168)
            del arg109_1
            del buf167
            buf169 = reinterpret_tensor(buf168, (8, 128, 3072), (393216, 3072, 1), 0); del buf168  # reuse
            # Topologically Sorted Source Nodes: [x_68, x_69, mul_32, pow_9, mul_33, add_34, mul_34, tanh_8, add_35, hidden_states_53], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_2.run(buf169, arg108_1, 3145728, stream=stream0)
            del arg108_1
            buf170 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_68, x_69, mul_32, pow_9, mul_33, add_34, mul_34, tanh_8, add_35, hidden_states_53, view_98, x_70], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf169, (1024, 3072), (3072, 1), 0), arg111_1, out=buf170)
            del arg111_1
            del buf169
            buf174 = empty_strided_cuda((8, 128, 768), (98304, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_70, x_71, hidden_states_55, hidden_states_56], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_3.run(buf163, buf170, arg110_1, arg112_1, arg113_1, buf174, 1024, 768, stream=stream0)
            del arg112_1
            del arg113_1
            buf175 = empty_strided_cuda((1024, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_70, x_71, hidden_states_55, hidden_states_56, view_100, x_72], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg114_1, reinterpret_tensor(buf174, (1024, 768), (768, 1), 0), arg115_1, alpha=1, beta=1, out=buf175)
            del arg114_1
            del arg115_1
            # Topologically Sorted Source Nodes: [x_73, split_9, view_104, query_states_19, view_102, key_states_19, view_103, value_states_19, attn_output_36], Original ATen: [aten.view, aten.split, aten.transpose, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf176 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf175, (8, 12, 128, 64), (294912, 64, 2304, 1), 0), reinterpret_tensor(buf175, (8, 12, 128, 64), (294912, 64, 2304, 1), 768), reinterpret_tensor(buf175, (8, 12, 128, 64), (294912, 64, 2304, 1), 1536), reinterpret_tensor(arg0_1, (8, 12, 128, 128), (0, 0, 128, 1), 0), False)
            del buf175
            buf177 = buf176[0]
            assert_size_stride(buf177, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf177, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf176
            buf181 = reinterpret_tensor(buf174, (1024, 768), (768, 1), 0); del buf174  # reuse
            # Topologically Sorted Source Nodes: [transpose_39, reshape_9, view_105, x_74], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf177, (1024, 768), (768, 1), 0), arg117_1, out=buf181)
            del arg117_1
            buf182 = reinterpret_tensor(buf181, (8, 128, 768), (98304, 768, 1), 0); del buf181  # reuse
            buf186 = reinterpret_tensor(buf177, (8, 128, 768), (98304, 768, 1), 0); del buf177  # reuse
            # Topologically Sorted Source Nodes: [x_70, x_71, hidden_states_55, x_74, x_75, hidden_states_57, hidden_states_58], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf182, arg116_1, buf163, buf170, arg110_1, arg118_1, arg119_1, buf186, 1024, 768, stream=stream0)
            del arg110_1
            del arg116_1
            del arg118_1
            del arg119_1
            del buf163
            del buf170
            buf187 = empty_strided_cuda((1024, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_58, view_107, x_76], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf186, (1024, 768), (768, 1), 0), arg121_1, out=buf187)
            del arg121_1
            del buf186
            buf188 = reinterpret_tensor(buf187, (8, 128, 3072), (393216, 3072, 1), 0); del buf187  # reuse
            # Topologically Sorted Source Nodes: [x_76, x_77, mul_36, pow_10, mul_37, add_38, mul_38, tanh_9, add_39, hidden_states_59], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_2.run(buf188, arg120_1, 3145728, stream=stream0)
            del arg120_1
            buf189 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_76, x_77, mul_36, pow_10, mul_37, add_38, mul_38, tanh_9, add_39, hidden_states_59, view_109, x_78], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf188, (1024, 3072), (3072, 1), 0), arg123_1, out=buf189)
            del arg123_1
            del buf188
            buf193 = empty_strided_cuda((8, 128, 768), (98304, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_78, x_79, hidden_states_61, hidden_states_62], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_3.run(buf182, buf189, arg122_1, arg124_1, arg125_1, buf193, 1024, 768, stream=stream0)
            del arg124_1
            del arg125_1
            buf194 = empty_strided_cuda((1024, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_78, x_79, hidden_states_61, hidden_states_62, view_111, x_80], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg126_1, reinterpret_tensor(buf193, (1024, 768), (768, 1), 0), arg127_1, alpha=1, beta=1, out=buf194)
            del arg126_1
            del arg127_1
            # Topologically Sorted Source Nodes: [x_81, split_10, view_115, query_states_21, view_113, key_states_21, view_114, value_states_21, attn_output_40], Original ATen: [aten.view, aten.split, aten.transpose, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf195 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf194, (8, 12, 128, 64), (294912, 64, 2304, 1), 0), reinterpret_tensor(buf194, (8, 12, 128, 64), (294912, 64, 2304, 1), 768), reinterpret_tensor(buf194, (8, 12, 128, 64), (294912, 64, 2304, 1), 1536), reinterpret_tensor(arg0_1, (8, 12, 128, 128), (0, 0, 128, 1), 0), False)
            del buf194
            buf196 = buf195[0]
            assert_size_stride(buf196, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf196, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf195
            buf200 = reinterpret_tensor(buf193, (1024, 768), (768, 1), 0); del buf193  # reuse
            # Topologically Sorted Source Nodes: [transpose_43, reshape_10, view_116, x_82], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf196, (1024, 768), (768, 1), 0), arg129_1, out=buf200)
            del arg129_1
            buf201 = reinterpret_tensor(buf200, (8, 128, 768), (98304, 768, 1), 0); del buf200  # reuse
            buf205 = reinterpret_tensor(buf196, (8, 128, 768), (98304, 768, 1), 0); del buf196  # reuse
            # Topologically Sorted Source Nodes: [x_78, x_79, hidden_states_61, x_82, x_83, hidden_states_63, hidden_states_64], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf201, arg128_1, buf182, buf189, arg122_1, arg130_1, arg131_1, buf205, 1024, 768, stream=stream0)
            del arg122_1
            del arg128_1
            del arg130_1
            del arg131_1
            del buf182
            del buf189
            buf206 = empty_strided_cuda((1024, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_64, view_118, x_84], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf205, (1024, 768), (768, 1), 0), arg133_1, out=buf206)
            del arg133_1
            del buf205
            buf207 = reinterpret_tensor(buf206, (8, 128, 3072), (393216, 3072, 1), 0); del buf206  # reuse
            # Topologically Sorted Source Nodes: [x_84, x_85, mul_40, pow_11, mul_41, add_42, mul_42, tanh_10, add_43, hidden_states_65], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_2.run(buf207, arg132_1, 3145728, stream=stream0)
            del arg132_1
            buf208 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_84, x_85, mul_40, pow_11, mul_41, add_42, mul_42, tanh_10, add_43, hidden_states_65, view_120, x_86], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf207, (1024, 3072), (3072, 1), 0), arg135_1, out=buf208)
            del arg135_1
            del buf207
            buf212 = empty_strided_cuda((8, 128, 768), (98304, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_86, x_87, hidden_states_67, hidden_states_68], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_3.run(buf201, buf208, arg134_1, arg136_1, arg137_1, buf212, 1024, 768, stream=stream0)
            del arg136_1
            del arg137_1
            buf213 = empty_strided_cuda((1024, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_86, x_87, hidden_states_67, hidden_states_68, view_122, x_88], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg138_1, reinterpret_tensor(buf212, (1024, 768), (768, 1), 0), arg139_1, alpha=1, beta=1, out=buf213)
            del arg138_1
            del arg139_1
            # Topologically Sorted Source Nodes: [x_89, split_11, view_126, query_states_23, view_124, key_states_23, view_125, value_states_23, attn_output_44], Original ATen: [aten.view, aten.split, aten.transpose, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf214 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf213, (8, 12, 128, 64), (294912, 64, 2304, 1), 0), reinterpret_tensor(buf213, (8, 12, 128, 64), (294912, 64, 2304, 1), 768), reinterpret_tensor(buf213, (8, 12, 128, 64), (294912, 64, 2304, 1), 1536), reinterpret_tensor(arg0_1, (8, 12, 128, 128), (0, 0, 128, 1), 0), False)
            del arg0_1
            del buf213
            buf215 = buf214[0]
            assert_size_stride(buf215, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf215, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf214
            buf219 = reinterpret_tensor(buf212, (1024, 768), (768, 1), 0); del buf212  # reuse
            # Topologically Sorted Source Nodes: [transpose_47, reshape_11, view_127, x_90], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf215, (1024, 768), (768, 1), 0), arg141_1, out=buf219)
            del arg141_1
            buf220 = reinterpret_tensor(buf219, (8, 128, 768), (98304, 768, 1), 0); del buf219  # reuse
            buf224 = reinterpret_tensor(buf215, (8, 128, 768), (98304, 768, 1), 0); del buf215  # reuse
            # Topologically Sorted Source Nodes: [x_86, x_87, hidden_states_67, x_90, x_91, hidden_states_69, hidden_states_70], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf220, arg140_1, buf201, buf208, arg134_1, arg142_1, arg143_1, buf224, 1024, 768, stream=stream0)
            del arg134_1
            del arg140_1
            del arg142_1
            del arg143_1
            del buf201
            del buf208
            buf225 = empty_strided_cuda((1024, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_70, view_129, x_92], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf224, (1024, 768), (768, 1), 0), arg145_1, out=buf225)
            del arg145_1
            del buf224
            buf226 = reinterpret_tensor(buf225, (8, 128, 3072), (393216, 3072, 1), 0); del buf225  # reuse
            # Topologically Sorted Source Nodes: [x_92, x_93, mul_44, pow_12, mul_45, add_46, mul_46, tanh_11, add_47, hidden_states_71], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_2.run(buf226, arg144_1, 3145728, stream=stream0)
            del arg144_1
            buf227 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_92, x_93, mul_44, pow_12, mul_45, add_46, mul_46, tanh_11, add_47, hidden_states_71, view_131, x_94], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf226, (1024, 3072), (3072, 1), 0), arg147_1, out=buf227)
            del arg147_1
            del buf226
            buf231 = buf220; del buf220  # reuse
            # Topologically Sorted Source Nodes: [x_94, x_95, hidden_states_73, hidden_states_74], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_5.run(buf231, buf227, arg146_1, arg148_1, arg149_1, 1024, 768, stream=stream0)
            del arg146_1
            del arg148_1
            del arg149_1
            del buf227
        return (buf231, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 1, 128, 128), (16384, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    arg2_1 = rand_strided((50257, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)


# ===== inductor generated file: /tmp/cnnbench-transformers-p50cz5fm/repeat_02/a1/torchinductor/tmpxdyb3vm9/wq/cwqiouzl5rqqzecyxn3st4tw5voktz55qizk6dw2eebgohjbtoi5.py =====
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


# kernel path: /tmp/cnnbench-transformers-p50cz5fm/repeat_02/a1/torchinductor/tmpxdyb3vm9/jq/cjqwss5atwrya52vg3gpmhy5m6itpshtoqaeg2ei6lm2otgmdhf5.py
# Topologically Sorted Source Nodes: [inputs_embeds, cache_position, position_ids, position_embeds, hidden_states, hidden_states_2], Original ATen: [aten.embedding, aten.arange, aten.unsqueeze, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   cache_position => iota
#   hidden_states => add
#   hidden_states_2 => add_1, add_2, mul, mul_1, rsqrt, sub, var_mean
#   inputs_embeds => embedding
#   position_embeds => embedding_1
#   position_ids => unsqueeze
# Graph fragment:
#   %arg1_1 : Tensor "i64[8, 128][128, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "f32[50257, 768][768, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg3_1 : Tensor "f32[1024, 768][768, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %getitem_1 : Tensor "f32[8, 128, 1][128, 1, 1024]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf1 : Tensor "f32[8, 128, 1][128, 1, 1024]cuda:0" = PlaceHolder[target=buf1]
#   %arg4_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %embedding : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %arg1_1), kwargs = {})
#   %iota : Tensor "i64[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze : Tensor "i64[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, 0), kwargs = {})
#   %embedding_1 : Tensor "f32[1, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %unsqueeze), kwargs = {})
#   %add : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %getitem_1), kwargs = {})
#   %add_1 : Tensor "f32[8, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[8, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg4_1), kwargs = {})
#   %add_2 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %arg5_1), kwargs = {})
#   return %getitem_1,%buf1,%add_2
triton_red_fused_add_arange_embedding_native_layer_norm_unsqueeze_0 = async_compile.triton('triton_red_fused_add_arange_embedding_native_layer_norm_unsqueeze_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_arange_embedding_native_layer_norm_unsqueeze_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_arange_embedding_native_layer_norm_unsqueeze_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 768
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    x0 = (xindex % 128)
    tmp10_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp10_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp10_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp7 = tl.load(in_ptr2 + (r0_2 + 768*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.full([XBLOCK, R0_BLOCK], 50257, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert(((0 <= tmp4) & (tmp4 < 50257)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 50257")
        tmp6 = tl.load(in_ptr1 + (r0_2 + 768*tmp4), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp10_mean_next, tmp10_m2_next, tmp10_weight_next = triton_helpers.welford_reduce(
            tmp9, tmp10_mean, tmp10_m2, tmp10_weight, roffset == 0
        )
        tmp10_mean = tl.where(r0_mask & xmask, tmp10_mean_next, tmp10_mean)
        tmp10_m2 = tl.where(r0_mask & xmask, tmp10_m2_next, tmp10_m2)
        tmp10_weight = tl.where(r0_mask & xmask, tmp10_weight_next, tmp10_weight)
    tmp11, tmp12, tmp13 = triton_helpers.welford(tmp10_mean, tmp10_m2, tmp10_weight, 1)
    tmp10 = tmp11[:, None]
    tmp14 = tmp12[:, None]
    tmp15 = tmp13[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp22 = tl.load(in_ptr2 + (r0_2 + 768*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp31 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp33 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.full([XBLOCK, R0_BLOCK], 50257, tl.int32)
        tmp17 = tmp0 + tmp16
        tmp18 = tmp0 < 0
        tmp19 = tl.where(tmp18, tmp17, tmp0)
        tl.device_assert(((0 <= tmp19) & (tmp19 < 50257)) | ~(xmask), "index out of bounds: 0 <= tmp19 < 50257")
        tmp21 = tl.load(in_ptr1 + (r0_2 + 768*tmp19), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp23 = tmp21 + tmp22
        tmp24 = tmp23 - tmp10
        tmp25 = 768.0
        tmp26 = (tmp14 / tmp25)
        tmp27 = 1e-05
        tmp28 = tmp26 + tmp27
        tmp29 = libdevice.rsqrt(tmp28)
        tmp30 = tmp24 * tmp29
        tmp32 = tmp30 * tmp31
        tmp34 = tmp32 + tmp33
        tl.store(out_ptr2 + (r0_2 + 768*x3), tmp34, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-transformers-p50cz5fm/repeat_02/a1/torchinductor/tmpxdyb3vm9/4h/c4hbd3bs4g2qthy7zrebrqivychrqd2oziojglrfg2tuj2xcnsvr.py
# Topologically Sorted Source Nodes: [inputs_embeds, cache_position, position_ids, position_embeds, hidden_states, x_2, x_3, hidden_states_3, hidden_states_4], Original ATen: [aten.embedding, aten.arange, aten.unsqueeze, aten.add, aten.addmm, aten.view, aten.native_layer_norm]
# Source node to ATen node mapping:
#   cache_position => iota
#   hidden_states => add
#   hidden_states_3 => add_3
#   hidden_states_4 => add_4, add_5, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
#   inputs_embeds => embedding
#   position_embeds => embedding_1
#   position_ids => unsqueeze
#   x_2 => add_tensor_35
#   x_3 => view_8
# Graph fragment:
#   %mm_default_35 : Tensor "f32[1024, 768][768, 1]cuda:0" = PlaceHolder[target=mm_default_35]
#   %arg8_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %arg1_1 : Tensor "i64[8, 128][128, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "f32[50257, 768][768, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg3_1 : Tensor "f32[1024, 768][768, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %add_3 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0" = PlaceHolder[target=add_3]
#   %getitem_10 : Tensor "f32[8, 128, 1][128, 1, 1024]cuda:0" = PlaceHolder[target=getitem_10]
#   %buf13 : Tensor "f32[8, 128, 1][128, 1, 1024]cuda:0" = PlaceHolder[target=buf13]
#   %arg10_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg10_1]
#   %arg11_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg11_1]
#   %embedding : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %arg1_1), kwargs = {})
#   %iota : Tensor "i64[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze : Tensor "i64[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, 0), kwargs = {})
#   %embedding_1 : Tensor "f32[1, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %unsqueeze), kwargs = {})
#   %add : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %add_tensor_35 : Tensor "f32[1024, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_35, %arg8_1), kwargs = {})
#   %view_8 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_35, [8, 128, 768]), kwargs = {})
#   %add_3 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_8, %add), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_3, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %getitem_10), kwargs = {})
#   %add_4 : Tensor "f32[8, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_9, 1e-05), kwargs = {})
#   %rsqrt_1 : Tensor "f32[8, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %mul_2 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_3 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %arg10_1), kwargs = {})
#   %add_5 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %arg11_1), kwargs = {})
#   return %add_3,%getitem_10,%buf13,%add_5
triton_per_fused_add_addmm_arange_embedding_native_layer_norm_unsqueeze_view_1 = async_compile.triton('triton_per_fused_add_addmm_arange_embedding_native_layer_norm_unsqueeze_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_arange_embedding_native_layer_norm_unsqueeze_view_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 6, 'num_reduction': 4, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_addmm_arange_embedding_native_layer_norm_unsqueeze_view_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
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
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (r0_2 + 768*x3), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (r0_2 + 768*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr5 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tl.full([XBLOCK, R0_BLOCK], 50257, tl.int32)
    tmp5 = tmp3 + tmp4
    tmp6 = tmp3 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp3)
    tl.device_assert(((0 <= tmp7) & (tmp7 < 50257)) | ~(xmask), "index out of bounds: 0 <= tmp7 < 50257")
    tmp9 = tl.load(in_ptr2 + (r0_2 + 768*tmp7), r0_mask & xmask, other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tmp2 + tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(r0_mask & xmask, tmp13, 0)
    tmp16 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp18 = tl.where(r0_mask & xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None].to(tl.float32)
    tmp20 = tl.full([XBLOCK, 1], 768, tl.int32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = (tmp19 / tmp21)
    tmp23 = tmp13 - tmp22
    tmp24 = tmp23 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
    tmp27 = tl.where(r0_mask & xmask, tmp25, 0)
    tmp28 = tl.sum(tmp27, 1)[:, None].to(tl.float32)
    tmp29 = tmp12 - tmp22
    tmp30 = 768.0
    tmp31 = (tmp28 / tmp30)
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.rsqrt(tmp33)
    tmp35 = tmp29 * tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp37 + tmp38
    tl.store(in_out_ptr0 + (r0_2 + 768*x3), tmp12, r0_mask & xmask)
    tl.store(out_ptr2 + (r0_2 + 768*x3), tmp39, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-transformers-p50cz5fm/repeat_02/a1/torchinductor/tmpxdyb3vm9/ea/ceaoz3uu6u24l5sygramjdrcfjvvi5huw4pndtejhhfpig2qrkiv.py
# Topologically Sorted Source Nodes: [x_4, x_5, mul, pow_1, mul_1, add_2, mul_2, tanh, add_3, hidden_states_5], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
# Source node to ATen node mapping:
#   add_2 => add_6
#   add_3 => add_7
#   hidden_states_5 => mul_7
#   mul => mul_4
#   mul_1 => mul_5
#   mul_2 => mul_6
#   pow_1 => pow_1
#   tanh => tanh
#   x_4 => add_tensor_34
#   x_5 => view_10
# Graph fragment:
#   %mm_default_34 : Tensor "f32[1024, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_default_34]
#   %arg12_1 : Tensor "f32[3072][1]cuda:0" = PlaceHolder[target=arg12_1]
#   %add_tensor_34 : Tensor "f32[1024, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_34, %arg12_1), kwargs = {})
#   %view_10 : Tensor "f32[8, 128, 3072][393216, 3072, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_34, [8, 128, 3072]), kwargs = {})
#   %mul_4 : Tensor "f32[8, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_10, 0.5), kwargs = {})
#   %pow_1 : Tensor "f32[8, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view_10, 3.0), kwargs = {})
#   %mul_5 : Tensor "f32[8, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_1, 0.044715), kwargs = {})
#   %add_6 : Tensor "f32[8, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_10, %mul_5), kwargs = {})
#   %mul_6 : Tensor "f32[8, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, 0.7978845608028654), kwargs = {})
#   %tanh : Tensor "f32[8, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_6,), kwargs = {})
#   %add_7 : Tensor "f32[8, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tanh, 1.0), kwargs = {})
#   %mul_7 : Tensor "f32[8, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %add_7), kwargs = {})
#   return %mul_7
triton_poi_fused_add_addmm_mul_pow_tanh_view_2 = async_compile.triton('triton_poi_fused_add_addmm_mul_pow_tanh_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_mul_pow_tanh_view_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 37761024}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_mul_pow_tanh_view_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 3072)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = tmp2 * tmp2
    tmp6 = tmp5 * tmp2
    tmp7 = 0.044715
    tmp8 = tmp6 * tmp7
    tmp9 = tmp2 + tmp8
    tmp10 = 0.7978845608028654
    tmp11 = tmp9 * tmp10
    tmp12 = libdevice.tanh(tmp11)
    tmp13 = 1.0
    tmp14 = tmp12 + tmp13
    tmp15 = tmp4 * tmp14
    tl.store(in_out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-transformers-p50cz5fm/repeat_02/a1/torchinductor/tmpxdyb3vm9/wz/cwznbdrhxienrtupkya6knusvpneohiudyrpaybqvypvjpmv4hgq.py
# Topologically Sorted Source Nodes: [x_6, x_7, hidden_states_7, hidden_states_8], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_7 => add_8
#   hidden_states_8 => add_10, add_9, mul_8, mul_9, rsqrt_2, sub_2, var_mean_2
#   x_6 => add_tensor_33
#   x_7 => view_12
# Graph fragment:
#   %add_3 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0" = PlaceHolder[target=add_3]
#   %mm_default_33 : Tensor "f32[1024, 768][768, 1]cuda:0" = PlaceHolder[target=mm_default_33]
#   %arg14_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg14_1]
#   %getitem_12 : Tensor "f32[8, 128, 1][128, 1, 1024]cuda:0" = PlaceHolder[target=getitem_12]
#   %buf20 : Tensor "f32[8, 128, 1][128, 1, 1024]cuda:0" = PlaceHolder[target=buf20]
#   %arg16_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg16_1]
#   %arg17_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg17_1]
#   %add_tensor_33 : Tensor "f32[1024, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_33, %arg14_1), kwargs = {})
#   %view_12 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_33, [8, 128, 768]), kwargs = {})
#   %add_8 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %view_12), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_8, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_8, %getitem_12), kwargs = {})
#   %add_9 : Tensor "f32[8, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_11, 1e-05), kwargs = {})
#   %rsqrt_2 : Tensor "f32[8, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_8 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_2), kwargs = {})
#   %mul_9 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %arg16_1), kwargs = {})
#   %add_10 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %arg17_1), kwargs = {})
#   return %getitem_12,%buf20,%add_10
triton_per_fused_add_addmm_native_layer_norm_view_3 = async_compile.triton('triton_per_fused_add_addmm_native_layer_norm_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_native_layer_norm_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 12592128}}
)
@triton.jit
def triton_per_fused_add_addmm_native_layer_norm_view_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp12 = tl.full([XBLOCK, 1], 768, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = (tmp11 / tmp13)
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(r0_mask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None].to(tl.float32)
    tmp21 = tmp4 - tmp14
    tmp22 = 768.0
    tmp23 = (tmp20 / tmp22)
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp31, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-transformers-p50cz5fm/repeat_02/a1/torchinductor/tmpxdyb3vm9/d6/cd6ybkiu75diiuhq75l2mwfs6w2h4zixpdedql34g3irkcc6jcqc.py
# Topologically Sorted Source Nodes: [x_6, x_7, hidden_states_7, x_10, x_11, hidden_states_9, hidden_states_10], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_10 => add_12, add_13, mul_10, mul_11, rsqrt_3, sub_3, var_mean_3
#   hidden_states_7 => add_8
#   hidden_states_9 => add_11
#   x_10 => add_tensor_32
#   x_11 => view_20
#   x_6 => add_tensor_33
#   x_7 => view_12
# Graph fragment:
#   %mm_default_32 : Tensor "f32[1024, 768][768, 1]cuda:0" = PlaceHolder[target=mm_default_32]
#   %arg20_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg20_1]
#   %add_3 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0" = PlaceHolder[target=add_3]
#   %mm_default_33 : Tensor "f32[1024, 768][768, 1]cuda:0" = PlaceHolder[target=mm_default_33]
#   %arg14_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg14_1]
#   %add_11 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0" = PlaceHolder[target=add_11]
#   %getitem_21 : Tensor "f32[8, 128, 1][128, 1, 1024]cuda:0" = PlaceHolder[target=getitem_21]
#   %buf32 : Tensor "f32[8, 128, 1][128, 1, 1024]cuda:0" = PlaceHolder[target=buf32]
#   %arg22_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg22_1]
#   %arg23_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg23_1]
#   %add_tensor_33 : Tensor "f32[1024, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_33, %arg14_1), kwargs = {})
#   %view_12 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_33, [8, 128, 768]), kwargs = {})
#   %add_8 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %view_12), kwargs = {})
#   %add_tensor_32 : Tensor "f32[1024, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_32, %arg20_1), kwargs = {})
#   %view_20 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_32, [8, 128, 768]), kwargs = {})
#   %add_11 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_20, %add_8), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_11, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_11, %getitem_21), kwargs = {})
#   %add_12 : Tensor "f32[8, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_20, 1e-05), kwargs = {})
#   %rsqrt_3 : Tensor "f32[8, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_12,), kwargs = {})
#   %mul_10 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_3), kwargs = {})
#   %mul_11 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %arg22_1), kwargs = {})
#   %add_13 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %arg23_1), kwargs = {})
#   return %add_11,%getitem_21,%buf32,%add_13
triton_per_fused_add_addmm_native_layer_norm_view_4 = async_compile.triton('triton_per_fused_add_addmm_native_layer_norm_view_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_native_layer_norm_view_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 22032384}}
)
@triton.jit
def triton_per_fused_add_addmm_native_layer_norm_view_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
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
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp6 = tmp4 + tmp5
    tmp7 = tmp3 + tmp6
    tmp8 = tmp2 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.where(r0_mask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp14 = tl.where(r0_mask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None].to(tl.float32)
    tmp16 = tl.full([XBLOCK, 1], 768, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = (tmp15 / tmp17)
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
    tmp23 = tl.where(r0_mask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None].to(tl.float32)
    tmp25 = tmp8 - tmp18
    tmp26 = 768.0
    tmp27 = (tmp24 / tmp26)
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp8, r0_mask & xmask)
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp35, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-transformers-p50cz5fm/repeat_02/a1/torchinductor/tmpxdyb3vm9/bo/cbogixtlgmsshy55xfrz275tizvetdt26ih37appoc6jgr7ajyk3.py
# Topologically Sorted Source Nodes: [x_94, x_95, hidden_states_73, hidden_states_74], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_73 => add_96
#   hidden_states_74 => add_97, add_98, mul_96, mul_97, rsqrt_24, sub_24, var_mean_24
#   x_94 => add_tensor
#   x_95 => view_144
# Graph fragment:
#   %add_91 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0" = PlaceHolder[target=add_91]
#   %mm_default : Tensor "f32[1024, 768][768, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %arg146_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg146_1]
#   %getitem_133 : Tensor "f32[8, 128, 1][128, 1, 1024]cuda:0" = PlaceHolder[target=getitem_133]
#   %buf229 : Tensor "f32[8, 128, 1][128, 1, 1024]cuda:0" = PlaceHolder[target=buf229]
#   %arg148_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg148_1]
#   %arg149_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg149_1]
#   %add_tensor : Tensor "f32[1024, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %arg146_1), kwargs = {})
#   %view_144 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor, [8, 128, 768]), kwargs = {})
#   %add_96 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_91, %view_144), kwargs = {})
#   %var_mean_24 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_96, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_24 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_96, %getitem_133), kwargs = {})
#   %add_97 : Tensor "f32[8, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_132, 1e-05), kwargs = {})
#   %rsqrt_24 : Tensor "f32[8, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_97,), kwargs = {})
#   %mul_96 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %rsqrt_24), kwargs = {})
#   %mul_97 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_96, %arg148_1), kwargs = {})
#   %add_98 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_97, %arg149_1), kwargs = {})
#   return %getitem_133,%buf229,%add_98
triton_per_fused_add_addmm_native_layer_norm_view_5 = async_compile.triton('triton_per_fused_add_addmm_native_layer_norm_view_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_native_layer_norm_view_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 12592128}}
)
@triton.jit
def triton_per_fused_add_addmm_native_layer_norm_view_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
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
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp12 = tl.full([XBLOCK, 1], 768, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = (tmp11 / tmp13)
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(r0_mask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None].to(tl.float32)
    tmp21 = tmp4 - tmp14
    tmp22 = 768.0
    tmp23 = (tmp20 / tmp22)
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp31, r0_mask & xmask)
''', device_str='cuda')


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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1 = args
        args.clear()
        assert_size_stride(arg0_1, (1, 1, 128, 128), (16384, 16384, 128, 1))
        assert_size_stride(arg1_1, (8, 128), (128, 1))
        assert_size_stride(arg2_1, (50257, 768), (768, 1))
        assert_size_stride(arg3_1, (1024, 768), (768, 1))
        assert_size_stride(arg4_1, (768, ), (1, ))
        assert_size_stride(arg5_1, (768, ), (1, ))
        assert_size_stride(arg6_1, (2304, ), (1, ))
        assert_size_stride(arg7_1, (768, 2304), (2304, 1))
        assert_size_stride(arg8_1, (768, ), (1, ))
        assert_size_stride(arg9_1, (768, 768), (768, 1))
        assert_size_stride(arg10_1, (768, ), (1, ))
        assert_size_stride(arg11_1, (768, ), (1, ))
        assert_size_stride(arg12_1, (3072, ), (1, ))
        assert_size_stride(arg13_1, (768, 3072), (3072, 1))
        assert_size_stride(arg14_1, (768, ), (1, ))
        assert_size_stride(arg15_1, (3072, 768), (768, 1))
        assert_size_stride(arg16_1, (768, ), (1, ))
        assert_size_stride(arg17_1, (768, ), (1, ))
        assert_size_stride(arg18_1, (2304, ), (1, ))
        assert_size_stride(arg19_1, (768, 2304), (2304, 1))
        assert_size_stride(arg20_1, (768, ), (1, ))
        assert_size_stride(arg21_1, (768, 768), (768, 1))
        assert_size_stride(arg22_1, (768, ), (1, ))
        assert_size_stride(arg23_1, (768, ), (1, ))
        assert_size_stride(arg24_1, (3072, ), (1, ))
        assert_size_stride(arg25_1, (768, 3072), (3072, 1))
        assert_size_stride(arg26_1, (768, ), (1, ))
        assert_size_stride(arg27_1, (3072, 768), (768, 1))
        assert_size_stride(arg28_1, (768, ), (1, ))
        assert_size_stride(arg29_1, (768, ), (1, ))
        assert_size_stride(arg30_1, (2304, ), (1, ))
        assert_size_stride(arg31_1, (768, 2304), (2304, 1))
        assert_size_stride(arg32_1, (768, ), (1, ))
        assert_size_stride(arg33_1, (768, 768), (768, 1))
        assert_size_stride(arg34_1, (768, ), (1, ))
        assert_size_stride(arg35_1, (768, ), (1, ))
        assert_size_stride(arg36_1, (3072, ), (1, ))
        assert_size_stride(arg37_1, (768, 3072), (3072, 1))
        assert_size_stride(arg38_1, (768, ), (1, ))
        assert_size_stride(arg39_1, (3072, 768), (768, 1))
        assert_size_stride(arg40_1, (768, ), (1, ))
        assert_size_stride(arg41_1, (768, ), (1, ))
        assert_size_stride(arg42_1, (2304, ), (1, ))
        assert_size_stride(arg43_1, (768, 2304), (2304, 1))
        assert_size_stride(arg44_1, (768, ), (1, ))
        assert_size_stride(arg45_1, (768, 768), (768, 1))
        assert_size_stride(arg46_1, (768, ), (1, ))
        assert_size_stride(arg47_1, (768, ), (1, ))
        assert_size_stride(arg48_1, (3072, ), (1, ))
        assert_size_stride(arg49_1, (768, 3072), (3072, 1))
        assert_size_stride(arg50_1, (768, ), (1, ))
        assert_size_stride(arg51_1, (3072, 768), (768, 1))
        assert_size_stride(arg52_1, (768, ), (1, ))
        assert_size_stride(arg53_1, (768, ), (1, ))
        assert_size_stride(arg54_1, (2304, ), (1, ))
        assert_size_stride(arg55_1, (768, 2304), (2304, 1))
        assert_size_stride(arg56_1, (768, ), (1, ))
        assert_size_stride(arg57_1, (768, 768), (768, 1))
        assert_size_stride(arg58_1, (768, ), (1, ))
        assert_size_stride(arg59_1, (768, ), (1, ))
        assert_size_stride(arg60_1, (3072, ), (1, ))
        assert_size_stride(arg61_1, (768, 3072), (3072, 1))
        assert_size_stride(arg62_1, (768, ), (1, ))
        assert_size_stride(arg63_1, (3072, 768), (768, 1))
        assert_size_stride(arg64_1, (768, ), (1, ))
        assert_size_stride(arg65_1, (768, ), (1, ))
        assert_size_stride(arg66_1, (2304, ), (1, ))
        assert_size_stride(arg67_1, (768, 2304), (2304, 1))
        assert_size_stride(arg68_1, (768, ), (1, ))
        assert_size_stride(arg69_1, (768, 768), (768, 1))
        assert_size_stride(arg70_1, (768, ), (1, ))
        assert_size_stride(arg71_1, (768, ), (1, ))
        assert_size_stride(arg72_1, (3072, ), (1, ))
        assert_size_stride(arg73_1, (768, 3072), (3072, 1))
        assert_size_stride(arg74_1, (768, ), (1, ))
        assert_size_stride(arg75_1, (3072, 768), (768, 1))
        assert_size_stride(arg76_1, (768, ), (1, ))
        assert_size_stride(arg77_1, (768, ), (1, ))
        assert_size_stride(arg78_1, (2304, ), (1, ))
        assert_size_stride(arg79_1, (768, 2304), (2304, 1))
        assert_size_stride(arg80_1, (768, ), (1, ))
        assert_size_stride(arg81_1, (768, 768), (768, 1))
        assert_size_stride(arg82_1, (768, ), (1, ))
        assert_size_stride(arg83_1, (768, ), (1, ))
        assert_size_stride(arg84_1, (3072, ), (1, ))
        assert_size_stride(arg85_1, (768, 3072), (3072, 1))
        assert_size_stride(arg86_1, (768, ), (1, ))
        assert_size_stride(arg87_1, (3072, 768), (768, 1))
        assert_size_stride(arg88_1, (768, ), (1, ))
        assert_size_stride(arg89_1, (768, ), (1, ))
        assert_size_stride(arg90_1, (2304, ), (1, ))
        assert_size_stride(arg91_1, (768, 2304), (2304, 1))
        assert_size_stride(arg92_1, (768, ), (1, ))
        assert_size_stride(arg93_1, (768, 768), (768, 1))
        assert_size_stride(arg94_1, (768, ), (1, ))
        assert_size_stride(arg95_1, (768, ), (1, ))
        assert_size_stride(arg96_1, (3072, ), (1, ))
        assert_size_stride(arg97_1, (768, 3072), (3072, 1))
        assert_size_stride(arg98_1, (768, ), (1, ))
        assert_size_stride(arg99_1, (3072, 768), (768, 1))
        assert_size_stride(arg100_1, (768, ), (1, ))
        assert_size_stride(arg101_1, (768, ), (1, ))
        assert_size_stride(arg102_1, (2304, ), (1, ))
        assert_size_stride(arg103_1, (768, 2304), (2304, 1))
        assert_size_stride(arg104_1, (768, ), (1, ))
        assert_size_stride(arg105_1, (768, 768), (768, 1))
        assert_size_stride(arg106_1, (768, ), (1, ))
        assert_size_stride(arg107_1, (768, ), (1, ))
        assert_size_stride(arg108_1, (3072, ), (1, ))
        assert_size_stride(arg109_1, (768, 3072), (3072, 1))
        assert_size_stride(arg110_1, (768, ), (1, ))
        assert_size_stride(arg111_1, (3072, 768), (768, 1))
        assert_size_stride(arg112_1, (768, ), (1, ))
        assert_size_stride(arg113_1, (768, ), (1, ))
        assert_size_stride(arg114_1, (2304, ), (1, ))
        assert_size_stride(arg115_1, (768, 2304), (2304, 1))
        assert_size_stride(arg116_1, (768, ), (1, ))
        assert_size_stride(arg117_1, (768, 768), (768, 1))
        assert_size_stride(arg118_1, (768, ), (1, ))
        assert_size_stride(arg119_1, (768, ), (1, ))
        assert_size_stride(arg120_1, (3072, ), (1, ))
        assert_size_stride(arg121_1, (768, 3072), (3072, 1))
        assert_size_stride(arg122_1, (768, ), (1, ))
        assert_size_stride(arg123_1, (3072, 768), (768, 1))
        assert_size_stride(arg124_1, (768, ), (1, ))
        assert_size_stride(arg125_1, (768, ), (1, ))
        assert_size_stride(arg126_1, (2304, ), (1, ))
        assert_size_stride(arg127_1, (768, 2304), (2304, 1))
        assert_size_stride(arg128_1, (768, ), (1, ))
        assert_size_stride(arg129_1, (768, 768), (768, 1))
        assert_size_stride(arg130_1, (768, ), (1, ))
        assert_size_stride(arg131_1, (768, ), (1, ))
        assert_size_stride(arg132_1, (3072, ), (1, ))
        assert_size_stride(arg133_1, (768, 3072), (3072, 1))
        assert_size_stride(arg134_1, (768, ), (1, ))
        assert_size_stride(arg135_1, (3072, 768), (768, 1))
        assert_size_stride(arg136_1, (768, ), (1, ))
        assert_size_stride(arg137_1, (768, ), (1, ))
        assert_size_stride(arg138_1, (2304, ), (1, ))
        assert_size_stride(arg139_1, (768, 2304), (2304, 1))
        assert_size_stride(arg140_1, (768, ), (1, ))
        assert_size_stride(arg141_1, (768, 768), (768, 1))
        assert_size_stride(arg142_1, (768, ), (1, ))
        assert_size_stride(arg143_1, (768, ), (1, ))
        assert_size_stride(arg144_1, (3072, ), (1, ))
        assert_size_stride(arg145_1, (768, 3072), (3072, 1))
        assert_size_stride(arg146_1, (768, ), (1, ))
        assert_size_stride(arg147_1, (3072, 768), (768, 1))
        assert_size_stride(arg148_1, (768, ), (1, ))
        assert_size_stride(arg149_1, (768, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf3 = empty_strided_cuda((8, 128, 768), (98304, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [inputs_embeds, cache_position, position_ids, position_embeds, hidden_states, hidden_states_2], Original ATen: [aten.embedding, aten.arange, aten.unsqueeze, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_arange_embedding_native_layer_norm_unsqueeze_0.run(arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, buf3, 1024, 768, stream=stream0)
            del arg4_1
            del arg5_1
            buf4 = empty_strided_cuda((1024, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [inputs_embeds, cache_position, position_ids, position_embeds, hidden_states, hidden_states_2, view_1, x], Original ATen: [aten.embedding, aten.arange, aten.unsqueeze, aten.add, aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.addmm(arg6_1, reinterpret_tensor(buf3, (1024, 768), (768, 1), 0), arg7_1, alpha=1, beta=1, out=buf4)
            del arg6_1
            del arg7_1
            # Topologically Sorted Source Nodes: [x_1, split, view_5, query_states_1, view_3, key_states_1, view_4, value_states_1, attn_output], Original ATen: [aten.view, aten.split, aten.transpose, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf4, (8, 12, 128, 64), (294912, 64, 2304, 1), 0), reinterpret_tensor(buf4, (8, 12, 128, 64), (294912, 64, 2304, 1), 768), reinterpret_tensor(buf4, (8, 12, 128, 64), (294912, 64, 2304, 1), 1536), reinterpret_tensor(arg0_1, (8, 12, 128, 128), (0, 0, 128, 1), 0), False)
            del buf4
            buf6 = buf5[0]
            assert_size_stride(buf6, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf6, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf5
            buf10 = reinterpret_tensor(buf3, (1024, 768), (768, 1), 0); del buf3  # reuse
            # Topologically Sorted Source Nodes: [transpose_3, reshape, view_6, x_2], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf6, (1024, 768), (768, 1), 0), arg9_1, out=buf10)
            del arg9_1
            buf11 = reinterpret_tensor(buf10, (8, 128, 768), (98304, 768, 1), 0); del buf10  # reuse
            buf15 = reinterpret_tensor(buf6, (8, 128, 768), (98304, 768, 1), 0); del buf6  # reuse
            # Topologically Sorted Source Nodes: [inputs_embeds, cache_position, position_ids, position_embeds, hidden_states, x_2, x_3, hidden_states_3, hidden_states_4], Original ATen: [aten.embedding, aten.arange, aten.unsqueeze, aten.add, aten.addmm, aten.view, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_arange_embedding_native_layer_norm_unsqueeze_view_1.run(buf11, arg8_1, arg1_1, arg2_1, arg3_1, arg10_1, arg11_1, buf15, 1024, 768, stream=stream0)
            del arg10_1
            del arg11_1
            del arg1_1
            del arg2_1
            del arg3_1
            del arg8_1
            buf16 = empty_strided_cuda((1024, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_4, view_8, x_4], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf15, (1024, 768), (768, 1), 0), arg13_1, out=buf16)
            del arg13_1
            del buf15
            buf17 = reinterpret_tensor(buf16, (8, 128, 3072), (393216, 3072, 1), 0); del buf16  # reuse
            # Topologically Sorted Source Nodes: [x_4, x_5, mul, pow_1, mul_1, add_2, mul_2, tanh, add_3, hidden_states_5], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_2.run(buf17, arg12_1, 3145728, stream=stream0)
            del arg12_1
            buf18 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_4, x_5, mul, pow_1, mul_1, add_2, mul_2, tanh, add_3, hidden_states_5, view_10, x_6], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf17, (1024, 3072), (3072, 1), 0), arg15_1, out=buf18)
            del arg15_1
            del buf17
            buf22 = empty_strided_cuda((8, 128, 768), (98304, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_6, x_7, hidden_states_7, hidden_states_8], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_3.run(buf11, buf18, arg14_1, arg16_1, arg17_1, buf22, 1024, 768, stream=stream0)
            del arg16_1
            del arg17_1
            buf23 = empty_strided_cuda((1024, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_6, x_7, hidden_states_7, hidden_states_8, view_12, x_8], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg18_1, reinterpret_tensor(buf22, (1024, 768), (768, 1), 0), arg19_1, alpha=1, beta=1, out=buf23)
            del arg18_1
            del arg19_1
            # Topologically Sorted Source Nodes: [x_9, split_1, view_16, query_states_3, view_14, key_states_3, view_15, value_states_3, attn_output_4], Original ATen: [aten.view, aten.split, aten.transpose, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf24 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf23, (8, 12, 128, 64), (294912, 64, 2304, 1), 0), reinterpret_tensor(buf23, (8, 12, 128, 64), (294912, 64, 2304, 1), 768), reinterpret_tensor(buf23, (8, 12, 128, 64), (294912, 64, 2304, 1), 1536), reinterpret_tensor(arg0_1, (8, 12, 128, 128), (0, 0, 128, 1), 0), False)
            del buf23
            buf25 = buf24[0]
            assert_size_stride(buf25, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf25, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf24
            buf29 = reinterpret_tensor(buf22, (1024, 768), (768, 1), 0); del buf22  # reuse
            # Topologically Sorted Source Nodes: [transpose_7, reshape_1, view_17, x_10], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf25, (1024, 768), (768, 1), 0), arg21_1, out=buf29)
            del arg21_1
            buf30 = reinterpret_tensor(buf29, (8, 128, 768), (98304, 768, 1), 0); del buf29  # reuse
            buf34 = reinterpret_tensor(buf25, (8, 128, 768), (98304, 768, 1), 0); del buf25  # reuse
            # Topologically Sorted Source Nodes: [x_6, x_7, hidden_states_7, x_10, x_11, hidden_states_9, hidden_states_10], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf30, arg20_1, buf11, buf18, arg14_1, arg22_1, arg23_1, buf34, 1024, 768, stream=stream0)
            del arg14_1
            del arg20_1
            del arg22_1
            del arg23_1
            del buf11
            del buf18
            buf35 = empty_strided_cuda((1024, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_10, view_19, x_12], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf34, (1024, 768), (768, 1), 0), arg25_1, out=buf35)
            del arg25_1
            del buf34
            buf36 = reinterpret_tensor(buf35, (8, 128, 3072), (393216, 3072, 1), 0); del buf35  # reuse
            # Topologically Sorted Source Nodes: [x_12, x_13, mul_4, pow_2, mul_5, add_6, mul_6, tanh_1, add_7, hidden_states_11], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_2.run(buf36, arg24_1, 3145728, stream=stream0)
            del arg24_1
            buf37 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_12, x_13, mul_4, pow_2, mul_5, add_6, mul_6, tanh_1, add_7, hidden_states_11, view_21, x_14], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf36, (1024, 3072), (3072, 1), 0), arg27_1, out=buf37)
            del arg27_1
            del buf36
            buf41 = empty_strided_cuda((8, 128, 768), (98304, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_14, x_15, hidden_states_13, hidden_states_14], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_3.run(buf30, buf37, arg26_1, arg28_1, arg29_1, buf41, 1024, 768, stream=stream0)
            del arg28_1
            del arg29_1
            buf42 = empty_strided_cuda((1024, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_14, x_15, hidden_states_13, hidden_states_14, view_23, x_16], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg30_1, reinterpret_tensor(buf41, (1024, 768), (768, 1), 0), arg31_1, alpha=1, beta=1, out=buf42)
            del arg30_1
            del arg31_1
            # Topologically Sorted Source Nodes: [x_17, split_2, view_27, query_states_5, view_25, key_states_5, view_26, value_states_5, attn_output_8], Original ATen: [aten.view, aten.split, aten.transpose, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf43 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf42, (8, 12, 128, 64), (294912, 64, 2304, 1), 0), reinterpret_tensor(buf42, (8, 12, 128, 64), (294912, 64, 2304, 1), 768), reinterpret_tensor(buf42, (8, 12, 128, 64), (294912, 64, 2304, 1), 1536), reinterpret_tensor(arg0_1, (8, 12, 128, 128), (0, 0, 128, 1), 0), False)
            del buf42
            buf44 = buf43[0]
            assert_size_stride(buf44, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf44, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf43
            buf48 = reinterpret_tensor(buf41, (1024, 768), (768, 1), 0); del buf41  # reuse
            # Topologically Sorted Source Nodes: [transpose_11, reshape_2, view_28, x_18], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf44, (1024, 768), (768, 1), 0), arg33_1, out=buf48)
            del arg33_1
            buf49 = reinterpret_tensor(buf48, (8, 128, 768), (98304, 768, 1), 0); del buf48  # reuse
            buf53 = reinterpret_tensor(buf44, (8, 128, 768), (98304, 768, 1), 0); del buf44  # reuse
            # Topologically Sorted Source Nodes: [x_14, x_15, hidden_states_13, x_18, x_19, hidden_states_15, hidden_states_16], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf49, arg32_1, buf30, buf37, arg26_1, arg34_1, arg35_1, buf53, 1024, 768, stream=stream0)
            del arg26_1
            del arg32_1
            del arg34_1
            del arg35_1
            del buf30
            del buf37
            buf54 = empty_strided_cuda((1024, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_16, view_30, x_20], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf53, (1024, 768), (768, 1), 0), arg37_1, out=buf54)
            del arg37_1
            del buf53
            buf55 = reinterpret_tensor(buf54, (8, 128, 3072), (393216, 3072, 1), 0); del buf54  # reuse
            # Topologically Sorted Source Nodes: [x_20, x_21, mul_8, pow_3, mul_9, add_10, mul_10, tanh_2, add_11, hidden_states_17], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_2.run(buf55, arg36_1, 3145728, stream=stream0)
            del arg36_1
            buf56 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_20, x_21, mul_8, pow_3, mul_9, add_10, mul_10, tanh_2, add_11, hidden_states_17, view_32, x_22], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf55, (1024, 3072), (3072, 1), 0), arg39_1, out=buf56)
            del arg39_1
            del buf55
            buf60 = empty_strided_cuda((8, 128, 768), (98304, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_22, x_23, hidden_states_19, hidden_states_20], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_3.run(buf49, buf56, arg38_1, arg40_1, arg41_1, buf60, 1024, 768, stream=stream0)
            del arg40_1
            del arg41_1
            buf61 = empty_strided_cuda((1024, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_22, x_23, hidden_states_19, hidden_states_20, view_34, x_24], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg42_1, reinterpret_tensor(buf60, (1024, 768), (768, 1), 0), arg43_1, alpha=1, beta=1, out=buf61)
            del arg42_1
            del arg43_1
            # Topologically Sorted Source Nodes: [x_25, split_3, view_38, query_states_7, view_36, key_states_7, view_37, value_states_7, attn_output_12], Original ATen: [aten.view, aten.split, aten.transpose, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf62 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf61, (8, 12, 128, 64), (294912, 64, 2304, 1), 0), reinterpret_tensor(buf61, (8, 12, 128, 64), (294912, 64, 2304, 1), 768), reinterpret_tensor(buf61, (8, 12, 128, 64), (294912, 64, 2304, 1), 1536), reinterpret_tensor(arg0_1, (8, 12, 128, 128), (0, 0, 128, 1), 0), False)
            del buf61
            buf63 = buf62[0]
            assert_size_stride(buf63, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf63, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf62
            buf67 = reinterpret_tensor(buf60, (1024, 768), (768, 1), 0); del buf60  # reuse
            # Topologically Sorted Source Nodes: [transpose_15, reshape_3, view_39, x_26], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf63, (1024, 768), (768, 1), 0), arg45_1, out=buf67)
            del arg45_1
            buf68 = reinterpret_tensor(buf67, (8, 128, 768), (98304, 768, 1), 0); del buf67  # reuse
            buf72 = reinterpret_tensor(buf63, (8, 128, 768), (98304, 768, 1), 0); del buf63  # reuse
            # Topologically Sorted Source Nodes: [x_22, x_23, hidden_states_19, x_26, x_27, hidden_states_21, hidden_states_22], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf68, arg44_1, buf49, buf56, arg38_1, arg46_1, arg47_1, buf72, 1024, 768, stream=stream0)
            del arg38_1
            del arg44_1
            del arg46_1
            del arg47_1
            del buf49
            del buf56
            buf73 = empty_strided_cuda((1024, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_22, view_41, x_28], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf72, (1024, 768), (768, 1), 0), arg49_1, out=buf73)
            del arg49_1
            del buf72
            buf74 = reinterpret_tensor(buf73, (8, 128, 3072), (393216, 3072, 1), 0); del buf73  # reuse
            # Topologically Sorted Source Nodes: [x_28, x_29, mul_12, pow_4, mul_13, add_14, mul_14, tanh_3, add_15, hidden_states_23], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_2.run(buf74, arg48_1, 3145728, stream=stream0)
            del arg48_1
            buf75 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_28, x_29, mul_12, pow_4, mul_13, add_14, mul_14, tanh_3, add_15, hidden_states_23, view_43, x_30], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf74, (1024, 3072), (3072, 1), 0), arg51_1, out=buf75)
            del arg51_1
            del buf74
            buf79 = empty_strided_cuda((8, 128, 768), (98304, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_30, x_31, hidden_states_25, hidden_states_26], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_3.run(buf68, buf75, arg50_1, arg52_1, arg53_1, buf79, 1024, 768, stream=stream0)
            del arg52_1
            del arg53_1
            buf80 = empty_strided_cuda((1024, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_30, x_31, hidden_states_25, hidden_states_26, view_45, x_32], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg54_1, reinterpret_tensor(buf79, (1024, 768), (768, 1), 0), arg55_1, alpha=1, beta=1, out=buf80)
            del arg54_1
            del arg55_1
            # Topologically Sorted Source Nodes: [x_33, split_4, view_49, query_states_9, view_47, key_states_9, view_48, value_states_9, attn_output_16], Original ATen: [aten.view, aten.split, aten.transpose, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf81 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf80, (8, 12, 128, 64), (294912, 64, 2304, 1), 0), reinterpret_tensor(buf80, (8, 12, 128, 64), (294912, 64, 2304, 1), 768), reinterpret_tensor(buf80, (8, 12, 128, 64), (294912, 64, 2304, 1), 1536), reinterpret_tensor(arg0_1, (8, 12, 128, 128), (0, 0, 128, 1), 0), False)
            del buf80
            buf82 = buf81[0]
            assert_size_stride(buf82, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf82, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf81
            buf86 = reinterpret_tensor(buf79, (1024, 768), (768, 1), 0); del buf79  # reuse
            # Topologically Sorted Source Nodes: [transpose_19, reshape_4, view_50, x_34], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf82, (1024, 768), (768, 1), 0), arg57_1, out=buf86)
            del arg57_1
            buf87 = reinterpret_tensor(buf86, (8, 128, 768), (98304, 768, 1), 0); del buf86  # reuse
            buf91 = reinterpret_tensor(buf82, (8, 128, 768), (98304, 768, 1), 0); del buf82  # reuse
            # Topologically Sorted Source Nodes: [x_30, x_31, hidden_states_25, x_34, x_35, hidden_states_27, hidden_states_28], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf87, arg56_1, buf68, buf75, arg50_1, arg58_1, arg59_1, buf91, 1024, 768, stream=stream0)
            del arg50_1
            del arg56_1
            del arg58_1
            del arg59_1
            del buf68
            del buf75
            buf92 = empty_strided_cuda((1024, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_28, view_52, x_36], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf91, (1024, 768), (768, 1), 0), arg61_1, out=buf92)
            del arg61_1
            del buf91
            buf93 = reinterpret_tensor(buf92, (8, 128, 3072), (393216, 3072, 1), 0); del buf92  # reuse
            # Topologically Sorted Source Nodes: [x_36, x_37, mul_16, pow_5, mul_17, add_18, mul_18, tanh_4, add_19, hidden_states_29], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_2.run(buf93, arg60_1, 3145728, stream=stream0)
            del arg60_1
            buf94 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_36, x_37, mul_16, pow_5, mul_17, add_18, mul_18, tanh_4, add_19, hidden_states_29, view_54, x_38], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf93, (1024, 3072), (3072, 1), 0), arg63_1, out=buf94)
            del arg63_1
            del buf93
            buf98 = empty_strided_cuda((8, 128, 768), (98304, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_38, x_39, hidden_states_31, hidden_states_32], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_3.run(buf87, buf94, arg62_1, arg64_1, arg65_1, buf98, 1024, 768, stream=stream0)
            del arg64_1
            del arg65_1
            buf99 = empty_strided_cuda((1024, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_38, x_39, hidden_states_31, hidden_states_32, view_56, x_40], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg66_1, reinterpret_tensor(buf98, (1024, 768), (768, 1), 0), arg67_1, alpha=1, beta=1, out=buf99)
            del arg66_1
            del arg67_1
            # Topologically Sorted Source Nodes: [x_41, split_5, view_60, query_states_11, view_58, key_states_11, view_59, value_states_11, attn_output_20], Original ATen: [aten.view, aten.split, aten.transpose, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf100 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf99, (8, 12, 128, 64), (294912, 64, 2304, 1), 0), reinterpret_tensor(buf99, (8, 12, 128, 64), (294912, 64, 2304, 1), 768), reinterpret_tensor(buf99, (8, 12, 128, 64), (294912, 64, 2304, 1), 1536), reinterpret_tensor(arg0_1, (8, 12, 128, 128), (0, 0, 128, 1), 0), False)
            del buf99
            buf101 = buf100[0]
            assert_size_stride(buf101, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf101, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf100
            buf105 = reinterpret_tensor(buf98, (1024, 768), (768, 1), 0); del buf98  # reuse
            # Topologically Sorted Source Nodes: [transpose_23, reshape_5, view_61, x_42], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf101, (1024, 768), (768, 1), 0), arg69_1, out=buf105)
            del arg69_1
            buf106 = reinterpret_tensor(buf105, (8, 128, 768), (98304, 768, 1), 0); del buf105  # reuse
            buf110 = reinterpret_tensor(buf101, (8, 128, 768), (98304, 768, 1), 0); del buf101  # reuse
            # Topologically Sorted Source Nodes: [x_38, x_39, hidden_states_31, x_42, x_43, hidden_states_33, hidden_states_34], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf106, arg68_1, buf87, buf94, arg62_1, arg70_1, arg71_1, buf110, 1024, 768, stream=stream0)
            del arg62_1
            del arg68_1
            del arg70_1
            del arg71_1
            del buf87
            del buf94
            buf111 = empty_strided_cuda((1024, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_34, view_63, x_44], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf110, (1024, 768), (768, 1), 0), arg73_1, out=buf111)
            del arg73_1
            del buf110
            buf112 = reinterpret_tensor(buf111, (8, 128, 3072), (393216, 3072, 1), 0); del buf111  # reuse
            # Topologically Sorted Source Nodes: [x_44, x_45, mul_20, pow_6, mul_21, add_22, mul_22, tanh_5, add_23, hidden_states_35], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_2.run(buf112, arg72_1, 3145728, stream=stream0)
            del arg72_1
            buf113 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_44, x_45, mul_20, pow_6, mul_21, add_22, mul_22, tanh_5, add_23, hidden_states_35, view_65, x_46], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf112, (1024, 3072), (3072, 1), 0), arg75_1, out=buf113)
            del arg75_1
            del buf112
            buf117 = empty_strided_cuda((8, 128, 768), (98304, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_46, x_47, hidden_states_37, hidden_states_38], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_3.run(buf106, buf113, arg74_1, arg76_1, arg77_1, buf117, 1024, 768, stream=stream0)
            del arg76_1
            del arg77_1
            buf118 = empty_strided_cuda((1024, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_46, x_47, hidden_states_37, hidden_states_38, view_67, x_48], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg78_1, reinterpret_tensor(buf117, (1024, 768), (768, 1), 0), arg79_1, alpha=1, beta=1, out=buf118)
            del arg78_1
            del arg79_1
            # Topologically Sorted Source Nodes: [x_49, split_6, view_71, query_states_13, view_69, key_states_13, view_70, value_states_13, attn_output_24], Original ATen: [aten.view, aten.split, aten.transpose, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf119 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf118, (8, 12, 128, 64), (294912, 64, 2304, 1), 0), reinterpret_tensor(buf118, (8, 12, 128, 64), (294912, 64, 2304, 1), 768), reinterpret_tensor(buf118, (8, 12, 128, 64), (294912, 64, 2304, 1), 1536), reinterpret_tensor(arg0_1, (8, 12, 128, 128), (0, 0, 128, 1), 0), False)
            del buf118
            buf120 = buf119[0]
            assert_size_stride(buf120, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf120, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf119
            buf124 = reinterpret_tensor(buf117, (1024, 768), (768, 1), 0); del buf117  # reuse
            # Topologically Sorted Source Nodes: [transpose_27, reshape_6, view_72, x_50], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf120, (1024, 768), (768, 1), 0), arg81_1, out=buf124)
            del arg81_1
            buf125 = reinterpret_tensor(buf124, (8, 128, 768), (98304, 768, 1), 0); del buf124  # reuse
            buf129 = reinterpret_tensor(buf120, (8, 128, 768), (98304, 768, 1), 0); del buf120  # reuse
            # Topologically Sorted Source Nodes: [x_46, x_47, hidden_states_37, x_50, x_51, hidden_states_39, hidden_states_40], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf125, arg80_1, buf106, buf113, arg74_1, arg82_1, arg83_1, buf129, 1024, 768, stream=stream0)
            del arg74_1
            del arg80_1
            del arg82_1
            del arg83_1
            del buf106
            del buf113
            buf130 = empty_strided_cuda((1024, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_40, view_74, x_52], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf129, (1024, 768), (768, 1), 0), arg85_1, out=buf130)
            del arg85_1
            del buf129
            buf131 = reinterpret_tensor(buf130, (8, 128, 3072), (393216, 3072, 1), 0); del buf130  # reuse
            # Topologically Sorted Source Nodes: [x_52, x_53, mul_24, pow_7, mul_25, add_26, mul_26, tanh_6, add_27, hidden_states_41], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_2.run(buf131, arg84_1, 3145728, stream=stream0)
            del arg84_1
            buf132 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_52, x_53, mul_24, pow_7, mul_25, add_26, mul_26, tanh_6, add_27, hidden_states_41, view_76, x_54], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf131, (1024, 3072), (3072, 1), 0), arg87_1, out=buf132)
            del arg87_1
            del buf131
            buf136 = empty_strided_cuda((8, 128, 768), (98304, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_54, x_55, hidden_states_43, hidden_states_44], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_3.run(buf125, buf132, arg86_1, arg88_1, arg89_1, buf136, 1024, 768, stream=stream0)
            del arg88_1
            del arg89_1
            buf137 = empty_strided_cuda((1024, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_54, x_55, hidden_states_43, hidden_states_44, view_78, x_56], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg90_1, reinterpret_tensor(buf136, (1024, 768), (768, 1), 0), arg91_1, alpha=1, beta=1, out=buf137)
            del arg90_1
            del arg91_1
            # Topologically Sorted Source Nodes: [x_57, split_7, view_82, query_states_15, view_80, key_states_15, view_81, value_states_15, attn_output_28], Original ATen: [aten.view, aten.split, aten.transpose, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf138 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf137, (8, 12, 128, 64), (294912, 64, 2304, 1), 0), reinterpret_tensor(buf137, (8, 12, 128, 64), (294912, 64, 2304, 1), 768), reinterpret_tensor(buf137, (8, 12, 128, 64), (294912, 64, 2304, 1), 1536), reinterpret_tensor(arg0_1, (8, 12, 128, 128), (0, 0, 128, 1), 0), False)
            del buf137
            buf139 = buf138[0]
            assert_size_stride(buf139, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf139, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf138
            buf143 = reinterpret_tensor(buf136, (1024, 768), (768, 1), 0); del buf136  # reuse
            # Topologically Sorted Source Nodes: [transpose_31, reshape_7, view_83, x_58], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf139, (1024, 768), (768, 1), 0), arg93_1, out=buf143)
            del arg93_1
            buf144 = reinterpret_tensor(buf143, (8, 128, 768), (98304, 768, 1), 0); del buf143  # reuse
            buf148 = reinterpret_tensor(buf139, (8, 128, 768), (98304, 768, 1), 0); del buf139  # reuse
            # Topologically Sorted Source Nodes: [x_54, x_55, hidden_states_43, x_58, x_59, hidden_states_45, hidden_states_46], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf144, arg92_1, buf125, buf132, arg86_1, arg94_1, arg95_1, buf148, 1024, 768, stream=stream0)
            del arg86_1
            del arg92_1
            del arg94_1
            del arg95_1
            del buf125
            del buf132
            buf149 = empty_strided_cuda((1024, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_46, view_85, x_60], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf148, (1024, 768), (768, 1), 0), arg97_1, out=buf149)
            del arg97_1
            del buf148
            buf150 = reinterpret_tensor(buf149, (8, 128, 3072), (393216, 3072, 1), 0); del buf149  # reuse
            # Topologically Sorted Source Nodes: [x_60, x_61, mul_28, pow_8, mul_29, add_30, mul_30, tanh_7, add_31, hidden_states_47], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_2.run(buf150, arg96_1, 3145728, stream=stream0)
            del arg96_1
            buf151 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_60, x_61, mul_28, pow_8, mul_29, add_30, mul_30, tanh_7, add_31, hidden_states_47, view_87, x_62], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf150, (1024, 3072), (3072, 1), 0), arg99_1, out=buf151)
            del arg99_1
            del buf150
            buf155 = empty_strided_cuda((8, 128, 768), (98304, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_62, x_63, hidden_states_49, hidden_states_50], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_3.run(buf144, buf151, arg98_1, arg100_1, arg101_1, buf155, 1024, 768, stream=stream0)
            del arg100_1
            del arg101_1
            buf156 = empty_strided_cuda((1024, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_62, x_63, hidden_states_49, hidden_states_50, view_89, x_64], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg102_1, reinterpret_tensor(buf155, (1024, 768), (768, 1), 0), arg103_1, alpha=1, beta=1, out=buf156)
            del arg102_1
            del arg103_1
            # Topologically Sorted Source Nodes: [x_65, split_8, view_93, query_states_17, view_91, key_states_17, view_92, value_states_17, attn_output_32], Original ATen: [aten.view, aten.split, aten.transpose, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf157 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf156, (8, 12, 128, 64), (294912, 64, 2304, 1), 0), reinterpret_tensor(buf156, (8, 12, 128, 64), (294912, 64, 2304, 1), 768), reinterpret_tensor(buf156, (8, 12, 128, 64), (294912, 64, 2304, 1), 1536), reinterpret_tensor(arg0_1, (8, 12, 128, 128), (0, 0, 128, 1), 0), False)
            del buf156
            buf158 = buf157[0]
            assert_size_stride(buf158, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf158, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf157
            buf162 = reinterpret_tensor(buf155, (1024, 768), (768, 1), 0); del buf155  # reuse
            # Topologically Sorted Source Nodes: [transpose_35, reshape_8, view_94, x_66], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf158, (1024, 768), (768, 1), 0), arg105_1, out=buf162)
            del arg105_1
            buf163 = reinterpret_tensor(buf162, (8, 128, 768), (98304, 768, 1), 0); del buf162  # reuse
            buf167 = reinterpret_tensor(buf158, (8, 128, 768), (98304, 768, 1), 0); del buf158  # reuse
            # Topologically Sorted Source Nodes: [x_62, x_63, hidden_states_49, x_66, x_67, hidden_states_51, hidden_states_52], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf163, arg104_1, buf144, buf151, arg98_1, arg106_1, arg107_1, buf167, 1024, 768, stream=stream0)
            del arg104_1
            del arg106_1
            del arg107_1
            del arg98_1
            del buf144
            del buf151
            buf168 = empty_strided_cuda((1024, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_52, view_96, x_68], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf167, (1024, 768), (768, 1), 0), arg109_1, out=buf168)
            del arg109_1
            del buf167
            buf169 = reinterpret_tensor(buf168, (8, 128, 3072), (393216, 3072, 1), 0); del buf168  # reuse
            # Topologically Sorted Source Nodes: [x_68, x_69, mul_32, pow_9, mul_33, add_34, mul_34, tanh_8, add_35, hidden_states_53], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_2.run(buf169, arg108_1, 3145728, stream=stream0)
            del arg108_1
            buf170 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_68, x_69, mul_32, pow_9, mul_33, add_34, mul_34, tanh_8, add_35, hidden_states_53, view_98, x_70], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf169, (1024, 3072), (3072, 1), 0), arg111_1, out=buf170)
            del arg111_1
            del buf169
            buf174 = empty_strided_cuda((8, 128, 768), (98304, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_70, x_71, hidden_states_55, hidden_states_56], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_3.run(buf163, buf170, arg110_1, arg112_1, arg113_1, buf174, 1024, 768, stream=stream0)
            del arg112_1
            del arg113_1
            buf175 = empty_strided_cuda((1024, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_70, x_71, hidden_states_55, hidden_states_56, view_100, x_72], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg114_1, reinterpret_tensor(buf174, (1024, 768), (768, 1), 0), arg115_1, alpha=1, beta=1, out=buf175)
            del arg114_1
            del arg115_1
            # Topologically Sorted Source Nodes: [x_73, split_9, view_104, query_states_19, view_102, key_states_19, view_103, value_states_19, attn_output_36], Original ATen: [aten.view, aten.split, aten.transpose, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf176 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf175, (8, 12, 128, 64), (294912, 64, 2304, 1), 0), reinterpret_tensor(buf175, (8, 12, 128, 64), (294912, 64, 2304, 1), 768), reinterpret_tensor(buf175, (8, 12, 128, 64), (294912, 64, 2304, 1), 1536), reinterpret_tensor(arg0_1, (8, 12, 128, 128), (0, 0, 128, 1), 0), False)
            del buf175
            buf177 = buf176[0]
            assert_size_stride(buf177, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf177, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf176
            buf181 = reinterpret_tensor(buf174, (1024, 768), (768, 1), 0); del buf174  # reuse
            # Topologically Sorted Source Nodes: [transpose_39, reshape_9, view_105, x_74], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf177, (1024, 768), (768, 1), 0), arg117_1, out=buf181)
            del arg117_1
            buf182 = reinterpret_tensor(buf181, (8, 128, 768), (98304, 768, 1), 0); del buf181  # reuse
            buf186 = reinterpret_tensor(buf177, (8, 128, 768), (98304, 768, 1), 0); del buf177  # reuse
            # Topologically Sorted Source Nodes: [x_70, x_71, hidden_states_55, x_74, x_75, hidden_states_57, hidden_states_58], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf182, arg116_1, buf163, buf170, arg110_1, arg118_1, arg119_1, buf186, 1024, 768, stream=stream0)
            del arg110_1
            del arg116_1
            del arg118_1
            del arg119_1
            del buf163
            del buf170
            buf187 = empty_strided_cuda((1024, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_58, view_107, x_76], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf186, (1024, 768), (768, 1), 0), arg121_1, out=buf187)
            del arg121_1
            del buf186
            buf188 = reinterpret_tensor(buf187, (8, 128, 3072), (393216, 3072, 1), 0); del buf187  # reuse
            # Topologically Sorted Source Nodes: [x_76, x_77, mul_36, pow_10, mul_37, add_38, mul_38, tanh_9, add_39, hidden_states_59], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_2.run(buf188, arg120_1, 3145728, stream=stream0)
            del arg120_1
            buf189 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_76, x_77, mul_36, pow_10, mul_37, add_38, mul_38, tanh_9, add_39, hidden_states_59, view_109, x_78], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf188, (1024, 3072), (3072, 1), 0), arg123_1, out=buf189)
            del arg123_1
            del buf188
            buf193 = empty_strided_cuda((8, 128, 768), (98304, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_78, x_79, hidden_states_61, hidden_states_62], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_3.run(buf182, buf189, arg122_1, arg124_1, arg125_1, buf193, 1024, 768, stream=stream0)
            del arg124_1
            del arg125_1
            buf194 = empty_strided_cuda((1024, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_78, x_79, hidden_states_61, hidden_states_62, view_111, x_80], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg126_1, reinterpret_tensor(buf193, (1024, 768), (768, 1), 0), arg127_1, alpha=1, beta=1, out=buf194)
            del arg126_1
            del arg127_1
            # Topologically Sorted Source Nodes: [x_81, split_10, view_115, query_states_21, view_113, key_states_21, view_114, value_states_21, attn_output_40], Original ATen: [aten.view, aten.split, aten.transpose, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf195 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf194, (8, 12, 128, 64), (294912, 64, 2304, 1), 0), reinterpret_tensor(buf194, (8, 12, 128, 64), (294912, 64, 2304, 1), 768), reinterpret_tensor(buf194, (8, 12, 128, 64), (294912, 64, 2304, 1), 1536), reinterpret_tensor(arg0_1, (8, 12, 128, 128), (0, 0, 128, 1), 0), False)
            del buf194
            buf196 = buf195[0]
            assert_size_stride(buf196, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf196, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf195
            buf200 = reinterpret_tensor(buf193, (1024, 768), (768, 1), 0); del buf193  # reuse
            # Topologically Sorted Source Nodes: [transpose_43, reshape_10, view_116, x_82], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf196, (1024, 768), (768, 1), 0), arg129_1, out=buf200)
            del arg129_1
            buf201 = reinterpret_tensor(buf200, (8, 128, 768), (98304, 768, 1), 0); del buf200  # reuse
            buf205 = reinterpret_tensor(buf196, (8, 128, 768), (98304, 768, 1), 0); del buf196  # reuse
            # Topologically Sorted Source Nodes: [x_78, x_79, hidden_states_61, x_82, x_83, hidden_states_63, hidden_states_64], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf201, arg128_1, buf182, buf189, arg122_1, arg130_1, arg131_1, buf205, 1024, 768, stream=stream0)
            del arg122_1
            del arg128_1
            del arg130_1
            del arg131_1
            del buf182
            del buf189
            buf206 = empty_strided_cuda((1024, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_64, view_118, x_84], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf205, (1024, 768), (768, 1), 0), arg133_1, out=buf206)
            del arg133_1
            del buf205
            buf207 = reinterpret_tensor(buf206, (8, 128, 3072), (393216, 3072, 1), 0); del buf206  # reuse
            # Topologically Sorted Source Nodes: [x_84, x_85, mul_40, pow_11, mul_41, add_42, mul_42, tanh_10, add_43, hidden_states_65], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_2.run(buf207, arg132_1, 3145728, stream=stream0)
            del arg132_1
            buf208 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_84, x_85, mul_40, pow_11, mul_41, add_42, mul_42, tanh_10, add_43, hidden_states_65, view_120, x_86], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf207, (1024, 3072), (3072, 1), 0), arg135_1, out=buf208)
            del arg135_1
            del buf207
            buf212 = empty_strided_cuda((8, 128, 768), (98304, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_86, x_87, hidden_states_67, hidden_states_68], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_3.run(buf201, buf208, arg134_1, arg136_1, arg137_1, buf212, 1024, 768, stream=stream0)
            del arg136_1
            del arg137_1
            buf213 = empty_strided_cuda((1024, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_86, x_87, hidden_states_67, hidden_states_68, view_122, x_88], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg138_1, reinterpret_tensor(buf212, (1024, 768), (768, 1), 0), arg139_1, alpha=1, beta=1, out=buf213)
            del arg138_1
            del arg139_1
            # Topologically Sorted Source Nodes: [x_89, split_11, view_126, query_states_23, view_124, key_states_23, view_125, value_states_23, attn_output_44], Original ATen: [aten.view, aten.split, aten.transpose, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf214 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf213, (8, 12, 128, 64), (294912, 64, 2304, 1), 0), reinterpret_tensor(buf213, (8, 12, 128, 64), (294912, 64, 2304, 1), 768), reinterpret_tensor(buf213, (8, 12, 128, 64), (294912, 64, 2304, 1), 1536), reinterpret_tensor(arg0_1, (8, 12, 128, 128), (0, 0, 128, 1), 0), False)
            del arg0_1
            del buf213
            buf215 = buf214[0]
            assert_size_stride(buf215, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf215, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf214
            buf219 = reinterpret_tensor(buf212, (1024, 768), (768, 1), 0); del buf212  # reuse
            # Topologically Sorted Source Nodes: [transpose_47, reshape_11, view_127, x_90], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf215, (1024, 768), (768, 1), 0), arg141_1, out=buf219)
            del arg141_1
            buf220 = reinterpret_tensor(buf219, (8, 128, 768), (98304, 768, 1), 0); del buf219  # reuse
            buf224 = reinterpret_tensor(buf215, (8, 128, 768), (98304, 768, 1), 0); del buf215  # reuse
            # Topologically Sorted Source Nodes: [x_86, x_87, hidden_states_67, x_90, x_91, hidden_states_69, hidden_states_70], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_4.run(buf220, arg140_1, buf201, buf208, arg134_1, arg142_1, arg143_1, buf224, 1024, 768, stream=stream0)
            del arg134_1
            del arg140_1
            del arg142_1
            del arg143_1
            del buf201
            del buf208
            buf225 = empty_strided_cuda((1024, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_70, view_129, x_92], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf224, (1024, 768), (768, 1), 0), arg145_1, out=buf225)
            del arg145_1
            del buf224
            buf226 = reinterpret_tensor(buf225, (8, 128, 3072), (393216, 3072, 1), 0); del buf225  # reuse
            # Topologically Sorted Source Nodes: [x_92, x_93, mul_44, pow_12, mul_45, add_46, mul_46, tanh_11, add_47, hidden_states_71], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_2.run(buf226, arg144_1, 3145728, stream=stream0)
            del arg144_1
            buf227 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_92, x_93, mul_44, pow_12, mul_45, add_46, mul_46, tanh_11, add_47, hidden_states_71, view_131, x_94], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf226, (1024, 3072), (3072, 1), 0), arg147_1, out=buf227)
            del arg147_1
            del buf226
            buf231 = buf220; del buf220  # reuse
            # Topologically Sorted Source Nodes: [x_94, x_95, hidden_states_73, hidden_states_74], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_5.run(buf231, buf227, arg146_1, arg148_1, arg149_1, 1024, 768, stream=stream0)
            del arg146_1
            del arg148_1
            del arg149_1
            del buf227
        return (buf231, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 1, 128, 128), (16384, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    arg2_1 = rand_strided((50257, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)


# ===== inductor generated file: /tmp/cnnbench-transformers-p50cz5fm/repeat_02/a1/torchinductor/tmpxdyb3vm9/4h/c4hbd3bs4g2qthy7zrebrqivychrqd2oziojglrfg2tuj2xcnsvr.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_arange_embedding_native_layer_norm_unsqueeze_view_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 6, 'num_reduction': 4, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_addmm_arange_embedding_native_layer_norm_unsqueeze_view_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
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
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (r0_2 + 768*x3), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (r0_2 + 768*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr5 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tl.full([XBLOCK, R0_BLOCK], 50257, tl.int32)
    tmp5 = tmp3 + tmp4
    tmp6 = tmp3 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp3)
    tl.device_assert(((0 <= tmp7) & (tmp7 < 50257)) | ~(xmask), "index out of bounds: 0 <= tmp7 < 50257")
    tmp9 = tl.load(in_ptr2 + (r0_2 + 768*tmp7), r0_mask & xmask, other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tmp2 + tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(r0_mask & xmask, tmp13, 0)
    tmp16 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp18 = tl.where(r0_mask & xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None].to(tl.float32)
    tmp20 = tl.full([XBLOCK, 1], 768, tl.int32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = (tmp19 / tmp21)
    tmp23 = tmp13 - tmp22
    tmp24 = tmp23 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
    tmp27 = tl.where(r0_mask & xmask, tmp25, 0)
    tmp28 = tl.sum(tmp27, 1)[:, None].to(tl.float32)
    tmp29 = tmp12 - tmp22
    tmp30 = 768.0
    tmp31 = (tmp28 / tmp30)
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.rsqrt(tmp33)
    tmp35 = tmp29 * tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp37 + tmp38
    tl.store(in_out_ptr0 + (r0_2 + 768*x3), tmp12, r0_mask & xmask)
    tl.store(out_ptr2 + (r0_2 + 768*x3), tmp39, r0_mask & xmask)


# ===== inductor generated file: /tmp/cnnbench-transformers-p50cz5fm/repeat_02/a1/torchinductor/tmpxdyb3vm9/bo/cbogixtlgmsshy55xfrz275tizvetdt26ih37appoc6jgr7ajyk3.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_native_layer_norm_view_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 12592128}}
)
@triton.jit
def triton_per_fused_add_addmm_native_layer_norm_view_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
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
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp12 = tl.full([XBLOCK, 1], 768, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = (tmp11 / tmp13)
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(r0_mask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None].to(tl.float32)
    tmp21 = tmp4 - tmp14
    tmp22 = 768.0
    tmp23 = (tmp20 / tmp22)
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp31, r0_mask & xmask)


# ===== inductor generated file: /tmp/cnnbench-transformers-p50cz5fm/repeat_02/a1/torchinductor/tmpxdyb3vm9/d6/cd6ybkiu75diiuhq75l2mwfs6w2h4zixpdedql34g3irkcc6jcqc.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_native_layer_norm_view_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 22032384}}
)
@triton.jit
def triton_per_fused_add_addmm_native_layer_norm_view_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
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
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp5 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp6 = tmp4 + tmp5
    tmp7 = tmp3 + tmp6
    tmp8 = tmp2 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.where(r0_mask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp14 = tl.where(r0_mask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None].to(tl.float32)
    tmp16 = tl.full([XBLOCK, 1], 768, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = (tmp15 / tmp17)
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
    tmp23 = tl.where(r0_mask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None].to(tl.float32)
    tmp25 = tmp8 - tmp18
    tmp26 = 768.0
    tmp27 = (tmp24 / tmp26)
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp8, r0_mask & xmask)
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp35, r0_mask & xmask)


# ===== inductor generated file: /tmp/cnnbench-transformers-p50cz5fm/repeat_02/a1/torchinductor/tmpxdyb3vm9/ea/ceaoz3uu6u24l5sygramjdrcfjvvi5huw4pndtejhhfpig2qrkiv.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_mul_pow_tanh_view_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 37761024}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_mul_pow_tanh_view_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 3072)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = tmp2 * tmp2
    tmp6 = tmp5 * tmp2
    tmp7 = 0.044715
    tmp8 = tmp6 * tmp7
    tmp9 = tmp2 + tmp8
    tmp10 = 0.7978845608028654
    tmp11 = tmp9 * tmp10
    tmp12 = libdevice.tanh(tmp11)
    tmp13 = 1.0
    tmp14 = tmp12 + tmp13
    tmp15 = tmp4 * tmp14
    tl.store(in_out_ptr0 + (x2), tmp15, None)


# ===== inductor generated file: /tmp/cnnbench-transformers-p50cz5fm/repeat_02/a1/torchinductor/tmpxdyb3vm9/jq/cjqwss5atwrya52vg3gpmhy5m6itpshtoqaeg2ei6lm2otgmdhf5.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_arange_embedding_native_layer_norm_unsqueeze_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_arange_embedding_native_layer_norm_unsqueeze_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 768
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    x0 = (xindex % 128)
    tmp10_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp10_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp10_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp7 = tl.load(in_ptr2 + (r0_2 + 768*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.full([XBLOCK, R0_BLOCK], 50257, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert(((0 <= tmp4) & (tmp4 < 50257)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 50257")
        tmp6 = tl.load(in_ptr1 + (r0_2 + 768*tmp4), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tmp6 + tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp10_mean_next, tmp10_m2_next, tmp10_weight_next = triton_helpers.welford_reduce(
            tmp9, tmp10_mean, tmp10_m2, tmp10_weight, roffset == 0
        )
        tmp10_mean = tl.where(r0_mask & xmask, tmp10_mean_next, tmp10_mean)
        tmp10_m2 = tl.where(r0_mask & xmask, tmp10_m2_next, tmp10_m2)
        tmp10_weight = tl.where(r0_mask & xmask, tmp10_weight_next, tmp10_weight)
    tmp11, tmp12, tmp13 = triton_helpers.welford(tmp10_mean, tmp10_m2, tmp10_weight, 1)
    tmp10 = tmp11[:, None]
    tmp14 = tmp12[:, None]
    tmp15 = tmp13[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp22 = tl.load(in_ptr2 + (r0_2 + 768*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp31 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp33 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.full([XBLOCK, R0_BLOCK], 50257, tl.int32)
        tmp17 = tmp0 + tmp16
        tmp18 = tmp0 < 0
        tmp19 = tl.where(tmp18, tmp17, tmp0)
        tl.device_assert(((0 <= tmp19) & (tmp19 < 50257)) | ~(xmask), "index out of bounds: 0 <= tmp19 < 50257")
        tmp21 = tl.load(in_ptr1 + (r0_2 + 768*tmp19), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp23 = tmp21 + tmp22
        tmp24 = tmp23 - tmp10
        tmp25 = 768.0
        tmp26 = (tmp14 / tmp25)
        tmp27 = 1e-05
        tmp28 = tmp26 + tmp27
        tmp29 = libdevice.rsqrt(tmp28)
        tmp30 = tmp24 * tmp29
        tmp32 = tmp30 * tmp31
        tmp34 = tmp32 + tmp33
        tl.store(out_ptr2 + (r0_2 + 768*x3), tmp34, r0_mask & xmask)


# ===== inductor generated file: /tmp/cnnbench-transformers-p50cz5fm/repeat_02/a1/torchinductor/tmpxdyb3vm9/wq/cwqiouzl5rqqzecyxn3st4tw5voktz55qizk6dw2eebgohjbtoi5.debug/fx_graph_readable.py =====
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[1, 1, 128, 128]", arg1_1: "i64[8, 128]", arg2_1: "f32[50257, 768]", arg3_1: "f32[1024, 768]", arg4_1: "f32[768]", arg5_1: "f32[768]", arg6_1: "f32[2304]", arg7_1: "f32[768, 2304]", arg8_1: "f32[768]", arg9_1: "f32[768, 768]", arg10_1: "f32[768]", arg11_1: "f32[768]", arg12_1: "f32[3072]", arg13_1: "f32[768, 3072]", arg14_1: "f32[768]", arg15_1: "f32[3072, 768]", arg16_1: "f32[768]", arg17_1: "f32[768]", arg18_1: "f32[2304]", arg19_1: "f32[768, 2304]", arg20_1: "f32[768]", arg21_1: "f32[768, 768]", arg22_1: "f32[768]", arg23_1: "f32[768]", arg24_1: "f32[3072]", arg25_1: "f32[768, 3072]", arg26_1: "f32[768]", arg27_1: "f32[3072, 768]", arg28_1: "f32[768]", arg29_1: "f32[768]", arg30_1: "f32[2304]", arg31_1: "f32[768, 2304]", arg32_1: "f32[768]", arg33_1: "f32[768, 768]", arg34_1: "f32[768]", arg35_1: "f32[768]", arg36_1: "f32[3072]", arg37_1: "f32[768, 3072]", arg38_1: "f32[768]", arg39_1: "f32[3072, 768]", arg40_1: "f32[768]", arg41_1: "f32[768]", arg42_1: "f32[2304]", arg43_1: "f32[768, 2304]", arg44_1: "f32[768]", arg45_1: "f32[768, 768]", arg46_1: "f32[768]", arg47_1: "f32[768]", arg48_1: "f32[3072]", arg49_1: "f32[768, 3072]", arg50_1: "f32[768]", arg51_1: "f32[3072, 768]", arg52_1: "f32[768]", arg53_1: "f32[768]", arg54_1: "f32[2304]", arg55_1: "f32[768, 2304]", arg56_1: "f32[768]", arg57_1: "f32[768, 768]", arg58_1: "f32[768]", arg59_1: "f32[768]", arg60_1: "f32[3072]", arg61_1: "f32[768, 3072]", arg62_1: "f32[768]", arg63_1: "f32[3072, 768]", arg64_1: "f32[768]", arg65_1: "f32[768]", arg66_1: "f32[2304]", arg67_1: "f32[768, 2304]", arg68_1: "f32[768]", arg69_1: "f32[768, 768]", arg70_1: "f32[768]", arg71_1: "f32[768]", arg72_1: "f32[3072]", arg73_1: "f32[768, 3072]", arg74_1: "f32[768]", arg75_1: "f32[3072, 768]", arg76_1: "f32[768]", arg77_1: "f32[768]", arg78_1: "f32[2304]", arg79_1: "f32[768, 2304]", arg80_1: "f32[768]", arg81_1: "f32[768, 768]", arg82_1: "f32[768]", arg83_1: "f32[768]", arg84_1: "f32[3072]", arg85_1: "f32[768, 3072]", arg86_1: "f32[768]", arg87_1: "f32[3072, 768]", arg88_1: "f32[768]", arg89_1: "f32[768]", arg90_1: "f32[2304]", arg91_1: "f32[768, 2304]", arg92_1: "f32[768]", arg93_1: "f32[768, 768]", arg94_1: "f32[768]", arg95_1: "f32[768]", arg96_1: "f32[3072]", arg97_1: "f32[768, 3072]", arg98_1: "f32[768]", arg99_1: "f32[3072, 768]", arg100_1: "f32[768]", arg101_1: "f32[768]", arg102_1: "f32[2304]", arg103_1: "f32[768, 2304]", arg104_1: "f32[768]", arg105_1: "f32[768, 768]", arg106_1: "f32[768]", arg107_1: "f32[768]", arg108_1: "f32[3072]", arg109_1: "f32[768, 3072]", arg110_1: "f32[768]", arg111_1: "f32[3072, 768]", arg112_1: "f32[768]", arg113_1: "f32[768]", arg114_1: "f32[2304]", arg115_1: "f32[768, 2304]", arg116_1: "f32[768]", arg117_1: "f32[768, 768]", arg118_1: "f32[768]", arg119_1: "f32[768]", arg120_1: "f32[3072]", arg121_1: "f32[768, 3072]", arg122_1: "f32[768]", arg123_1: "f32[3072, 768]", arg124_1: "f32[768]", arg125_1: "f32[768]", arg126_1: "f32[2304]", arg127_1: "f32[768, 2304]", arg128_1: "f32[768]", arg129_1: "f32[768, 768]", arg130_1: "f32[768]", arg131_1: "f32[768]", arg132_1: "f32[3072]", arg133_1: "f32[768, 3072]", arg134_1: "f32[768]", arg135_1: "f32[3072, 768]", arg136_1: "f32[768]", arg137_1: "f32[768]", arg138_1: "f32[2304]", arg139_1: "f32[768, 2304]", arg140_1: "f32[768]", arg141_1: "f32[768, 768]", arg142_1: "f32[768]", arg143_1: "f32[768]", arg144_1: "f32[3072]", arg145_1: "f32[768, 3072]", arg146_1: "f32[768]", arg147_1: "f32[3072, 768]", arg148_1: "f32[768]", arg149_1: "f32[768]"):
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:857 in forward, code: inputs_embeds = self.wte(input_ids)
        embedding: "f32[8, 128, 768]" = torch.ops.aten.embedding.default(arg2_1, arg1_1);  arg2_1 = arg1_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:861 in forward, code: cache_position = torch.arange(
        iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:865 in forward, code: position_ids = cache_position.unsqueeze(0)
        unsqueeze: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:867 in forward, code: position_embeds = self.wpe(position_ids)
        embedding_1: "f32[1, 128, 768]" = torch.ops.aten.embedding.default(arg3_1, unsqueeze);  arg3_1 = unsqueeze = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:868 in forward, code: hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)
        add: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 128, 1]" = var_mean[0]
        getitem_1: "f32[8, 128, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add, getitem_1);  getitem_1 = None
        mul: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = arg4_1 = None
        add_2: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_1: "f32[1024, 768]" = torch.ops.aten.view.default(add_2, [-1, 768]);  add_2 = None
        addmm: "f32[1024, 2304]" = torch.ops.aten.addmm.default(arg6_1, view_1, arg7_1);  arg6_1 = view_1 = arg7_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_2: "f32[8, 128, 2304]" = torch.ops.aten.view.default(addmm, [8, 128, 2304]);  addmm = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split = torch.ops.aten.split.Tensor(view_2, 768, 2);  view_2 = None
        getitem_2: "f32[8, 128, 768]" = split[0]
        getitem_3: "f32[8, 128, 768]" = split[1]
        getitem_4: "f32[8, 128, 768]" = split[2];  split = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_3: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(getitem_3, [8, 128, -1, 64]);  getitem_3 = None
        permute: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_4: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(getitem_4, [8, 128, -1, 64]);  getitem_4 = None
        permute_1: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_5: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(getitem_2, [8, 128, -1, 64]);  getitem_2 = None
        permute_2: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(arg0_1, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_2, permute, permute_1, expand, False);  permute_2 = permute = permute_1 = expand = None
        getitem_5: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention[0];  _scaled_dot_product_efficient_attention = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_3: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_6: "f32[8, 128, 768]" = torch.ops.aten.view.default(permute_3, [8, 128, -1]);  permute_3 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_7: "f32[1024, 768]" = torch.ops.aten.view.default(view_6, [-1, 768]);  view_6 = None
        addmm_1: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg8_1, view_7, arg9_1);  arg8_1 = view_7 = arg9_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_8: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_1, [8, 128, 768]);  addmm_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_3: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_8, add);  view_8 = add = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
        getitem_9: "f32[8, 128, 1]" = var_mean_1[0]
        getitem_10: "f32[8, 128, 1]" = var_mean_1[1];  var_mean_1 = None
        add_4: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_9, 1e-05);  getitem_9 = None
        rsqrt_1: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_1: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_3, getitem_10);  getitem_10 = None
        mul_2: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_3: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_2, arg10_1);  mul_2 = arg10_1 = None
        add_5: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_3, arg11_1);  mul_3 = arg11_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_9: "f32[1024, 768]" = torch.ops.aten.view.default(add_5, [-1, 768]);  add_5 = None
        addmm_2: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg12_1, view_9, arg13_1);  arg12_1 = view_9 = arg13_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_10: "f32[8, 128, 3072]" = torch.ops.aten.view.default(addmm_2, [8, 128, 3072]);  addmm_2 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_4: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_10, 0.5)
        pow_1: "f32[8, 128, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_10, 3.0)
        mul_5: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_6: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(view_10, mul_5);  view_10 = mul_5 = None
        mul_6: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(add_6, 0.7978845608028654);  add_6 = None
        tanh: "f32[8, 128, 3072]" = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
        add_7: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_7: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_4, add_7);  mul_4 = add_7 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_11: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_7, [-1, 3072]);  mul_7 = None
        addmm_3: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg14_1, view_11, arg15_1);  arg14_1 = view_11 = arg15_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_12: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_3, [8, 128, 768]);  addmm_3 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_8: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(add_3, view_12);  add_3 = view_12 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
        getitem_11: "f32[8, 128, 1]" = var_mean_2[0]
        getitem_12: "f32[8, 128, 1]" = var_mean_2[1];  var_mean_2 = None
        add_9: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
        rsqrt_2: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        sub_2: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_8, getitem_12);  getitem_12 = None
        mul_8: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
        mul_9: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_8, arg16_1);  mul_8 = arg16_1 = None
        add_10: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_9, arg17_1);  mul_9 = arg17_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_13: "f32[1024, 768]" = torch.ops.aten.view.default(add_10, [-1, 768]);  add_10 = None
        addmm_4: "f32[1024, 2304]" = torch.ops.aten.addmm.default(arg18_1, view_13, arg19_1);  arg18_1 = view_13 = arg19_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_14: "f32[8, 128, 2304]" = torch.ops.aten.view.default(addmm_4, [8, 128, 2304]);  addmm_4 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_1 = torch.ops.aten.split.Tensor(view_14, 768, 2);  view_14 = None
        getitem_13: "f32[8, 128, 768]" = split_1[0]
        getitem_14: "f32[8, 128, 768]" = split_1[1]
        getitem_15: "f32[8, 128, 768]" = split_1[2];  split_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_15: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(getitem_14, [8, 128, -1, 64]);  getitem_14 = None
        permute_4: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_16: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(getitem_15, [8, 128, -1, 64]);  getitem_15 = None
        permute_5: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_17: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(getitem_13, [8, 128, -1, 64]);  getitem_13 = None
        permute_6: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_1: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(arg0_1, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_6, permute_4, permute_5, expand_1, False);  permute_6 = permute_4 = permute_5 = expand_1 = None
        getitem_16: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_1[0];  _scaled_dot_product_efficient_attention_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3]);  getitem_16 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_18: "f32[8, 128, 768]" = torch.ops.aten.view.default(permute_7, [8, 128, -1]);  permute_7 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_19: "f32[1024, 768]" = torch.ops.aten.view.default(view_18, [-1, 768]);  view_18 = None
        addmm_5: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg20_1, view_19, arg21_1);  arg20_1 = view_19 = arg21_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_20: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_5, [8, 128, 768]);  addmm_5 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_11: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_20, add_8);  view_20 = add_8 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_3 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
        getitem_20: "f32[8, 128, 1]" = var_mean_3[0]
        getitem_21: "f32[8, 128, 1]" = var_mean_3[1];  var_mean_3 = None
        add_12: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_3: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        sub_3: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_21);  getitem_21 = None
        mul_10: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
        mul_11: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_10, arg22_1);  mul_10 = arg22_1 = None
        add_13: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_11, arg23_1);  mul_11 = arg23_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_21: "f32[1024, 768]" = torch.ops.aten.view.default(add_13, [-1, 768]);  add_13 = None
        addmm_6: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg24_1, view_21, arg25_1);  arg24_1 = view_21 = arg25_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_22: "f32[8, 128, 3072]" = torch.ops.aten.view.default(addmm_6, [8, 128, 3072]);  addmm_6 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_12: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_22, 0.5)
        pow_2: "f32[8, 128, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_22, 3.0)
        mul_13: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_14: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(view_22, mul_13);  view_22 = mul_13 = None
        mul_14: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(add_14, 0.7978845608028654);  add_14 = None
        tanh_1: "f32[8, 128, 3072]" = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
        add_15: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_15: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_12, add_15);  mul_12 = add_15 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_23: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_15, [-1, 3072]);  mul_15 = None
        addmm_7: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg26_1, view_23, arg27_1);  arg26_1 = view_23 = arg27_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_24: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_7, [8, 128, 768]);  addmm_7 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_16: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(add_11, view_24);  add_11 = view_24 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_4 = torch.ops.aten.var_mean.correction(add_16, [2], correction = 0, keepdim = True)
        getitem_22: "f32[8, 128, 1]" = var_mean_4[0]
        getitem_23: "f32[8, 128, 1]" = var_mean_4[1];  var_mean_4 = None
        add_17: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_4: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        sub_4: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_16, getitem_23);  getitem_23 = None
        mul_16: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
        mul_17: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_16, arg28_1);  mul_16 = arg28_1 = None
        add_18: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_17, arg29_1);  mul_17 = arg29_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_25: "f32[1024, 768]" = torch.ops.aten.view.default(add_18, [-1, 768]);  add_18 = None
        addmm_8: "f32[1024, 2304]" = torch.ops.aten.addmm.default(arg30_1, view_25, arg31_1);  arg30_1 = view_25 = arg31_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_26: "f32[8, 128, 2304]" = torch.ops.aten.view.default(addmm_8, [8, 128, 2304]);  addmm_8 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_2 = torch.ops.aten.split.Tensor(view_26, 768, 2);  view_26 = None
        getitem_24: "f32[8, 128, 768]" = split_2[0]
        getitem_25: "f32[8, 128, 768]" = split_2[1]
        getitem_26: "f32[8, 128, 768]" = split_2[2];  split_2 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_27: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(getitem_25, [8, 128, -1, 64]);  getitem_25 = None
        permute_8: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_28: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(getitem_26, [8, 128, -1, 64]);  getitem_26 = None
        permute_9: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_29: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(getitem_24, [8, 128, -1, 64]);  getitem_24 = None
        permute_10: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_2: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(arg0_1, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_10, permute_8, permute_9, expand_2, False);  permute_10 = permute_8 = permute_9 = expand_2 = None
        getitem_27: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_11: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3]);  getitem_27 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_30: "f32[8, 128, 768]" = torch.ops.aten.view.default(permute_11, [8, 128, -1]);  permute_11 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_31: "f32[1024, 768]" = torch.ops.aten.view.default(view_30, [-1, 768]);  view_30 = None
        addmm_9: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg32_1, view_31, arg33_1);  arg32_1 = view_31 = arg33_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_32: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_9, [8, 128, 768]);  addmm_9 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_19: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_32, add_16);  view_32 = add_16 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_5 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
        getitem_31: "f32[8, 128, 1]" = var_mean_5[0]
        getitem_32: "f32[8, 128, 1]" = var_mean_5[1];  var_mean_5 = None
        add_20: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_31, 1e-05);  getitem_31 = None
        rsqrt_5: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        sub_5: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_32);  getitem_32 = None
        mul_18: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
        mul_19: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_18, arg34_1);  mul_18 = arg34_1 = None
        add_21: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_19, arg35_1);  mul_19 = arg35_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_33: "f32[1024, 768]" = torch.ops.aten.view.default(add_21, [-1, 768]);  add_21 = None
        addmm_10: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg36_1, view_33, arg37_1);  arg36_1 = view_33 = arg37_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_34: "f32[8, 128, 3072]" = torch.ops.aten.view.default(addmm_10, [8, 128, 3072]);  addmm_10 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_20: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_34, 0.5)
        pow_3: "f32[8, 128, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_34, 3.0)
        mul_21: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_22: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(view_34, mul_21);  view_34 = mul_21 = None
        mul_22: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(add_22, 0.7978845608028654);  add_22 = None
        tanh_2: "f32[8, 128, 3072]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
        add_23: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_23: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_20, add_23);  mul_20 = add_23 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_35: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_23, [-1, 3072]);  mul_23 = None
        addmm_11: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg38_1, view_35, arg39_1);  arg38_1 = view_35 = arg39_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_36: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_11, [8, 128, 768]);  addmm_11 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_24: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(add_19, view_36);  add_19 = view_36 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_6 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
        getitem_33: "f32[8, 128, 1]" = var_mean_6[0]
        getitem_34: "f32[8, 128, 1]" = var_mean_6[1];  var_mean_6 = None
        add_25: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_33, 1e-05);  getitem_33 = None
        rsqrt_6: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        sub_6: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_24, getitem_34);  getitem_34 = None
        mul_24: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
        mul_25: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_24, arg40_1);  mul_24 = arg40_1 = None
        add_26: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_25, arg41_1);  mul_25 = arg41_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_37: "f32[1024, 768]" = torch.ops.aten.view.default(add_26, [-1, 768]);  add_26 = None
        addmm_12: "f32[1024, 2304]" = torch.ops.aten.addmm.default(arg42_1, view_37, arg43_1);  arg42_1 = view_37 = arg43_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_38: "f32[8, 128, 2304]" = torch.ops.aten.view.default(addmm_12, [8, 128, 2304]);  addmm_12 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_3 = torch.ops.aten.split.Tensor(view_38, 768, 2);  view_38 = None
        getitem_35: "f32[8, 128, 768]" = split_3[0]
        getitem_36: "f32[8, 128, 768]" = split_3[1]
        getitem_37: "f32[8, 128, 768]" = split_3[2];  split_3 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_39: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(getitem_36, [8, 128, -1, 64]);  getitem_36 = None
        permute_12: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_39, [0, 2, 1, 3]);  view_39 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_40: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(getitem_37, [8, 128, -1, 64]);  getitem_37 = None
        permute_13: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_41: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(getitem_35, [8, 128, -1, 64]);  getitem_35 = None
        permute_14: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_3: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(arg0_1, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_14, permute_12, permute_13, expand_3, False);  permute_14 = permute_12 = permute_13 = expand_3 = None
        getitem_38: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_15: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3]);  getitem_38 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_42: "f32[8, 128, 768]" = torch.ops.aten.view.default(permute_15, [8, 128, -1]);  permute_15 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_43: "f32[1024, 768]" = torch.ops.aten.view.default(view_42, [-1, 768]);  view_42 = None
        addmm_13: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg44_1, view_43, arg45_1);  arg44_1 = view_43 = arg45_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_44: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_13, [8, 128, 768]);  addmm_13 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_27: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_44, add_24);  view_44 = add_24 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_7 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
        getitem_42: "f32[8, 128, 1]" = var_mean_7[0]
        getitem_43: "f32[8, 128, 1]" = var_mean_7[1];  var_mean_7 = None
        add_28: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_7: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_7: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_43);  getitem_43 = None
        mul_26: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
        mul_27: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_26, arg46_1);  mul_26 = arg46_1 = None
        add_29: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_27, arg47_1);  mul_27 = arg47_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_45: "f32[1024, 768]" = torch.ops.aten.view.default(add_29, [-1, 768]);  add_29 = None
        addmm_14: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg48_1, view_45, arg49_1);  arg48_1 = view_45 = arg49_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_46: "f32[8, 128, 3072]" = torch.ops.aten.view.default(addmm_14, [8, 128, 3072]);  addmm_14 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_28: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_46, 0.5)
        pow_4: "f32[8, 128, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_46, 3.0)
        mul_29: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_30: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(view_46, mul_29);  view_46 = mul_29 = None
        mul_30: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(add_30, 0.7978845608028654);  add_30 = None
        tanh_3: "f32[8, 128, 3072]" = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
        add_31: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_31: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_28, add_31);  mul_28 = add_31 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_47: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_31, [-1, 3072]);  mul_31 = None
        addmm_15: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg50_1, view_47, arg51_1);  arg50_1 = view_47 = arg51_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_48: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_15, [8, 128, 768]);  addmm_15 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_32: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(add_27, view_48);  add_27 = view_48 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_8 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
        getitem_44: "f32[8, 128, 1]" = var_mean_8[0]
        getitem_45: "f32[8, 128, 1]" = var_mean_8[1];  var_mean_8 = None
        add_33: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_8: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        sub_8: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_32, getitem_45);  getitem_45 = None
        mul_32: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
        mul_33: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_32, arg52_1);  mul_32 = arg52_1 = None
        add_34: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_33, arg53_1);  mul_33 = arg53_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_49: "f32[1024, 768]" = torch.ops.aten.view.default(add_34, [-1, 768]);  add_34 = None
        addmm_16: "f32[1024, 2304]" = torch.ops.aten.addmm.default(arg54_1, view_49, arg55_1);  arg54_1 = view_49 = arg55_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_50: "f32[8, 128, 2304]" = torch.ops.aten.view.default(addmm_16, [8, 128, 2304]);  addmm_16 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_4 = torch.ops.aten.split.Tensor(view_50, 768, 2);  view_50 = None
        getitem_46: "f32[8, 128, 768]" = split_4[0]
        getitem_47: "f32[8, 128, 768]" = split_4[1]
        getitem_48: "f32[8, 128, 768]" = split_4[2];  split_4 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_51: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(getitem_47, [8, 128, -1, 64]);  getitem_47 = None
        permute_16: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_52: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(getitem_48, [8, 128, -1, 64]);  getitem_48 = None
        permute_17: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_53: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(getitem_46, [8, 128, -1, 64]);  getitem_46 = None
        permute_18: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_4: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(arg0_1, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_18, permute_16, permute_17, expand_4, False);  permute_18 = permute_16 = permute_17 = expand_4 = None
        getitem_49: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_4[0];  _scaled_dot_product_efficient_attention_4 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_19: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3]);  getitem_49 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_54: "f32[8, 128, 768]" = torch.ops.aten.view.default(permute_19, [8, 128, -1]);  permute_19 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_55: "f32[1024, 768]" = torch.ops.aten.view.default(view_54, [-1, 768]);  view_54 = None
        addmm_17: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg56_1, view_55, arg57_1);  arg56_1 = view_55 = arg57_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_56: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_17, [8, 128, 768]);  addmm_17 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_35: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_56, add_32);  view_56 = add_32 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_9 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_53: "f32[8, 128, 1]" = var_mean_9[0]
        getitem_54: "f32[8, 128, 1]" = var_mean_9[1];  var_mean_9 = None
        add_36: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_53, 1e-05);  getitem_53 = None
        rsqrt_9: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_9: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_54);  getitem_54 = None
        mul_34: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None
        mul_35: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_34, arg58_1);  mul_34 = arg58_1 = None
        add_37: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_35, arg59_1);  mul_35 = arg59_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_57: "f32[1024, 768]" = torch.ops.aten.view.default(add_37, [-1, 768]);  add_37 = None
        addmm_18: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg60_1, view_57, arg61_1);  arg60_1 = view_57 = arg61_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_58: "f32[8, 128, 3072]" = torch.ops.aten.view.default(addmm_18, [8, 128, 3072]);  addmm_18 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_36: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.5)
        pow_5: "f32[8, 128, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_58, 3.0)
        mul_37: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_38: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(view_58, mul_37);  view_58 = mul_37 = None
        mul_38: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(add_38, 0.7978845608028654);  add_38 = None
        tanh_4: "f32[8, 128, 3072]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_39: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_39: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_36, add_39);  mul_36 = add_39 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_59: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_39, [-1, 3072]);  mul_39 = None
        addmm_19: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg62_1, view_59, arg63_1);  arg62_1 = view_59 = arg63_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_60: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_19, [8, 128, 768]);  addmm_19 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_40: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(add_35, view_60);  add_35 = view_60 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_10 = torch.ops.aten.var_mean.correction(add_40, [2], correction = 0, keepdim = True)
        getitem_55: "f32[8, 128, 1]" = var_mean_10[0]
        getitem_56: "f32[8, 128, 1]" = var_mean_10[1];  var_mean_10 = None
        add_41: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_55, 1e-05);  getitem_55 = None
        rsqrt_10: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
        sub_10: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_40, getitem_56);  getitem_56 = None
        mul_40: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None
        mul_41: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_40, arg64_1);  mul_40 = arg64_1 = None
        add_42: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_41, arg65_1);  mul_41 = arg65_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_61: "f32[1024, 768]" = torch.ops.aten.view.default(add_42, [-1, 768]);  add_42 = None
        addmm_20: "f32[1024, 2304]" = torch.ops.aten.addmm.default(arg66_1, view_61, arg67_1);  arg66_1 = view_61 = arg67_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_62: "f32[8, 128, 2304]" = torch.ops.aten.view.default(addmm_20, [8, 128, 2304]);  addmm_20 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_5 = torch.ops.aten.split.Tensor(view_62, 768, 2);  view_62 = None
        getitem_57: "f32[8, 128, 768]" = split_5[0]
        getitem_58: "f32[8, 128, 768]" = split_5[1]
        getitem_59: "f32[8, 128, 768]" = split_5[2];  split_5 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_63: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(getitem_58, [8, 128, -1, 64]);  getitem_58 = None
        permute_20: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_63, [0, 2, 1, 3]);  view_63 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_64: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(getitem_59, [8, 128, -1, 64]);  getitem_59 = None
        permute_21: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_65: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(getitem_57, [8, 128, -1, 64]);  getitem_57 = None
        permute_22: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_5: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(arg0_1, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_22, permute_20, permute_21, expand_5, False);  permute_22 = permute_20 = permute_21 = expand_5 = None
        getitem_60: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_5[0];  _scaled_dot_product_efficient_attention_5 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_23: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_66: "f32[8, 128, 768]" = torch.ops.aten.view.default(permute_23, [8, 128, -1]);  permute_23 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_67: "f32[1024, 768]" = torch.ops.aten.view.default(view_66, [-1, 768]);  view_66 = None
        addmm_21: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg68_1, view_67, arg69_1);  arg68_1 = view_67 = arg69_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_68: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_21, [8, 128, 768]);  addmm_21 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_43: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_68, add_40);  view_68 = add_40 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_11 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
        getitem_64: "f32[8, 128, 1]" = var_mean_11[0]
        getitem_65: "f32[8, 128, 1]" = var_mean_11[1];  var_mean_11 = None
        add_44: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_11: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        sub_11: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_65);  getitem_65 = None
        mul_42: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
        mul_43: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_42, arg70_1);  mul_42 = arg70_1 = None
        add_45: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_43, arg71_1);  mul_43 = arg71_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_69: "f32[1024, 768]" = torch.ops.aten.view.default(add_45, [-1, 768]);  add_45 = None
        addmm_22: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg72_1, view_69, arg73_1);  arg72_1 = view_69 = arg73_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_70: "f32[8, 128, 3072]" = torch.ops.aten.view.default(addmm_22, [8, 128, 3072]);  addmm_22 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_44: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_70, 0.5)
        pow_6: "f32[8, 128, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_70, 3.0)
        mul_45: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_46: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(view_70, mul_45);  view_70 = mul_45 = None
        mul_46: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(add_46, 0.7978845608028654);  add_46 = None
        tanh_5: "f32[8, 128, 3072]" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
        add_47: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_47: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_44, add_47);  mul_44 = add_47 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_71: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_47, [-1, 3072]);  mul_47 = None
        addmm_23: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg74_1, view_71, arg75_1);  arg74_1 = view_71 = arg75_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_72: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_23, [8, 128, 768]);  addmm_23 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_48: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(add_43, view_72);  add_43 = view_72 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_12 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
        getitem_66: "f32[8, 128, 1]" = var_mean_12[0]
        getitem_67: "f32[8, 128, 1]" = var_mean_12[1];  var_mean_12 = None
        add_49: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_12: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        sub_12: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_48, getitem_67);  getitem_67 = None
        mul_48: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
        mul_49: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_48, arg76_1);  mul_48 = arg76_1 = None
        add_50: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_49, arg77_1);  mul_49 = arg77_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_73: "f32[1024, 768]" = torch.ops.aten.view.default(add_50, [-1, 768]);  add_50 = None
        addmm_24: "f32[1024, 2304]" = torch.ops.aten.addmm.default(arg78_1, view_73, arg79_1);  arg78_1 = view_73 = arg79_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_74: "f32[8, 128, 2304]" = torch.ops.aten.view.default(addmm_24, [8, 128, 2304]);  addmm_24 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_6 = torch.ops.aten.split.Tensor(view_74, 768, 2);  view_74 = None
        getitem_68: "f32[8, 128, 768]" = split_6[0]
        getitem_69: "f32[8, 128, 768]" = split_6[1]
        getitem_70: "f32[8, 128, 768]" = split_6[2];  split_6 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_75: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(getitem_69, [8, 128, -1, 64]);  getitem_69 = None
        permute_24: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_76: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(getitem_70, [8, 128, -1, 64]);  getitem_70 = None
        permute_25: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_77: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(getitem_68, [8, 128, -1, 64]);  getitem_68 = None
        permute_26: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_77, [0, 2, 1, 3]);  view_77 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_6: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(arg0_1, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_26, permute_24, permute_25, expand_6, False);  permute_26 = permute_24 = permute_25 = expand_6 = None
        getitem_71: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_6[0];  _scaled_dot_product_efficient_attention_6 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_27: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_71, [0, 2, 1, 3]);  getitem_71 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_78: "f32[8, 128, 768]" = torch.ops.aten.view.default(permute_27, [8, 128, -1]);  permute_27 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_79: "f32[1024, 768]" = torch.ops.aten.view.default(view_78, [-1, 768]);  view_78 = None
        addmm_25: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg80_1, view_79, arg81_1);  arg80_1 = view_79 = arg81_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_80: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_25, [8, 128, 768]);  addmm_25 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_51: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_80, add_48);  view_80 = add_48 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_13 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
        getitem_75: "f32[8, 128, 1]" = var_mean_13[0]
        getitem_76: "f32[8, 128, 1]" = var_mean_13[1];  var_mean_13 = None
        add_52: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_75, 1e-05);  getitem_75 = None
        rsqrt_13: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        sub_13: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_51, getitem_76);  getitem_76 = None
        mul_50: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = rsqrt_13 = None
        mul_51: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_50, arg82_1);  mul_50 = arg82_1 = None
        add_53: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_51, arg83_1);  mul_51 = arg83_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_81: "f32[1024, 768]" = torch.ops.aten.view.default(add_53, [-1, 768]);  add_53 = None
        addmm_26: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg84_1, view_81, arg85_1);  arg84_1 = view_81 = arg85_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_82: "f32[8, 128, 3072]" = torch.ops.aten.view.default(addmm_26, [8, 128, 3072]);  addmm_26 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_52: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_82, 0.5)
        pow_7: "f32[8, 128, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_82, 3.0)
        mul_53: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_54: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(view_82, mul_53);  view_82 = mul_53 = None
        mul_54: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(add_54, 0.7978845608028654);  add_54 = None
        tanh_6: "f32[8, 128, 3072]" = torch.ops.aten.tanh.default(mul_54);  mul_54 = None
        add_55: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
        mul_55: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_52, add_55);  mul_52 = add_55 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_83: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_55, [-1, 3072]);  mul_55 = None
        addmm_27: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg86_1, view_83, arg87_1);  arg86_1 = view_83 = arg87_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_84: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_27, [8, 128, 768]);  addmm_27 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_56: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(add_51, view_84);  add_51 = view_84 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_14 = torch.ops.aten.var_mean.correction(add_56, [2], correction = 0, keepdim = True)
        getitem_77: "f32[8, 128, 1]" = var_mean_14[0]
        getitem_78: "f32[8, 128, 1]" = var_mean_14[1];  var_mean_14 = None
        add_57: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_77, 1e-05);  getitem_77 = None
        rsqrt_14: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
        sub_14: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_56, getitem_78);  getitem_78 = None
        mul_56: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = rsqrt_14 = None
        mul_57: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_56, arg88_1);  mul_56 = arg88_1 = None
        add_58: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_57, arg89_1);  mul_57 = arg89_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_85: "f32[1024, 768]" = torch.ops.aten.view.default(add_58, [-1, 768]);  add_58 = None
        addmm_28: "f32[1024, 2304]" = torch.ops.aten.addmm.default(arg90_1, view_85, arg91_1);  arg90_1 = view_85 = arg91_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_86: "f32[8, 128, 2304]" = torch.ops.aten.view.default(addmm_28, [8, 128, 2304]);  addmm_28 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_7 = torch.ops.aten.split.Tensor(view_86, 768, 2);  view_86 = None
        getitem_79: "f32[8, 128, 768]" = split_7[0]
        getitem_80: "f32[8, 128, 768]" = split_7[1]
        getitem_81: "f32[8, 128, 768]" = split_7[2];  split_7 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_87: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(getitem_80, [8, 128, -1, 64]);  getitem_80 = None
        permute_28: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_87, [0, 2, 1, 3]);  view_87 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_88: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(getitem_81, [8, 128, -1, 64]);  getitem_81 = None
        permute_29: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_89: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(getitem_79, [8, 128, -1, 64]);  getitem_79 = None
        permute_30: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_7: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(arg0_1, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_30, permute_28, permute_29, expand_7, False);  permute_30 = permute_28 = permute_29 = expand_7 = None
        getitem_82: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_7[0];  _scaled_dot_product_efficient_attention_7 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_31: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3]);  getitem_82 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_90: "f32[8, 128, 768]" = torch.ops.aten.view.default(permute_31, [8, 128, -1]);  permute_31 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_91: "f32[1024, 768]" = torch.ops.aten.view.default(view_90, [-1, 768]);  view_90 = None
        addmm_29: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg92_1, view_91, arg93_1);  arg92_1 = view_91 = arg93_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_92: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_29, [8, 128, 768]);  addmm_29 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_59: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_92, add_56);  view_92 = add_56 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_15 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
        getitem_86: "f32[8, 128, 1]" = var_mean_15[0]
        getitem_87: "f32[8, 128, 1]" = var_mean_15[1];  var_mean_15 = None
        add_60: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
        rsqrt_15: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        sub_15: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_87);  getitem_87 = None
        mul_58: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None
        mul_59: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_58, arg94_1);  mul_58 = arg94_1 = None
        add_61: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_59, arg95_1);  mul_59 = arg95_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_93: "f32[1024, 768]" = torch.ops.aten.view.default(add_61, [-1, 768]);  add_61 = None
        addmm_30: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg96_1, view_93, arg97_1);  arg96_1 = view_93 = arg97_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_94: "f32[8, 128, 3072]" = torch.ops.aten.view.default(addmm_30, [8, 128, 3072]);  addmm_30 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_60: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_94, 0.5)
        pow_8: "f32[8, 128, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_94, 3.0)
        mul_61: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_62: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(view_94, mul_61);  view_94 = mul_61 = None
        mul_62: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(add_62, 0.7978845608028654);  add_62 = None
        tanh_7: "f32[8, 128, 3072]" = torch.ops.aten.tanh.default(mul_62);  mul_62 = None
        add_63: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
        mul_63: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_60, add_63);  mul_60 = add_63 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_95: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_63, [-1, 3072]);  mul_63 = None
        addmm_31: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg98_1, view_95, arg99_1);  arg98_1 = view_95 = arg99_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_96: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_31, [8, 128, 768]);  addmm_31 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_64: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(add_59, view_96);  add_59 = view_96 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_16 = torch.ops.aten.var_mean.correction(add_64, [2], correction = 0, keepdim = True)
        getitem_88: "f32[8, 128, 1]" = var_mean_16[0]
        getitem_89: "f32[8, 128, 1]" = var_mean_16[1];  var_mean_16 = None
        add_65: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_16: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        sub_16: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_64, getitem_89);  getitem_89 = None
        mul_64: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = rsqrt_16 = None
        mul_65: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_64, arg100_1);  mul_64 = arg100_1 = None
        add_66: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_65, arg101_1);  mul_65 = arg101_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_97: "f32[1024, 768]" = torch.ops.aten.view.default(add_66, [-1, 768]);  add_66 = None
        addmm_32: "f32[1024, 2304]" = torch.ops.aten.addmm.default(arg102_1, view_97, arg103_1);  arg102_1 = view_97 = arg103_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_98: "f32[8, 128, 2304]" = torch.ops.aten.view.default(addmm_32, [8, 128, 2304]);  addmm_32 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_8 = torch.ops.aten.split.Tensor(view_98, 768, 2);  view_98 = None
        getitem_90: "f32[8, 128, 768]" = split_8[0]
        getitem_91: "f32[8, 128, 768]" = split_8[1]
        getitem_92: "f32[8, 128, 768]" = split_8[2];  split_8 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_99: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(getitem_91, [8, 128, -1, 64]);  getitem_91 = None
        permute_32: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_100: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(getitem_92, [8, 128, -1, 64]);  getitem_92 = None
        permute_33: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_101: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(getitem_90, [8, 128, -1, 64]);  getitem_90 = None
        permute_34: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_101, [0, 2, 1, 3]);  view_101 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_8: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(arg0_1, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_34, permute_32, permute_33, expand_8, False);  permute_34 = permute_32 = permute_33 = expand_8 = None
        getitem_93: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_8[0];  _scaled_dot_product_efficient_attention_8 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_35: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_93, [0, 2, 1, 3]);  getitem_93 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_102: "f32[8, 128, 768]" = torch.ops.aten.view.default(permute_35, [8, 128, -1]);  permute_35 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_103: "f32[1024, 768]" = torch.ops.aten.view.default(view_102, [-1, 768]);  view_102 = None
        addmm_33: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg104_1, view_103, arg105_1);  arg104_1 = view_103 = arg105_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_104: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_33, [8, 128, 768]);  addmm_33 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_67: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_104, add_64);  view_104 = add_64 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_17 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
        getitem_97: "f32[8, 128, 1]" = var_mean_17[0]
        getitem_98: "f32[8, 128, 1]" = var_mean_17[1];  var_mean_17 = None
        add_68: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_97, 1e-05);  getitem_97 = None
        rsqrt_17: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        sub_17: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_67, getitem_98);  getitem_98 = None
        mul_66: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = rsqrt_17 = None
        mul_67: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_66, arg106_1);  mul_66 = arg106_1 = None
        add_69: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_67, arg107_1);  mul_67 = arg107_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_105: "f32[1024, 768]" = torch.ops.aten.view.default(add_69, [-1, 768]);  add_69 = None
        addmm_34: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg108_1, view_105, arg109_1);  arg108_1 = view_105 = arg109_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_106: "f32[8, 128, 3072]" = torch.ops.aten.view.default(addmm_34, [8, 128, 3072]);  addmm_34 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_68: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_106, 0.5)
        pow_9: "f32[8, 128, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_106, 3.0)
        mul_69: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_70: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(view_106, mul_69);  view_106 = mul_69 = None
        mul_70: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(add_70, 0.7978845608028654);  add_70 = None
        tanh_8: "f32[8, 128, 3072]" = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
        add_71: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
        mul_71: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_68, add_71);  mul_68 = add_71 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_107: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_71, [-1, 3072]);  mul_71 = None
        addmm_35: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg110_1, view_107, arg111_1);  arg110_1 = view_107 = arg111_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_108: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_35, [8, 128, 768]);  addmm_35 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_72: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(add_67, view_108);  add_67 = view_108 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_18 = torch.ops.aten.var_mean.correction(add_72, [2], correction = 0, keepdim = True)
        getitem_99: "f32[8, 128, 1]" = var_mean_18[0]
        getitem_100: "f32[8, 128, 1]" = var_mean_18[1];  var_mean_18 = None
        add_73: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_99, 1e-05);  getitem_99 = None
        rsqrt_18: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
        sub_18: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_72, getitem_100);  getitem_100 = None
        mul_72: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = rsqrt_18 = None
        mul_73: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_72, arg112_1);  mul_72 = arg112_1 = None
        add_74: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_73, arg113_1);  mul_73 = arg113_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_109: "f32[1024, 768]" = torch.ops.aten.view.default(add_74, [-1, 768]);  add_74 = None
        addmm_36: "f32[1024, 2304]" = torch.ops.aten.addmm.default(arg114_1, view_109, arg115_1);  arg114_1 = view_109 = arg115_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_110: "f32[8, 128, 2304]" = torch.ops.aten.view.default(addmm_36, [8, 128, 2304]);  addmm_36 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_9 = torch.ops.aten.split.Tensor(view_110, 768, 2);  view_110 = None
        getitem_101: "f32[8, 128, 768]" = split_9[0]
        getitem_102: "f32[8, 128, 768]" = split_9[1]
        getitem_103: "f32[8, 128, 768]" = split_9[2];  split_9 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_111: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(getitem_102, [8, 128, -1, 64]);  getitem_102 = None
        permute_36: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_111, [0, 2, 1, 3]);  view_111 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_112: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(getitem_103, [8, 128, -1, 64]);  getitem_103 = None
        permute_37: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_113: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(getitem_101, [8, 128, -1, 64]);  getitem_101 = None
        permute_38: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_9: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(arg0_1, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_38, permute_36, permute_37, expand_9, False);  permute_38 = permute_36 = permute_37 = expand_9 = None
        getitem_104: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_9[0];  _scaled_dot_product_efficient_attention_9 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_39: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_104, [0, 2, 1, 3]);  getitem_104 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_114: "f32[8, 128, 768]" = torch.ops.aten.view.default(permute_39, [8, 128, -1]);  permute_39 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_115: "f32[1024, 768]" = torch.ops.aten.view.default(view_114, [-1, 768]);  view_114 = None
        addmm_37: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg116_1, view_115, arg117_1);  arg116_1 = view_115 = arg117_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_116: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_37, [8, 128, 768]);  addmm_37 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_75: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_116, add_72);  view_116 = add_72 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_19 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
        getitem_108: "f32[8, 128, 1]" = var_mean_19[0]
        getitem_109: "f32[8, 128, 1]" = var_mean_19[1];  var_mean_19 = None
        add_76: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
        rsqrt_19: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        sub_19: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_75, getitem_109);  getitem_109 = None
        mul_74: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = rsqrt_19 = None
        mul_75: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_74, arg118_1);  mul_74 = arg118_1 = None
        add_77: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_75, arg119_1);  mul_75 = arg119_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_117: "f32[1024, 768]" = torch.ops.aten.view.default(add_77, [-1, 768]);  add_77 = None
        addmm_38: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg120_1, view_117, arg121_1);  arg120_1 = view_117 = arg121_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_118: "f32[8, 128, 3072]" = torch.ops.aten.view.default(addmm_38, [8, 128, 3072]);  addmm_38 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_76: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_118, 0.5)
        pow_10: "f32[8, 128, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_118, 3.0)
        mul_77: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_78: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(view_118, mul_77);  view_118 = mul_77 = None
        mul_78: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(add_78, 0.7978845608028654);  add_78 = None
        tanh_9: "f32[8, 128, 3072]" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
        add_79: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
        mul_79: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_76, add_79);  mul_76 = add_79 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_119: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_79, [-1, 3072]);  mul_79 = None
        addmm_39: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg122_1, view_119, arg123_1);  arg122_1 = view_119 = arg123_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_120: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_39, [8, 128, 768]);  addmm_39 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_80: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(add_75, view_120);  add_75 = view_120 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_20 = torch.ops.aten.var_mean.correction(add_80, [2], correction = 0, keepdim = True)
        getitem_110: "f32[8, 128, 1]" = var_mean_20[0]
        getitem_111: "f32[8, 128, 1]" = var_mean_20[1];  var_mean_20 = None
        add_81: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
        rsqrt_20: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
        sub_20: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_80, getitem_111);  getitem_111 = None
        mul_80: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = rsqrt_20 = None
        mul_81: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_80, arg124_1);  mul_80 = arg124_1 = None
        add_82: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_81, arg125_1);  mul_81 = arg125_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_121: "f32[1024, 768]" = torch.ops.aten.view.default(add_82, [-1, 768]);  add_82 = None
        addmm_40: "f32[1024, 2304]" = torch.ops.aten.addmm.default(arg126_1, view_121, arg127_1);  arg126_1 = view_121 = arg127_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_122: "f32[8, 128, 2304]" = torch.ops.aten.view.default(addmm_40, [8, 128, 2304]);  addmm_40 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_10 = torch.ops.aten.split.Tensor(view_122, 768, 2);  view_122 = None
        getitem_112: "f32[8, 128, 768]" = split_10[0]
        getitem_113: "f32[8, 128, 768]" = split_10[1]
        getitem_114: "f32[8, 128, 768]" = split_10[2];  split_10 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_123: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(getitem_113, [8, 128, -1, 64]);  getitem_113 = None
        permute_40: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_124: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(getitem_114, [8, 128, -1, 64]);  getitem_114 = None
        permute_41: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_125: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(getitem_112, [8, 128, -1, 64]);  getitem_112 = None
        permute_42: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_125, [0, 2, 1, 3]);  view_125 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_10: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(arg0_1, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_42, permute_40, permute_41, expand_10, False);  permute_42 = permute_40 = permute_41 = expand_10 = None
        getitem_115: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_10[0];  _scaled_dot_product_efficient_attention_10 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_43: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_115, [0, 2, 1, 3]);  getitem_115 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_126: "f32[8, 128, 768]" = torch.ops.aten.view.default(permute_43, [8, 128, -1]);  permute_43 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_127: "f32[1024, 768]" = torch.ops.aten.view.default(view_126, [-1, 768]);  view_126 = None
        addmm_41: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg128_1, view_127, arg129_1);  arg128_1 = view_127 = arg129_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_128: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_41, [8, 128, 768]);  addmm_41 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_83: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_128, add_80);  view_128 = add_80 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_21 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
        getitem_119: "f32[8, 128, 1]" = var_mean_21[0]
        getitem_120: "f32[8, 128, 1]" = var_mean_21[1];  var_mean_21 = None
        add_84: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_119, 1e-05);  getitem_119 = None
        rsqrt_21: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        sub_21: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_83, getitem_120);  getitem_120 = None
        mul_82: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = rsqrt_21 = None
        mul_83: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_82, arg130_1);  mul_82 = arg130_1 = None
        add_85: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_83, arg131_1);  mul_83 = arg131_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_129: "f32[1024, 768]" = torch.ops.aten.view.default(add_85, [-1, 768]);  add_85 = None
        addmm_42: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg132_1, view_129, arg133_1);  arg132_1 = view_129 = arg133_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_130: "f32[8, 128, 3072]" = torch.ops.aten.view.default(addmm_42, [8, 128, 3072]);  addmm_42 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_84: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_130, 0.5)
        pow_11: "f32[8, 128, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_130, 3.0)
        mul_85: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_86: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(view_130, mul_85);  view_130 = mul_85 = None
        mul_86: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(add_86, 0.7978845608028654);  add_86 = None
        tanh_10: "f32[8, 128, 3072]" = torch.ops.aten.tanh.default(mul_86);  mul_86 = None
        add_87: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
        mul_87: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_84, add_87);  mul_84 = add_87 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_131: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_87, [-1, 3072]);  mul_87 = None
        addmm_43: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg134_1, view_131, arg135_1);  arg134_1 = view_131 = arg135_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_132: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_43, [8, 128, 768]);  addmm_43 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_88: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(add_83, view_132);  add_83 = view_132 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_22 = torch.ops.aten.var_mean.correction(add_88, [2], correction = 0, keepdim = True)
        getitem_121: "f32[8, 128, 1]" = var_mean_22[0]
        getitem_122: "f32[8, 128, 1]" = var_mean_22[1];  var_mean_22 = None
        add_89: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_121, 1e-05);  getitem_121 = None
        rsqrt_22: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
        sub_22: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_88, getitem_122);  getitem_122 = None
        mul_88: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = rsqrt_22 = None
        mul_89: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_88, arg136_1);  mul_88 = arg136_1 = None
        add_90: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_89, arg137_1);  mul_89 = arg137_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_133: "f32[1024, 768]" = torch.ops.aten.view.default(add_90, [-1, 768]);  add_90 = None
        addmm_44: "f32[1024, 2304]" = torch.ops.aten.addmm.default(arg138_1, view_133, arg139_1);  arg138_1 = view_133 = arg139_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_134: "f32[8, 128, 2304]" = torch.ops.aten.view.default(addmm_44, [8, 128, 2304]);  addmm_44 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_11 = torch.ops.aten.split.Tensor(view_134, 768, 2);  view_134 = None
        getitem_123: "f32[8, 128, 768]" = split_11[0]
        getitem_124: "f32[8, 128, 768]" = split_11[1]
        getitem_125: "f32[8, 128, 768]" = split_11[2];  split_11 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_135: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(getitem_124, [8, 128, -1, 64]);  getitem_124 = None
        permute_44: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_135, [0, 2, 1, 3]);  view_135 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_136: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(getitem_125, [8, 128, -1, 64]);  getitem_125 = None
        permute_45: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_137: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(getitem_123, [8, 128, -1, 64]);  getitem_123 = None
        permute_46: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_11: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(arg0_1, [8, 12, 128, 128]);  arg0_1 = None
        _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_46, permute_44, permute_45, expand_11, False);  permute_46 = permute_44 = permute_45 = expand_11 = None
        getitem_126: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_11[0];  _scaled_dot_product_efficient_attention_11 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_47: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3]);  getitem_126 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_138: "f32[8, 128, 768]" = torch.ops.aten.view.default(permute_47, [8, 128, -1]);  permute_47 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_139: "f32[1024, 768]" = torch.ops.aten.view.default(view_138, [-1, 768]);  view_138 = None
        addmm_45: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg140_1, view_139, arg141_1);  arg140_1 = view_139 = arg141_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_140: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_45, [8, 128, 768]);  addmm_45 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_91: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_140, add_88);  view_140 = add_88 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_23 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
        getitem_130: "f32[8, 128, 1]" = var_mean_23[0]
        getitem_131: "f32[8, 128, 1]" = var_mean_23[1];  var_mean_23 = None
        add_92: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
        rsqrt_23: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        sub_23: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_91, getitem_131);  getitem_131 = None
        mul_90: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = rsqrt_23 = None
        mul_91: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_90, arg142_1);  mul_90 = arg142_1 = None
        add_93: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_91, arg143_1);  mul_91 = arg143_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_141: "f32[1024, 768]" = torch.ops.aten.view.default(add_93, [-1, 768]);  add_93 = None
        addmm_46: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg144_1, view_141, arg145_1);  arg144_1 = view_141 = arg145_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_142: "f32[8, 128, 3072]" = torch.ops.aten.view.default(addmm_46, [8, 128, 3072]);  addmm_46 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_92: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_142, 0.5)
        pow_12: "f32[8, 128, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_142, 3.0)
        mul_93: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_94: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(view_142, mul_93);  view_142 = mul_93 = None
        mul_94: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(add_94, 0.7978845608028654);  add_94 = None
        tanh_11: "f32[8, 128, 3072]" = torch.ops.aten.tanh.default(mul_94);  mul_94 = None
        add_95: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
        mul_95: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_92, add_95);  mul_92 = add_95 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_143: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_95, [-1, 3072]);  mul_95 = None
        addmm_47: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg146_1, view_143, arg147_1);  arg146_1 = view_143 = arg147_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_144: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_47, [8, 128, 768]);  addmm_47 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_96: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(add_91, view_144);  add_91 = view_144 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:953 in forward, code: hidden_states = self.ln_f(hidden_states)
        var_mean_24 = torch.ops.aten.var_mean.correction(add_96, [2], correction = 0, keepdim = True)
        getitem_132: "f32[8, 128, 1]" = var_mean_24[0]
        getitem_133: "f32[8, 128, 1]" = var_mean_24[1];  var_mean_24 = None
        add_97: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-05);  getitem_132 = None
        rsqrt_24: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
        sub_24: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_96, getitem_133);  add_96 = getitem_133 = None
        mul_96: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = rsqrt_24 = None
        mul_97: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_96, arg148_1);  mul_96 = arg148_1 = None
        add_98: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_97, arg149_1);  mul_97 = arg149_1 = None
        return (add_98,)
        

# ===== inductor generated file: /tmp/cnnbench-transformers-p50cz5fm/repeat_02/a1/torchinductor/tmpxdyb3vm9/wq/cwqiouzl5rqqzecyxn3st4tw5voktz55qizk6dw2eebgohjbtoi5.debug/fx_graph_runnable.py =====

import os
os.environ['RUN_INDUCTOR'] = '1'
os.environ['TORCHINDUCTOR_FORCE_DISABLE_CACHES'] = '1'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/cnnbench-transformers-p50cz5fm/repeat_02/a1/torchinductor/tmpxdyb3vm9'
os.environ['TRITON_CACHE_DIR'] = '/tmp/cnnbench-transformers-p50cz5fm/repeat_02/a1/torchinductor/tmpxdyb3vm9/triton'

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims



import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.assume_static_by_default = True
torch._dynamo.config.automatic_dynamic_shapes = False
torch._inductor.config.triton.cudagraphs = False
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None




# torch version: 2.9.0+cu128
# torch cuda version: 12.8
# torch git version: 0fabc3ba44823f257e70ce397d989c8de5e362c1


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2025 NVIDIA Corporation 
# Built on Fri_Feb_21_20:23:50_PST_2025 
# Cuda compilation tools, release 12.8, V12.8.93 
# Build cuda_12.8.r12.8/compiler.35583870_0 

# GPU Hardware Info: 
# NVIDIA GeForce RTX 3050 : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1):
        embedding = torch.ops.aten.embedding.default(arg2_1, arg1_1);  arg2_1 = arg1_1 = None
        iota = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
        embedding_1 = torch.ops.aten.embedding.default(arg3_1, unsqueeze);  arg3_1 = unsqueeze = None
        add = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_1 = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub = torch.ops.aten.sub.Tensor(add, getitem_1);  getitem_1 = None
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = arg4_1 = None
        add_2 = torch.ops.aten.add.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None
        view_1 = torch.ops.aten.view.default(add_2, [-1, 768]);  add_2 = None
        addmm = torch.ops.aten.addmm.default(arg6_1, view_1, arg7_1);  arg6_1 = view_1 = arg7_1 = None
        view_2 = torch.ops.aten.view.default(addmm, [8, 128, 2304]);  addmm = None
        split = torch.ops.aten.split.Tensor(view_2, 768, 2);  view_2 = None
        getitem_2 = split[0]
        getitem_3 = split[1]
        getitem_4 = split[2];  split = None
        view_3 = torch.ops.aten.view.default(getitem_3, [8, 128, -1, 64]);  getitem_3 = None
        permute = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        view_4 = torch.ops.aten.view.default(getitem_4, [8, 128, -1, 64]);  getitem_4 = None
        permute_1 = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        view_5 = torch.ops.aten.view.default(getitem_2, [8, 128, -1, 64]);  getitem_2 = None
        permute_2 = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        expand = torch.ops.aten.expand.default(arg0_1, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_2, permute, permute_1, expand, False);  permute_2 = permute = permute_1 = expand = None
        getitem_5 = _scaled_dot_product_efficient_attention[0];  _scaled_dot_product_efficient_attention = None
        permute_3 = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None
        view_6 = torch.ops.aten.view.default(permute_3, [8, 128, -1]);  permute_3 = None
        view_7 = torch.ops.aten.view.default(view_6, [-1, 768]);  view_6 = None
        addmm_1 = torch.ops.aten.addmm.default(arg8_1, view_7, arg9_1);  arg8_1 = view_7 = arg9_1 = None
        view_8 = torch.ops.aten.view.default(addmm_1, [8, 128, 768]);  addmm_1 = None
        add_3 = torch.ops.aten.add.Tensor(view_8, add);  view_8 = add = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
        getitem_9 = var_mean_1[0]
        getitem_10 = var_mean_1[1];  var_mean_1 = None
        add_4 = torch.ops.aten.add.Tensor(getitem_9, 1e-05);  getitem_9 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_1 = torch.ops.aten.sub.Tensor(add_3, getitem_10);  getitem_10 = None
        mul_2 = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_3 = torch.ops.aten.mul.Tensor(mul_2, arg10_1);  mul_2 = arg10_1 = None
        add_5 = torch.ops.aten.add.Tensor(mul_3, arg11_1);  mul_3 = arg11_1 = None
        view_9 = torch.ops.aten.view.default(add_5, [-1, 768]);  add_5 = None
        addmm_2 = torch.ops.aten.addmm.default(arg12_1, view_9, arg13_1);  arg12_1 = view_9 = arg13_1 = None
        view_10 = torch.ops.aten.view.default(addmm_2, [8, 128, 3072]);  addmm_2 = None
        mul_4 = torch.ops.aten.mul.Tensor(view_10, 0.5)
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(view_10, 3.0)
        mul_5 = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_6 = torch.ops.aten.add.Tensor(view_10, mul_5);  view_10 = mul_5 = None
        mul_6 = torch.ops.aten.mul.Tensor(add_6, 0.7978845608028654);  add_6 = None
        tanh = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
        add_7 = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_7 = torch.ops.aten.mul.Tensor(mul_4, add_7);  mul_4 = add_7 = None
        view_11 = torch.ops.aten.view.default(mul_7, [-1, 3072]);  mul_7 = None
        addmm_3 = torch.ops.aten.addmm.default(arg14_1, view_11, arg15_1);  arg14_1 = view_11 = arg15_1 = None
        view_12 = torch.ops.aten.view.default(addmm_3, [8, 128, 768]);  addmm_3 = None
        add_8 = torch.ops.aten.add.Tensor(add_3, view_12);  add_3 = view_12 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
        getitem_11 = var_mean_2[0]
        getitem_12 = var_mean_2[1];  var_mean_2 = None
        add_9 = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        sub_2 = torch.ops.aten.sub.Tensor(add_8, getitem_12);  getitem_12 = None
        mul_8 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
        mul_9 = torch.ops.aten.mul.Tensor(mul_8, arg16_1);  mul_8 = arg16_1 = None
        add_10 = torch.ops.aten.add.Tensor(mul_9, arg17_1);  mul_9 = arg17_1 = None
        view_13 = torch.ops.aten.view.default(add_10, [-1, 768]);  add_10 = None
        addmm_4 = torch.ops.aten.addmm.default(arg18_1, view_13, arg19_1);  arg18_1 = view_13 = arg19_1 = None
        view_14 = torch.ops.aten.view.default(addmm_4, [8, 128, 2304]);  addmm_4 = None
        split_1 = torch.ops.aten.split.Tensor(view_14, 768, 2);  view_14 = None
        getitem_13 = split_1[0]
        getitem_14 = split_1[1]
        getitem_15 = split_1[2];  split_1 = None
        view_15 = torch.ops.aten.view.default(getitem_14, [8, 128, -1, 64]);  getitem_14 = None
        permute_4 = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None
        view_16 = torch.ops.aten.view.default(getitem_15, [8, 128, -1, 64]);  getitem_15 = None
        permute_5 = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        view_17 = torch.ops.aten.view.default(getitem_13, [8, 128, -1, 64]);  getitem_13 = None
        permute_6 = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None
        expand_1 = torch.ops.aten.expand.default(arg0_1, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_6, permute_4, permute_5, expand_1, False);  permute_6 = permute_4 = permute_5 = expand_1 = None
        getitem_16 = _scaled_dot_product_efficient_attention_1[0];  _scaled_dot_product_efficient_attention_1 = None
        permute_7 = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3]);  getitem_16 = None
        view_18 = torch.ops.aten.view.default(permute_7, [8, 128, -1]);  permute_7 = None
        view_19 = torch.ops.aten.view.default(view_18, [-1, 768]);  view_18 = None
        addmm_5 = torch.ops.aten.addmm.default(arg20_1, view_19, arg21_1);  arg20_1 = view_19 = arg21_1 = None
        view_20 = torch.ops.aten.view.default(addmm_5, [8, 128, 768]);  addmm_5 = None
        add_11 = torch.ops.aten.add.Tensor(view_20, add_8);  view_20 = add_8 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
        getitem_20 = var_mean_3[0]
        getitem_21 = var_mean_3[1];  var_mean_3 = None
        add_12 = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        sub_3 = torch.ops.aten.sub.Tensor(add_11, getitem_21);  getitem_21 = None
        mul_10 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
        mul_11 = torch.ops.aten.mul.Tensor(mul_10, arg22_1);  mul_10 = arg22_1 = None
        add_13 = torch.ops.aten.add.Tensor(mul_11, arg23_1);  mul_11 = arg23_1 = None
        view_21 = torch.ops.aten.view.default(add_13, [-1, 768]);  add_13 = None
        addmm_6 = torch.ops.aten.addmm.default(arg24_1, view_21, arg25_1);  arg24_1 = view_21 = arg25_1 = None
        view_22 = torch.ops.aten.view.default(addmm_6, [8, 128, 3072]);  addmm_6 = None
        mul_12 = torch.ops.aten.mul.Tensor(view_22, 0.5)
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(view_22, 3.0)
        mul_13 = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_14 = torch.ops.aten.add.Tensor(view_22, mul_13);  view_22 = mul_13 = None
        mul_14 = torch.ops.aten.mul.Tensor(add_14, 0.7978845608028654);  add_14 = None
        tanh_1 = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
        add_15 = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_15 = torch.ops.aten.mul.Tensor(mul_12, add_15);  mul_12 = add_15 = None
        view_23 = torch.ops.aten.view.default(mul_15, [-1, 3072]);  mul_15 = None
        addmm_7 = torch.ops.aten.addmm.default(arg26_1, view_23, arg27_1);  arg26_1 = view_23 = arg27_1 = None
        view_24 = torch.ops.aten.view.default(addmm_7, [8, 128, 768]);  addmm_7 = None
        add_16 = torch.ops.aten.add.Tensor(add_11, view_24);  add_11 = view_24 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_16, [2], correction = 0, keepdim = True)
        getitem_22 = var_mean_4[0]
        getitem_23 = var_mean_4[1];  var_mean_4 = None
        add_17 = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        sub_4 = torch.ops.aten.sub.Tensor(add_16, getitem_23);  getitem_23 = None
        mul_16 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
        mul_17 = torch.ops.aten.mul.Tensor(mul_16, arg28_1);  mul_16 = arg28_1 = None
        add_18 = torch.ops.aten.add.Tensor(mul_17, arg29_1);  mul_17 = arg29_1 = None
        view_25 = torch.ops.aten.view.default(add_18, [-1, 768]);  add_18 = None
        addmm_8 = torch.ops.aten.addmm.default(arg30_1, view_25, arg31_1);  arg30_1 = view_25 = arg31_1 = None
        view_26 = torch.ops.aten.view.default(addmm_8, [8, 128, 2304]);  addmm_8 = None
        split_2 = torch.ops.aten.split.Tensor(view_26, 768, 2);  view_26 = None
        getitem_24 = split_2[0]
        getitem_25 = split_2[1]
        getitem_26 = split_2[2];  split_2 = None
        view_27 = torch.ops.aten.view.default(getitem_25, [8, 128, -1, 64]);  getitem_25 = None
        permute_8 = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None
        view_28 = torch.ops.aten.view.default(getitem_26, [8, 128, -1, 64]);  getitem_26 = None
        permute_9 = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
        view_29 = torch.ops.aten.view.default(getitem_24, [8, 128, -1, 64]);  getitem_24 = None
        permute_10 = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
        expand_2 = torch.ops.aten.expand.default(arg0_1, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_10, permute_8, permute_9, expand_2, False);  permute_10 = permute_8 = permute_9 = expand_2 = None
        getitem_27 = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None
        permute_11 = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3]);  getitem_27 = None
        view_30 = torch.ops.aten.view.default(permute_11, [8, 128, -1]);  permute_11 = None
        view_31 = torch.ops.aten.view.default(view_30, [-1, 768]);  view_30 = None
        addmm_9 = torch.ops.aten.addmm.default(arg32_1, view_31, arg33_1);  arg32_1 = view_31 = arg33_1 = None
        view_32 = torch.ops.aten.view.default(addmm_9, [8, 128, 768]);  addmm_9 = None
        add_19 = torch.ops.aten.add.Tensor(view_32, add_16);  view_32 = add_16 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
        getitem_31 = var_mean_5[0]
        getitem_32 = var_mean_5[1];  var_mean_5 = None
        add_20 = torch.ops.aten.add.Tensor(getitem_31, 1e-05);  getitem_31 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        sub_5 = torch.ops.aten.sub.Tensor(add_19, getitem_32);  getitem_32 = None
        mul_18 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
        mul_19 = torch.ops.aten.mul.Tensor(mul_18, arg34_1);  mul_18 = arg34_1 = None
        add_21 = torch.ops.aten.add.Tensor(mul_19, arg35_1);  mul_19 = arg35_1 = None
        view_33 = torch.ops.aten.view.default(add_21, [-1, 768]);  add_21 = None
        addmm_10 = torch.ops.aten.addmm.default(arg36_1, view_33, arg37_1);  arg36_1 = view_33 = arg37_1 = None
        view_34 = torch.ops.aten.view.default(addmm_10, [8, 128, 3072]);  addmm_10 = None
        mul_20 = torch.ops.aten.mul.Tensor(view_34, 0.5)
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(view_34, 3.0)
        mul_21 = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_22 = torch.ops.aten.add.Tensor(view_34, mul_21);  view_34 = mul_21 = None
        mul_22 = torch.ops.aten.mul.Tensor(add_22, 0.7978845608028654);  add_22 = None
        tanh_2 = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
        add_23 = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_23 = torch.ops.aten.mul.Tensor(mul_20, add_23);  mul_20 = add_23 = None
        view_35 = torch.ops.aten.view.default(mul_23, [-1, 3072]);  mul_23 = None
        addmm_11 = torch.ops.aten.addmm.default(arg38_1, view_35, arg39_1);  arg38_1 = view_35 = arg39_1 = None
        view_36 = torch.ops.aten.view.default(addmm_11, [8, 128, 768]);  addmm_11 = None
        add_24 = torch.ops.aten.add.Tensor(add_19, view_36);  add_19 = view_36 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
        getitem_33 = var_mean_6[0]
        getitem_34 = var_mean_6[1];  var_mean_6 = None
        add_25 = torch.ops.aten.add.Tensor(getitem_33, 1e-05);  getitem_33 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        sub_6 = torch.ops.aten.sub.Tensor(add_24, getitem_34);  getitem_34 = None
        mul_24 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
        mul_25 = torch.ops.aten.mul.Tensor(mul_24, arg40_1);  mul_24 = arg40_1 = None
        add_26 = torch.ops.aten.add.Tensor(mul_25, arg41_1);  mul_25 = arg41_1 = None
        view_37 = torch.ops.aten.view.default(add_26, [-1, 768]);  add_26 = None
        addmm_12 = torch.ops.aten.addmm.default(arg42_1, view_37, arg43_1);  arg42_1 = view_37 = arg43_1 = None
        view_38 = torch.ops.aten.view.default(addmm_12, [8, 128, 2304]);  addmm_12 = None
        split_3 = torch.ops.aten.split.Tensor(view_38, 768, 2);  view_38 = None
        getitem_35 = split_3[0]
        getitem_36 = split_3[1]
        getitem_37 = split_3[2];  split_3 = None
        view_39 = torch.ops.aten.view.default(getitem_36, [8, 128, -1, 64]);  getitem_36 = None
        permute_12 = torch.ops.aten.permute.default(view_39, [0, 2, 1, 3]);  view_39 = None
        view_40 = torch.ops.aten.view.default(getitem_37, [8, 128, -1, 64]);  getitem_37 = None
        permute_13 = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        view_41 = torch.ops.aten.view.default(getitem_35, [8, 128, -1, 64]);  getitem_35 = None
        permute_14 = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
        expand_3 = torch.ops.aten.expand.default(arg0_1, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_14, permute_12, permute_13, expand_3, False);  permute_14 = permute_12 = permute_13 = expand_3 = None
        getitem_38 = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None
        permute_15 = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3]);  getitem_38 = None
        view_42 = torch.ops.aten.view.default(permute_15, [8, 128, -1]);  permute_15 = None
        view_43 = torch.ops.aten.view.default(view_42, [-1, 768]);  view_42 = None
        addmm_13 = torch.ops.aten.addmm.default(arg44_1, view_43, arg45_1);  arg44_1 = view_43 = arg45_1 = None
        view_44 = torch.ops.aten.view.default(addmm_13, [8, 128, 768]);  addmm_13 = None
        add_27 = torch.ops.aten.add.Tensor(view_44, add_24);  view_44 = add_24 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
        getitem_42 = var_mean_7[0]
        getitem_43 = var_mean_7[1];  var_mean_7 = None
        add_28 = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_7 = torch.ops.aten.sub.Tensor(add_27, getitem_43);  getitem_43 = None
        mul_26 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
        mul_27 = torch.ops.aten.mul.Tensor(mul_26, arg46_1);  mul_26 = arg46_1 = None
        add_29 = torch.ops.aten.add.Tensor(mul_27, arg47_1);  mul_27 = arg47_1 = None
        view_45 = torch.ops.aten.view.default(add_29, [-1, 768]);  add_29 = None
        addmm_14 = torch.ops.aten.addmm.default(arg48_1, view_45, arg49_1);  arg48_1 = view_45 = arg49_1 = None
        view_46 = torch.ops.aten.view.default(addmm_14, [8, 128, 3072]);  addmm_14 = None
        mul_28 = torch.ops.aten.mul.Tensor(view_46, 0.5)
        pow_4 = torch.ops.aten.pow.Tensor_Scalar(view_46, 3.0)
        mul_29 = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_30 = torch.ops.aten.add.Tensor(view_46, mul_29);  view_46 = mul_29 = None
        mul_30 = torch.ops.aten.mul.Tensor(add_30, 0.7978845608028654);  add_30 = None
        tanh_3 = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
        add_31 = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_31 = torch.ops.aten.mul.Tensor(mul_28, add_31);  mul_28 = add_31 = None
        view_47 = torch.ops.aten.view.default(mul_31, [-1, 3072]);  mul_31 = None
        addmm_15 = torch.ops.aten.addmm.default(arg50_1, view_47, arg51_1);  arg50_1 = view_47 = arg51_1 = None
        view_48 = torch.ops.aten.view.default(addmm_15, [8, 128, 768]);  addmm_15 = None
        add_32 = torch.ops.aten.add.Tensor(add_27, view_48);  add_27 = view_48 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
        getitem_44 = var_mean_8[0]
        getitem_45 = var_mean_8[1];  var_mean_8 = None
        add_33 = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        sub_8 = torch.ops.aten.sub.Tensor(add_32, getitem_45);  getitem_45 = None
        mul_32 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
        mul_33 = torch.ops.aten.mul.Tensor(mul_32, arg52_1);  mul_32 = arg52_1 = None
        add_34 = torch.ops.aten.add.Tensor(mul_33, arg53_1);  mul_33 = arg53_1 = None
        view_49 = torch.ops.aten.view.default(add_34, [-1, 768]);  add_34 = None
        addmm_16 = torch.ops.aten.addmm.default(arg54_1, view_49, arg55_1);  arg54_1 = view_49 = arg55_1 = None
        view_50 = torch.ops.aten.view.default(addmm_16, [8, 128, 2304]);  addmm_16 = None
        split_4 = torch.ops.aten.split.Tensor(view_50, 768, 2);  view_50 = None
        getitem_46 = split_4[0]
        getitem_47 = split_4[1]
        getitem_48 = split_4[2];  split_4 = None
        view_51 = torch.ops.aten.view.default(getitem_47, [8, 128, -1, 64]);  getitem_47 = None
        permute_16 = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
        view_52 = torch.ops.aten.view.default(getitem_48, [8, 128, -1, 64]);  getitem_48 = None
        permute_17 = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
        view_53 = torch.ops.aten.view.default(getitem_46, [8, 128, -1, 64]);  getitem_46 = None
        permute_18 = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
        expand_4 = torch.ops.aten.expand.default(arg0_1, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_18, permute_16, permute_17, expand_4, False);  permute_18 = permute_16 = permute_17 = expand_4 = None
        getitem_49 = _scaled_dot_product_efficient_attention_4[0];  _scaled_dot_product_efficient_attention_4 = None
        permute_19 = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3]);  getitem_49 = None
        view_54 = torch.ops.aten.view.default(permute_19, [8, 128, -1]);  permute_19 = None
        view_55 = torch.ops.aten.view.default(view_54, [-1, 768]);  view_54 = None
        addmm_17 = torch.ops.aten.addmm.default(arg56_1, view_55, arg57_1);  arg56_1 = view_55 = arg57_1 = None
        view_56 = torch.ops.aten.view.default(addmm_17, [8, 128, 768]);  addmm_17 = None
        add_35 = torch.ops.aten.add.Tensor(view_56, add_32);  view_56 = add_32 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_53 = var_mean_9[0]
        getitem_54 = var_mean_9[1];  var_mean_9 = None
        add_36 = torch.ops.aten.add.Tensor(getitem_53, 1e-05);  getitem_53 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_9 = torch.ops.aten.sub.Tensor(add_35, getitem_54);  getitem_54 = None
        mul_34 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None
        mul_35 = torch.ops.aten.mul.Tensor(mul_34, arg58_1);  mul_34 = arg58_1 = None
        add_37 = torch.ops.aten.add.Tensor(mul_35, arg59_1);  mul_35 = arg59_1 = None
        view_57 = torch.ops.aten.view.default(add_37, [-1, 768]);  add_37 = None
        addmm_18 = torch.ops.aten.addmm.default(arg60_1, view_57, arg61_1);  arg60_1 = view_57 = arg61_1 = None
        view_58 = torch.ops.aten.view.default(addmm_18, [8, 128, 3072]);  addmm_18 = None
        mul_36 = torch.ops.aten.mul.Tensor(view_58, 0.5)
        pow_5 = torch.ops.aten.pow.Tensor_Scalar(view_58, 3.0)
        mul_37 = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_38 = torch.ops.aten.add.Tensor(view_58, mul_37);  view_58 = mul_37 = None
        mul_38 = torch.ops.aten.mul.Tensor(add_38, 0.7978845608028654);  add_38 = None
        tanh_4 = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_39 = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_39 = torch.ops.aten.mul.Tensor(mul_36, add_39);  mul_36 = add_39 = None
        view_59 = torch.ops.aten.view.default(mul_39, [-1, 3072]);  mul_39 = None
        addmm_19 = torch.ops.aten.addmm.default(arg62_1, view_59, arg63_1);  arg62_1 = view_59 = arg63_1 = None
        view_60 = torch.ops.aten.view.default(addmm_19, [8, 128, 768]);  addmm_19 = None
        add_40 = torch.ops.aten.add.Tensor(add_35, view_60);  add_35 = view_60 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_40, [2], correction = 0, keepdim = True)
        getitem_55 = var_mean_10[0]
        getitem_56 = var_mean_10[1];  var_mean_10 = None
        add_41 = torch.ops.aten.add.Tensor(getitem_55, 1e-05);  getitem_55 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
        sub_10 = torch.ops.aten.sub.Tensor(add_40, getitem_56);  getitem_56 = None
        mul_40 = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_40, arg64_1);  mul_40 = arg64_1 = None
        add_42 = torch.ops.aten.add.Tensor(mul_41, arg65_1);  mul_41 = arg65_1 = None
        view_61 = torch.ops.aten.view.default(add_42, [-1, 768]);  add_42 = None
        addmm_20 = torch.ops.aten.addmm.default(arg66_1, view_61, arg67_1);  arg66_1 = view_61 = arg67_1 = None
        view_62 = torch.ops.aten.view.default(addmm_20, [8, 128, 2304]);  addmm_20 = None
        split_5 = torch.ops.aten.split.Tensor(view_62, 768, 2);  view_62 = None
        getitem_57 = split_5[0]
        getitem_58 = split_5[1]
        getitem_59 = split_5[2];  split_5 = None
        view_63 = torch.ops.aten.view.default(getitem_58, [8, 128, -1, 64]);  getitem_58 = None
        permute_20 = torch.ops.aten.permute.default(view_63, [0, 2, 1, 3]);  view_63 = None
        view_64 = torch.ops.aten.view.default(getitem_59, [8, 128, -1, 64]);  getitem_59 = None
        permute_21 = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None
        view_65 = torch.ops.aten.view.default(getitem_57, [8, 128, -1, 64]);  getitem_57 = None
        permute_22 = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None
        expand_5 = torch.ops.aten.expand.default(arg0_1, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_22, permute_20, permute_21, expand_5, False);  permute_22 = permute_20 = permute_21 = expand_5 = None
        getitem_60 = _scaled_dot_product_efficient_attention_5[0];  _scaled_dot_product_efficient_attention_5 = None
        permute_23 = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None
        view_66 = torch.ops.aten.view.default(permute_23, [8, 128, -1]);  permute_23 = None
        view_67 = torch.ops.aten.view.default(view_66, [-1, 768]);  view_66 = None
        addmm_21 = torch.ops.aten.addmm.default(arg68_1, view_67, arg69_1);  arg68_1 = view_67 = arg69_1 = None
        view_68 = torch.ops.aten.view.default(addmm_21, [8, 128, 768]);  addmm_21 = None
        add_43 = torch.ops.aten.add.Tensor(view_68, add_40);  view_68 = add_40 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
        getitem_64 = var_mean_11[0]
        getitem_65 = var_mean_11[1];  var_mean_11 = None
        add_44 = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        sub_11 = torch.ops.aten.sub.Tensor(add_43, getitem_65);  getitem_65 = None
        mul_42 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
        mul_43 = torch.ops.aten.mul.Tensor(mul_42, arg70_1);  mul_42 = arg70_1 = None
        add_45 = torch.ops.aten.add.Tensor(mul_43, arg71_1);  mul_43 = arg71_1 = None
        view_69 = torch.ops.aten.view.default(add_45, [-1, 768]);  add_45 = None
        addmm_22 = torch.ops.aten.addmm.default(arg72_1, view_69, arg73_1);  arg72_1 = view_69 = arg73_1 = None
        view_70 = torch.ops.aten.view.default(addmm_22, [8, 128, 3072]);  addmm_22 = None
        mul_44 = torch.ops.aten.mul.Tensor(view_70, 0.5)
        pow_6 = torch.ops.aten.pow.Tensor_Scalar(view_70, 3.0)
        mul_45 = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_46 = torch.ops.aten.add.Tensor(view_70, mul_45);  view_70 = mul_45 = None
        mul_46 = torch.ops.aten.mul.Tensor(add_46, 0.7978845608028654);  add_46 = None
        tanh_5 = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
        add_47 = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_47 = torch.ops.aten.mul.Tensor(mul_44, add_47);  mul_44 = add_47 = None
        view_71 = torch.ops.aten.view.default(mul_47, [-1, 3072]);  mul_47 = None
        addmm_23 = torch.ops.aten.addmm.default(arg74_1, view_71, arg75_1);  arg74_1 = view_71 = arg75_1 = None
        view_72 = torch.ops.aten.view.default(addmm_23, [8, 128, 768]);  addmm_23 = None
        add_48 = torch.ops.aten.add.Tensor(add_43, view_72);  add_43 = view_72 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
        getitem_66 = var_mean_12[0]
        getitem_67 = var_mean_12[1];  var_mean_12 = None
        add_49 = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        sub_12 = torch.ops.aten.sub.Tensor(add_48, getitem_67);  getitem_67 = None
        mul_48 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
        mul_49 = torch.ops.aten.mul.Tensor(mul_48, arg76_1);  mul_48 = arg76_1 = None
        add_50 = torch.ops.aten.add.Tensor(mul_49, arg77_1);  mul_49 = arg77_1 = None
        view_73 = torch.ops.aten.view.default(add_50, [-1, 768]);  add_50 = None
        addmm_24 = torch.ops.aten.addmm.default(arg78_1, view_73, arg79_1);  arg78_1 = view_73 = arg79_1 = None
        view_74 = torch.ops.aten.view.default(addmm_24, [8, 128, 2304]);  addmm_24 = None
        split_6 = torch.ops.aten.split.Tensor(view_74, 768, 2);  view_74 = None
        getitem_68 = split_6[0]
        getitem_69 = split_6[1]
        getitem_70 = split_6[2];  split_6 = None
        view_75 = torch.ops.aten.view.default(getitem_69, [8, 128, -1, 64]);  getitem_69 = None
        permute_24 = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
        view_76 = torch.ops.aten.view.default(getitem_70, [8, 128, -1, 64]);  getitem_70 = None
        permute_25 = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
        view_77 = torch.ops.aten.view.default(getitem_68, [8, 128, -1, 64]);  getitem_68 = None
        permute_26 = torch.ops.aten.permute.default(view_77, [0, 2, 1, 3]);  view_77 = None
        expand_6 = torch.ops.aten.expand.default(arg0_1, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_26, permute_24, permute_25, expand_6, False);  permute_26 = permute_24 = permute_25 = expand_6 = None
        getitem_71 = _scaled_dot_product_efficient_attention_6[0];  _scaled_dot_product_efficient_attention_6 = None
        permute_27 = torch.ops.aten.permute.default(getitem_71, [0, 2, 1, 3]);  getitem_71 = None
        view_78 = torch.ops.aten.view.default(permute_27, [8, 128, -1]);  permute_27 = None
        view_79 = torch.ops.aten.view.default(view_78, [-1, 768]);  view_78 = None
        addmm_25 = torch.ops.aten.addmm.default(arg80_1, view_79, arg81_1);  arg80_1 = view_79 = arg81_1 = None
        view_80 = torch.ops.aten.view.default(addmm_25, [8, 128, 768]);  addmm_25 = None
        add_51 = torch.ops.aten.add.Tensor(view_80, add_48);  view_80 = add_48 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
        getitem_75 = var_mean_13[0]
        getitem_76 = var_mean_13[1];  var_mean_13 = None
        add_52 = torch.ops.aten.add.Tensor(getitem_75, 1e-05);  getitem_75 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        sub_13 = torch.ops.aten.sub.Tensor(add_51, getitem_76);  getitem_76 = None
        mul_50 = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = rsqrt_13 = None
        mul_51 = torch.ops.aten.mul.Tensor(mul_50, arg82_1);  mul_50 = arg82_1 = None
        add_53 = torch.ops.aten.add.Tensor(mul_51, arg83_1);  mul_51 = arg83_1 = None
        view_81 = torch.ops.aten.view.default(add_53, [-1, 768]);  add_53 = None
        addmm_26 = torch.ops.aten.addmm.default(arg84_1, view_81, arg85_1);  arg84_1 = view_81 = arg85_1 = None
        view_82 = torch.ops.aten.view.default(addmm_26, [8, 128, 3072]);  addmm_26 = None
        mul_52 = torch.ops.aten.mul.Tensor(view_82, 0.5)
        pow_7 = torch.ops.aten.pow.Tensor_Scalar(view_82, 3.0)
        mul_53 = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_54 = torch.ops.aten.add.Tensor(view_82, mul_53);  view_82 = mul_53 = None
        mul_54 = torch.ops.aten.mul.Tensor(add_54, 0.7978845608028654);  add_54 = None
        tanh_6 = torch.ops.aten.tanh.default(mul_54);  mul_54 = None
        add_55 = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
        mul_55 = torch.ops.aten.mul.Tensor(mul_52, add_55);  mul_52 = add_55 = None
        view_83 = torch.ops.aten.view.default(mul_55, [-1, 3072]);  mul_55 = None
        addmm_27 = torch.ops.aten.addmm.default(arg86_1, view_83, arg87_1);  arg86_1 = view_83 = arg87_1 = None
        view_84 = torch.ops.aten.view.default(addmm_27, [8, 128, 768]);  addmm_27 = None
        add_56 = torch.ops.aten.add.Tensor(add_51, view_84);  add_51 = view_84 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_56, [2], correction = 0, keepdim = True)
        getitem_77 = var_mean_14[0]
        getitem_78 = var_mean_14[1];  var_mean_14 = None
        add_57 = torch.ops.aten.add.Tensor(getitem_77, 1e-05);  getitem_77 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
        sub_14 = torch.ops.aten.sub.Tensor(add_56, getitem_78);  getitem_78 = None
        mul_56 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = rsqrt_14 = None
        mul_57 = torch.ops.aten.mul.Tensor(mul_56, arg88_1);  mul_56 = arg88_1 = None
        add_58 = torch.ops.aten.add.Tensor(mul_57, arg89_1);  mul_57 = arg89_1 = None
        view_85 = torch.ops.aten.view.default(add_58, [-1, 768]);  add_58 = None
        addmm_28 = torch.ops.aten.addmm.default(arg90_1, view_85, arg91_1);  arg90_1 = view_85 = arg91_1 = None
        view_86 = torch.ops.aten.view.default(addmm_28, [8, 128, 2304]);  addmm_28 = None
        split_7 = torch.ops.aten.split.Tensor(view_86, 768, 2);  view_86 = None
        getitem_79 = split_7[0]
        getitem_80 = split_7[1]
        getitem_81 = split_7[2];  split_7 = None
        view_87 = torch.ops.aten.view.default(getitem_80, [8, 128, -1, 64]);  getitem_80 = None
        permute_28 = torch.ops.aten.permute.default(view_87, [0, 2, 1, 3]);  view_87 = None
        view_88 = torch.ops.aten.view.default(getitem_81, [8, 128, -1, 64]);  getitem_81 = None
        permute_29 = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
        view_89 = torch.ops.aten.view.default(getitem_79, [8, 128, -1, 64]);  getitem_79 = None
        permute_30 = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None
        expand_7 = torch.ops.aten.expand.default(arg0_1, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_30, permute_28, permute_29, expand_7, False);  permute_30 = permute_28 = permute_29 = expand_7 = None
        getitem_82 = _scaled_dot_product_efficient_attention_7[0];  _scaled_dot_product_efficient_attention_7 = None
        permute_31 = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3]);  getitem_82 = None
        view_90 = torch.ops.aten.view.default(permute_31, [8, 128, -1]);  permute_31 = None
        view_91 = torch.ops.aten.view.default(view_90, [-1, 768]);  view_90 = None
        addmm_29 = torch.ops.aten.addmm.default(arg92_1, view_91, arg93_1);  arg92_1 = view_91 = arg93_1 = None
        view_92 = torch.ops.aten.view.default(addmm_29, [8, 128, 768]);  addmm_29 = None
        add_59 = torch.ops.aten.add.Tensor(view_92, add_56);  view_92 = add_56 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
        getitem_86 = var_mean_15[0]
        getitem_87 = var_mean_15[1];  var_mean_15 = None
        add_60 = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        sub_15 = torch.ops.aten.sub.Tensor(add_59, getitem_87);  getitem_87 = None
        mul_58 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None
        mul_59 = torch.ops.aten.mul.Tensor(mul_58, arg94_1);  mul_58 = arg94_1 = None
        add_61 = torch.ops.aten.add.Tensor(mul_59, arg95_1);  mul_59 = arg95_1 = None
        view_93 = torch.ops.aten.view.default(add_61, [-1, 768]);  add_61 = None
        addmm_30 = torch.ops.aten.addmm.default(arg96_1, view_93, arg97_1);  arg96_1 = view_93 = arg97_1 = None
        view_94 = torch.ops.aten.view.default(addmm_30, [8, 128, 3072]);  addmm_30 = None
        mul_60 = torch.ops.aten.mul.Tensor(view_94, 0.5)
        pow_8 = torch.ops.aten.pow.Tensor_Scalar(view_94, 3.0)
        mul_61 = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_62 = torch.ops.aten.add.Tensor(view_94, mul_61);  view_94 = mul_61 = None
        mul_62 = torch.ops.aten.mul.Tensor(add_62, 0.7978845608028654);  add_62 = None
        tanh_7 = torch.ops.aten.tanh.default(mul_62);  mul_62 = None
        add_63 = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
        mul_63 = torch.ops.aten.mul.Tensor(mul_60, add_63);  mul_60 = add_63 = None
        view_95 = torch.ops.aten.view.default(mul_63, [-1, 3072]);  mul_63 = None
        addmm_31 = torch.ops.aten.addmm.default(arg98_1, view_95, arg99_1);  arg98_1 = view_95 = arg99_1 = None
        view_96 = torch.ops.aten.view.default(addmm_31, [8, 128, 768]);  addmm_31 = None
        add_64 = torch.ops.aten.add.Tensor(add_59, view_96);  add_59 = view_96 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_64, [2], correction = 0, keepdim = True)
        getitem_88 = var_mean_16[0]
        getitem_89 = var_mean_16[1];  var_mean_16 = None
        add_65 = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        sub_16 = torch.ops.aten.sub.Tensor(add_64, getitem_89);  getitem_89 = None
        mul_64 = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = rsqrt_16 = None
        mul_65 = torch.ops.aten.mul.Tensor(mul_64, arg100_1);  mul_64 = arg100_1 = None
        add_66 = torch.ops.aten.add.Tensor(mul_65, arg101_1);  mul_65 = arg101_1 = None
        view_97 = torch.ops.aten.view.default(add_66, [-1, 768]);  add_66 = None
        addmm_32 = torch.ops.aten.addmm.default(arg102_1, view_97, arg103_1);  arg102_1 = view_97 = arg103_1 = None
        view_98 = torch.ops.aten.view.default(addmm_32, [8, 128, 2304]);  addmm_32 = None
        split_8 = torch.ops.aten.split.Tensor(view_98, 768, 2);  view_98 = None
        getitem_90 = split_8[0]
        getitem_91 = split_8[1]
        getitem_92 = split_8[2];  split_8 = None
        view_99 = torch.ops.aten.view.default(getitem_91, [8, 128, -1, 64]);  getitem_91 = None
        permute_32 = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None
        view_100 = torch.ops.aten.view.default(getitem_92, [8, 128, -1, 64]);  getitem_92 = None
        permute_33 = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None
        view_101 = torch.ops.aten.view.default(getitem_90, [8, 128, -1, 64]);  getitem_90 = None
        permute_34 = torch.ops.aten.permute.default(view_101, [0, 2, 1, 3]);  view_101 = None
        expand_8 = torch.ops.aten.expand.default(arg0_1, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_34, permute_32, permute_33, expand_8, False);  permute_34 = permute_32 = permute_33 = expand_8 = None
        getitem_93 = _scaled_dot_product_efficient_attention_8[0];  _scaled_dot_product_efficient_attention_8 = None
        permute_35 = torch.ops.aten.permute.default(getitem_93, [0, 2, 1, 3]);  getitem_93 = None
        view_102 = torch.ops.aten.view.default(permute_35, [8, 128, -1]);  permute_35 = None
        view_103 = torch.ops.aten.view.default(view_102, [-1, 768]);  view_102 = None
        addmm_33 = torch.ops.aten.addmm.default(arg104_1, view_103, arg105_1);  arg104_1 = view_103 = arg105_1 = None
        view_104 = torch.ops.aten.view.default(addmm_33, [8, 128, 768]);  addmm_33 = None
        add_67 = torch.ops.aten.add.Tensor(view_104, add_64);  view_104 = add_64 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
        getitem_97 = var_mean_17[0]
        getitem_98 = var_mean_17[1];  var_mean_17 = None
        add_68 = torch.ops.aten.add.Tensor(getitem_97, 1e-05);  getitem_97 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        sub_17 = torch.ops.aten.sub.Tensor(add_67, getitem_98);  getitem_98 = None
        mul_66 = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = rsqrt_17 = None
        mul_67 = torch.ops.aten.mul.Tensor(mul_66, arg106_1);  mul_66 = arg106_1 = None
        add_69 = torch.ops.aten.add.Tensor(mul_67, arg107_1);  mul_67 = arg107_1 = None
        view_105 = torch.ops.aten.view.default(add_69, [-1, 768]);  add_69 = None
        addmm_34 = torch.ops.aten.addmm.default(arg108_1, view_105, arg109_1);  arg108_1 = view_105 = arg109_1 = None
        view_106 = torch.ops.aten.view.default(addmm_34, [8, 128, 3072]);  addmm_34 = None
        mul_68 = torch.ops.aten.mul.Tensor(view_106, 0.5)
        pow_9 = torch.ops.aten.pow.Tensor_Scalar(view_106, 3.0)
        mul_69 = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_70 = torch.ops.aten.add.Tensor(view_106, mul_69);  view_106 = mul_69 = None
        mul_70 = torch.ops.aten.mul.Tensor(add_70, 0.7978845608028654);  add_70 = None
        tanh_8 = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
        add_71 = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
        mul_71 = torch.ops.aten.mul.Tensor(mul_68, add_71);  mul_68 = add_71 = None
        view_107 = torch.ops.aten.view.default(mul_71, [-1, 3072]);  mul_71 = None
        addmm_35 = torch.ops.aten.addmm.default(arg110_1, view_107, arg111_1);  arg110_1 = view_107 = arg111_1 = None
        view_108 = torch.ops.aten.view.default(addmm_35, [8, 128, 768]);  addmm_35 = None
        add_72 = torch.ops.aten.add.Tensor(add_67, view_108);  add_67 = view_108 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_72, [2], correction = 0, keepdim = True)
        getitem_99 = var_mean_18[0]
        getitem_100 = var_mean_18[1];  var_mean_18 = None
        add_73 = torch.ops.aten.add.Tensor(getitem_99, 1e-05);  getitem_99 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
        sub_18 = torch.ops.aten.sub.Tensor(add_72, getitem_100);  getitem_100 = None
        mul_72 = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = rsqrt_18 = None
        mul_73 = torch.ops.aten.mul.Tensor(mul_72, arg112_1);  mul_72 = arg112_1 = None
        add_74 = torch.ops.aten.add.Tensor(mul_73, arg113_1);  mul_73 = arg113_1 = None
        view_109 = torch.ops.aten.view.default(add_74, [-1, 768]);  add_74 = None
        addmm_36 = torch.ops.aten.addmm.default(arg114_1, view_109, arg115_1);  arg114_1 = view_109 = arg115_1 = None
        view_110 = torch.ops.aten.view.default(addmm_36, [8, 128, 2304]);  addmm_36 = None
        split_9 = torch.ops.aten.split.Tensor(view_110, 768, 2);  view_110 = None
        getitem_101 = split_9[0]
        getitem_102 = split_9[1]
        getitem_103 = split_9[2];  split_9 = None
        view_111 = torch.ops.aten.view.default(getitem_102, [8, 128, -1, 64]);  getitem_102 = None
        permute_36 = torch.ops.aten.permute.default(view_111, [0, 2, 1, 3]);  view_111 = None
        view_112 = torch.ops.aten.view.default(getitem_103, [8, 128, -1, 64]);  getitem_103 = None
        permute_37 = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
        view_113 = torch.ops.aten.view.default(getitem_101, [8, 128, -1, 64]);  getitem_101 = None
        permute_38 = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None
        expand_9 = torch.ops.aten.expand.default(arg0_1, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_38, permute_36, permute_37, expand_9, False);  permute_38 = permute_36 = permute_37 = expand_9 = None
        getitem_104 = _scaled_dot_product_efficient_attention_9[0];  _scaled_dot_product_efficient_attention_9 = None
        permute_39 = torch.ops.aten.permute.default(getitem_104, [0, 2, 1, 3]);  getitem_104 = None
        view_114 = torch.ops.aten.view.default(permute_39, [8, 128, -1]);  permute_39 = None
        view_115 = torch.ops.aten.view.default(view_114, [-1, 768]);  view_114 = None
        addmm_37 = torch.ops.aten.addmm.default(arg116_1, view_115, arg117_1);  arg116_1 = view_115 = arg117_1 = None
        view_116 = torch.ops.aten.view.default(addmm_37, [8, 128, 768]);  addmm_37 = None
        add_75 = torch.ops.aten.add.Tensor(view_116, add_72);  view_116 = add_72 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
        getitem_108 = var_mean_19[0]
        getitem_109 = var_mean_19[1];  var_mean_19 = None
        add_76 = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        sub_19 = torch.ops.aten.sub.Tensor(add_75, getitem_109);  getitem_109 = None
        mul_74 = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = rsqrt_19 = None
        mul_75 = torch.ops.aten.mul.Tensor(mul_74, arg118_1);  mul_74 = arg118_1 = None
        add_77 = torch.ops.aten.add.Tensor(mul_75, arg119_1);  mul_75 = arg119_1 = None
        view_117 = torch.ops.aten.view.default(add_77, [-1, 768]);  add_77 = None
        addmm_38 = torch.ops.aten.addmm.default(arg120_1, view_117, arg121_1);  arg120_1 = view_117 = arg121_1 = None
        view_118 = torch.ops.aten.view.default(addmm_38, [8, 128, 3072]);  addmm_38 = None
        mul_76 = torch.ops.aten.mul.Tensor(view_118, 0.5)
        pow_10 = torch.ops.aten.pow.Tensor_Scalar(view_118, 3.0)
        mul_77 = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_78 = torch.ops.aten.add.Tensor(view_118, mul_77);  view_118 = mul_77 = None
        mul_78 = torch.ops.aten.mul.Tensor(add_78, 0.7978845608028654);  add_78 = None
        tanh_9 = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
        add_79 = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
        mul_79 = torch.ops.aten.mul.Tensor(mul_76, add_79);  mul_76 = add_79 = None
        view_119 = torch.ops.aten.view.default(mul_79, [-1, 3072]);  mul_79 = None
        addmm_39 = torch.ops.aten.addmm.default(arg122_1, view_119, arg123_1);  arg122_1 = view_119 = arg123_1 = None
        view_120 = torch.ops.aten.view.default(addmm_39, [8, 128, 768]);  addmm_39 = None
        add_80 = torch.ops.aten.add.Tensor(add_75, view_120);  add_75 = view_120 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_80, [2], correction = 0, keepdim = True)
        getitem_110 = var_mean_20[0]
        getitem_111 = var_mean_20[1];  var_mean_20 = None
        add_81 = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
        sub_20 = torch.ops.aten.sub.Tensor(add_80, getitem_111);  getitem_111 = None
        mul_80 = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = rsqrt_20 = None
        mul_81 = torch.ops.aten.mul.Tensor(mul_80, arg124_1);  mul_80 = arg124_1 = None
        add_82 = torch.ops.aten.add.Tensor(mul_81, arg125_1);  mul_81 = arg125_1 = None
        view_121 = torch.ops.aten.view.default(add_82, [-1, 768]);  add_82 = None
        addmm_40 = torch.ops.aten.addmm.default(arg126_1, view_121, arg127_1);  arg126_1 = view_121 = arg127_1 = None
        view_122 = torch.ops.aten.view.default(addmm_40, [8, 128, 2304]);  addmm_40 = None
        split_10 = torch.ops.aten.split.Tensor(view_122, 768, 2);  view_122 = None
        getitem_112 = split_10[0]
        getitem_113 = split_10[1]
        getitem_114 = split_10[2];  split_10 = None
        view_123 = torch.ops.aten.view.default(getitem_113, [8, 128, -1, 64]);  getitem_113 = None
        permute_40 = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None
        view_124 = torch.ops.aten.view.default(getitem_114, [8, 128, -1, 64]);  getitem_114 = None
        permute_41 = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
        view_125 = torch.ops.aten.view.default(getitem_112, [8, 128, -1, 64]);  getitem_112 = None
        permute_42 = torch.ops.aten.permute.default(view_125, [0, 2, 1, 3]);  view_125 = None
        expand_10 = torch.ops.aten.expand.default(arg0_1, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_42, permute_40, permute_41, expand_10, False);  permute_42 = permute_40 = permute_41 = expand_10 = None
        getitem_115 = _scaled_dot_product_efficient_attention_10[0];  _scaled_dot_product_efficient_attention_10 = None
        permute_43 = torch.ops.aten.permute.default(getitem_115, [0, 2, 1, 3]);  getitem_115 = None
        view_126 = torch.ops.aten.view.default(permute_43, [8, 128, -1]);  permute_43 = None
        view_127 = torch.ops.aten.view.default(view_126, [-1, 768]);  view_126 = None
        addmm_41 = torch.ops.aten.addmm.default(arg128_1, view_127, arg129_1);  arg128_1 = view_127 = arg129_1 = None
        view_128 = torch.ops.aten.view.default(addmm_41, [8, 128, 768]);  addmm_41 = None
        add_83 = torch.ops.aten.add.Tensor(view_128, add_80);  view_128 = add_80 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
        getitem_119 = var_mean_21[0]
        getitem_120 = var_mean_21[1];  var_mean_21 = None
        add_84 = torch.ops.aten.add.Tensor(getitem_119, 1e-05);  getitem_119 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        sub_21 = torch.ops.aten.sub.Tensor(add_83, getitem_120);  getitem_120 = None
        mul_82 = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = rsqrt_21 = None
        mul_83 = torch.ops.aten.mul.Tensor(mul_82, arg130_1);  mul_82 = arg130_1 = None
        add_85 = torch.ops.aten.add.Tensor(mul_83, arg131_1);  mul_83 = arg131_1 = None
        view_129 = torch.ops.aten.view.default(add_85, [-1, 768]);  add_85 = None
        addmm_42 = torch.ops.aten.addmm.default(arg132_1, view_129, arg133_1);  arg132_1 = view_129 = arg133_1 = None
        view_130 = torch.ops.aten.view.default(addmm_42, [8, 128, 3072]);  addmm_42 = None
        mul_84 = torch.ops.aten.mul.Tensor(view_130, 0.5)
        pow_11 = torch.ops.aten.pow.Tensor_Scalar(view_130, 3.0)
        mul_85 = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_86 = torch.ops.aten.add.Tensor(view_130, mul_85);  view_130 = mul_85 = None
        mul_86 = torch.ops.aten.mul.Tensor(add_86, 0.7978845608028654);  add_86 = None
        tanh_10 = torch.ops.aten.tanh.default(mul_86);  mul_86 = None
        add_87 = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
        mul_87 = torch.ops.aten.mul.Tensor(mul_84, add_87);  mul_84 = add_87 = None
        view_131 = torch.ops.aten.view.default(mul_87, [-1, 3072]);  mul_87 = None
        addmm_43 = torch.ops.aten.addmm.default(arg134_1, view_131, arg135_1);  arg134_1 = view_131 = arg135_1 = None
        view_132 = torch.ops.aten.view.default(addmm_43, [8, 128, 768]);  addmm_43 = None
        add_88 = torch.ops.aten.add.Tensor(add_83, view_132);  add_83 = view_132 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_88, [2], correction = 0, keepdim = True)
        getitem_121 = var_mean_22[0]
        getitem_122 = var_mean_22[1];  var_mean_22 = None
        add_89 = torch.ops.aten.add.Tensor(getitem_121, 1e-05);  getitem_121 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
        sub_22 = torch.ops.aten.sub.Tensor(add_88, getitem_122);  getitem_122 = None
        mul_88 = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = rsqrt_22 = None
        mul_89 = torch.ops.aten.mul.Tensor(mul_88, arg136_1);  mul_88 = arg136_1 = None
        add_90 = torch.ops.aten.add.Tensor(mul_89, arg137_1);  mul_89 = arg137_1 = None
        view_133 = torch.ops.aten.view.default(add_90, [-1, 768]);  add_90 = None
        addmm_44 = torch.ops.aten.addmm.default(arg138_1, view_133, arg139_1);  arg138_1 = view_133 = arg139_1 = None
        view_134 = torch.ops.aten.view.default(addmm_44, [8, 128, 2304]);  addmm_44 = None
        split_11 = torch.ops.aten.split.Tensor(view_134, 768, 2);  view_134 = None
        getitem_123 = split_11[0]
        getitem_124 = split_11[1]
        getitem_125 = split_11[2];  split_11 = None
        view_135 = torch.ops.aten.view.default(getitem_124, [8, 128, -1, 64]);  getitem_124 = None
        permute_44 = torch.ops.aten.permute.default(view_135, [0, 2, 1, 3]);  view_135 = None
        view_136 = torch.ops.aten.view.default(getitem_125, [8, 128, -1, 64]);  getitem_125 = None
        permute_45 = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
        view_137 = torch.ops.aten.view.default(getitem_123, [8, 128, -1, 64]);  getitem_123 = None
        permute_46 = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None
        expand_11 = torch.ops.aten.expand.default(arg0_1, [8, 12, 128, 128]);  arg0_1 = None
        _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_46, permute_44, permute_45, expand_11, False);  permute_46 = permute_44 = permute_45 = expand_11 = None
        getitem_126 = _scaled_dot_product_efficient_attention_11[0];  _scaled_dot_product_efficient_attention_11 = None
        permute_47 = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3]);  getitem_126 = None
        view_138 = torch.ops.aten.view.default(permute_47, [8, 128, -1]);  permute_47 = None
        view_139 = torch.ops.aten.view.default(view_138, [-1, 768]);  view_138 = None
        addmm_45 = torch.ops.aten.addmm.default(arg140_1, view_139, arg141_1);  arg140_1 = view_139 = arg141_1 = None
        view_140 = torch.ops.aten.view.default(addmm_45, [8, 128, 768]);  addmm_45 = None
        add_91 = torch.ops.aten.add.Tensor(view_140, add_88);  view_140 = add_88 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
        getitem_130 = var_mean_23[0]
        getitem_131 = var_mean_23[1];  var_mean_23 = None
        add_92 = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        sub_23 = torch.ops.aten.sub.Tensor(add_91, getitem_131);  getitem_131 = None
        mul_90 = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = rsqrt_23 = None
        mul_91 = torch.ops.aten.mul.Tensor(mul_90, arg142_1);  mul_90 = arg142_1 = None
        add_93 = torch.ops.aten.add.Tensor(mul_91, arg143_1);  mul_91 = arg143_1 = None
        view_141 = torch.ops.aten.view.default(add_93, [-1, 768]);  add_93 = None
        addmm_46 = torch.ops.aten.addmm.default(arg144_1, view_141, arg145_1);  arg144_1 = view_141 = arg145_1 = None
        view_142 = torch.ops.aten.view.default(addmm_46, [8, 128, 3072]);  addmm_46 = None
        mul_92 = torch.ops.aten.mul.Tensor(view_142, 0.5)
        pow_12 = torch.ops.aten.pow.Tensor_Scalar(view_142, 3.0)
        mul_93 = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_94 = torch.ops.aten.add.Tensor(view_142, mul_93);  view_142 = mul_93 = None
        mul_94 = torch.ops.aten.mul.Tensor(add_94, 0.7978845608028654);  add_94 = None
        tanh_11 = torch.ops.aten.tanh.default(mul_94);  mul_94 = None
        add_95 = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
        mul_95 = torch.ops.aten.mul.Tensor(mul_92, add_95);  mul_92 = add_95 = None
        view_143 = torch.ops.aten.view.default(mul_95, [-1, 3072]);  mul_95 = None
        addmm_47 = torch.ops.aten.addmm.default(arg146_1, view_143, arg147_1);  arg146_1 = view_143 = arg147_1 = None
        view_144 = torch.ops.aten.view.default(addmm_47, [8, 128, 768]);  addmm_47 = None
        add_96 = torch.ops.aten.add.Tensor(add_91, view_144);  add_91 = view_144 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_96, [2], correction = 0, keepdim = True)
        getitem_132 = var_mean_24[0]
        getitem_133 = var_mean_24[1];  var_mean_24 = None
        add_97 = torch.ops.aten.add.Tensor(getitem_132, 1e-05);  getitem_132 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
        sub_24 = torch.ops.aten.sub.Tensor(add_96, getitem_133);  add_96 = getitem_133 = None
        mul_96 = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = rsqrt_24 = None
        mul_97 = torch.ops.aten.mul.Tensor(mul_96, arg148_1);  mul_96 = arg148_1 = None
        add_98 = torch.ops.aten.add.Tensor(mul_97, arg149_1);  mul_97 = arg149_1 = None
        return (add_98,)
        
def load_args(reader):
    buf0 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf0, (1, 1, 128, 128), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 8192, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf1, (8, 128), dtype=torch.int64, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 154389504, device=device(type='cuda', index=0))
    reader.tensor(buf2, (50257, 768), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf3, (1024, 768), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf4, (768,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf5, (768,), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf6, (2304,), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf7, (768, 2304), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf8, (768,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf9, (768, 768), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf10, (768,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf11, (768,), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf12, (3072,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf13, (768, 3072), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf14, (768,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf15, (3072, 768), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf16, (768,), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf17, (768,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf18, (2304,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf19, (768, 2304), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf20, (768,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf21, (768, 768), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf22, (768,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf23, (768,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf24, (3072,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf25, (768, 3072), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf26, (768,), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf27, (3072, 768), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf28, (768,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf29, (768,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf30, (2304,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf31, (768, 2304), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf32, (768,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf33, (768, 768), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf34, (768,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf35, (768,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf36, (3072,), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf37, (768, 3072), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf38, (768,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf39, (3072, 768), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf40, (768,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf41, (768,), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf42, (2304,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf43, (768, 2304), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf44, (768,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf45, (768, 768), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf46, (768,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf47, (768,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf48, (3072,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf49, (768, 3072), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf50, (768,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf51, (3072, 768), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf52, (768,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf53, (768,), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf54, (2304,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf55, (768, 2304), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf56, (768,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf57, (768, 768), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf58, (768,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf59, (768,), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf60, (3072,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf61, (768, 3072), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf62, (768,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf63, (3072, 768), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf64, (768,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf65, (768,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf66, (2304,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf67, (768, 2304), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf68, (768,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf69, (768, 768), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf70, (768,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf71, (768,), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf72, (3072,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf73, (768, 3072), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf74, (768,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf75, (3072, 768), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf76, (768,), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf77, (768,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf78, (2304,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf79, (768, 2304), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf80, (768,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf81, (768, 768), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf82, (768,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf83, (768,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf84, (3072,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf85, (768, 3072), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf86, (768,), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf87, (3072, 768), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf88, (768,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf89, (768,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf90, (2304,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf91, (768, 2304), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf92, (768,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf93, (768, 768), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf94, (768,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf95, (768,), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf96, (3072,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf97, (768, 3072), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf98, (768,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf99, (3072, 768), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf100, (768,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf101, (768,), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf102, (2304,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf103, (768, 2304), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf104, (768,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf105, (768, 768), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf106, (768,), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf107, (768,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf108, (3072,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf109, (768, 3072), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf110, (768,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf111, (3072, 768), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf112, (768,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf113, (768,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf114, (2304,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf115, (768, 2304), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf116, (768,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf117, (768, 768), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf118, (768,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf119, (768,), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf120, (3072,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf121, (768, 3072), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf122, (768,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf123, (3072, 768), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf124, (768,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf125, (768,), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf126, (2304,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf127, (768, 2304), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf128, (768,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf129, (768, 768), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf130, (768,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf131, (768,), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf132, (3072,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf133, (768, 3072), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf134, (768,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf135, (3072, 768), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf136, (768,), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf137, (768,), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf138, (2304,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf139, (768, 2304), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf140, (768,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf141, (768, 768), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf142, (768,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf143, (768,), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf144, (3072,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf145, (768, 3072), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf146, (768,), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf147, (3072, 768), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf148, (768,), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf149, (768,), is_leaf=True)  # arg149_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)

# ===== inductor generated file: /tmp/cnnbench-transformers-p50cz5fm/repeat_02/a1/torchinductor/tmpxdyb3vm9/wq/cwqiouzl5rqqzecyxn3st4tw5voktz55qizk6dw2eebgohjbtoi5.debug/fx_graph_transformed.py =====
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[1, 1, 128, 128]", arg1_1: "i64[8, 128]", arg2_1: "f32[50257, 768]", arg3_1: "f32[1024, 768]", arg4_1: "f32[768]", arg5_1: "f32[768]", arg6_1: "f32[2304]", arg7_1: "f32[768, 2304]", arg8_1: "f32[768]", arg9_1: "f32[768, 768]", arg10_1: "f32[768]", arg11_1: "f32[768]", arg12_1: "f32[3072]", arg13_1: "f32[768, 3072]", arg14_1: "f32[768]", arg15_1: "f32[3072, 768]", arg16_1: "f32[768]", arg17_1: "f32[768]", arg18_1: "f32[2304]", arg19_1: "f32[768, 2304]", arg20_1: "f32[768]", arg21_1: "f32[768, 768]", arg22_1: "f32[768]", arg23_1: "f32[768]", arg24_1: "f32[3072]", arg25_1: "f32[768, 3072]", arg26_1: "f32[768]", arg27_1: "f32[3072, 768]", arg28_1: "f32[768]", arg29_1: "f32[768]", arg30_1: "f32[2304]", arg31_1: "f32[768, 2304]", arg32_1: "f32[768]", arg33_1: "f32[768, 768]", arg34_1: "f32[768]", arg35_1: "f32[768]", arg36_1: "f32[3072]", arg37_1: "f32[768, 3072]", arg38_1: "f32[768]", arg39_1: "f32[3072, 768]", arg40_1: "f32[768]", arg41_1: "f32[768]", arg42_1: "f32[2304]", arg43_1: "f32[768, 2304]", arg44_1: "f32[768]", arg45_1: "f32[768, 768]", arg46_1: "f32[768]", arg47_1: "f32[768]", arg48_1: "f32[3072]", arg49_1: "f32[768, 3072]", arg50_1: "f32[768]", arg51_1: "f32[3072, 768]", arg52_1: "f32[768]", arg53_1: "f32[768]", arg54_1: "f32[2304]", arg55_1: "f32[768, 2304]", arg56_1: "f32[768]", arg57_1: "f32[768, 768]", arg58_1: "f32[768]", arg59_1: "f32[768]", arg60_1: "f32[3072]", arg61_1: "f32[768, 3072]", arg62_1: "f32[768]", arg63_1: "f32[3072, 768]", arg64_1: "f32[768]", arg65_1: "f32[768]", arg66_1: "f32[2304]", arg67_1: "f32[768, 2304]", arg68_1: "f32[768]", arg69_1: "f32[768, 768]", arg70_1: "f32[768]", arg71_1: "f32[768]", arg72_1: "f32[3072]", arg73_1: "f32[768, 3072]", arg74_1: "f32[768]", arg75_1: "f32[3072, 768]", arg76_1: "f32[768]", arg77_1: "f32[768]", arg78_1: "f32[2304]", arg79_1: "f32[768, 2304]", arg80_1: "f32[768]", arg81_1: "f32[768, 768]", arg82_1: "f32[768]", arg83_1: "f32[768]", arg84_1: "f32[3072]", arg85_1: "f32[768, 3072]", arg86_1: "f32[768]", arg87_1: "f32[3072, 768]", arg88_1: "f32[768]", arg89_1: "f32[768]", arg90_1: "f32[2304]", arg91_1: "f32[768, 2304]", arg92_1: "f32[768]", arg93_1: "f32[768, 768]", arg94_1: "f32[768]", arg95_1: "f32[768]", arg96_1: "f32[3072]", arg97_1: "f32[768, 3072]", arg98_1: "f32[768]", arg99_1: "f32[3072, 768]", arg100_1: "f32[768]", arg101_1: "f32[768]", arg102_1: "f32[2304]", arg103_1: "f32[768, 2304]", arg104_1: "f32[768]", arg105_1: "f32[768, 768]", arg106_1: "f32[768]", arg107_1: "f32[768]", arg108_1: "f32[3072]", arg109_1: "f32[768, 3072]", arg110_1: "f32[768]", arg111_1: "f32[3072, 768]", arg112_1: "f32[768]", arg113_1: "f32[768]", arg114_1: "f32[2304]", arg115_1: "f32[768, 2304]", arg116_1: "f32[768]", arg117_1: "f32[768, 768]", arg118_1: "f32[768]", arg119_1: "f32[768]", arg120_1: "f32[3072]", arg121_1: "f32[768, 3072]", arg122_1: "f32[768]", arg123_1: "f32[3072, 768]", arg124_1: "f32[768]", arg125_1: "f32[768]", arg126_1: "f32[2304]", arg127_1: "f32[768, 2304]", arg128_1: "f32[768]", arg129_1: "f32[768, 768]", arg130_1: "f32[768]", arg131_1: "f32[768]", arg132_1: "f32[3072]", arg133_1: "f32[768, 3072]", arg134_1: "f32[768]", arg135_1: "f32[3072, 768]", arg136_1: "f32[768]", arg137_1: "f32[768]", arg138_1: "f32[2304]", arg139_1: "f32[768, 2304]", arg140_1: "f32[768]", arg141_1: "f32[768, 768]", arg142_1: "f32[768]", arg143_1: "f32[768]", arg144_1: "f32[3072]", arg145_1: "f32[768, 3072]", arg146_1: "f32[768]", arg147_1: "f32[3072, 768]", arg148_1: "f32[768]", arg149_1: "f32[768]"):
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:857 in forward, code: inputs_embeds = self.wte(input_ids)
        embedding: "f32[8, 128, 768]" = torch.ops.aten.embedding.default(arg2_1, arg1_1);  arg2_1 = arg1_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:861 in forward, code: cache_position = torch.arange(
        iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:865 in forward, code: position_ids = cache_position.unsqueeze(0)
        unsqueeze: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:867 in forward, code: position_embeds = self.wpe(position_ids)
        embedding_1: "f32[1, 128, 768]" = torch.ops.aten.embedding.default(arg3_1, unsqueeze);  arg3_1 = unsqueeze = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:868 in forward, code: hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)
        add: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 128, 1]" = var_mean[0]
        getitem_1: "f32[8, 128, 1]" = var_mean[1];  var_mean = None
        sub: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add, getitem_1);  getitem_1 = None
        add_1: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = arg4_1 = None
        add_2: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_1: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_2, [-1, 768]);  add_2 = None
        addmm: "f32[1024, 2304]" = torch.ops.aten.addmm.default(arg6_1, view_1, arg7_1);  arg6_1 = view_1 = arg7_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_2: "f32[8, 128, 2304]" = torch.ops.aten.reshape.default(addmm, [8, 128, 2304]);  addmm = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split = torch.ops.aten.split.Tensor(view_2, 768, 2);  view_2 = None
        getitem_2: "f32[8, 128, 768]" = split[0]
        getitem_3: "f32[8, 128, 768]" = split[1]
        getitem_4: "f32[8, 128, 768]" = split[2];  split = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_5: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(getitem_2, [8, 128, -1, 64]);  getitem_2 = None
        permute_2: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_3: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(getitem_3, [8, 128, -1, 64]);  getitem_3 = None
        permute: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_4: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(getitem_4, [8, 128, -1, 64]);  getitem_4 = None
        permute_1: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(arg0_1, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_2, permute, permute_1, expand, False);  permute_2 = permute = permute_1 = expand = None
        getitem_5: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention[0];  _scaled_dot_product_efficient_attention = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_3: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_6: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(permute_3, [8, 128, -1]);  permute_3 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_7: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_6, [-1, 768]);  view_6 = None
        mm_default_35: "f32[1024, 768]" = torch.ops.aten.mm.default(view_7, arg9_1);  view_7 = arg9_1 = None
        add_tensor_35: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_35, arg8_1);  mm_default_35 = arg8_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_8: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_35, [8, 128, 768]);  add_tensor_35 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_3: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_8, add);  view_8 = add = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
        getitem_9: "f32[8, 128, 1]" = var_mean_1[0]
        getitem_10: "f32[8, 128, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_1: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_3, getitem_10);  getitem_10 = None
        add_4: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_9, 1e-05);  getitem_9 = None
        rsqrt_1: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul_2: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_3: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_2, arg10_1);  mul_2 = arg10_1 = None
        add_5: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_3, arg11_1);  mul_3 = arg11_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_9: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_5, [-1, 768]);  add_5 = None
        mm_default_34: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_9, arg13_1);  view_9 = arg13_1 = None
        add_tensor_34: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_34, arg12_1);  mm_default_34 = arg12_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_10: "f32[8, 128, 3072]" = torch.ops.aten.reshape.default(add_tensor_34, [8, 128, 3072]);  add_tensor_34 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_4: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_10, 0.5)
        pow_1: "f32[8, 128, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_10, 3.0)
        mul_5: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_6: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(view_10, mul_5);  view_10 = mul_5 = None
        mul_6: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(add_6, 0.7978845608028654);  add_6 = None
        tanh: "f32[8, 128, 3072]" = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
        add_7: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_7: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_4, add_7);  mul_4 = add_7 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_11: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_7, [-1, 3072]);  mul_7 = None
        mm_default_33: "f32[1024, 768]" = torch.ops.aten.mm.default(view_11, arg15_1);  view_11 = arg15_1 = None
        add_tensor_33: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_33, arg14_1);  mm_default_33 = arg14_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_12: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_33, [8, 128, 768]);  add_tensor_33 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_8: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(add_3, view_12);  add_3 = view_12 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
        getitem_11: "f32[8, 128, 1]" = var_mean_2[0]
        getitem_12: "f32[8, 128, 1]" = var_mean_2[1];  var_mean_2 = None
        sub_2: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_8, getitem_12);  getitem_12 = None
        add_9: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
        rsqrt_2: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        mul_8: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
        mul_9: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_8, arg16_1);  mul_8 = arg16_1 = None
        add_10: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_9, arg17_1);  mul_9 = arg17_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_13: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_10, [-1, 768]);  add_10 = None
        addmm_4: "f32[1024, 2304]" = torch.ops.aten.addmm.default(arg18_1, view_13, arg19_1);  arg18_1 = view_13 = arg19_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_14: "f32[8, 128, 2304]" = torch.ops.aten.reshape.default(addmm_4, [8, 128, 2304]);  addmm_4 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_1 = torch.ops.aten.split.Tensor(view_14, 768, 2);  view_14 = None
        getitem_13: "f32[8, 128, 768]" = split_1[0]
        getitem_14: "f32[8, 128, 768]" = split_1[1]
        getitem_15: "f32[8, 128, 768]" = split_1[2];  split_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_17: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(getitem_13, [8, 128, -1, 64]);  getitem_13 = None
        permute_6: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_15: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(getitem_14, [8, 128, -1, 64]);  getitem_14 = None
        permute_4: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_16: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(getitem_15, [8, 128, -1, 64]);  getitem_15 = None
        permute_5: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_1: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(arg0_1, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_6, permute_4, permute_5, expand_1, False);  permute_6 = permute_4 = permute_5 = expand_1 = None
        getitem_16: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_1[0];  _scaled_dot_product_efficient_attention_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3]);  getitem_16 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_18: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(permute_7, [8, 128, -1]);  permute_7 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_19: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_18, [-1, 768]);  view_18 = None
        mm_default_32: "f32[1024, 768]" = torch.ops.aten.mm.default(view_19, arg21_1);  view_19 = arg21_1 = None
        add_tensor_32: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_32, arg20_1);  mm_default_32 = arg20_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_20: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_32, [8, 128, 768]);  add_tensor_32 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_11: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_20, add_8);  view_20 = add_8 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_3 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
        getitem_20: "f32[8, 128, 1]" = var_mean_3[0]
        getitem_21: "f32[8, 128, 1]" = var_mean_3[1];  var_mean_3 = None
        sub_3: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_21);  getitem_21 = None
        add_12: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_3: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_10: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
        mul_11: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_10, arg22_1);  mul_10 = arg22_1 = None
        add_13: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_11, arg23_1);  mul_11 = arg23_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_21: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_13, [-1, 768]);  add_13 = None
        mm_default_31: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_21, arg25_1);  view_21 = arg25_1 = None
        add_tensor_31: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_31, arg24_1);  mm_default_31 = arg24_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_22: "f32[8, 128, 3072]" = torch.ops.aten.reshape.default(add_tensor_31, [8, 128, 3072]);  add_tensor_31 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_12: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_22, 0.5)
        pow_2: "f32[8, 128, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_22, 3.0)
        mul_13: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_14: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(view_22, mul_13);  view_22 = mul_13 = None
        mul_14: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(add_14, 0.7978845608028654);  add_14 = None
        tanh_1: "f32[8, 128, 3072]" = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
        add_15: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_15: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_12, add_15);  mul_12 = add_15 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_23: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_15, [-1, 3072]);  mul_15 = None
        mm_default_30: "f32[1024, 768]" = torch.ops.aten.mm.default(view_23, arg27_1);  view_23 = arg27_1 = None
        add_tensor_30: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_30, arg26_1);  mm_default_30 = arg26_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_24: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_30, [8, 128, 768]);  add_tensor_30 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_16: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(add_11, view_24);  add_11 = view_24 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_4 = torch.ops.aten.var_mean.correction(add_16, [2], correction = 0, keepdim = True)
        getitem_22: "f32[8, 128, 1]" = var_mean_4[0]
        getitem_23: "f32[8, 128, 1]" = var_mean_4[1];  var_mean_4 = None
        sub_4: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_16, getitem_23);  getitem_23 = None
        add_17: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_4: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        mul_16: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
        mul_17: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_16, arg28_1);  mul_16 = arg28_1 = None
        add_18: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_17, arg29_1);  mul_17 = arg29_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_25: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_18, [-1, 768]);  add_18 = None
        addmm_8: "f32[1024, 2304]" = torch.ops.aten.addmm.default(arg30_1, view_25, arg31_1);  arg30_1 = view_25 = arg31_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_26: "f32[8, 128, 2304]" = torch.ops.aten.reshape.default(addmm_8, [8, 128, 2304]);  addmm_8 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_2 = torch.ops.aten.split.Tensor(view_26, 768, 2);  view_26 = None
        getitem_24: "f32[8, 128, 768]" = split_2[0]
        getitem_25: "f32[8, 128, 768]" = split_2[1]
        getitem_26: "f32[8, 128, 768]" = split_2[2];  split_2 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_29: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(getitem_24, [8, 128, -1, 64]);  getitem_24 = None
        permute_10: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_27: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(getitem_25, [8, 128, -1, 64]);  getitem_25 = None
        permute_8: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_28: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(getitem_26, [8, 128, -1, 64]);  getitem_26 = None
        permute_9: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_2: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(arg0_1, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_10, permute_8, permute_9, expand_2, False);  permute_10 = permute_8 = permute_9 = expand_2 = None
        getitem_27: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_11: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3]);  getitem_27 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_30: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(permute_11, [8, 128, -1]);  permute_11 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_31: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_30, [-1, 768]);  view_30 = None
        mm_default_29: "f32[1024, 768]" = torch.ops.aten.mm.default(view_31, arg33_1);  view_31 = arg33_1 = None
        add_tensor_29: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_29, arg32_1);  mm_default_29 = arg32_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_32: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_29, [8, 128, 768]);  add_tensor_29 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_19: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_32, add_16);  view_32 = add_16 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_5 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
        getitem_31: "f32[8, 128, 1]" = var_mean_5[0]
        getitem_32: "f32[8, 128, 1]" = var_mean_5[1];  var_mean_5 = None
        sub_5: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_32);  getitem_32 = None
        add_20: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_31, 1e-05);  getitem_31 = None
        rsqrt_5: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        mul_18: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
        mul_19: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_18, arg34_1);  mul_18 = arg34_1 = None
        add_21: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_19, arg35_1);  mul_19 = arg35_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_33: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_21, [-1, 768]);  add_21 = None
        mm_default_28: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_33, arg37_1);  view_33 = arg37_1 = None
        add_tensor_28: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_28, arg36_1);  mm_default_28 = arg36_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_34: "f32[8, 128, 3072]" = torch.ops.aten.reshape.default(add_tensor_28, [8, 128, 3072]);  add_tensor_28 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_20: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_34, 0.5)
        pow_3: "f32[8, 128, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_34, 3.0)
        mul_21: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_22: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(view_34, mul_21);  view_34 = mul_21 = None
        mul_22: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(add_22, 0.7978845608028654);  add_22 = None
        tanh_2: "f32[8, 128, 3072]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
        add_23: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_23: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_20, add_23);  mul_20 = add_23 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_35: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_23, [-1, 3072]);  mul_23 = None
        mm_default_27: "f32[1024, 768]" = torch.ops.aten.mm.default(view_35, arg39_1);  view_35 = arg39_1 = None
        add_tensor_27: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_27, arg38_1);  mm_default_27 = arg38_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_36: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_27, [8, 128, 768]);  add_tensor_27 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_24: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(add_19, view_36);  add_19 = view_36 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_6 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
        getitem_33: "f32[8, 128, 1]" = var_mean_6[0]
        getitem_34: "f32[8, 128, 1]" = var_mean_6[1];  var_mean_6 = None
        sub_6: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_24, getitem_34);  getitem_34 = None
        add_25: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_33, 1e-05);  getitem_33 = None
        rsqrt_6: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        mul_24: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
        mul_25: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_24, arg40_1);  mul_24 = arg40_1 = None
        add_26: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_25, arg41_1);  mul_25 = arg41_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_37: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_26, [-1, 768]);  add_26 = None
        addmm_12: "f32[1024, 2304]" = torch.ops.aten.addmm.default(arg42_1, view_37, arg43_1);  arg42_1 = view_37 = arg43_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_38: "f32[8, 128, 2304]" = torch.ops.aten.reshape.default(addmm_12, [8, 128, 2304]);  addmm_12 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_3 = torch.ops.aten.split.Tensor(view_38, 768, 2);  view_38 = None
        getitem_35: "f32[8, 128, 768]" = split_3[0]
        getitem_36: "f32[8, 128, 768]" = split_3[1]
        getitem_37: "f32[8, 128, 768]" = split_3[2];  split_3 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_41: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(getitem_35, [8, 128, -1, 64]);  getitem_35 = None
        permute_14: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_39: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(getitem_36, [8, 128, -1, 64]);  getitem_36 = None
        permute_12: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_39, [0, 2, 1, 3]);  view_39 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_40: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(getitem_37, [8, 128, -1, 64]);  getitem_37 = None
        permute_13: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_3: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(arg0_1, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_14, permute_12, permute_13, expand_3, False);  permute_14 = permute_12 = permute_13 = expand_3 = None
        getitem_38: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_15: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3]);  getitem_38 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_42: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(permute_15, [8, 128, -1]);  permute_15 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_43: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_42, [-1, 768]);  view_42 = None
        mm_default_26: "f32[1024, 768]" = torch.ops.aten.mm.default(view_43, arg45_1);  view_43 = arg45_1 = None
        add_tensor_26: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_26, arg44_1);  mm_default_26 = arg44_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_44: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_26, [8, 128, 768]);  add_tensor_26 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_27: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_44, add_24);  view_44 = add_24 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_7 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
        getitem_42: "f32[8, 128, 1]" = var_mean_7[0]
        getitem_43: "f32[8, 128, 1]" = var_mean_7[1];  var_mean_7 = None
        sub_7: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_43);  getitem_43 = None
        add_28: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_7: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_26: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
        mul_27: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_26, arg46_1);  mul_26 = arg46_1 = None
        add_29: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_27, arg47_1);  mul_27 = arg47_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_45: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_29, [-1, 768]);  add_29 = None
        mm_default_25: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_45, arg49_1);  view_45 = arg49_1 = None
        add_tensor_25: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_25, arg48_1);  mm_default_25 = arg48_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_46: "f32[8, 128, 3072]" = torch.ops.aten.reshape.default(add_tensor_25, [8, 128, 3072]);  add_tensor_25 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_28: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_46, 0.5)
        pow_4: "f32[8, 128, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_46, 3.0)
        mul_29: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_30: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(view_46, mul_29);  view_46 = mul_29 = None
        mul_30: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(add_30, 0.7978845608028654);  add_30 = None
        tanh_3: "f32[8, 128, 3072]" = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
        add_31: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_31: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_28, add_31);  mul_28 = add_31 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_47: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_31, [-1, 3072]);  mul_31 = None
        mm_default_24: "f32[1024, 768]" = torch.ops.aten.mm.default(view_47, arg51_1);  view_47 = arg51_1 = None
        add_tensor_24: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_24, arg50_1);  mm_default_24 = arg50_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_48: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_24, [8, 128, 768]);  add_tensor_24 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_32: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(add_27, view_48);  add_27 = view_48 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_8 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
        getitem_44: "f32[8, 128, 1]" = var_mean_8[0]
        getitem_45: "f32[8, 128, 1]" = var_mean_8[1];  var_mean_8 = None
        sub_8: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_32, getitem_45);  getitem_45 = None
        add_33: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_8: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        mul_32: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
        mul_33: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_32, arg52_1);  mul_32 = arg52_1 = None
        add_34: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_33, arg53_1);  mul_33 = arg53_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_49: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_34, [-1, 768]);  add_34 = None
        addmm_16: "f32[1024, 2304]" = torch.ops.aten.addmm.default(arg54_1, view_49, arg55_1);  arg54_1 = view_49 = arg55_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_50: "f32[8, 128, 2304]" = torch.ops.aten.reshape.default(addmm_16, [8, 128, 2304]);  addmm_16 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_4 = torch.ops.aten.split.Tensor(view_50, 768, 2);  view_50 = None
        getitem_46: "f32[8, 128, 768]" = split_4[0]
        getitem_47: "f32[8, 128, 768]" = split_4[1]
        getitem_48: "f32[8, 128, 768]" = split_4[2];  split_4 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_53: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(getitem_46, [8, 128, -1, 64]);  getitem_46 = None
        permute_18: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_51: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(getitem_47, [8, 128, -1, 64]);  getitem_47 = None
        permute_16: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_52: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(getitem_48, [8, 128, -1, 64]);  getitem_48 = None
        permute_17: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_4: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(arg0_1, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_18, permute_16, permute_17, expand_4, False);  permute_18 = permute_16 = permute_17 = expand_4 = None
        getitem_49: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_4[0];  _scaled_dot_product_efficient_attention_4 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_19: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3]);  getitem_49 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_54: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(permute_19, [8, 128, -1]);  permute_19 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_55: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_54, [-1, 768]);  view_54 = None
        mm_default_23: "f32[1024, 768]" = torch.ops.aten.mm.default(view_55, arg57_1);  view_55 = arg57_1 = None
        add_tensor_23: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_23, arg56_1);  mm_default_23 = arg56_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_56: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_23, [8, 128, 768]);  add_tensor_23 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_35: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_56, add_32);  view_56 = add_32 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_9 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_53: "f32[8, 128, 1]" = var_mean_9[0]
        getitem_54: "f32[8, 128, 1]" = var_mean_9[1];  var_mean_9 = None
        sub_9: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_54);  getitem_54 = None
        add_36: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_53, 1e-05);  getitem_53 = None
        rsqrt_9: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        mul_34: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None
        mul_35: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_34, arg58_1);  mul_34 = arg58_1 = None
        add_37: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_35, arg59_1);  mul_35 = arg59_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_57: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_37, [-1, 768]);  add_37 = None
        mm_default_22: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_57, arg61_1);  view_57 = arg61_1 = None
        add_tensor_22: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_22, arg60_1);  mm_default_22 = arg60_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_58: "f32[8, 128, 3072]" = torch.ops.aten.reshape.default(add_tensor_22, [8, 128, 3072]);  add_tensor_22 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_36: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.5)
        pow_5: "f32[8, 128, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_58, 3.0)
        mul_37: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_38: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(view_58, mul_37);  view_58 = mul_37 = None
        mul_38: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(add_38, 0.7978845608028654);  add_38 = None
        tanh_4: "f32[8, 128, 3072]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_39: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_39: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_36, add_39);  mul_36 = add_39 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_59: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_39, [-1, 3072]);  mul_39 = None
        mm_default_21: "f32[1024, 768]" = torch.ops.aten.mm.default(view_59, arg63_1);  view_59 = arg63_1 = None
        add_tensor_21: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_21, arg62_1);  mm_default_21 = arg62_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_60: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_21, [8, 128, 768]);  add_tensor_21 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_40: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(add_35, view_60);  add_35 = view_60 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_10 = torch.ops.aten.var_mean.correction(add_40, [2], correction = 0, keepdim = True)
        getitem_55: "f32[8, 128, 1]" = var_mean_10[0]
        getitem_56: "f32[8, 128, 1]" = var_mean_10[1];  var_mean_10 = None
        sub_10: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_40, getitem_56);  getitem_56 = None
        add_41: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_55, 1e-05);  getitem_55 = None
        rsqrt_10: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
        mul_40: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None
        mul_41: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_40, arg64_1);  mul_40 = arg64_1 = None
        add_42: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_41, arg65_1);  mul_41 = arg65_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_61: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_42, [-1, 768]);  add_42 = None
        addmm_20: "f32[1024, 2304]" = torch.ops.aten.addmm.default(arg66_1, view_61, arg67_1);  arg66_1 = view_61 = arg67_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_62: "f32[8, 128, 2304]" = torch.ops.aten.reshape.default(addmm_20, [8, 128, 2304]);  addmm_20 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_5 = torch.ops.aten.split.Tensor(view_62, 768, 2);  view_62 = None
        getitem_57: "f32[8, 128, 768]" = split_5[0]
        getitem_58: "f32[8, 128, 768]" = split_5[1]
        getitem_59: "f32[8, 128, 768]" = split_5[2];  split_5 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_65: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(getitem_57, [8, 128, -1, 64]);  getitem_57 = None
        permute_22: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_63: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(getitem_58, [8, 128, -1, 64]);  getitem_58 = None
        permute_20: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_63, [0, 2, 1, 3]);  view_63 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_64: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(getitem_59, [8, 128, -1, 64]);  getitem_59 = None
        permute_21: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_5: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(arg0_1, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_22, permute_20, permute_21, expand_5, False);  permute_22 = permute_20 = permute_21 = expand_5 = None
        getitem_60: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_5[0];  _scaled_dot_product_efficient_attention_5 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_23: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_66: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(permute_23, [8, 128, -1]);  permute_23 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_67: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_66, [-1, 768]);  view_66 = None
        mm_default_20: "f32[1024, 768]" = torch.ops.aten.mm.default(view_67, arg69_1);  view_67 = arg69_1 = None
        add_tensor_20: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_20, arg68_1);  mm_default_20 = arg68_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_68: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_20, [8, 128, 768]);  add_tensor_20 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_43: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_68, add_40);  view_68 = add_40 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_11 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
        getitem_64: "f32[8, 128, 1]" = var_mean_11[0]
        getitem_65: "f32[8, 128, 1]" = var_mean_11[1];  var_mean_11 = None
        sub_11: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_65);  getitem_65 = None
        add_44: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_11: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        mul_42: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
        mul_43: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_42, arg70_1);  mul_42 = arg70_1 = None
        add_45: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_43, arg71_1);  mul_43 = arg71_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_69: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_45, [-1, 768]);  add_45 = None
        mm_default_19: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_69, arg73_1);  view_69 = arg73_1 = None
        add_tensor_19: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_19, arg72_1);  mm_default_19 = arg72_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_70: "f32[8, 128, 3072]" = torch.ops.aten.reshape.default(add_tensor_19, [8, 128, 3072]);  add_tensor_19 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_44: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_70, 0.5)
        pow_6: "f32[8, 128, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_70, 3.0)
        mul_45: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_46: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(view_70, mul_45);  view_70 = mul_45 = None
        mul_46: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(add_46, 0.7978845608028654);  add_46 = None
        tanh_5: "f32[8, 128, 3072]" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
        add_47: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_47: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_44, add_47);  mul_44 = add_47 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_71: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_47, [-1, 3072]);  mul_47 = None
        mm_default_18: "f32[1024, 768]" = torch.ops.aten.mm.default(view_71, arg75_1);  view_71 = arg75_1 = None
        add_tensor_18: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_18, arg74_1);  mm_default_18 = arg74_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_72: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_18, [8, 128, 768]);  add_tensor_18 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_48: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(add_43, view_72);  add_43 = view_72 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_12 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
        getitem_66: "f32[8, 128, 1]" = var_mean_12[0]
        getitem_67: "f32[8, 128, 1]" = var_mean_12[1];  var_mean_12 = None
        sub_12: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_48, getitem_67);  getitem_67 = None
        add_49: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_12: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        mul_48: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
        mul_49: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_48, arg76_1);  mul_48 = arg76_1 = None
        add_50: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_49, arg77_1);  mul_49 = arg77_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_73: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_50, [-1, 768]);  add_50 = None
        addmm_24: "f32[1024, 2304]" = torch.ops.aten.addmm.default(arg78_1, view_73, arg79_1);  arg78_1 = view_73 = arg79_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_74: "f32[8, 128, 2304]" = torch.ops.aten.reshape.default(addmm_24, [8, 128, 2304]);  addmm_24 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_6 = torch.ops.aten.split.Tensor(view_74, 768, 2);  view_74 = None
        getitem_68: "f32[8, 128, 768]" = split_6[0]
        getitem_69: "f32[8, 128, 768]" = split_6[1]
        getitem_70: "f32[8, 128, 768]" = split_6[2];  split_6 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_77: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(getitem_68, [8, 128, -1, 64]);  getitem_68 = None
        permute_26: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_77, [0, 2, 1, 3]);  view_77 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_75: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(getitem_69, [8, 128, -1, 64]);  getitem_69 = None
        permute_24: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_76: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(getitem_70, [8, 128, -1, 64]);  getitem_70 = None
        permute_25: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_6: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(arg0_1, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_26, permute_24, permute_25, expand_6, False);  permute_26 = permute_24 = permute_25 = expand_6 = None
        getitem_71: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_6[0];  _scaled_dot_product_efficient_attention_6 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_27: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_71, [0, 2, 1, 3]);  getitem_71 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_78: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(permute_27, [8, 128, -1]);  permute_27 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_79: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_78, [-1, 768]);  view_78 = None
        mm_default_17: "f32[1024, 768]" = torch.ops.aten.mm.default(view_79, arg81_1);  view_79 = arg81_1 = None
        add_tensor_17: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_17, arg80_1);  mm_default_17 = arg80_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_80: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_17, [8, 128, 768]);  add_tensor_17 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_51: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_80, add_48);  view_80 = add_48 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_13 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
        getitem_75: "f32[8, 128, 1]" = var_mean_13[0]
        getitem_76: "f32[8, 128, 1]" = var_mean_13[1];  var_mean_13 = None
        sub_13: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_51, getitem_76);  getitem_76 = None
        add_52: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_75, 1e-05);  getitem_75 = None
        rsqrt_13: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        mul_50: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = rsqrt_13 = None
        mul_51: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_50, arg82_1);  mul_50 = arg82_1 = None
        add_53: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_51, arg83_1);  mul_51 = arg83_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_81: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_53, [-1, 768]);  add_53 = None
        mm_default_16: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_81, arg85_1);  view_81 = arg85_1 = None
        add_tensor_16: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_16, arg84_1);  mm_default_16 = arg84_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_82: "f32[8, 128, 3072]" = torch.ops.aten.reshape.default(add_tensor_16, [8, 128, 3072]);  add_tensor_16 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_52: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_82, 0.5)
        pow_7: "f32[8, 128, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_82, 3.0)
        mul_53: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_54: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(view_82, mul_53);  view_82 = mul_53 = None
        mul_54: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(add_54, 0.7978845608028654);  add_54 = None
        tanh_6: "f32[8, 128, 3072]" = torch.ops.aten.tanh.default(mul_54);  mul_54 = None
        add_55: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
        mul_55: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_52, add_55);  mul_52 = add_55 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_83: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_55, [-1, 3072]);  mul_55 = None
        mm_default_15: "f32[1024, 768]" = torch.ops.aten.mm.default(view_83, arg87_1);  view_83 = arg87_1 = None
        add_tensor_15: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_15, arg86_1);  mm_default_15 = arg86_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_84: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_15, [8, 128, 768]);  add_tensor_15 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_56: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(add_51, view_84);  add_51 = view_84 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_14 = torch.ops.aten.var_mean.correction(add_56, [2], correction = 0, keepdim = True)
        getitem_77: "f32[8, 128, 1]" = var_mean_14[0]
        getitem_78: "f32[8, 128, 1]" = var_mean_14[1];  var_mean_14 = None
        sub_14: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_56, getitem_78);  getitem_78 = None
        add_57: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_77, 1e-05);  getitem_77 = None
        rsqrt_14: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
        mul_56: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = rsqrt_14 = None
        mul_57: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_56, arg88_1);  mul_56 = arg88_1 = None
        add_58: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_57, arg89_1);  mul_57 = arg89_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_85: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_58, [-1, 768]);  add_58 = None
        addmm_28: "f32[1024, 2304]" = torch.ops.aten.addmm.default(arg90_1, view_85, arg91_1);  arg90_1 = view_85 = arg91_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_86: "f32[8, 128, 2304]" = torch.ops.aten.reshape.default(addmm_28, [8, 128, 2304]);  addmm_28 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_7 = torch.ops.aten.split.Tensor(view_86, 768, 2);  view_86 = None
        getitem_79: "f32[8, 128, 768]" = split_7[0]
        getitem_80: "f32[8, 128, 768]" = split_7[1]
        getitem_81: "f32[8, 128, 768]" = split_7[2];  split_7 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_89: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(getitem_79, [8, 128, -1, 64]);  getitem_79 = None
        permute_30: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_87: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(getitem_80, [8, 128, -1, 64]);  getitem_80 = None
        permute_28: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_87, [0, 2, 1, 3]);  view_87 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_88: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(getitem_81, [8, 128, -1, 64]);  getitem_81 = None
        permute_29: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_7: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(arg0_1, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_30, permute_28, permute_29, expand_7, False);  permute_30 = permute_28 = permute_29 = expand_7 = None
        getitem_82: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_7[0];  _scaled_dot_product_efficient_attention_7 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_31: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3]);  getitem_82 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_90: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(permute_31, [8, 128, -1]);  permute_31 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_91: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_90, [-1, 768]);  view_90 = None
        mm_default_14: "f32[1024, 768]" = torch.ops.aten.mm.default(view_91, arg93_1);  view_91 = arg93_1 = None
        add_tensor_14: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_14, arg92_1);  mm_default_14 = arg92_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_92: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_14, [8, 128, 768]);  add_tensor_14 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_59: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_92, add_56);  view_92 = add_56 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_15 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
        getitem_86: "f32[8, 128, 1]" = var_mean_15[0]
        getitem_87: "f32[8, 128, 1]" = var_mean_15[1];  var_mean_15 = None
        sub_15: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_87);  getitem_87 = None
        add_60: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
        rsqrt_15: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        mul_58: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None
        mul_59: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_58, arg94_1);  mul_58 = arg94_1 = None
        add_61: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_59, arg95_1);  mul_59 = arg95_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_93: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_61, [-1, 768]);  add_61 = None
        mm_default_13: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_93, arg97_1);  view_93 = arg97_1 = None
        add_tensor_13: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_13, arg96_1);  mm_default_13 = arg96_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_94: "f32[8, 128, 3072]" = torch.ops.aten.reshape.default(add_tensor_13, [8, 128, 3072]);  add_tensor_13 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_60: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_94, 0.5)
        pow_8: "f32[8, 128, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_94, 3.0)
        mul_61: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_62: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(view_94, mul_61);  view_94 = mul_61 = None
        mul_62: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(add_62, 0.7978845608028654);  add_62 = None
        tanh_7: "f32[8, 128, 3072]" = torch.ops.aten.tanh.default(mul_62);  mul_62 = None
        add_63: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
        mul_63: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_60, add_63);  mul_60 = add_63 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_95: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_63, [-1, 3072]);  mul_63 = None
        mm_default_12: "f32[1024, 768]" = torch.ops.aten.mm.default(view_95, arg99_1);  view_95 = arg99_1 = None
        add_tensor_12: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_12, arg98_1);  mm_default_12 = arg98_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_96: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_12, [8, 128, 768]);  add_tensor_12 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_64: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(add_59, view_96);  add_59 = view_96 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_16 = torch.ops.aten.var_mean.correction(add_64, [2], correction = 0, keepdim = True)
        getitem_88: "f32[8, 128, 1]" = var_mean_16[0]
        getitem_89: "f32[8, 128, 1]" = var_mean_16[1];  var_mean_16 = None
        sub_16: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_64, getitem_89);  getitem_89 = None
        add_65: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_16: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        mul_64: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = rsqrt_16 = None
        mul_65: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_64, arg100_1);  mul_64 = arg100_1 = None
        add_66: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_65, arg101_1);  mul_65 = arg101_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_97: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_66, [-1, 768]);  add_66 = None
        addmm_32: "f32[1024, 2304]" = torch.ops.aten.addmm.default(arg102_1, view_97, arg103_1);  arg102_1 = view_97 = arg103_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_98: "f32[8, 128, 2304]" = torch.ops.aten.reshape.default(addmm_32, [8, 128, 2304]);  addmm_32 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_8 = torch.ops.aten.split.Tensor(view_98, 768, 2);  view_98 = None
        getitem_90: "f32[8, 128, 768]" = split_8[0]
        getitem_91: "f32[8, 128, 768]" = split_8[1]
        getitem_92: "f32[8, 128, 768]" = split_8[2];  split_8 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_101: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(getitem_90, [8, 128, -1, 64]);  getitem_90 = None
        permute_34: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_101, [0, 2, 1, 3]);  view_101 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_99: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(getitem_91, [8, 128, -1, 64]);  getitem_91 = None
        permute_32: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_100: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(getitem_92, [8, 128, -1, 64]);  getitem_92 = None
        permute_33: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_8: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(arg0_1, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_34, permute_32, permute_33, expand_8, False);  permute_34 = permute_32 = permute_33 = expand_8 = None
        getitem_93: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_8[0];  _scaled_dot_product_efficient_attention_8 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_35: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_93, [0, 2, 1, 3]);  getitem_93 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_102: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(permute_35, [8, 128, -1]);  permute_35 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_103: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_102, [-1, 768]);  view_102 = None
        mm_default_11: "f32[1024, 768]" = torch.ops.aten.mm.default(view_103, arg105_1);  view_103 = arg105_1 = None
        add_tensor_11: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_11, arg104_1);  mm_default_11 = arg104_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_104: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_11, [8, 128, 768]);  add_tensor_11 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_67: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_104, add_64);  view_104 = add_64 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_17 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
        getitem_97: "f32[8, 128, 1]" = var_mean_17[0]
        getitem_98: "f32[8, 128, 1]" = var_mean_17[1];  var_mean_17 = None
        sub_17: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_67, getitem_98);  getitem_98 = None
        add_68: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_97, 1e-05);  getitem_97 = None
        rsqrt_17: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        mul_66: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = rsqrt_17 = None
        mul_67: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_66, arg106_1);  mul_66 = arg106_1 = None
        add_69: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_67, arg107_1);  mul_67 = arg107_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_105: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_69, [-1, 768]);  add_69 = None
        mm_default_10: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_105, arg109_1);  view_105 = arg109_1 = None
        add_tensor_10: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_10, arg108_1);  mm_default_10 = arg108_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_106: "f32[8, 128, 3072]" = torch.ops.aten.reshape.default(add_tensor_10, [8, 128, 3072]);  add_tensor_10 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_68: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_106, 0.5)
        pow_9: "f32[8, 128, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_106, 3.0)
        mul_69: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_70: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(view_106, mul_69);  view_106 = mul_69 = None
        mul_70: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(add_70, 0.7978845608028654);  add_70 = None
        tanh_8: "f32[8, 128, 3072]" = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
        add_71: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
        mul_71: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_68, add_71);  mul_68 = add_71 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_107: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_71, [-1, 3072]);  mul_71 = None
        mm_default_9: "f32[1024, 768]" = torch.ops.aten.mm.default(view_107, arg111_1);  view_107 = arg111_1 = None
        add_tensor_9: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_9, arg110_1);  mm_default_9 = arg110_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_108: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_9, [8, 128, 768]);  add_tensor_9 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_72: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(add_67, view_108);  add_67 = view_108 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_18 = torch.ops.aten.var_mean.correction(add_72, [2], correction = 0, keepdim = True)
        getitem_99: "f32[8, 128, 1]" = var_mean_18[0]
        getitem_100: "f32[8, 128, 1]" = var_mean_18[1];  var_mean_18 = None
        sub_18: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_72, getitem_100);  getitem_100 = None
        add_73: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_99, 1e-05);  getitem_99 = None
        rsqrt_18: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
        mul_72: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = rsqrt_18 = None
        mul_73: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_72, arg112_1);  mul_72 = arg112_1 = None
        add_74: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_73, arg113_1);  mul_73 = arg113_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_109: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_74, [-1, 768]);  add_74 = None
        addmm_36: "f32[1024, 2304]" = torch.ops.aten.addmm.default(arg114_1, view_109, arg115_1);  arg114_1 = view_109 = arg115_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_110: "f32[8, 128, 2304]" = torch.ops.aten.reshape.default(addmm_36, [8, 128, 2304]);  addmm_36 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_9 = torch.ops.aten.split.Tensor(view_110, 768, 2);  view_110 = None
        getitem_101: "f32[8, 128, 768]" = split_9[0]
        getitem_102: "f32[8, 128, 768]" = split_9[1]
        getitem_103: "f32[8, 128, 768]" = split_9[2];  split_9 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_113: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(getitem_101, [8, 128, -1, 64]);  getitem_101 = None
        permute_38: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_111: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(getitem_102, [8, 128, -1, 64]);  getitem_102 = None
        permute_36: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_111, [0, 2, 1, 3]);  view_111 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_112: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(getitem_103, [8, 128, -1, 64]);  getitem_103 = None
        permute_37: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_9: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(arg0_1, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_38, permute_36, permute_37, expand_9, False);  permute_38 = permute_36 = permute_37 = expand_9 = None
        getitem_104: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_9[0];  _scaled_dot_product_efficient_attention_9 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_39: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_104, [0, 2, 1, 3]);  getitem_104 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_114: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(permute_39, [8, 128, -1]);  permute_39 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_115: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_114, [-1, 768]);  view_114 = None
        mm_default_8: "f32[1024, 768]" = torch.ops.aten.mm.default(view_115, arg117_1);  view_115 = arg117_1 = None
        add_tensor_8: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_8, arg116_1);  mm_default_8 = arg116_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_116: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_8, [8, 128, 768]);  add_tensor_8 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_75: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_116, add_72);  view_116 = add_72 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_19 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
        getitem_108: "f32[8, 128, 1]" = var_mean_19[0]
        getitem_109: "f32[8, 128, 1]" = var_mean_19[1];  var_mean_19 = None
        sub_19: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_75, getitem_109);  getitem_109 = None
        add_76: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
        rsqrt_19: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        mul_74: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = rsqrt_19 = None
        mul_75: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_74, arg118_1);  mul_74 = arg118_1 = None
        add_77: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_75, arg119_1);  mul_75 = arg119_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_117: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_77, [-1, 768]);  add_77 = None
        mm_default_7: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_117, arg121_1);  view_117 = arg121_1 = None
        add_tensor_7: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_7, arg120_1);  mm_default_7 = arg120_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_118: "f32[8, 128, 3072]" = torch.ops.aten.reshape.default(add_tensor_7, [8, 128, 3072]);  add_tensor_7 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_76: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_118, 0.5)
        pow_10: "f32[8, 128, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_118, 3.0)
        mul_77: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_78: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(view_118, mul_77);  view_118 = mul_77 = None
        mul_78: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(add_78, 0.7978845608028654);  add_78 = None
        tanh_9: "f32[8, 128, 3072]" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
        add_79: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
        mul_79: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_76, add_79);  mul_76 = add_79 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_119: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_79, [-1, 3072]);  mul_79 = None
        mm_default_6: "f32[1024, 768]" = torch.ops.aten.mm.default(view_119, arg123_1);  view_119 = arg123_1 = None
        add_tensor_6: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_6, arg122_1);  mm_default_6 = arg122_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_120: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_6, [8, 128, 768]);  add_tensor_6 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_80: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(add_75, view_120);  add_75 = view_120 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_20 = torch.ops.aten.var_mean.correction(add_80, [2], correction = 0, keepdim = True)
        getitem_110: "f32[8, 128, 1]" = var_mean_20[0]
        getitem_111: "f32[8, 128, 1]" = var_mean_20[1];  var_mean_20 = None
        sub_20: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_80, getitem_111);  getitem_111 = None
        add_81: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
        rsqrt_20: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
        mul_80: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = rsqrt_20 = None
        mul_81: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_80, arg124_1);  mul_80 = arg124_1 = None
        add_82: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_81, arg125_1);  mul_81 = arg125_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_121: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_82, [-1, 768]);  add_82 = None
        addmm_40: "f32[1024, 2304]" = torch.ops.aten.addmm.default(arg126_1, view_121, arg127_1);  arg126_1 = view_121 = arg127_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_122: "f32[8, 128, 2304]" = torch.ops.aten.reshape.default(addmm_40, [8, 128, 2304]);  addmm_40 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_10 = torch.ops.aten.split.Tensor(view_122, 768, 2);  view_122 = None
        getitem_112: "f32[8, 128, 768]" = split_10[0]
        getitem_113: "f32[8, 128, 768]" = split_10[1]
        getitem_114: "f32[8, 128, 768]" = split_10[2];  split_10 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_125: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(getitem_112, [8, 128, -1, 64]);  getitem_112 = None
        permute_42: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_125, [0, 2, 1, 3]);  view_125 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_123: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(getitem_113, [8, 128, -1, 64]);  getitem_113 = None
        permute_40: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_124: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(getitem_114, [8, 128, -1, 64]);  getitem_114 = None
        permute_41: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_10: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(arg0_1, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_42, permute_40, permute_41, expand_10, False);  permute_42 = permute_40 = permute_41 = expand_10 = None
        getitem_115: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_10[0];  _scaled_dot_product_efficient_attention_10 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_43: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_115, [0, 2, 1, 3]);  getitem_115 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_126: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(permute_43, [8, 128, -1]);  permute_43 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_127: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_126, [-1, 768]);  view_126 = None
        mm_default_5: "f32[1024, 768]" = torch.ops.aten.mm.default(view_127, arg129_1);  view_127 = arg129_1 = None
        add_tensor_5: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_5, arg128_1);  mm_default_5 = arg128_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_128: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_5, [8, 128, 768]);  add_tensor_5 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_83: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_128, add_80);  view_128 = add_80 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_21 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
        getitem_119: "f32[8, 128, 1]" = var_mean_21[0]
        getitem_120: "f32[8, 128, 1]" = var_mean_21[1];  var_mean_21 = None
        sub_21: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_83, getitem_120);  getitem_120 = None
        add_84: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_119, 1e-05);  getitem_119 = None
        rsqrt_21: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        mul_82: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = rsqrt_21 = None
        mul_83: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_82, arg130_1);  mul_82 = arg130_1 = None
        add_85: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_83, arg131_1);  mul_83 = arg131_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_129: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_85, [-1, 768]);  add_85 = None
        mm_default_4: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_129, arg133_1);  view_129 = arg133_1 = None
        add_tensor_4: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_4, arg132_1);  mm_default_4 = arg132_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_130: "f32[8, 128, 3072]" = torch.ops.aten.reshape.default(add_tensor_4, [8, 128, 3072]);  add_tensor_4 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_84: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_130, 0.5)
        pow_11: "f32[8, 128, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_130, 3.0)
        mul_85: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_86: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(view_130, mul_85);  view_130 = mul_85 = None
        mul_86: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(add_86, 0.7978845608028654);  add_86 = None
        tanh_10: "f32[8, 128, 3072]" = torch.ops.aten.tanh.default(mul_86);  mul_86 = None
        add_87: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
        mul_87: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_84, add_87);  mul_84 = add_87 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_131: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_87, [-1, 3072]);  mul_87 = None
        mm_default_3: "f32[1024, 768]" = torch.ops.aten.mm.default(view_131, arg135_1);  view_131 = arg135_1 = None
        add_tensor_3: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_3, arg134_1);  mm_default_3 = arg134_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_132: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_3, [8, 128, 768]);  add_tensor_3 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_88: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(add_83, view_132);  add_83 = view_132 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_22 = torch.ops.aten.var_mean.correction(add_88, [2], correction = 0, keepdim = True)
        getitem_121: "f32[8, 128, 1]" = var_mean_22[0]
        getitem_122: "f32[8, 128, 1]" = var_mean_22[1];  var_mean_22 = None
        sub_22: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_88, getitem_122);  getitem_122 = None
        add_89: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_121, 1e-05);  getitem_121 = None
        rsqrt_22: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
        mul_88: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = rsqrt_22 = None
        mul_89: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_88, arg136_1);  mul_88 = arg136_1 = None
        add_90: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_89, arg137_1);  mul_89 = arg137_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_133: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_90, [-1, 768]);  add_90 = None
        addmm_44: "f32[1024, 2304]" = torch.ops.aten.addmm.default(arg138_1, view_133, arg139_1);  arg138_1 = view_133 = arg139_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_134: "f32[8, 128, 2304]" = torch.ops.aten.reshape.default(addmm_44, [8, 128, 2304]);  addmm_44 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_11 = torch.ops.aten.split.Tensor(view_134, 768, 2);  view_134 = None
        getitem_123: "f32[8, 128, 768]" = split_11[0]
        getitem_124: "f32[8, 128, 768]" = split_11[1]
        getitem_125: "f32[8, 128, 768]" = split_11[2];  split_11 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_137: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(getitem_123, [8, 128, -1, 64]);  getitem_123 = None
        permute_46: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_135: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(getitem_124, [8, 128, -1, 64]);  getitem_124 = None
        permute_44: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_135, [0, 2, 1, 3]);  view_135 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_136: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(getitem_125, [8, 128, -1, 64]);  getitem_125 = None
        permute_45: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_11: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(arg0_1, [8, 12, 128, 128]);  arg0_1 = None
        _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_46, permute_44, permute_45, expand_11, False);  permute_46 = permute_44 = permute_45 = expand_11 = None
        getitem_126: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_11[0];  _scaled_dot_product_efficient_attention_11 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_47: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3]);  getitem_126 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_138: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(permute_47, [8, 128, -1]);  permute_47 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_139: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_138, [-1, 768]);  view_138 = None
        mm_default_2: "f32[1024, 768]" = torch.ops.aten.mm.default(view_139, arg141_1);  view_139 = arg141_1 = None
        add_tensor_2: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_2, arg140_1);  mm_default_2 = arg140_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_140: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_2, [8, 128, 768]);  add_tensor_2 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_91: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_140, add_88);  view_140 = add_88 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_23 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
        getitem_130: "f32[8, 128, 1]" = var_mean_23[0]
        getitem_131: "f32[8, 128, 1]" = var_mean_23[1];  var_mean_23 = None
        sub_23: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_91, getitem_131);  getitem_131 = None
        add_92: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
        rsqrt_23: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        mul_90: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = rsqrt_23 = None
        mul_91: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_90, arg142_1);  mul_90 = arg142_1 = None
        add_93: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_91, arg143_1);  mul_91 = arg143_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_141: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_93, [-1, 768]);  add_93 = None
        mm_default_1: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_141, arg145_1);  view_141 = arg145_1 = None
        add_tensor_1: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_1, arg144_1);  mm_default_1 = arg144_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_142: "f32[8, 128, 3072]" = torch.ops.aten.reshape.default(add_tensor_1, [8, 128, 3072]);  add_tensor_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_92: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_142, 0.5)
        pow_12: "f32[8, 128, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_142, 3.0)
        mul_93: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_94: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(view_142, mul_93);  view_142 = mul_93 = None
        mul_94: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(add_94, 0.7978845608028654);  add_94 = None
        tanh_11: "f32[8, 128, 3072]" = torch.ops.aten.tanh.default(mul_94);  mul_94 = None
        add_95: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
        mul_95: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_92, add_95);  mul_92 = add_95 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_143: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_95, [-1, 3072]);  mul_95 = None
        mm_default: "f32[1024, 768]" = torch.ops.aten.mm.default(view_143, arg147_1);  view_143 = arg147_1 = None
        add_tensor: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default, arg146_1);  mm_default = arg146_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_144: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor, [8, 128, 768]);  add_tensor = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_96: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(add_91, view_144);  add_91 = view_144 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:953 in forward, code: hidden_states = self.ln_f(hidden_states)
        var_mean_24 = torch.ops.aten.var_mean.correction(add_96, [2], correction = 0, keepdim = True)
        getitem_132: "f32[8, 128, 1]" = var_mean_24[0]
        getitem_133: "f32[8, 128, 1]" = var_mean_24[1];  var_mean_24 = None
        sub_24: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_96, getitem_133);  add_96 = getitem_133 = None
        add_97: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-05);  getitem_132 = None
        rsqrt_24: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
        mul_96: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = rsqrt_24 = None
        mul_97: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_96, arg148_1);  mul_96 = arg148_1 = None
        add_98: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_97, arg149_1);  mul_97 = arg149_1 = None
        return (add_98,)
        

# ===== inductor generated file: /tmp/cnnbench-transformers-p50cz5fm/repeat_02/a1/torchinductor/tmpxdyb3vm9/wz/cwznbdrhxienrtupkya6knusvpneohiudyrpaybqvypvjpmv4hgq.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_native_layer_norm_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 12592128}}
)
@triton.jit
def triton_per_fused_add_addmm_native_layer_norm_view_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp12 = tl.full([XBLOCK, 1], 768, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = (tmp11 / tmp13)
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(r0_mask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None].to(tl.float32)
    tmp21 = tmp4 - tmp14
    tmp22 = 768.0
    tmp23 = (tmp20 / tmp22)
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp31, r0_mask & xmask)
