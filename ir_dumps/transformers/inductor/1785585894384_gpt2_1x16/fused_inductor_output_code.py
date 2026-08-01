# ===== inductor generated file: /tmp/cnnbench-transformers-ob_ijj07/repeat_03/a1/torchinductor/tmpuqb_59nb/tb/ctbeuecaa5nknwy6bvucluzl476p3mx7abciczu4tjqirt7duxrh.debug/output_code.py =====
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


# kernel path: /tmp/cnnbench-transformers-ob_ijj07/repeat_03/a1/torchinductor/tmpuqb_59nb/lj/cljm2yw4ttzhprxdzyni7ggrjibfnhmatq2ohteunj2nfqlnqaqr.py
# Topologically Sorted Source Nodes: [inputs_embeds, cache_position, position_ids, position_embeds, hidden_states, hidden_states_2], Original ATen: [aten.embedding, aten.arange, aten.unsqueeze, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   cache_position => iota
#   hidden_states => add
#   hidden_states_2 => add_2, add_3, mul, mul_1, rsqrt, sub_2, var_mean
#   inputs_embeds => embedding
#   position_embeds => embedding_1
#   position_ids => unsqueeze
# Graph fragment:
#   %arg0_1 : Tensor "i64[1, 16][16, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[50257, 768][768, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "f32[1024, 768][768, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %getitem_1 : Tensor "f32[1, 16, 1][16, 1, 16]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf1 : Tensor "f32[1, 16, 1][16, 1, 16]cuda:0" = PlaceHolder[target=buf1]
#   %arg3_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %embedding : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %arg0_1), kwargs = {})
#   %iota : Tensor "i64[16][1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze : Tensor "i64[1, 16][16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, 0), kwargs = {})
#   %embedding_1 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %unsqueeze), kwargs = {})
#   %add : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %getitem_1), kwargs = {})
#   %add_2 : Tensor "f32[1, 16, 1][16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[1, 16, 1][16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg3_1), kwargs = {})
#   %add_3 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %arg4_1), kwargs = {})
#   return %getitem_1,%buf1,%add_3
triton_red_fused_add_arange_embedding_native_layer_norm_unsqueeze_0 = async_compile.triton('triton_red_fused_add_arange_embedding_native_layer_norm_unsqueeze_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16, 'r0_': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_arange_embedding_native_layer_norm_unsqueeze_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_arange_embedding_native_layer_norm_unsqueeze_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 16
    r0_numel = 768
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp10_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp10_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp10_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp7 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.full([XBLOCK, R0_BLOCK], 50257, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert(((0 <= tmp4) & (tmp4 < 50257)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 50257")
        tmp6 = tl.load(in_ptr1 + (r0_1 + 768*tmp4), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
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
        r0_1 = r0_index
        tmp22 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp31 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp33 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.full([XBLOCK, R0_BLOCK], 50257, tl.int32)
        tmp17 = tmp0 + tmp16
        tmp18 = tmp0 < 0
        tmp19 = tl.where(tmp18, tmp17, tmp0)
        tl.device_assert(((0 <= tmp19) & (tmp19 < 50257)) | ~(xmask), "index out of bounds: 0 <= tmp19 < 50257")
        tmp21 = tl.load(in_ptr1 + (r0_1 + 768*tmp19), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
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
        tl.store(out_ptr2 + (r0_1 + 768*x0), tmp34, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-transformers-ob_ijj07/repeat_03/a1/torchinductor/tmpuqb_59nb/dx/cdxx2tckrq2rfzdtdjpsecgjjg5rvlcjjvpv5ckasole7tffxdye.py
# Topologically Sorted Source Nodes: [cache_position, position_ids, getitem, first_dummy_value, position_diff, ne, packed_sequence_mask], Original ATen: [aten.arange, aten.unsqueeze, aten.slice, aten.sub, aten.cat, aten.ne, aten.cumsum]
# Source node to ATen node mapping:
#   cache_position => iota
#   first_dummy_value => sub
#   getitem => slice_1
#   ne => ne
#   packed_sequence_mask => cumsum
#   position_diff => cat, slice_2, slice_3, sub_1
#   position_ids => unsqueeze
# Graph fragment:
#   %iota : Tensor "i64[16][1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze : Tensor "i64[1, 16][16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, 0), kwargs = {})
#   %slice_1 : Tensor "i64[1, 1][16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%unsqueeze, 1, 0, 1), kwargs = {})
#   %sub : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_1, 1), kwargs = {})
#   %cat : Tensor "i64[1, 17][17, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%sub, %unsqueeze], -1), kwargs = {})
#   %slice_3 : Tensor "i64[1, 16][17, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat, -1, 1, 17), kwargs = {})
#   %slice_2 : Tensor "i64[1, 16][17, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat, -1, 0, 16), kwargs = {})
#   %sub_1 : Tensor "i64[1, 16][16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_3, %slice_2), kwargs = {})
#   %ne : Tensor "b8[1, 16][16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%sub_1, 1), kwargs = {})
#   %cumsum : Tensor "i64[1, 16][16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.cumsum.default](args = (%ne, -1), kwargs = {})
#   return %cumsum
triton_per_fused_arange_cat_cumsum_ne_slice_sub_unsqueeze_1 = async_compile.triton('triton_per_fused_arange_cat_cumsum_ne_slice_sub_unsqueeze_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def _triton_helper_fn_add0(arg0_0, arg1_0):
    tmp0 = arg0_0 + arg1_0
    return tmp0

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_arange_cat_cumsum_ne_slice_sub_unsqueeze_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 256}}
)
@triton.jit
def triton_per_fused_arange_cat_cumsum_ne_slice_sub_unsqueeze_1(out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 16
    R0_BLOCK: tl.constexpr = 16
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = 1 + r0_0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.broadcast_to((-1) + (1 + r0_0), [XBLOCK, R0_BLOCK])
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 17, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp15 = r0_0
    tmp16 = tmp15 >= tmp1
    tmp17 = tmp15 < tmp3
    tmp18 = tl.broadcast_to((-1) + (r0_0), [XBLOCK, R0_BLOCK])
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp15 >= tmp3
    tmp22 = tmp15 < tmp9
    tmp23 = tl.broadcast_to((-1) + r0_0, [XBLOCK, R0_BLOCK])
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp21, tmp23, tmp24)
    tmp26 = tl.where(tmp17, tmp20, tmp25)
    tmp27 = tmp14 - tmp26
    tmp28 = tmp27 != tmp3
    tmp29 = tmp28.to(tl.int64)
    tmp30 = tmp29.to(tl.int64)
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK, R0_BLOCK])
    tmp32, = tl.associative_scan((tmp31,), 1, _triton_helper_fn_add0)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp32, None)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-transformers-ob_ijj07/repeat_03/a1/torchinductor/tmpuqb_59nb/ja/cjaucbdxwmlrtgyrskzzyoh4pgwb47uid5kjp6f6k3rgfh2zx7km.py
# Topologically Sorted Source Nodes: [cache_position, x_1, split, view_5, query_states_1, view_3, key_states_1, view_4, value_states_1, result_1, kv_arange, kv_arange_1, le, batch_arange, index, eq, index_1, result_2, batched_outputs_2, attn_output, x_9, split_1, view_16, query_states_3, view_14, key_states_3, view_15, value_states_3, attn_output_4, x_17, split_2, view_27, query_states_5, view_25, key_states_5, view_26, value_states_5, attn_output_8, x_25, split_3, view_38, query_states_7, view_36, key_states_7, view_37, value_states_7, attn_output_12, x_33, split_4, view_49, query_states_9, view_47, key_states_9, view_48, value_states_9, attn_output_16], Original ATen: [aten.arange, aten.view, aten.split, aten.transpose, aten.add, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   attn_output => _scaled_dot_product_efficient_attention, expand_1, full_default_1, full_default_2, where
#   attn_output_12 => _scaled_dot_product_efficient_attention_3, expand_4, full_default_7, full_default_8, where_3
#   attn_output_16 => _scaled_dot_product_efficient_attention_4, expand_5, full_default_10, full_default_9, where_4
#   attn_output_4 => _scaled_dot_product_efficient_attention_1, expand_2, full_default_3, full_default_4, where_1
#   attn_output_8 => _scaled_dot_product_efficient_attention_2, expand_3, full_default_5, full_default_6, where_2
#   batch_arange => iota_2
#   batched_outputs_2 => view_7
#   cache_position => iota
#   eq => eq, view_5, view_6
#   index => index, view_3
#   index_1 => index_1, view_4
#   key_states_1 => permute
#   key_states_3 => permute_4
#   key_states_5 => permute_8
#   key_states_7 => permute_12
#   key_states_9 => permute_16
#   kv_arange => iota_1
#   kv_arange_1 => add_1
#   le => le, view_1
#   query_states_1 => permute_2
#   query_states_3 => permute_6
#   query_states_5 => permute_10
#   query_states_7 => permute_14
#   query_states_9 => permute_18
#   result_1 => bitwise_and, full_default
#   result_2 => bitwise_and_1
#   split => split
#   split_1 => split_1
#   split_2 => split_2
#   split_3 => split_3
#   split_4 => split_4
#   value_states_1 => permute_1
#   value_states_3 => permute_5
#   value_states_5 => permute_9
#   value_states_7 => permute_13
#   value_states_9 => permute_17
#   view_14 => view_22
#   view_15 => view_23
#   view_16 => view_24
#   view_25 => view_34
#   view_26 => view_35
#   view_27 => view_36
#   view_3 => view_10
#   view_36 => view_46
#   view_37 => view_47
#   view_38 => view_48
#   view_4 => view_11
#   view_47 => view_58
#   view_48 => view_59
#   view_49 => view_60
#   view_5 => view_12
#   x_1 => view_9
#   x_17 => view_33
#   x_25 => view_45
#   x_33 => view_57
#   x_9 => view_21
# Graph fragment:
#   %cumsum : Tensor "i64[1, 16][16, 1]cuda:0" = PlaceHolder[target=cumsum]
#   %iota : Tensor "i64[16][1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %view_9 : Tensor "f32[1, 16, 2304][36864, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [1, 16, 2304]), kwargs = {})
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_9, 768, 2), kwargs = {})
#   %view_12 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_2, [1, 16, -1, 64]), kwargs = {})
#   %permute_2 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_12, [0, 2, 1, 3]), kwargs = {})
#   %view_10 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_3, [1, 16, -1, 64]), kwargs = {})
#   %permute : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_10, [0, 2, 1, 3]), kwargs = {})
#   %view_11 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_4, [1, 16, -1, 64]), kwargs = {})
#   %permute_1 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_11, [0, 2, 1, 3]), kwargs = {})
#   %full_default : Tensor "b8[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([16, 1], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %iota_1 : Tensor "i64[16][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add_1 : Tensor "i64[16][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota_1, 0), kwargs = {})
#   %view_1 : Tensor "i64[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%iota, [16, 1]), kwargs = {})
#   %le : Tensor "b8[16, 16][16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Tensor](args = (%add_1, %view_1), kwargs = {})
#   %bitwise_and : Tensor "b8[16, 16][16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%full_default, %le), kwargs = {})
#   %iota_2 : Tensor "i64[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %view_3 : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%iota_2, [1, 1]), kwargs = {})
#   %index : Tensor "i64[1, 16][16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%cumsum, [%view_3, %iota]), kwargs = {})
#   %view_5 : Tensor "i64[1, 16, 1][16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%index, [1, 16, 1]), kwargs = {})
#   %view_4 : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%iota_2, [1, 1]), kwargs = {})
#   %index_1 : Tensor "i64[1, 16][16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%cumsum, [%view_4, %add_1]), kwargs = {})
#   %view_6 : Tensor "i64[1, 1, 16][16, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%index_1, [1, 1, 16]), kwargs = {})
#   %eq : Tensor "b8[1, 16, 16][256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%view_5, %view_6), kwargs = {})
#   %bitwise_and_1 : Tensor "b8[1, 16, 16][256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and, %eq), kwargs = {})
#   %view_7 : Tensor "b8[1, 1, 16, 16][256, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bitwise_and_1, [1, 1, 16, 16]), kwargs = {})
#   %full_default_2 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "f32[1, 1, 16, 16][256, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand, %full_default_2, %full_default_1), kwargs = {})
#   %expand_1 : Tensor "f32[1, 12, 16, 16][256, 0, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%where, [1, 12, 16, 16]), kwargs = {})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_2, %permute, %permute_1, %expand_1, False), kwargs = {})
#   %view_21 : Tensor "f32[1, 16, 2304][36864, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_4, [1, 16, 2304]), kwargs = {})
#   %split_1 : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_21, 768, 2), kwargs = {})
#   %view_24 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_13, [1, 16, -1, 64]), kwargs = {})
#   %permute_6 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_24, [0, 2, 1, 3]), kwargs = {})
#   %view_22 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_14, [1, 16, -1, 64]), kwargs = {})
#   %permute_4 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_22, [0, 2, 1, 3]), kwargs = {})
#   %view_23 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_15, [1, 16, -1, 64]), kwargs = {})
#   %permute_5 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_23, [0, 2, 1, 3]), kwargs = {})
#   %full_default_4 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_3 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : Tensor "f32[1, 1, 16, 16][256, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand, %full_default_4, %full_default_3), kwargs = {})
#   %expand_2 : Tensor "f32[1, 12, 16, 16][256, 0, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%where_1, [1, 12, 16, 16]), kwargs = {})
#   %_scaled_dot_product_efficient_attention_1 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_6, %permute_4, %permute_5, %expand_2, False), kwargs = {})
#   %view_33 : Tensor "f32[1, 16, 2304][36864, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_8, [1, 16, 2304]), kwargs = {})
#   %split_2 : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_33, 768, 2), kwargs = {})
#   %view_36 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_24, [1, 16, -1, 64]), kwargs = {})
#   %permute_10 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_36, [0, 2, 1, 3]), kwargs = {})
#   %view_34 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_25, [1, 16, -1, 64]), kwargs = {})
#   %permute_8 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_34, [0, 2, 1, 3]), kwargs = {})
#   %view_35 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_26, [1, 16, -1, 64]), kwargs = {})
#   %permute_9 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_35, [0, 2, 1, 3]), kwargs = {})
#   %full_default_6 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_5 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : Tensor "f32[1, 1, 16, 16][256, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand, %full_default_6, %full_default_5), kwargs = {})
#   %expand_3 : Tensor "f32[1, 12, 16, 16][256, 0, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%where_2, [1, 12, 16, 16]), kwargs = {})
#   %_scaled_dot_product_efficient_attention_2 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_10, %permute_8, %permute_9, %expand_3, False), kwargs = {})
#   %view_45 : Tensor "f32[1, 16, 2304][36864, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_12, [1, 16, 2304]), kwargs = {})
#   %split_3 : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_45, 768, 2), kwargs = {})
#   %view_48 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_35, [1, 16, -1, 64]), kwargs = {})
#   %permute_14 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_48, [0, 2, 1, 3]), kwargs = {})
#   %view_46 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_36, [1, 16, -1, 64]), kwargs = {})
#   %permute_12 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_46, [0, 2, 1, 3]), kwargs = {})
#   %view_47 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_37, [1, 16, -1, 64]), kwargs = {})
#   %permute_13 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_47, [0, 2, 1, 3]), kwargs = {})
#   %full_default_8 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_7 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_3 : Tensor "f32[1, 1, 16, 16][256, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand, %full_default_8, %full_default_7), kwargs = {})
#   %expand_4 : Tensor "f32[1, 12, 16, 16][256, 0, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%where_3, [1, 12, 16, 16]), kwargs = {})
#   %_scaled_dot_product_efficient_attention_3 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_14, %permute_12, %permute_13, %expand_4, False), kwargs = {})
#   %view_57 : Tensor "f32[1, 16, 2304][36864, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_16, [1, 16, 2304]), kwargs = {})
#   %split_4 : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_57, 768, 2), kwargs = {})
#   %view_60 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_46, [1, 16, -1, 64]), kwargs = {})
#   %permute_18 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_60, [0, 2, 1, 3]), kwargs = {})
#   %view_58 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_47, [1, 16, -1, 64]), kwargs = {})
#   %permute_16 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_58, [0, 2, 1, 3]), kwargs = {})
#   %view_59 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_48, [1, 16, -1, 64]), kwargs = {})
#   %permute_17 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_59, [0, 2, 1, 3]), kwargs = {})
#   %full_default_10 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_9 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_4 : Tensor "f32[1, 1, 16, 16][256, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand, %full_default_10, %full_default_9), kwargs = {})
#   %expand_5 : Tensor "f32[1, 12, 16, 16][256, 0, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%where_4, [1, 12, 16, 16]), kwargs = {})
#   %_scaled_dot_product_efficient_attention_4 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_18, %permute_16, %permute_17, %expand_5, False), kwargs = {})
#   return %buf6,%buf26,%buf46,%buf66,%buf86
triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_bitwise_and_eq_expand_index_le_scalar_tensor_split_transpose_view_where_2 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_bitwise_and_eq_expand_index_le_scalar_tensor_split_transpose_view_where_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_bitwise_and_eq_expand_index_le_scalar_tensor_split_transpose_view_where_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 10368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_bitwise_and_eq_expand_index_le_scalar_tensor_split_transpose_view_where_2(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp5 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = x1
    tmp2 = tmp0 <= tmp1
    tmp3 = tl.full([1], True, tl.int1)
    tmp4 = tmp3 & tmp2
    tmp7 = tmp5 == tmp6
    tmp8 = tmp4 & tmp7
    tmp9 = 0.0
    tmp10 = float("-inf")
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tl.store(out_ptr0 + (x2), tmp11, xmask)
    tl.store(out_ptr1 + (x2), tmp11, xmask)
    tl.store(out_ptr2 + (x2), tmp11, xmask)
    tl.store(out_ptr3 + (x2), tmp11, xmask)
    tl.store(out_ptr4 + (x2), tmp11, xmask)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-transformers-ob_ijj07/repeat_03/a1/torchinductor/tmpuqb_59nb/tg/ctgkujgbvk26wf25wgqfcccy3meikawb6tsgv2vykji5s3geakfs.py
# Topologically Sorted Source Nodes: [inputs_embeds, cache_position, position_ids, position_embeds, hidden_states, x_2, x_3, hidden_states_3, hidden_states_4], Original ATen: [aten.embedding, aten.arange, aten.unsqueeze, aten.add, aten.addmm, aten.view, aten.native_layer_norm]
# Source node to ATen node mapping:
#   cache_position => iota
#   hidden_states => add
#   hidden_states_3 => add_4
#   hidden_states_4 => add_5, add_6, mul_2, mul_3, rsqrt_1, sub_3, var_mean_1
#   inputs_embeds => embedding
#   position_embeds => embedding_1
#   position_ids => unsqueeze
#   x_2 => add_tensor_35
#   x_3 => view_15
# Graph fragment:
#   %mm_default_35 : Tensor "f32[16, 768][768, 1]cuda:0" = PlaceHolder[target=mm_default_35]
#   %arg7_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg7_1]
#   %arg0_1 : Tensor "i64[1, 16][16, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[50257, 768][768, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "f32[1024, 768][768, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %add_4 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0" = PlaceHolder[target=add_4]
#   %getitem_10 : Tensor "f32[1, 16, 1][16, 1, 16]cuda:0" = PlaceHolder[target=getitem_10]
#   %buf15 : Tensor "f32[1, 16, 1][16, 1, 16]cuda:0" = PlaceHolder[target=buf15]
#   %arg9_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg9_1]
#   %arg10_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg10_1]
#   %embedding : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %arg0_1), kwargs = {})
#   %iota : Tensor "i64[16][1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze : Tensor "i64[1, 16][16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, 0), kwargs = {})
#   %embedding_1 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %unsqueeze), kwargs = {})
#   %add : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %add_tensor_35 : Tensor "f32[16, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_35, %arg7_1), kwargs = {})
#   %view_15 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_35, [1, 16, 768]), kwargs = {})
#   %add_4 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_15, %add), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_4, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_4, %getitem_10), kwargs = {})
#   %add_5 : Tensor "f32[1, 16, 1][16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_9, 1e-05), kwargs = {})
#   %rsqrt_1 : Tensor "f32[1, 16, 1][16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
#   %mul_2 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_1), kwargs = {})
#   %mul_3 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %arg9_1), kwargs = {})
#   %add_6 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %arg10_1), kwargs = {})
#   return %add_4,%getitem_10,%buf15,%add_6
triton_per_fused_add_addmm_arange_embedding_native_layer_norm_unsqueeze_view_3 = async_compile.triton('triton_per_fused_add_addmm_arange_embedding_native_layer_norm_unsqueeze_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_arange_embedding_native_layer_norm_unsqueeze_view_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 6, 'num_reduction': 4, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_addmm_arange_embedding_native_layer_norm_unsqueeze_view_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16
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
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp36 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tl.full([XBLOCK, R0_BLOCK], 50257, tl.int32)
    tmp5 = tmp3 + tmp4
    tmp6 = tmp3 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp3)
    tl.device_assert(((0 <= tmp7) & (tmp7 < 50257)) | ~(xmask), "index out of bounds: 0 <= tmp7 < 50257")
    tmp9 = tl.load(in_ptr2 + (r0_1 + 768*tmp7), r0_mask & xmask, other=0.0)
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
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp12, r0_mask & xmask)
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp39, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-transformers-ob_ijj07/repeat_03/a1/torchinductor/tmpuqb_59nb/s5/cs5rl3zynqzmsjh2xshqzukk3gtxthgzyqi3b5jc7jul5gcv3uc4.py
# Topologically Sorted Source Nodes: [x_4, x_5, mul, pow_1, mul_1, add_2, mul_2, tanh, add_3, hidden_states_5], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
# Source node to ATen node mapping:
#   add_2 => add_7
#   add_3 => add_8
#   hidden_states_5 => mul_7
#   mul => mul_4
#   mul_1 => mul_5
#   mul_2 => mul_6
#   pow_1 => pow_1
#   tanh => tanh
#   x_4 => add_tensor_34
#   x_5 => view_17
# Graph fragment:
#   %mm_default_34 : Tensor "f32[16, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_default_34]
#   %arg11_1 : Tensor "f32[3072][1]cuda:0" = PlaceHolder[target=arg11_1]
#   %add_tensor_34 : Tensor "f32[16, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_34, %arg11_1), kwargs = {})
#   %view_17 : Tensor "f32[1, 16, 3072][49152, 3072, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_34, [1, 16, 3072]), kwargs = {})
#   %mul_4 : Tensor "f32[1, 16, 3072][49152, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_17, 0.5), kwargs = {})
#   %pow_1 : Tensor "f32[1, 16, 3072][49152, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view_17, 3.0), kwargs = {})
#   %mul_5 : Tensor "f32[1, 16, 3072][49152, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_1, 0.044715), kwargs = {})
#   %add_7 : Tensor "f32[1, 16, 3072][49152, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_17, %mul_5), kwargs = {})
#   %mul_6 : Tensor "f32[1, 16, 3072][49152, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_7, 0.7978845608028654), kwargs = {})
#   %tanh : Tensor "f32[1, 16, 3072][49152, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_6,), kwargs = {})
#   %add_8 : Tensor "f32[1, 16, 3072][49152, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tanh, 1.0), kwargs = {})
#   %mul_7 : Tensor "f32[1, 16, 3072][49152, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %add_8), kwargs = {})
#   return %mul_7
triton_poi_fused_add_addmm_mul_pow_tanh_view_4 = async_compile.triton('triton_poi_fused_add_addmm_mul_pow_tanh_view_4', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_mul_pow_tanh_view_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 602112}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_mul_pow_tanh_view_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
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


# kernel path: /tmp/cnnbench-transformers-ob_ijj07/repeat_03/a1/torchinductor/tmpuqb_59nb/2e/c2eekjtavfj2jegok4t3geyjeblhnw2aljimpfddxzj6eolzgwjx.py
# Topologically Sorted Source Nodes: [x_6, x_7, hidden_states_7, hidden_states_8], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_7 => add_9
#   hidden_states_8 => add_10, add_11, mul_8, mul_9, rsqrt_2, sub_4, var_mean_2
#   x_6 => add_tensor_33
#   x_7 => view_19
# Graph fragment:
#   %add_4 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0" = PlaceHolder[target=add_4]
#   %mm_default_33 : Tensor "f32[16, 768][768, 1]cuda:0" = PlaceHolder[target=mm_default_33]
#   %arg13_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg13_1]
#   %getitem_12 : Tensor "f32[1, 16, 1][16, 1, 16]cuda:0" = PlaceHolder[target=getitem_12]
#   %buf22 : Tensor "f32[1, 16, 1][16, 1, 16]cuda:0" = PlaceHolder[target=buf22]
#   %arg15_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg15_1]
#   %arg16_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg16_1]
#   %add_tensor_33 : Tensor "f32[16, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_33, %arg13_1), kwargs = {})
#   %view_19 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_33, [1, 16, 768]), kwargs = {})
#   %add_9 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %view_19), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_9, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_4 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_9, %getitem_12), kwargs = {})
#   %add_10 : Tensor "f32[1, 16, 1][16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_11, 1e-05), kwargs = {})
#   %rsqrt_2 : Tensor "f32[1, 16, 1][16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_10,), kwargs = {})
#   %mul_8 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_2), kwargs = {})
#   %mul_9 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %arg15_1), kwargs = {})
#   %add_11 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %arg16_1), kwargs = {})
#   return %getitem_12,%buf22,%add_11
triton_per_fused_add_addmm_native_layer_norm_view_5 = async_compile.triton('triton_per_fused_add_addmm_native_layer_norm_view_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_native_layer_norm_view_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 205824}}
)
@triton.jit
def triton_per_fused_add_addmm_native_layer_norm_view_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16
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


# kernel path: /tmp/cnnbench-transformers-ob_ijj07/repeat_03/a1/torchinductor/tmpuqb_59nb/m5/cm5y4qzqfsa7zrgnjswk65q2jceoci2grzwwh22it65glp6s5qcx.py
# Topologically Sorted Source Nodes: [x_6, x_7, hidden_states_7, x_10, x_11, hidden_states_9, hidden_states_10], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_10 => add_13, add_14, mul_10, mul_11, rsqrt_3, sub_5, var_mean_3
#   hidden_states_7 => add_9
#   hidden_states_9 => add_12
#   x_10 => add_tensor_32
#   x_11 => view_27
#   x_6 => add_tensor_33
#   x_7 => view_19
# Graph fragment:
#   %mm_default_32 : Tensor "f32[16, 768][768, 1]cuda:0" = PlaceHolder[target=mm_default_32]
#   %arg19_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg19_1]
#   %add_4 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0" = PlaceHolder[target=add_4]
#   %mm_default_33 : Tensor "f32[16, 768][768, 1]cuda:0" = PlaceHolder[target=mm_default_33]
#   %arg13_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg13_1]
#   %add_12 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0" = PlaceHolder[target=add_12]
#   %getitem_21 : Tensor "f32[1, 16, 1][16, 1, 16]cuda:0" = PlaceHolder[target=getitem_21]
#   %buf35 : Tensor "f32[1, 16, 1][16, 1, 16]cuda:0" = PlaceHolder[target=buf35]
#   %arg21_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg21_1]
#   %arg22_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg22_1]
#   %add_tensor_33 : Tensor "f32[16, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_33, %arg13_1), kwargs = {})
#   %view_19 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_33, [1, 16, 768]), kwargs = {})
#   %add_9 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %view_19), kwargs = {})
#   %add_tensor_32 : Tensor "f32[16, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_32, %arg19_1), kwargs = {})
#   %view_27 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_32, [1, 16, 768]), kwargs = {})
#   %add_12 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_27, %add_9), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_12, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_5 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_12, %getitem_21), kwargs = {})
#   %add_13 : Tensor "f32[1, 16, 1][16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_20, 1e-05), kwargs = {})
#   %rsqrt_3 : Tensor "f32[1, 16, 1][16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_13,), kwargs = {})
#   %mul_10 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_3), kwargs = {})
#   %mul_11 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %arg21_1), kwargs = {})
#   %add_14 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %arg22_1), kwargs = {})
#   return %add_12,%getitem_21,%buf35,%add_14
triton_per_fused_add_addmm_native_layer_norm_view_6 = async_compile.triton('triton_per_fused_add_addmm_native_layer_norm_view_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_native_layer_norm_view_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 356352}}
)
@triton.jit
def triton_per_fused_add_addmm_native_layer_norm_view_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16
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


# kernel path: /tmp/cnnbench-transformers-ob_ijj07/repeat_03/a1/torchinductor/tmpuqb_59nb/n5/cn5drgj55mabns5bc5i7nj2iwlnlly62rbfoy3roikqqnbm2fwkk.py
# Topologically Sorted Source Nodes: [cache_position, result_1, kv_arange, kv_arange_1, le, batch_arange, index, eq, index_1, result_2, batched_outputs_2, x_81, split_10, view_115, query_states_21, view_113, key_states_21, view_114, value_states_21, attn_output_40, x_89, split_11, view_126, query_states_23, view_124, key_states_23, view_125, value_states_23, attn_output_44], Original ATen: [aten.arange, aten.view, aten.add, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.split, aten.transpose, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   attn_output_40 => _scaled_dot_product_efficient_attention_10, expand_11, full_default_21, full_default_22, where_10
#   attn_output_44 => _scaled_dot_product_efficient_attention_11, expand_12, full_default_23, full_default_24, where_11
#   batch_arange => iota_2
#   batched_outputs_2 => view_7
#   cache_position => iota
#   eq => eq, view_5, view_6
#   index => index, view_3
#   index_1 => index_1, view_4
#   key_states_21 => permute_40
#   key_states_23 => permute_44
#   kv_arange => iota_1
#   kv_arange_1 => add_1
#   le => le, view_1
#   query_states_21 => permute_42
#   query_states_23 => permute_46
#   result_1 => bitwise_and, full_default
#   result_2 => bitwise_and_1
#   split_10 => split_10
#   split_11 => split_11
#   value_states_21 => permute_41
#   value_states_23 => permute_45
#   view_113 => view_130
#   view_114 => view_131
#   view_115 => view_132
#   view_124 => view_142
#   view_125 => view_143
#   view_126 => view_144
#   x_81 => view_129
#   x_89 => view_141
# Graph fragment:
#   %cumsum : Tensor "i64[1, 16][16, 1]cuda:0" = PlaceHolder[target=cumsum]
#   %iota : Tensor "i64[16][1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %full_default : Tensor "b8[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([16, 1], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %iota_1 : Tensor "i64[16][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add_1 : Tensor "i64[16][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota_1, 0), kwargs = {})
#   %view_1 : Tensor "i64[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%iota, [16, 1]), kwargs = {})
#   %le : Tensor "b8[16, 16][16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Tensor](args = (%add_1, %view_1), kwargs = {})
#   %bitwise_and : Tensor "b8[16, 16][16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%full_default, %le), kwargs = {})
#   %iota_2 : Tensor "i64[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %view_3 : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%iota_2, [1, 1]), kwargs = {})
#   %index : Tensor "i64[1, 16][16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%cumsum, [%view_3, %iota]), kwargs = {})
#   %view_5 : Tensor "i64[1, 16, 1][16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%index, [1, 16, 1]), kwargs = {})
#   %view_4 : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%iota_2, [1, 1]), kwargs = {})
#   %index_1 : Tensor "i64[1, 16][16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%cumsum, [%view_4, %add_1]), kwargs = {})
#   %view_6 : Tensor "i64[1, 1, 16][16, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%index_1, [1, 1, 16]), kwargs = {})
#   %eq : Tensor "b8[1, 16, 16][256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%view_5, %view_6), kwargs = {})
#   %bitwise_and_1 : Tensor "b8[1, 16, 16][256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and, %eq), kwargs = {})
#   %view_7 : Tensor "b8[1, 1, 16, 16][256, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bitwise_and_1, [1, 1, 16, 16]), kwargs = {})
#   %view_129 : Tensor "f32[1, 16, 2304][36864, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_40, [1, 16, 2304]), kwargs = {})
#   %split_10 : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_129, 768, 2), kwargs = {})
#   %view_132 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_112, [1, 16, -1, 64]), kwargs = {})
#   %permute_42 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_132, [0, 2, 1, 3]), kwargs = {})
#   %view_130 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_113, [1, 16, -1, 64]), kwargs = {})
#   %permute_40 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_130, [0, 2, 1, 3]), kwargs = {})
#   %view_131 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_114, [1, 16, -1, 64]), kwargs = {})
#   %permute_41 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_131, [0, 2, 1, 3]), kwargs = {})
#   %full_default_22 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_21 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_10 : Tensor "f32[1, 1, 16, 16][256, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand, %full_default_22, %full_default_21), kwargs = {})
#   %expand_11 : Tensor "f32[1, 12, 16, 16][256, 0, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%where_10, [1, 12, 16, 16]), kwargs = {})
#   %_scaled_dot_product_efficient_attention_10 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_42, %permute_40, %permute_41, %expand_11, False), kwargs = {})
#   %view_141 : Tensor "f32[1, 16, 2304][36864, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_44, [1, 16, 2304]), kwargs = {})
#   %split_11 : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_141, 768, 2), kwargs = {})
#   %view_144 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_123, [1, 16, -1, 64]), kwargs = {})
#   %permute_46 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_144, [0, 2, 1, 3]), kwargs = {})
#   %view_142 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_124, [1, 16, -1, 64]), kwargs = {})
#   %permute_44 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_142, [0, 2, 1, 3]), kwargs = {})
#   %view_143 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_125, [1, 16, -1, 64]), kwargs = {})
#   %permute_45 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_143, [0, 2, 1, 3]), kwargs = {})
#   %full_default_24 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_23 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_11 : Tensor "f32[1, 1, 16, 16][256, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand, %full_default_24, %full_default_23), kwargs = {})
#   %expand_12 : Tensor "f32[1, 12, 16, 16][256, 0, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%where_11, [1, 12, 16, 16]), kwargs = {})
#   %_scaled_dot_product_efficient_attention_11 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_46, %permute_44, %permute_45, %expand_12, False), kwargs = {})
#   return %buf206,%buf226
triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_bitwise_and_eq_expand_index_le_scalar_tensor_split_transpose_view_where_7 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_bitwise_and_eq_expand_index_le_scalar_tensor_split_transpose_view_where_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_bitwise_and_eq_expand_index_le_scalar_tensor_split_transpose_view_where_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4224}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_bitwise_and_eq_expand_index_le_scalar_tensor_split_transpose_view_where_7(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp5 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = x1
    tmp2 = tmp0 <= tmp1
    tmp3 = tl.full([1], True, tl.int1)
    tmp4 = tmp3 & tmp2
    tmp7 = tmp5 == tmp6
    tmp8 = tmp4 & tmp7
    tmp9 = 0.0
    tmp10 = float("-inf")
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tl.store(out_ptr0 + (x2), tmp11, xmask)
    tl.store(out_ptr1 + (x2), tmp11, xmask)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-transformers-ob_ijj07/repeat_03/a1/torchinductor/tmpuqb_59nb/qx/cqxp4qkuvdyerf6x47rzmpv55xsjkmwal5kymfolzw5gb7gk6fal.py
# Topologically Sorted Source Nodes: [x_94, x_95, hidden_states_73, hidden_states_74], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_73 => add_97
#   hidden_states_74 => add_98, add_99, mul_96, mul_97, rsqrt_24, sub_26, var_mean_24
#   x_94 => add_tensor
#   x_95 => view_151
# Graph fragment:
#   %add_92 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0" = PlaceHolder[target=add_92]
#   %mm_default : Tensor "f32[16, 768][768, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %arg145_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg145_1]
#   %getitem_133 : Tensor "f32[1, 16, 1][16, 1, 16]cuda:0" = PlaceHolder[target=getitem_133]
#   %buf242 : Tensor "f32[1, 16, 1][16, 1, 16]cuda:0" = PlaceHolder[target=buf242]
#   %arg147_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg147_1]
#   %arg148_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg148_1]
#   %add_tensor : Tensor "f32[16, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %arg145_1), kwargs = {})
#   %view_151 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor, [1, 16, 768]), kwargs = {})
#   %add_97 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_92, %view_151), kwargs = {})
#   %var_mean_24 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_97, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_26 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_97, %getitem_133), kwargs = {})
#   %add_98 : Tensor "f32[1, 16, 1][16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_132, 1e-05), kwargs = {})
#   %rsqrt_24 : Tensor "f32[1, 16, 1][16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_98,), kwargs = {})
#   %mul_96 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %rsqrt_24), kwargs = {})
#   %mul_97 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_96, %arg147_1), kwargs = {})
#   %add_99 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_97, %arg148_1), kwargs = {})
#   return %getitem_133,%buf242,%add_99
triton_per_fused_add_addmm_native_layer_norm_view_8 = async_compile.triton('triton_per_fused_add_addmm_native_layer_norm_view_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_native_layer_norm_view_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 205824}}
)
@triton.jit
def triton_per_fused_add_addmm_native_layer_norm_view_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1 = args
        args.clear()
        assert_size_stride(arg0_1, (1, 16), (16, 1))
        assert_size_stride(arg1_1, (50257, 768), (768, 1))
        assert_size_stride(arg2_1, (1024, 768), (768, 1))
        assert_size_stride(arg3_1, (768, ), (1, ))
        assert_size_stride(arg4_1, (768, ), (1, ))
        assert_size_stride(arg5_1, (2304, ), (1, ))
        assert_size_stride(arg6_1, (768, 2304), (2304, 1))
        assert_size_stride(arg7_1, (768, ), (1, ))
        assert_size_stride(arg8_1, (768, 768), (768, 1))
        assert_size_stride(arg9_1, (768, ), (1, ))
        assert_size_stride(arg10_1, (768, ), (1, ))
        assert_size_stride(arg11_1, (3072, ), (1, ))
        assert_size_stride(arg12_1, (768, 3072), (3072, 1))
        assert_size_stride(arg13_1, (768, ), (1, ))
        assert_size_stride(arg14_1, (3072, 768), (768, 1))
        assert_size_stride(arg15_1, (768, ), (1, ))
        assert_size_stride(arg16_1, (768, ), (1, ))
        assert_size_stride(arg17_1, (2304, ), (1, ))
        assert_size_stride(arg18_1, (768, 2304), (2304, 1))
        assert_size_stride(arg19_1, (768, ), (1, ))
        assert_size_stride(arg20_1, (768, 768), (768, 1))
        assert_size_stride(arg21_1, (768, ), (1, ))
        assert_size_stride(arg22_1, (768, ), (1, ))
        assert_size_stride(arg23_1, (3072, ), (1, ))
        assert_size_stride(arg24_1, (768, 3072), (3072, 1))
        assert_size_stride(arg25_1, (768, ), (1, ))
        assert_size_stride(arg26_1, (3072, 768), (768, 1))
        assert_size_stride(arg27_1, (768, ), (1, ))
        assert_size_stride(arg28_1, (768, ), (1, ))
        assert_size_stride(arg29_1, (2304, ), (1, ))
        assert_size_stride(arg30_1, (768, 2304), (2304, 1))
        assert_size_stride(arg31_1, (768, ), (1, ))
        assert_size_stride(arg32_1, (768, 768), (768, 1))
        assert_size_stride(arg33_1, (768, ), (1, ))
        assert_size_stride(arg34_1, (768, ), (1, ))
        assert_size_stride(arg35_1, (3072, ), (1, ))
        assert_size_stride(arg36_1, (768, 3072), (3072, 1))
        assert_size_stride(arg37_1, (768, ), (1, ))
        assert_size_stride(arg38_1, (3072, 768), (768, 1))
        assert_size_stride(arg39_1, (768, ), (1, ))
        assert_size_stride(arg40_1, (768, ), (1, ))
        assert_size_stride(arg41_1, (2304, ), (1, ))
        assert_size_stride(arg42_1, (768, 2304), (2304, 1))
        assert_size_stride(arg43_1, (768, ), (1, ))
        assert_size_stride(arg44_1, (768, 768), (768, 1))
        assert_size_stride(arg45_1, (768, ), (1, ))
        assert_size_stride(arg46_1, (768, ), (1, ))
        assert_size_stride(arg47_1, (3072, ), (1, ))
        assert_size_stride(arg48_1, (768, 3072), (3072, 1))
        assert_size_stride(arg49_1, (768, ), (1, ))
        assert_size_stride(arg50_1, (3072, 768), (768, 1))
        assert_size_stride(arg51_1, (768, ), (1, ))
        assert_size_stride(arg52_1, (768, ), (1, ))
        assert_size_stride(arg53_1, (2304, ), (1, ))
        assert_size_stride(arg54_1, (768, 2304), (2304, 1))
        assert_size_stride(arg55_1, (768, ), (1, ))
        assert_size_stride(arg56_1, (768, 768), (768, 1))
        assert_size_stride(arg57_1, (768, ), (1, ))
        assert_size_stride(arg58_1, (768, ), (1, ))
        assert_size_stride(arg59_1, (3072, ), (1, ))
        assert_size_stride(arg60_1, (768, 3072), (3072, 1))
        assert_size_stride(arg61_1, (768, ), (1, ))
        assert_size_stride(arg62_1, (3072, 768), (768, 1))
        assert_size_stride(arg63_1, (768, ), (1, ))
        assert_size_stride(arg64_1, (768, ), (1, ))
        assert_size_stride(arg65_1, (2304, ), (1, ))
        assert_size_stride(arg66_1, (768, 2304), (2304, 1))
        assert_size_stride(arg67_1, (768, ), (1, ))
        assert_size_stride(arg68_1, (768, 768), (768, 1))
        assert_size_stride(arg69_1, (768, ), (1, ))
        assert_size_stride(arg70_1, (768, ), (1, ))
        assert_size_stride(arg71_1, (3072, ), (1, ))
        assert_size_stride(arg72_1, (768, 3072), (3072, 1))
        assert_size_stride(arg73_1, (768, ), (1, ))
        assert_size_stride(arg74_1, (3072, 768), (768, 1))
        assert_size_stride(arg75_1, (768, ), (1, ))
        assert_size_stride(arg76_1, (768, ), (1, ))
        assert_size_stride(arg77_1, (2304, ), (1, ))
        assert_size_stride(arg78_1, (768, 2304), (2304, 1))
        assert_size_stride(arg79_1, (768, ), (1, ))
        assert_size_stride(arg80_1, (768, 768), (768, 1))
        assert_size_stride(arg81_1, (768, ), (1, ))
        assert_size_stride(arg82_1, (768, ), (1, ))
        assert_size_stride(arg83_1, (3072, ), (1, ))
        assert_size_stride(arg84_1, (768, 3072), (3072, 1))
        assert_size_stride(arg85_1, (768, ), (1, ))
        assert_size_stride(arg86_1, (3072, 768), (768, 1))
        assert_size_stride(arg87_1, (768, ), (1, ))
        assert_size_stride(arg88_1, (768, ), (1, ))
        assert_size_stride(arg89_1, (2304, ), (1, ))
        assert_size_stride(arg90_1, (768, 2304), (2304, 1))
        assert_size_stride(arg91_1, (768, ), (1, ))
        assert_size_stride(arg92_1, (768, 768), (768, 1))
        assert_size_stride(arg93_1, (768, ), (1, ))
        assert_size_stride(arg94_1, (768, ), (1, ))
        assert_size_stride(arg95_1, (3072, ), (1, ))
        assert_size_stride(arg96_1, (768, 3072), (3072, 1))
        assert_size_stride(arg97_1, (768, ), (1, ))
        assert_size_stride(arg98_1, (3072, 768), (768, 1))
        assert_size_stride(arg99_1, (768, ), (1, ))
        assert_size_stride(arg100_1, (768, ), (1, ))
        assert_size_stride(arg101_1, (2304, ), (1, ))
        assert_size_stride(arg102_1, (768, 2304), (2304, 1))
        assert_size_stride(arg103_1, (768, ), (1, ))
        assert_size_stride(arg104_1, (768, 768), (768, 1))
        assert_size_stride(arg105_1, (768, ), (1, ))
        assert_size_stride(arg106_1, (768, ), (1, ))
        assert_size_stride(arg107_1, (3072, ), (1, ))
        assert_size_stride(arg108_1, (768, 3072), (3072, 1))
        assert_size_stride(arg109_1, (768, ), (1, ))
        assert_size_stride(arg110_1, (3072, 768), (768, 1))
        assert_size_stride(arg111_1, (768, ), (1, ))
        assert_size_stride(arg112_1, (768, ), (1, ))
        assert_size_stride(arg113_1, (2304, ), (1, ))
        assert_size_stride(arg114_1, (768, 2304), (2304, 1))
        assert_size_stride(arg115_1, (768, ), (1, ))
        assert_size_stride(arg116_1, (768, 768), (768, 1))
        assert_size_stride(arg117_1, (768, ), (1, ))
        assert_size_stride(arg118_1, (768, ), (1, ))
        assert_size_stride(arg119_1, (3072, ), (1, ))
        assert_size_stride(arg120_1, (768, 3072), (3072, 1))
        assert_size_stride(arg121_1, (768, ), (1, ))
        assert_size_stride(arg122_1, (3072, 768), (768, 1))
        assert_size_stride(arg123_1, (768, ), (1, ))
        assert_size_stride(arg124_1, (768, ), (1, ))
        assert_size_stride(arg125_1, (2304, ), (1, ))
        assert_size_stride(arg126_1, (768, 2304), (2304, 1))
        assert_size_stride(arg127_1, (768, ), (1, ))
        assert_size_stride(arg128_1, (768, 768), (768, 1))
        assert_size_stride(arg129_1, (768, ), (1, ))
        assert_size_stride(arg130_1, (768, ), (1, ))
        assert_size_stride(arg131_1, (3072, ), (1, ))
        assert_size_stride(arg132_1, (768, 3072), (3072, 1))
        assert_size_stride(arg133_1, (768, ), (1, ))
        assert_size_stride(arg134_1, (3072, 768), (768, 1))
        assert_size_stride(arg135_1, (768, ), (1, ))
        assert_size_stride(arg136_1, (768, ), (1, ))
        assert_size_stride(arg137_1, (2304, ), (1, ))
        assert_size_stride(arg138_1, (768, 2304), (2304, 1))
        assert_size_stride(arg139_1, (768, ), (1, ))
        assert_size_stride(arg140_1, (768, 768), (768, 1))
        assert_size_stride(arg141_1, (768, ), (1, ))
        assert_size_stride(arg142_1, (768, ), (1, ))
        assert_size_stride(arg143_1, (3072, ), (1, ))
        assert_size_stride(arg144_1, (768, 3072), (3072, 1))
        assert_size_stride(arg145_1, (768, ), (1, ))
        assert_size_stride(arg146_1, (3072, 768), (768, 1))
        assert_size_stride(arg147_1, (768, ), (1, ))
        assert_size_stride(arg148_1, (768, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf3 = empty_strided_cuda((1, 16, 768), (12288, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [inputs_embeds, cache_position, position_ids, position_embeds, hidden_states, hidden_states_2], Original ATen: [aten.embedding, aten.arange, aten.unsqueeze, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_arange_embedding_native_layer_norm_unsqueeze_0.run(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, buf3, 16, 768, stream=stream0)
            del arg3_1
            del arg4_1
            buf4 = empty_strided_cuda((16, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [inputs_embeds, cache_position, position_ids, position_embeds, hidden_states, hidden_states_2, view_1, x], Original ATen: [aten.embedding, aten.arange, aten.unsqueeze, aten.add, aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.addmm(arg5_1, reinterpret_tensor(buf3, (16, 768), (768, 1), 0), arg6_1, alpha=1, beta=1, out=buf4)
            del arg5_1
            del arg6_1
            buf5 = empty_strided_cuda((1, 16), (16, 1), torch.int64)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem, first_dummy_value, position_diff, ne, packed_sequence_mask], Original ATen: [aten.arange, aten.unsqueeze, aten.slice, aten.sub, aten.cat, aten.ne, aten.cumsum]
            stream0 = get_raw_stream(0)
            triton_per_fused_arange_cat_cumsum_ne_slice_sub_unsqueeze_1.run(buf5, 1, 16, stream=stream0)
            buf6 = empty_strided_cuda((1, 1, 16, 16), (256, 0, 16, 1), torch.float32)
            buf26 = empty_strided_cuda((1, 1, 16, 16), (256, 0, 16, 1), torch.float32)
            buf46 = empty_strided_cuda((1, 1, 16, 16), (256, 0, 16, 1), torch.float32)
            buf66 = empty_strided_cuda((1, 1, 16, 16), (256, 0, 16, 1), torch.float32)
            buf86 = empty_strided_cuda((1, 1, 16, 16), (256, 0, 16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [cache_position, x_1, split, view_5, query_states_1, view_3, key_states_1, view_4, value_states_1, result_1, kv_arange, kv_arange_1, le, batch_arange, index, eq, index_1, result_2, batched_outputs_2, attn_output, x_9, split_1, view_16, query_states_3, view_14, key_states_3, view_15, value_states_3, attn_output_4, x_17, split_2, view_27, query_states_5, view_25, key_states_5, view_26, value_states_5, attn_output_8, x_25, split_3, view_38, query_states_7, view_36, key_states_7, view_37, value_states_7, attn_output_12, x_33, split_4, view_49, query_states_9, view_47, key_states_9, view_48, value_states_9, attn_output_16], Original ATen: [aten.arange, aten.view, aten.split, aten.transpose, aten.add, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_bitwise_and_eq_expand_index_le_scalar_tensor_split_transpose_view_where_2.run(buf5, buf6, buf26, buf46, buf66, buf86, 256, stream=stream0)
            # Topologically Sorted Source Nodes: [cache_position, x_1, split, view_5, query_states_1, view_3, key_states_1, view_4, value_states_1, result_1, kv_arange, kv_arange_1, le, batch_arange, index, eq, index_1, result_2, batched_outputs_2, attn_output], Original ATen: [aten.arange, aten.view, aten.split, aten.transpose, aten.add, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf4, (1, 12, 16, 64), (0, 64, 2304, 1), 0), reinterpret_tensor(buf4, (1, 12, 16, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf4, (1, 12, 16, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf6, (1, 12, 16, 16), (256, 0, 16, 1), 0), False)
            del buf4
            del buf6
            buf8 = buf7[0]
            assert_size_stride(buf8, (1, 12, 16, 64), (12288, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf8, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf7
            buf12 = reinterpret_tensor(buf3, (16, 768), (768, 1), 0); del buf3  # reuse
            # Topologically Sorted Source Nodes: [transpose_3, reshape, view_6, x_2], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf8, (16, 768), (768, 1), 0), arg8_1, out=buf12)
            del arg8_1
            buf13 = reinterpret_tensor(buf12, (1, 16, 768), (12288, 768, 1), 0); del buf12  # reuse
            buf17 = reinterpret_tensor(buf8, (1, 16, 768), (12288, 768, 1), 0); del buf8  # reuse
            # Topologically Sorted Source Nodes: [inputs_embeds, cache_position, position_ids, position_embeds, hidden_states, x_2, x_3, hidden_states_3, hidden_states_4], Original ATen: [aten.embedding, aten.arange, aten.unsqueeze, aten.add, aten.addmm, aten.view, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_arange_embedding_native_layer_norm_unsqueeze_view_3.run(buf13, arg7_1, arg0_1, arg1_1, arg2_1, arg9_1, arg10_1, buf17, 16, 768, stream=stream0)
            del arg0_1
            del arg10_1
            del arg1_1
            del arg2_1
            del arg7_1
            del arg9_1
            buf18 = empty_strided_cuda((16, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_4, view_8, x_4], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf17, (16, 768), (768, 1), 0), arg12_1, out=buf18)
            del arg12_1
            del buf17
            buf19 = reinterpret_tensor(buf18, (1, 16, 3072), (49152, 3072, 1), 0); del buf18  # reuse
            # Topologically Sorted Source Nodes: [x_4, x_5, mul, pow_1, mul_1, add_2, mul_2, tanh, add_3, hidden_states_5], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_4.run(buf19, arg11_1, 49152, stream=stream0)
            del arg11_1
            buf20 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_4, x_5, mul, pow_1, mul_1, add_2, mul_2, tanh, add_3, hidden_states_5, view_10, x_6], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf19, (16, 3072), (3072, 1), 0), arg14_1, out=buf20)
            del arg14_1
            del buf19
            buf24 = empty_strided_cuda((1, 16, 768), (12288, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_6, x_7, hidden_states_7, hidden_states_8], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_5.run(buf13, buf20, arg13_1, arg15_1, arg16_1, buf24, 16, 768, stream=stream0)
            del arg15_1
            del arg16_1
            buf25 = empty_strided_cuda((16, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_6, x_7, hidden_states_7, hidden_states_8, view_12, x_8], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg17_1, reinterpret_tensor(buf24, (16, 768), (768, 1), 0), arg18_1, alpha=1, beta=1, out=buf25)
            del arg17_1
            del arg18_1
            # Topologically Sorted Source Nodes: [cache_position, result_1, kv_arange, kv_arange_1, le, batch_arange, index, eq, index_1, result_2, batched_outputs_2, x_9, split_1, view_16, query_states_3, view_14, key_states_3, view_15, value_states_3, attn_output_4], Original ATen: [aten.arange, aten.view, aten.add, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.split, aten.transpose, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf27 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf25, (1, 12, 16, 64), (0, 64, 2304, 1), 0), reinterpret_tensor(buf25, (1, 12, 16, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf25, (1, 12, 16, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf26, (1, 12, 16, 16), (256, 0, 16, 1), 0), False)
            del buf25
            buf28 = buf27[0]
            assert_size_stride(buf28, (1, 12, 16, 64), (12288, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf28, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf27
            buf32 = reinterpret_tensor(buf24, (16, 768), (768, 1), 0); del buf24  # reuse
            # Topologically Sorted Source Nodes: [transpose_7, reshape_1, view_17, x_10], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf28, (16, 768), (768, 1), 0), arg20_1, out=buf32)
            del arg20_1
            buf33 = reinterpret_tensor(buf32, (1, 16, 768), (12288, 768, 1), 0); del buf32  # reuse
            buf37 = reinterpret_tensor(buf28, (1, 16, 768), (12288, 768, 1), 0); del buf28  # reuse
            # Topologically Sorted Source Nodes: [x_6, x_7, hidden_states_7, x_10, x_11, hidden_states_9, hidden_states_10], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_6.run(buf33, arg19_1, buf13, buf20, arg13_1, arg21_1, arg22_1, buf37, 16, 768, stream=stream0)
            del arg13_1
            del arg19_1
            del arg21_1
            del arg22_1
            del buf13
            del buf20
            buf38 = empty_strided_cuda((16, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_10, view_19, x_12], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf37, (16, 768), (768, 1), 0), arg24_1, out=buf38)
            del arg24_1
            del buf37
            buf39 = reinterpret_tensor(buf38, (1, 16, 3072), (49152, 3072, 1), 0); del buf38  # reuse
            # Topologically Sorted Source Nodes: [x_12, x_13, mul_4, pow_2, mul_5, add_6, mul_6, tanh_1, add_7, hidden_states_11], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_4.run(buf39, arg23_1, 49152, stream=stream0)
            del arg23_1
            buf40 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_12, x_13, mul_4, pow_2, mul_5, add_6, mul_6, tanh_1, add_7, hidden_states_11, view_21, x_14], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf39, (16, 3072), (3072, 1), 0), arg26_1, out=buf40)
            del arg26_1
            del buf39
            buf44 = empty_strided_cuda((1, 16, 768), (12288, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_14, x_15, hidden_states_13, hidden_states_14], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_5.run(buf33, buf40, arg25_1, arg27_1, arg28_1, buf44, 16, 768, stream=stream0)
            del arg27_1
            del arg28_1
            buf45 = empty_strided_cuda((16, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_14, x_15, hidden_states_13, hidden_states_14, view_23, x_16], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg29_1, reinterpret_tensor(buf44, (16, 768), (768, 1), 0), arg30_1, alpha=1, beta=1, out=buf45)
            del arg29_1
            del arg30_1
            # Topologically Sorted Source Nodes: [cache_position, result_1, kv_arange, kv_arange_1, le, batch_arange, index, eq, index_1, result_2, batched_outputs_2, x_17, split_2, view_27, query_states_5, view_25, key_states_5, view_26, value_states_5, attn_output_8], Original ATen: [aten.arange, aten.view, aten.add, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.split, aten.transpose, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf47 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf45, (1, 12, 16, 64), (0, 64, 2304, 1), 0), reinterpret_tensor(buf45, (1, 12, 16, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf45, (1, 12, 16, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf46, (1, 12, 16, 16), (256, 0, 16, 1), 0), False)
            del buf45
            buf48 = buf47[0]
            assert_size_stride(buf48, (1, 12, 16, 64), (12288, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf48, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf47
            buf52 = reinterpret_tensor(buf44, (16, 768), (768, 1), 0); del buf44  # reuse
            # Topologically Sorted Source Nodes: [transpose_11, reshape_2, view_28, x_18], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf48, (16, 768), (768, 1), 0), arg32_1, out=buf52)
            del arg32_1
            buf53 = reinterpret_tensor(buf52, (1, 16, 768), (12288, 768, 1), 0); del buf52  # reuse
            buf57 = reinterpret_tensor(buf48, (1, 16, 768), (12288, 768, 1), 0); del buf48  # reuse
            # Topologically Sorted Source Nodes: [x_14, x_15, hidden_states_13, x_18, x_19, hidden_states_15, hidden_states_16], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_6.run(buf53, arg31_1, buf33, buf40, arg25_1, arg33_1, arg34_1, buf57, 16, 768, stream=stream0)
            del arg25_1
            del arg31_1
            del arg33_1
            del arg34_1
            del buf33
            del buf40
            buf58 = empty_strided_cuda((16, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_16, view_30, x_20], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf57, (16, 768), (768, 1), 0), arg36_1, out=buf58)
            del arg36_1
            del buf57
            buf59 = reinterpret_tensor(buf58, (1, 16, 3072), (49152, 3072, 1), 0); del buf58  # reuse
            # Topologically Sorted Source Nodes: [x_20, x_21, mul_8, pow_3, mul_9, add_10, mul_10, tanh_2, add_11, hidden_states_17], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_4.run(buf59, arg35_1, 49152, stream=stream0)
            del arg35_1
            buf60 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_20, x_21, mul_8, pow_3, mul_9, add_10, mul_10, tanh_2, add_11, hidden_states_17, view_32, x_22], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf59, (16, 3072), (3072, 1), 0), arg38_1, out=buf60)
            del arg38_1
            del buf59
            buf64 = empty_strided_cuda((1, 16, 768), (12288, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_22, x_23, hidden_states_19, hidden_states_20], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_5.run(buf53, buf60, arg37_1, arg39_1, arg40_1, buf64, 16, 768, stream=stream0)
            del arg39_1
            del arg40_1
            buf65 = empty_strided_cuda((16, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_22, x_23, hidden_states_19, hidden_states_20, view_34, x_24], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg41_1, reinterpret_tensor(buf64, (16, 768), (768, 1), 0), arg42_1, alpha=1, beta=1, out=buf65)
            del arg41_1
            del arg42_1
            # Topologically Sorted Source Nodes: [cache_position, result_1, kv_arange, kv_arange_1, le, batch_arange, index, eq, index_1, result_2, batched_outputs_2, x_25, split_3, view_38, query_states_7, view_36, key_states_7, view_37, value_states_7, attn_output_12], Original ATen: [aten.arange, aten.view, aten.add, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.split, aten.transpose, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf67 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf65, (1, 12, 16, 64), (0, 64, 2304, 1), 0), reinterpret_tensor(buf65, (1, 12, 16, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf65, (1, 12, 16, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf66, (1, 12, 16, 16), (256, 0, 16, 1), 0), False)
            del buf65
            buf68 = buf67[0]
            assert_size_stride(buf68, (1, 12, 16, 64), (12288, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf68, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf67
            buf72 = reinterpret_tensor(buf64, (16, 768), (768, 1), 0); del buf64  # reuse
            # Topologically Sorted Source Nodes: [transpose_15, reshape_3, view_39, x_26], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf68, (16, 768), (768, 1), 0), arg44_1, out=buf72)
            del arg44_1
            buf73 = reinterpret_tensor(buf72, (1, 16, 768), (12288, 768, 1), 0); del buf72  # reuse
            buf77 = reinterpret_tensor(buf68, (1, 16, 768), (12288, 768, 1), 0); del buf68  # reuse
            # Topologically Sorted Source Nodes: [x_22, x_23, hidden_states_19, x_26, x_27, hidden_states_21, hidden_states_22], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_6.run(buf73, arg43_1, buf53, buf60, arg37_1, arg45_1, arg46_1, buf77, 16, 768, stream=stream0)
            del arg37_1
            del arg43_1
            del arg45_1
            del arg46_1
            del buf53
            del buf60
            buf78 = empty_strided_cuda((16, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_22, view_41, x_28], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf77, (16, 768), (768, 1), 0), arg48_1, out=buf78)
            del arg48_1
            del buf77
            buf79 = reinterpret_tensor(buf78, (1, 16, 3072), (49152, 3072, 1), 0); del buf78  # reuse
            # Topologically Sorted Source Nodes: [x_28, x_29, mul_12, pow_4, mul_13, add_14, mul_14, tanh_3, add_15, hidden_states_23], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_4.run(buf79, arg47_1, 49152, stream=stream0)
            del arg47_1
            buf80 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_28, x_29, mul_12, pow_4, mul_13, add_14, mul_14, tanh_3, add_15, hidden_states_23, view_43, x_30], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf79, (16, 3072), (3072, 1), 0), arg50_1, out=buf80)
            del arg50_1
            del buf79
            buf84 = empty_strided_cuda((1, 16, 768), (12288, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_30, x_31, hidden_states_25, hidden_states_26], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_5.run(buf73, buf80, arg49_1, arg51_1, arg52_1, buf84, 16, 768, stream=stream0)
            del arg51_1
            del arg52_1
            buf85 = empty_strided_cuda((16, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_30, x_31, hidden_states_25, hidden_states_26, view_45, x_32], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg53_1, reinterpret_tensor(buf84, (16, 768), (768, 1), 0), arg54_1, alpha=1, beta=1, out=buf85)
            del arg53_1
            del arg54_1
            # Topologically Sorted Source Nodes: [cache_position, result_1, kv_arange, kv_arange_1, le, batch_arange, index, eq, index_1, result_2, batched_outputs_2, x_33, split_4, view_49, query_states_9, view_47, key_states_9, view_48, value_states_9, attn_output_16], Original ATen: [aten.arange, aten.view, aten.add, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.split, aten.transpose, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf87 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf85, (1, 12, 16, 64), (0, 64, 2304, 1), 0), reinterpret_tensor(buf85, (1, 12, 16, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf85, (1, 12, 16, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf86, (1, 12, 16, 16), (256, 0, 16, 1), 0), False)
            del buf85
            buf88 = buf87[0]
            assert_size_stride(buf88, (1, 12, 16, 64), (12288, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf88, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf87
            buf92 = reinterpret_tensor(buf84, (16, 768), (768, 1), 0); del buf84  # reuse
            # Topologically Sorted Source Nodes: [transpose_19, reshape_4, view_50, x_34], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf88, (16, 768), (768, 1), 0), arg56_1, out=buf92)
            del arg56_1
            buf93 = reinterpret_tensor(buf92, (1, 16, 768), (12288, 768, 1), 0); del buf92  # reuse
            buf97 = reinterpret_tensor(buf88, (1, 16, 768), (12288, 768, 1), 0); del buf88  # reuse
            # Topologically Sorted Source Nodes: [x_30, x_31, hidden_states_25, x_34, x_35, hidden_states_27, hidden_states_28], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_6.run(buf93, arg55_1, buf73, buf80, arg49_1, arg57_1, arg58_1, buf97, 16, 768, stream=stream0)
            del arg49_1
            del arg55_1
            del arg57_1
            del arg58_1
            del buf73
            del buf80
            buf98 = empty_strided_cuda((16, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_28, view_52, x_36], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf97, (16, 768), (768, 1), 0), arg60_1, out=buf98)
            del arg60_1
            del buf97
            buf99 = reinterpret_tensor(buf98, (1, 16, 3072), (49152, 3072, 1), 0); del buf98  # reuse
            # Topologically Sorted Source Nodes: [x_36, x_37, mul_16, pow_5, mul_17, add_18, mul_18, tanh_4, add_19, hidden_states_29], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_4.run(buf99, arg59_1, 49152, stream=stream0)
            del arg59_1
            buf100 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_36, x_37, mul_16, pow_5, mul_17, add_18, mul_18, tanh_4, add_19, hidden_states_29, view_54, x_38], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf99, (16, 3072), (3072, 1), 0), arg62_1, out=buf100)
            del arg62_1
            del buf99
            buf104 = empty_strided_cuda((1, 16, 768), (12288, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_38, x_39, hidden_states_31, hidden_states_32], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_5.run(buf93, buf100, arg61_1, arg63_1, arg64_1, buf104, 16, 768, stream=stream0)
            del arg63_1
            del arg64_1
            buf105 = empty_strided_cuda((16, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_38, x_39, hidden_states_31, hidden_states_32, view_56, x_40], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg65_1, reinterpret_tensor(buf104, (16, 768), (768, 1), 0), arg66_1, alpha=1, beta=1, out=buf105)
            del arg65_1
            del arg66_1
            buf106 = buf86; del buf86  # reuse
            buf126 = buf66; del buf66  # reuse
            buf146 = buf46; del buf46  # reuse
            buf166 = buf26; del buf26  # reuse
            buf186 = empty_strided_cuda((1, 1, 16, 16), (256, 0, 16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [cache_position, result_1, kv_arange, kv_arange_1, le, batch_arange, index, eq, index_1, result_2, batched_outputs_2, x_41, split_5, view_60, query_states_11, view_58, key_states_11, view_59, value_states_11, attn_output_20, x_49, split_6, view_71, query_states_13, view_69, key_states_13, view_70, value_states_13, attn_output_24, x_57, split_7, view_82, query_states_15, view_80, key_states_15, view_81, value_states_15, attn_output_28, x_65, split_8, view_93, query_states_17, view_91, key_states_17, view_92, value_states_17, attn_output_32, x_73, split_9, view_104, query_states_19, view_102, key_states_19, view_103, value_states_19, attn_output_36], Original ATen: [aten.arange, aten.view, aten.add, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.split, aten.transpose, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_bitwise_and_eq_expand_index_le_scalar_tensor_split_transpose_view_where_2.run(buf5, buf106, buf126, buf146, buf166, buf186, 256, stream=stream0)
            # Topologically Sorted Source Nodes: [cache_position, result_1, kv_arange, kv_arange_1, le, batch_arange, index, eq, index_1, result_2, batched_outputs_2, x_41, split_5, view_60, query_states_11, view_58, key_states_11, view_59, value_states_11, attn_output_20], Original ATen: [aten.arange, aten.view, aten.add, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.split, aten.transpose, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf107 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf105, (1, 12, 16, 64), (0, 64, 2304, 1), 0), reinterpret_tensor(buf105, (1, 12, 16, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf105, (1, 12, 16, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf106, (1, 12, 16, 16), (256, 0, 16, 1), 0), False)
            del buf105
            del buf106
            buf108 = buf107[0]
            assert_size_stride(buf108, (1, 12, 16, 64), (12288, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf108, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf107
            buf112 = reinterpret_tensor(buf104, (16, 768), (768, 1), 0); del buf104  # reuse
            # Topologically Sorted Source Nodes: [transpose_23, reshape_5, view_61, x_42], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf108, (16, 768), (768, 1), 0), arg68_1, out=buf112)
            del arg68_1
            buf113 = reinterpret_tensor(buf112, (1, 16, 768), (12288, 768, 1), 0); del buf112  # reuse
            buf117 = reinterpret_tensor(buf108, (1, 16, 768), (12288, 768, 1), 0); del buf108  # reuse
            # Topologically Sorted Source Nodes: [x_38, x_39, hidden_states_31, x_42, x_43, hidden_states_33, hidden_states_34], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_6.run(buf113, arg67_1, buf93, buf100, arg61_1, arg69_1, arg70_1, buf117, 16, 768, stream=stream0)
            del arg61_1
            del arg67_1
            del arg69_1
            del arg70_1
            del buf100
            del buf93
            buf118 = empty_strided_cuda((16, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_34, view_63, x_44], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf117, (16, 768), (768, 1), 0), arg72_1, out=buf118)
            del arg72_1
            del buf117
            buf119 = reinterpret_tensor(buf118, (1, 16, 3072), (49152, 3072, 1), 0); del buf118  # reuse
            # Topologically Sorted Source Nodes: [x_44, x_45, mul_20, pow_6, mul_21, add_22, mul_22, tanh_5, add_23, hidden_states_35], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_4.run(buf119, arg71_1, 49152, stream=stream0)
            del arg71_1
            buf120 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_44, x_45, mul_20, pow_6, mul_21, add_22, mul_22, tanh_5, add_23, hidden_states_35, view_65, x_46], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf119, (16, 3072), (3072, 1), 0), arg74_1, out=buf120)
            del arg74_1
            del buf119
            buf124 = empty_strided_cuda((1, 16, 768), (12288, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_46, x_47, hidden_states_37, hidden_states_38], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_5.run(buf113, buf120, arg73_1, arg75_1, arg76_1, buf124, 16, 768, stream=stream0)
            del arg75_1
            del arg76_1
            buf125 = empty_strided_cuda((16, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_46, x_47, hidden_states_37, hidden_states_38, view_67, x_48], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg77_1, reinterpret_tensor(buf124, (16, 768), (768, 1), 0), arg78_1, alpha=1, beta=1, out=buf125)
            del arg77_1
            del arg78_1
            # Topologically Sorted Source Nodes: [cache_position, result_1, kv_arange, kv_arange_1, le, batch_arange, index, eq, index_1, result_2, batched_outputs_2, x_49, split_6, view_71, query_states_13, view_69, key_states_13, view_70, value_states_13, attn_output_24], Original ATen: [aten.arange, aten.view, aten.add, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.split, aten.transpose, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf127 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf125, (1, 12, 16, 64), (0, 64, 2304, 1), 0), reinterpret_tensor(buf125, (1, 12, 16, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf125, (1, 12, 16, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf126, (1, 12, 16, 16), (256, 0, 16, 1), 0), False)
            del buf125
            del buf126
            buf128 = buf127[0]
            assert_size_stride(buf128, (1, 12, 16, 64), (12288, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf128, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf127
            buf132 = reinterpret_tensor(buf124, (16, 768), (768, 1), 0); del buf124  # reuse
            # Topologically Sorted Source Nodes: [transpose_27, reshape_6, view_72, x_50], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf128, (16, 768), (768, 1), 0), arg80_1, out=buf132)
            del arg80_1
            buf133 = reinterpret_tensor(buf132, (1, 16, 768), (12288, 768, 1), 0); del buf132  # reuse
            buf137 = reinterpret_tensor(buf128, (1, 16, 768), (12288, 768, 1), 0); del buf128  # reuse
            # Topologically Sorted Source Nodes: [x_46, x_47, hidden_states_37, x_50, x_51, hidden_states_39, hidden_states_40], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_6.run(buf133, arg79_1, buf113, buf120, arg73_1, arg81_1, arg82_1, buf137, 16, 768, stream=stream0)
            del arg73_1
            del arg79_1
            del arg81_1
            del arg82_1
            del buf113
            del buf120
            buf138 = empty_strided_cuda((16, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_40, view_74, x_52], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf137, (16, 768), (768, 1), 0), arg84_1, out=buf138)
            del arg84_1
            del buf137
            buf139 = reinterpret_tensor(buf138, (1, 16, 3072), (49152, 3072, 1), 0); del buf138  # reuse
            # Topologically Sorted Source Nodes: [x_52, x_53, mul_24, pow_7, mul_25, add_26, mul_26, tanh_6, add_27, hidden_states_41], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_4.run(buf139, arg83_1, 49152, stream=stream0)
            del arg83_1
            buf140 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_52, x_53, mul_24, pow_7, mul_25, add_26, mul_26, tanh_6, add_27, hidden_states_41, view_76, x_54], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf139, (16, 3072), (3072, 1), 0), arg86_1, out=buf140)
            del arg86_1
            del buf139
            buf144 = empty_strided_cuda((1, 16, 768), (12288, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_54, x_55, hidden_states_43, hidden_states_44], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_5.run(buf133, buf140, arg85_1, arg87_1, arg88_1, buf144, 16, 768, stream=stream0)
            del arg87_1
            del arg88_1
            buf145 = empty_strided_cuda((16, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_54, x_55, hidden_states_43, hidden_states_44, view_78, x_56], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg89_1, reinterpret_tensor(buf144, (16, 768), (768, 1), 0), arg90_1, alpha=1, beta=1, out=buf145)
            del arg89_1
            del arg90_1
            # Topologically Sorted Source Nodes: [cache_position, result_1, kv_arange, kv_arange_1, le, batch_arange, index, eq, index_1, result_2, batched_outputs_2, x_57, split_7, view_82, query_states_15, view_80, key_states_15, view_81, value_states_15, attn_output_28], Original ATen: [aten.arange, aten.view, aten.add, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.split, aten.transpose, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf147 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf145, (1, 12, 16, 64), (0, 64, 2304, 1), 0), reinterpret_tensor(buf145, (1, 12, 16, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf145, (1, 12, 16, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf146, (1, 12, 16, 16), (256, 0, 16, 1), 0), False)
            del buf145
            del buf146
            buf148 = buf147[0]
            assert_size_stride(buf148, (1, 12, 16, 64), (12288, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf148, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf147
            buf152 = reinterpret_tensor(buf144, (16, 768), (768, 1), 0); del buf144  # reuse
            # Topologically Sorted Source Nodes: [transpose_31, reshape_7, view_83, x_58], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf148, (16, 768), (768, 1), 0), arg92_1, out=buf152)
            del arg92_1
            buf153 = reinterpret_tensor(buf152, (1, 16, 768), (12288, 768, 1), 0); del buf152  # reuse
            buf157 = reinterpret_tensor(buf148, (1, 16, 768), (12288, 768, 1), 0); del buf148  # reuse
            # Topologically Sorted Source Nodes: [x_54, x_55, hidden_states_43, x_58, x_59, hidden_states_45, hidden_states_46], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_6.run(buf153, arg91_1, buf133, buf140, arg85_1, arg93_1, arg94_1, buf157, 16, 768, stream=stream0)
            del arg85_1
            del arg91_1
            del arg93_1
            del arg94_1
            del buf133
            del buf140
            buf158 = empty_strided_cuda((16, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_46, view_85, x_60], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf157, (16, 768), (768, 1), 0), arg96_1, out=buf158)
            del arg96_1
            del buf157
            buf159 = reinterpret_tensor(buf158, (1, 16, 3072), (49152, 3072, 1), 0); del buf158  # reuse
            # Topologically Sorted Source Nodes: [x_60, x_61, mul_28, pow_8, mul_29, add_30, mul_30, tanh_7, add_31, hidden_states_47], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_4.run(buf159, arg95_1, 49152, stream=stream0)
            del arg95_1
            buf160 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_60, x_61, mul_28, pow_8, mul_29, add_30, mul_30, tanh_7, add_31, hidden_states_47, view_87, x_62], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf159, (16, 3072), (3072, 1), 0), arg98_1, out=buf160)
            del arg98_1
            del buf159
            buf164 = empty_strided_cuda((1, 16, 768), (12288, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_62, x_63, hidden_states_49, hidden_states_50], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_5.run(buf153, buf160, arg97_1, arg99_1, arg100_1, buf164, 16, 768, stream=stream0)
            del arg100_1
            del arg99_1
            buf165 = empty_strided_cuda((16, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_62, x_63, hidden_states_49, hidden_states_50, view_89, x_64], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg101_1, reinterpret_tensor(buf164, (16, 768), (768, 1), 0), arg102_1, alpha=1, beta=1, out=buf165)
            del arg101_1
            del arg102_1
            # Topologically Sorted Source Nodes: [cache_position, result_1, kv_arange, kv_arange_1, le, batch_arange, index, eq, index_1, result_2, batched_outputs_2, x_65, split_8, view_93, query_states_17, view_91, key_states_17, view_92, value_states_17, attn_output_32], Original ATen: [aten.arange, aten.view, aten.add, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.split, aten.transpose, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf167 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf165, (1, 12, 16, 64), (0, 64, 2304, 1), 0), reinterpret_tensor(buf165, (1, 12, 16, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf165, (1, 12, 16, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf166, (1, 12, 16, 16), (256, 0, 16, 1), 0), False)
            del buf165
            buf168 = buf167[0]
            assert_size_stride(buf168, (1, 12, 16, 64), (12288, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf168, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf167
            buf172 = reinterpret_tensor(buf164, (16, 768), (768, 1), 0); del buf164  # reuse
            # Topologically Sorted Source Nodes: [transpose_35, reshape_8, view_94, x_66], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf168, (16, 768), (768, 1), 0), arg104_1, out=buf172)
            del arg104_1
            buf173 = reinterpret_tensor(buf172, (1, 16, 768), (12288, 768, 1), 0); del buf172  # reuse
            buf177 = reinterpret_tensor(buf168, (1, 16, 768), (12288, 768, 1), 0); del buf168  # reuse
            # Topologically Sorted Source Nodes: [x_62, x_63, hidden_states_49, x_66, x_67, hidden_states_51, hidden_states_52], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_6.run(buf173, arg103_1, buf153, buf160, arg97_1, arg105_1, arg106_1, buf177, 16, 768, stream=stream0)
            del arg103_1
            del arg105_1
            del arg106_1
            del arg97_1
            del buf153
            del buf160
            buf178 = empty_strided_cuda((16, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_52, view_96, x_68], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf177, (16, 768), (768, 1), 0), arg108_1, out=buf178)
            del arg108_1
            del buf177
            buf179 = reinterpret_tensor(buf178, (1, 16, 3072), (49152, 3072, 1), 0); del buf178  # reuse
            # Topologically Sorted Source Nodes: [x_68, x_69, mul_32, pow_9, mul_33, add_34, mul_34, tanh_8, add_35, hidden_states_53], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_4.run(buf179, arg107_1, 49152, stream=stream0)
            del arg107_1
            buf180 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_68, x_69, mul_32, pow_9, mul_33, add_34, mul_34, tanh_8, add_35, hidden_states_53, view_98, x_70], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf179, (16, 3072), (3072, 1), 0), arg110_1, out=buf180)
            del arg110_1
            del buf179
            buf184 = empty_strided_cuda((1, 16, 768), (12288, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_70, x_71, hidden_states_55, hidden_states_56], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_5.run(buf173, buf180, arg109_1, arg111_1, arg112_1, buf184, 16, 768, stream=stream0)
            del arg111_1
            del arg112_1
            buf185 = empty_strided_cuda((16, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_70, x_71, hidden_states_55, hidden_states_56, view_100, x_72], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg113_1, reinterpret_tensor(buf184, (16, 768), (768, 1), 0), arg114_1, alpha=1, beta=1, out=buf185)
            del arg113_1
            del arg114_1
            # Topologically Sorted Source Nodes: [cache_position, result_1, kv_arange, kv_arange_1, le, batch_arange, index, eq, index_1, result_2, batched_outputs_2, x_73, split_9, view_104, query_states_19, view_102, key_states_19, view_103, value_states_19, attn_output_36], Original ATen: [aten.arange, aten.view, aten.add, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.split, aten.transpose, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf187 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf185, (1, 12, 16, 64), (0, 64, 2304, 1), 0), reinterpret_tensor(buf185, (1, 12, 16, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf185, (1, 12, 16, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf186, (1, 12, 16, 16), (256, 0, 16, 1), 0), False)
            del buf185
            buf188 = buf187[0]
            assert_size_stride(buf188, (1, 12, 16, 64), (12288, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf188, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf187
            buf192 = reinterpret_tensor(buf184, (16, 768), (768, 1), 0); del buf184  # reuse
            # Topologically Sorted Source Nodes: [transpose_39, reshape_9, view_105, x_74], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf188, (16, 768), (768, 1), 0), arg116_1, out=buf192)
            del arg116_1
            buf193 = reinterpret_tensor(buf192, (1, 16, 768), (12288, 768, 1), 0); del buf192  # reuse
            buf197 = reinterpret_tensor(buf188, (1, 16, 768), (12288, 768, 1), 0); del buf188  # reuse
            # Topologically Sorted Source Nodes: [x_70, x_71, hidden_states_55, x_74, x_75, hidden_states_57, hidden_states_58], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_6.run(buf193, arg115_1, buf173, buf180, arg109_1, arg117_1, arg118_1, buf197, 16, 768, stream=stream0)
            del arg109_1
            del arg115_1
            del arg117_1
            del arg118_1
            del buf173
            del buf180
            buf198 = empty_strided_cuda((16, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_58, view_107, x_76], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf197, (16, 768), (768, 1), 0), arg120_1, out=buf198)
            del arg120_1
            del buf197
            buf199 = reinterpret_tensor(buf198, (1, 16, 3072), (49152, 3072, 1), 0); del buf198  # reuse
            # Topologically Sorted Source Nodes: [x_76, x_77, mul_36, pow_10, mul_37, add_38, mul_38, tanh_9, add_39, hidden_states_59], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_4.run(buf199, arg119_1, 49152, stream=stream0)
            del arg119_1
            buf200 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_76, x_77, mul_36, pow_10, mul_37, add_38, mul_38, tanh_9, add_39, hidden_states_59, view_109, x_78], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf199, (16, 3072), (3072, 1), 0), arg122_1, out=buf200)
            del arg122_1
            del buf199
            buf204 = empty_strided_cuda((1, 16, 768), (12288, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_78, x_79, hidden_states_61, hidden_states_62], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_5.run(buf193, buf200, arg121_1, arg123_1, arg124_1, buf204, 16, 768, stream=stream0)
            del arg123_1
            del arg124_1
            buf205 = empty_strided_cuda((16, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_78, x_79, hidden_states_61, hidden_states_62, view_111, x_80], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg125_1, reinterpret_tensor(buf204, (16, 768), (768, 1), 0), arg126_1, alpha=1, beta=1, out=buf205)
            del arg125_1
            del arg126_1
            buf206 = buf186; del buf186  # reuse
            buf226 = buf166; del buf166  # reuse
            # Topologically Sorted Source Nodes: [cache_position, result_1, kv_arange, kv_arange_1, le, batch_arange, index, eq, index_1, result_2, batched_outputs_2, x_81, split_10, view_115, query_states_21, view_113, key_states_21, view_114, value_states_21, attn_output_40, x_89, split_11, view_126, query_states_23, view_124, key_states_23, view_125, value_states_23, attn_output_44], Original ATen: [aten.arange, aten.view, aten.add, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.split, aten.transpose, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_bitwise_and_eq_expand_index_le_scalar_tensor_split_transpose_view_where_7.run(buf5, buf206, buf226, 256, stream=stream0)
            del buf5
            # Topologically Sorted Source Nodes: [cache_position, result_1, kv_arange, kv_arange_1, le, batch_arange, index, eq, index_1, result_2, batched_outputs_2, x_81, split_10, view_115, query_states_21, view_113, key_states_21, view_114, value_states_21, attn_output_40], Original ATen: [aten.arange, aten.view, aten.add, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.split, aten.transpose, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf207 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf205, (1, 12, 16, 64), (0, 64, 2304, 1), 0), reinterpret_tensor(buf205, (1, 12, 16, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf205, (1, 12, 16, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf206, (1, 12, 16, 16), (256, 0, 16, 1), 0), False)
            del buf205
            del buf206
            buf208 = buf207[0]
            assert_size_stride(buf208, (1, 12, 16, 64), (12288, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf208, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf207
            buf212 = reinterpret_tensor(buf204, (16, 768), (768, 1), 0); del buf204  # reuse
            # Topologically Sorted Source Nodes: [transpose_43, reshape_10, view_116, x_82], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf208, (16, 768), (768, 1), 0), arg128_1, out=buf212)
            del arg128_1
            buf213 = reinterpret_tensor(buf212, (1, 16, 768), (12288, 768, 1), 0); del buf212  # reuse
            buf217 = reinterpret_tensor(buf208, (1, 16, 768), (12288, 768, 1), 0); del buf208  # reuse
            # Topologically Sorted Source Nodes: [x_78, x_79, hidden_states_61, x_82, x_83, hidden_states_63, hidden_states_64], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_6.run(buf213, arg127_1, buf193, buf200, arg121_1, arg129_1, arg130_1, buf217, 16, 768, stream=stream0)
            del arg121_1
            del arg127_1
            del arg129_1
            del arg130_1
            del buf193
            del buf200
            buf218 = empty_strided_cuda((16, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_64, view_118, x_84], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf217, (16, 768), (768, 1), 0), arg132_1, out=buf218)
            del arg132_1
            del buf217
            buf219 = reinterpret_tensor(buf218, (1, 16, 3072), (49152, 3072, 1), 0); del buf218  # reuse
            # Topologically Sorted Source Nodes: [x_84, x_85, mul_40, pow_11, mul_41, add_42, mul_42, tanh_10, add_43, hidden_states_65], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_4.run(buf219, arg131_1, 49152, stream=stream0)
            del arg131_1
            buf220 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_84, x_85, mul_40, pow_11, mul_41, add_42, mul_42, tanh_10, add_43, hidden_states_65, view_120, x_86], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf219, (16, 3072), (3072, 1), 0), arg134_1, out=buf220)
            del arg134_1
            del buf219
            buf224 = empty_strided_cuda((1, 16, 768), (12288, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_86, x_87, hidden_states_67, hidden_states_68], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_5.run(buf213, buf220, arg133_1, arg135_1, arg136_1, buf224, 16, 768, stream=stream0)
            del arg135_1
            del arg136_1
            buf225 = empty_strided_cuda((16, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_86, x_87, hidden_states_67, hidden_states_68, view_122, x_88], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg137_1, reinterpret_tensor(buf224, (16, 768), (768, 1), 0), arg138_1, alpha=1, beta=1, out=buf225)
            del arg137_1
            del arg138_1
            # Topologically Sorted Source Nodes: [cache_position, result_1, kv_arange, kv_arange_1, le, batch_arange, index, eq, index_1, result_2, batched_outputs_2, x_89, split_11, view_126, query_states_23, view_124, key_states_23, view_125, value_states_23, attn_output_44], Original ATen: [aten.arange, aten.view, aten.add, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.split, aten.transpose, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf227 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf225, (1, 12, 16, 64), (0, 64, 2304, 1), 0), reinterpret_tensor(buf225, (1, 12, 16, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf225, (1, 12, 16, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf226, (1, 12, 16, 16), (256, 0, 16, 1), 0), False)
            del buf225
            del buf226
            buf228 = buf227[0]
            assert_size_stride(buf228, (1, 12, 16, 64), (12288, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf228, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf227
            buf232 = reinterpret_tensor(buf224, (16, 768), (768, 1), 0); del buf224  # reuse
            # Topologically Sorted Source Nodes: [transpose_47, reshape_11, view_127, x_90], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf228, (16, 768), (768, 1), 0), arg140_1, out=buf232)
            del arg140_1
            buf233 = reinterpret_tensor(buf232, (1, 16, 768), (12288, 768, 1), 0); del buf232  # reuse
            buf237 = reinterpret_tensor(buf228, (1, 16, 768), (12288, 768, 1), 0); del buf228  # reuse
            # Topologically Sorted Source Nodes: [x_86, x_87, hidden_states_67, x_90, x_91, hidden_states_69, hidden_states_70], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_6.run(buf233, arg139_1, buf213, buf220, arg133_1, arg141_1, arg142_1, buf237, 16, 768, stream=stream0)
            del arg133_1
            del arg139_1
            del arg141_1
            del arg142_1
            del buf213
            del buf220
            buf238 = empty_strided_cuda((16, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_70, view_129, x_92], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf237, (16, 768), (768, 1), 0), arg144_1, out=buf238)
            del arg144_1
            del buf237
            buf239 = reinterpret_tensor(buf238, (1, 16, 3072), (49152, 3072, 1), 0); del buf238  # reuse
            # Topologically Sorted Source Nodes: [x_92, x_93, mul_44, pow_12, mul_45, add_46, mul_46, tanh_11, add_47, hidden_states_71], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_4.run(buf239, arg143_1, 49152, stream=stream0)
            del arg143_1
            buf240 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_92, x_93, mul_44, pow_12, mul_45, add_46, mul_46, tanh_11, add_47, hidden_states_71, view_131, x_94], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf239, (16, 3072), (3072, 1), 0), arg146_1, out=buf240)
            del arg146_1
            del buf239
            buf244 = buf233; del buf233  # reuse
            # Topologically Sorted Source Nodes: [x_94, x_95, hidden_states_73, hidden_states_74], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_8.run(buf244, buf240, arg145_1, arg147_1, arg148_1, 16, 768, stream=stream0)
            del arg145_1
            del arg147_1
            del arg148_1
            del buf240
        return (buf244, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 16), (16, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((50257, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)


# ===== inductor generated file: /tmp/cnnbench-transformers-ob_ijj07/repeat_03/a1/torchinductor/tmpuqb_59nb/tb/ctbeuecaa5nknwy6bvucluzl476p3mx7abciczu4tjqirt7duxrh.py =====
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


# kernel path: /tmp/cnnbench-transformers-ob_ijj07/repeat_03/a1/torchinductor/tmpuqb_59nb/lj/cljm2yw4ttzhprxdzyni7ggrjibfnhmatq2ohteunj2nfqlnqaqr.py
# Topologically Sorted Source Nodes: [inputs_embeds, cache_position, position_ids, position_embeds, hidden_states, hidden_states_2], Original ATen: [aten.embedding, aten.arange, aten.unsqueeze, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   cache_position => iota
#   hidden_states => add
#   hidden_states_2 => add_2, add_3, mul, mul_1, rsqrt, sub_2, var_mean
#   inputs_embeds => embedding
#   position_embeds => embedding_1
#   position_ids => unsqueeze
# Graph fragment:
#   %arg0_1 : Tensor "i64[1, 16][16, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[50257, 768][768, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "f32[1024, 768][768, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %getitem_1 : Tensor "f32[1, 16, 1][16, 1, 16]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf1 : Tensor "f32[1, 16, 1][16, 1, 16]cuda:0" = PlaceHolder[target=buf1]
#   %arg3_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %embedding : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %arg0_1), kwargs = {})
#   %iota : Tensor "i64[16][1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze : Tensor "i64[1, 16][16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, 0), kwargs = {})
#   %embedding_1 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %unsqueeze), kwargs = {})
#   %add : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %getitem_1), kwargs = {})
#   %add_2 : Tensor "f32[1, 16, 1][16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[1, 16, 1][16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg3_1), kwargs = {})
#   %add_3 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %arg4_1), kwargs = {})
#   return %getitem_1,%buf1,%add_3
triton_red_fused_add_arange_embedding_native_layer_norm_unsqueeze_0 = async_compile.triton('triton_red_fused_add_arange_embedding_native_layer_norm_unsqueeze_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16, 'r0_': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_arange_embedding_native_layer_norm_unsqueeze_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_arange_embedding_native_layer_norm_unsqueeze_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 16
    r0_numel = 768
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp10_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp10_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp10_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp7 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.full([XBLOCK, R0_BLOCK], 50257, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert(((0 <= tmp4) & (tmp4 < 50257)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 50257")
        tmp6 = tl.load(in_ptr1 + (r0_1 + 768*tmp4), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
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
        r0_1 = r0_index
        tmp22 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp31 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp33 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.full([XBLOCK, R0_BLOCK], 50257, tl.int32)
        tmp17 = tmp0 + tmp16
        tmp18 = tmp0 < 0
        tmp19 = tl.where(tmp18, tmp17, tmp0)
        tl.device_assert(((0 <= tmp19) & (tmp19 < 50257)) | ~(xmask), "index out of bounds: 0 <= tmp19 < 50257")
        tmp21 = tl.load(in_ptr1 + (r0_1 + 768*tmp19), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
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
        tl.store(out_ptr2 + (r0_1 + 768*x0), tmp34, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-transformers-ob_ijj07/repeat_03/a1/torchinductor/tmpuqb_59nb/dx/cdxx2tckrq2rfzdtdjpsecgjjg5rvlcjjvpv5ckasole7tffxdye.py
# Topologically Sorted Source Nodes: [cache_position, position_ids, getitem, first_dummy_value, position_diff, ne, packed_sequence_mask], Original ATen: [aten.arange, aten.unsqueeze, aten.slice, aten.sub, aten.cat, aten.ne, aten.cumsum]
# Source node to ATen node mapping:
#   cache_position => iota
#   first_dummy_value => sub
#   getitem => slice_1
#   ne => ne
#   packed_sequence_mask => cumsum
#   position_diff => cat, slice_2, slice_3, sub_1
#   position_ids => unsqueeze
# Graph fragment:
#   %iota : Tensor "i64[16][1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze : Tensor "i64[1, 16][16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, 0), kwargs = {})
#   %slice_1 : Tensor "i64[1, 1][16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%unsqueeze, 1, 0, 1), kwargs = {})
#   %sub : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_1, 1), kwargs = {})
#   %cat : Tensor "i64[1, 17][17, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%sub, %unsqueeze], -1), kwargs = {})
#   %slice_3 : Tensor "i64[1, 16][17, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat, -1, 1, 17), kwargs = {})
#   %slice_2 : Tensor "i64[1, 16][17, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat, -1, 0, 16), kwargs = {})
#   %sub_1 : Tensor "i64[1, 16][16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_3, %slice_2), kwargs = {})
#   %ne : Tensor "b8[1, 16][16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%sub_1, 1), kwargs = {})
#   %cumsum : Tensor "i64[1, 16][16, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.cumsum.default](args = (%ne, -1), kwargs = {})
#   return %cumsum
triton_per_fused_arange_cat_cumsum_ne_slice_sub_unsqueeze_1 = async_compile.triton('triton_per_fused_arange_cat_cumsum_ne_slice_sub_unsqueeze_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def _triton_helper_fn_add0(arg0_0, arg1_0):
    tmp0 = arg0_0 + arg1_0
    return tmp0

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_arange_cat_cumsum_ne_slice_sub_unsqueeze_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 256}}
)
@triton.jit
def triton_per_fused_arange_cat_cumsum_ne_slice_sub_unsqueeze_1(out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 16
    R0_BLOCK: tl.constexpr = 16
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = 1 + r0_0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.broadcast_to((-1) + (1 + r0_0), [XBLOCK, R0_BLOCK])
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 17, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp15 = r0_0
    tmp16 = tmp15 >= tmp1
    tmp17 = tmp15 < tmp3
    tmp18 = tl.broadcast_to((-1) + (r0_0), [XBLOCK, R0_BLOCK])
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp15 >= tmp3
    tmp22 = tmp15 < tmp9
    tmp23 = tl.broadcast_to((-1) + r0_0, [XBLOCK, R0_BLOCK])
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp21, tmp23, tmp24)
    tmp26 = tl.where(tmp17, tmp20, tmp25)
    tmp27 = tmp14 - tmp26
    tmp28 = tmp27 != tmp3
    tmp29 = tmp28.to(tl.int64)
    tmp30 = tmp29.to(tl.int64)
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK, R0_BLOCK])
    tmp32, = tl.associative_scan((tmp31,), 1, _triton_helper_fn_add0)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp32, None)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-transformers-ob_ijj07/repeat_03/a1/torchinductor/tmpuqb_59nb/ja/cjaucbdxwmlrtgyrskzzyoh4pgwb47uid5kjp6f6k3rgfh2zx7km.py
# Topologically Sorted Source Nodes: [cache_position, x_1, split, view_5, query_states_1, view_3, key_states_1, view_4, value_states_1, result_1, kv_arange, kv_arange_1, le, batch_arange, index, eq, index_1, result_2, batched_outputs_2, attn_output, x_9, split_1, view_16, query_states_3, view_14, key_states_3, view_15, value_states_3, attn_output_4, x_17, split_2, view_27, query_states_5, view_25, key_states_5, view_26, value_states_5, attn_output_8, x_25, split_3, view_38, query_states_7, view_36, key_states_7, view_37, value_states_7, attn_output_12, x_33, split_4, view_49, query_states_9, view_47, key_states_9, view_48, value_states_9, attn_output_16], Original ATen: [aten.arange, aten.view, aten.split, aten.transpose, aten.add, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   attn_output => _scaled_dot_product_efficient_attention, expand_1, full_default_1, full_default_2, where
#   attn_output_12 => _scaled_dot_product_efficient_attention_3, expand_4, full_default_7, full_default_8, where_3
#   attn_output_16 => _scaled_dot_product_efficient_attention_4, expand_5, full_default_10, full_default_9, where_4
#   attn_output_4 => _scaled_dot_product_efficient_attention_1, expand_2, full_default_3, full_default_4, where_1
#   attn_output_8 => _scaled_dot_product_efficient_attention_2, expand_3, full_default_5, full_default_6, where_2
#   batch_arange => iota_2
#   batched_outputs_2 => view_7
#   cache_position => iota
#   eq => eq, view_5, view_6
#   index => index, view_3
#   index_1 => index_1, view_4
#   key_states_1 => permute
#   key_states_3 => permute_4
#   key_states_5 => permute_8
#   key_states_7 => permute_12
#   key_states_9 => permute_16
#   kv_arange => iota_1
#   kv_arange_1 => add_1
#   le => le, view_1
#   query_states_1 => permute_2
#   query_states_3 => permute_6
#   query_states_5 => permute_10
#   query_states_7 => permute_14
#   query_states_9 => permute_18
#   result_1 => bitwise_and, full_default
#   result_2 => bitwise_and_1
#   split => split
#   split_1 => split_1
#   split_2 => split_2
#   split_3 => split_3
#   split_4 => split_4
#   value_states_1 => permute_1
#   value_states_3 => permute_5
#   value_states_5 => permute_9
#   value_states_7 => permute_13
#   value_states_9 => permute_17
#   view_14 => view_22
#   view_15 => view_23
#   view_16 => view_24
#   view_25 => view_34
#   view_26 => view_35
#   view_27 => view_36
#   view_3 => view_10
#   view_36 => view_46
#   view_37 => view_47
#   view_38 => view_48
#   view_4 => view_11
#   view_47 => view_58
#   view_48 => view_59
#   view_49 => view_60
#   view_5 => view_12
#   x_1 => view_9
#   x_17 => view_33
#   x_25 => view_45
#   x_33 => view_57
#   x_9 => view_21
# Graph fragment:
#   %cumsum : Tensor "i64[1, 16][16, 1]cuda:0" = PlaceHolder[target=cumsum]
#   %iota : Tensor "i64[16][1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %view_9 : Tensor "f32[1, 16, 2304][36864, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [1, 16, 2304]), kwargs = {})
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_9, 768, 2), kwargs = {})
#   %view_12 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_2, [1, 16, -1, 64]), kwargs = {})
#   %permute_2 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_12, [0, 2, 1, 3]), kwargs = {})
#   %view_10 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_3, [1, 16, -1, 64]), kwargs = {})
#   %permute : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_10, [0, 2, 1, 3]), kwargs = {})
#   %view_11 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_4, [1, 16, -1, 64]), kwargs = {})
#   %permute_1 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_11, [0, 2, 1, 3]), kwargs = {})
#   %full_default : Tensor "b8[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([16, 1], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %iota_1 : Tensor "i64[16][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add_1 : Tensor "i64[16][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota_1, 0), kwargs = {})
#   %view_1 : Tensor "i64[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%iota, [16, 1]), kwargs = {})
#   %le : Tensor "b8[16, 16][16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Tensor](args = (%add_1, %view_1), kwargs = {})
#   %bitwise_and : Tensor "b8[16, 16][16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%full_default, %le), kwargs = {})
#   %iota_2 : Tensor "i64[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %view_3 : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%iota_2, [1, 1]), kwargs = {})
#   %index : Tensor "i64[1, 16][16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%cumsum, [%view_3, %iota]), kwargs = {})
#   %view_5 : Tensor "i64[1, 16, 1][16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%index, [1, 16, 1]), kwargs = {})
#   %view_4 : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%iota_2, [1, 1]), kwargs = {})
#   %index_1 : Tensor "i64[1, 16][16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%cumsum, [%view_4, %add_1]), kwargs = {})
#   %view_6 : Tensor "i64[1, 1, 16][16, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%index_1, [1, 1, 16]), kwargs = {})
#   %eq : Tensor "b8[1, 16, 16][256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%view_5, %view_6), kwargs = {})
#   %bitwise_and_1 : Tensor "b8[1, 16, 16][256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and, %eq), kwargs = {})
#   %view_7 : Tensor "b8[1, 1, 16, 16][256, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bitwise_and_1, [1, 1, 16, 16]), kwargs = {})
#   %full_default_2 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "f32[1, 1, 16, 16][256, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand, %full_default_2, %full_default_1), kwargs = {})
#   %expand_1 : Tensor "f32[1, 12, 16, 16][256, 0, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%where, [1, 12, 16, 16]), kwargs = {})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_2, %permute, %permute_1, %expand_1, False), kwargs = {})
#   %view_21 : Tensor "f32[1, 16, 2304][36864, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_4, [1, 16, 2304]), kwargs = {})
#   %split_1 : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_21, 768, 2), kwargs = {})
#   %view_24 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_13, [1, 16, -1, 64]), kwargs = {})
#   %permute_6 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_24, [0, 2, 1, 3]), kwargs = {})
#   %view_22 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_14, [1, 16, -1, 64]), kwargs = {})
#   %permute_4 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_22, [0, 2, 1, 3]), kwargs = {})
#   %view_23 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_15, [1, 16, -1, 64]), kwargs = {})
#   %permute_5 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_23, [0, 2, 1, 3]), kwargs = {})
#   %full_default_4 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_3 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : Tensor "f32[1, 1, 16, 16][256, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand, %full_default_4, %full_default_3), kwargs = {})
#   %expand_2 : Tensor "f32[1, 12, 16, 16][256, 0, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%where_1, [1, 12, 16, 16]), kwargs = {})
#   %_scaled_dot_product_efficient_attention_1 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_6, %permute_4, %permute_5, %expand_2, False), kwargs = {})
#   %view_33 : Tensor "f32[1, 16, 2304][36864, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_8, [1, 16, 2304]), kwargs = {})
#   %split_2 : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_33, 768, 2), kwargs = {})
#   %view_36 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_24, [1, 16, -1, 64]), kwargs = {})
#   %permute_10 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_36, [0, 2, 1, 3]), kwargs = {})
#   %view_34 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_25, [1, 16, -1, 64]), kwargs = {})
#   %permute_8 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_34, [0, 2, 1, 3]), kwargs = {})
#   %view_35 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_26, [1, 16, -1, 64]), kwargs = {})
#   %permute_9 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_35, [0, 2, 1, 3]), kwargs = {})
#   %full_default_6 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_5 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : Tensor "f32[1, 1, 16, 16][256, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand, %full_default_6, %full_default_5), kwargs = {})
#   %expand_3 : Tensor "f32[1, 12, 16, 16][256, 0, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%where_2, [1, 12, 16, 16]), kwargs = {})
#   %_scaled_dot_product_efficient_attention_2 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_10, %permute_8, %permute_9, %expand_3, False), kwargs = {})
#   %view_45 : Tensor "f32[1, 16, 2304][36864, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_12, [1, 16, 2304]), kwargs = {})
#   %split_3 : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_45, 768, 2), kwargs = {})
#   %view_48 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_35, [1, 16, -1, 64]), kwargs = {})
#   %permute_14 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_48, [0, 2, 1, 3]), kwargs = {})
#   %view_46 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_36, [1, 16, -1, 64]), kwargs = {})
#   %permute_12 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_46, [0, 2, 1, 3]), kwargs = {})
#   %view_47 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_37, [1, 16, -1, 64]), kwargs = {})
#   %permute_13 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_47, [0, 2, 1, 3]), kwargs = {})
#   %full_default_8 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_7 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_3 : Tensor "f32[1, 1, 16, 16][256, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand, %full_default_8, %full_default_7), kwargs = {})
#   %expand_4 : Tensor "f32[1, 12, 16, 16][256, 0, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%where_3, [1, 12, 16, 16]), kwargs = {})
#   %_scaled_dot_product_efficient_attention_3 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_14, %permute_12, %permute_13, %expand_4, False), kwargs = {})
#   %view_57 : Tensor "f32[1, 16, 2304][36864, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_16, [1, 16, 2304]), kwargs = {})
#   %split_4 : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_57, 768, 2), kwargs = {})
#   %view_60 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_46, [1, 16, -1, 64]), kwargs = {})
#   %permute_18 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_60, [0, 2, 1, 3]), kwargs = {})
#   %view_58 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_47, [1, 16, -1, 64]), kwargs = {})
#   %permute_16 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_58, [0, 2, 1, 3]), kwargs = {})
#   %view_59 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_48, [1, 16, -1, 64]), kwargs = {})
#   %permute_17 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_59, [0, 2, 1, 3]), kwargs = {})
#   %full_default_10 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_9 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_4 : Tensor "f32[1, 1, 16, 16][256, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand, %full_default_10, %full_default_9), kwargs = {})
#   %expand_5 : Tensor "f32[1, 12, 16, 16][256, 0, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%where_4, [1, 12, 16, 16]), kwargs = {})
#   %_scaled_dot_product_efficient_attention_4 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_18, %permute_16, %permute_17, %expand_5, False), kwargs = {})
#   return %buf6,%buf26,%buf46,%buf66,%buf86
triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_bitwise_and_eq_expand_index_le_scalar_tensor_split_transpose_view_where_2 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_bitwise_and_eq_expand_index_le_scalar_tensor_split_transpose_view_where_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_bitwise_and_eq_expand_index_le_scalar_tensor_split_transpose_view_where_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 10368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_bitwise_and_eq_expand_index_le_scalar_tensor_split_transpose_view_where_2(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp5 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = x1
    tmp2 = tmp0 <= tmp1
    tmp3 = tl.full([1], True, tl.int1)
    tmp4 = tmp3 & tmp2
    tmp7 = tmp5 == tmp6
    tmp8 = tmp4 & tmp7
    tmp9 = 0.0
    tmp10 = float("-inf")
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tl.store(out_ptr0 + (x2), tmp11, xmask)
    tl.store(out_ptr1 + (x2), tmp11, xmask)
    tl.store(out_ptr2 + (x2), tmp11, xmask)
    tl.store(out_ptr3 + (x2), tmp11, xmask)
    tl.store(out_ptr4 + (x2), tmp11, xmask)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-transformers-ob_ijj07/repeat_03/a1/torchinductor/tmpuqb_59nb/tg/ctgkujgbvk26wf25wgqfcccy3meikawb6tsgv2vykji5s3geakfs.py
# Topologically Sorted Source Nodes: [inputs_embeds, cache_position, position_ids, position_embeds, hidden_states, x_2, x_3, hidden_states_3, hidden_states_4], Original ATen: [aten.embedding, aten.arange, aten.unsqueeze, aten.add, aten.addmm, aten.view, aten.native_layer_norm]
# Source node to ATen node mapping:
#   cache_position => iota
#   hidden_states => add
#   hidden_states_3 => add_4
#   hidden_states_4 => add_5, add_6, mul_2, mul_3, rsqrt_1, sub_3, var_mean_1
#   inputs_embeds => embedding
#   position_embeds => embedding_1
#   position_ids => unsqueeze
#   x_2 => add_tensor_35
#   x_3 => view_15
# Graph fragment:
#   %mm_default_35 : Tensor "f32[16, 768][768, 1]cuda:0" = PlaceHolder[target=mm_default_35]
#   %arg7_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg7_1]
#   %arg0_1 : Tensor "i64[1, 16][16, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f32[50257, 768][768, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "f32[1024, 768][768, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %add_4 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0" = PlaceHolder[target=add_4]
#   %getitem_10 : Tensor "f32[1, 16, 1][16, 1, 16]cuda:0" = PlaceHolder[target=getitem_10]
#   %buf15 : Tensor "f32[1, 16, 1][16, 1, 16]cuda:0" = PlaceHolder[target=buf15]
#   %arg9_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg9_1]
#   %arg10_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg10_1]
#   %embedding : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %arg0_1), kwargs = {})
#   %iota : Tensor "i64[16][1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze : Tensor "i64[1, 16][16, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, 0), kwargs = {})
#   %embedding_1 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg2_1, %unsqueeze), kwargs = {})
#   %add : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %add_tensor_35 : Tensor "f32[16, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_35, %arg7_1), kwargs = {})
#   %view_15 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_35, [1, 16, 768]), kwargs = {})
#   %add_4 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_15, %add), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_4, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_4, %getitem_10), kwargs = {})
#   %add_5 : Tensor "f32[1, 16, 1][16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_9, 1e-05), kwargs = {})
#   %rsqrt_1 : Tensor "f32[1, 16, 1][16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
#   %mul_2 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_1), kwargs = {})
#   %mul_3 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %arg9_1), kwargs = {})
#   %add_6 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %arg10_1), kwargs = {})
#   return %add_4,%getitem_10,%buf15,%add_6
triton_per_fused_add_addmm_arange_embedding_native_layer_norm_unsqueeze_view_3 = async_compile.triton('triton_per_fused_add_addmm_arange_embedding_native_layer_norm_unsqueeze_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_arange_embedding_native_layer_norm_unsqueeze_view_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 6, 'num_reduction': 4, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_addmm_arange_embedding_native_layer_norm_unsqueeze_view_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16
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
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp36 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tl.full([XBLOCK, R0_BLOCK], 50257, tl.int32)
    tmp5 = tmp3 + tmp4
    tmp6 = tmp3 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp3)
    tl.device_assert(((0 <= tmp7) & (tmp7 < 50257)) | ~(xmask), "index out of bounds: 0 <= tmp7 < 50257")
    tmp9 = tl.load(in_ptr2 + (r0_1 + 768*tmp7), r0_mask & xmask, other=0.0)
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
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp12, r0_mask & xmask)
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp39, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-transformers-ob_ijj07/repeat_03/a1/torchinductor/tmpuqb_59nb/s5/cs5rl3zynqzmsjh2xshqzukk3gtxthgzyqi3b5jc7jul5gcv3uc4.py
# Topologically Sorted Source Nodes: [x_4, x_5, mul, pow_1, mul_1, add_2, mul_2, tanh, add_3, hidden_states_5], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
# Source node to ATen node mapping:
#   add_2 => add_7
#   add_3 => add_8
#   hidden_states_5 => mul_7
#   mul => mul_4
#   mul_1 => mul_5
#   mul_2 => mul_6
#   pow_1 => pow_1
#   tanh => tanh
#   x_4 => add_tensor_34
#   x_5 => view_17
# Graph fragment:
#   %mm_default_34 : Tensor "f32[16, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_default_34]
#   %arg11_1 : Tensor "f32[3072][1]cuda:0" = PlaceHolder[target=arg11_1]
#   %add_tensor_34 : Tensor "f32[16, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_34, %arg11_1), kwargs = {})
#   %view_17 : Tensor "f32[1, 16, 3072][49152, 3072, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_34, [1, 16, 3072]), kwargs = {})
#   %mul_4 : Tensor "f32[1, 16, 3072][49152, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_17, 0.5), kwargs = {})
#   %pow_1 : Tensor "f32[1, 16, 3072][49152, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view_17, 3.0), kwargs = {})
#   %mul_5 : Tensor "f32[1, 16, 3072][49152, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_1, 0.044715), kwargs = {})
#   %add_7 : Tensor "f32[1, 16, 3072][49152, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_17, %mul_5), kwargs = {})
#   %mul_6 : Tensor "f32[1, 16, 3072][49152, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_7, 0.7978845608028654), kwargs = {})
#   %tanh : Tensor "f32[1, 16, 3072][49152, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_6,), kwargs = {})
#   %add_8 : Tensor "f32[1, 16, 3072][49152, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tanh, 1.0), kwargs = {})
#   %mul_7 : Tensor "f32[1, 16, 3072][49152, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %add_8), kwargs = {})
#   return %mul_7
triton_poi_fused_add_addmm_mul_pow_tanh_view_4 = async_compile.triton('triton_poi_fused_add_addmm_mul_pow_tanh_view_4', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_mul_pow_tanh_view_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 602112}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_mul_pow_tanh_view_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
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


# kernel path: /tmp/cnnbench-transformers-ob_ijj07/repeat_03/a1/torchinductor/tmpuqb_59nb/2e/c2eekjtavfj2jegok4t3geyjeblhnw2aljimpfddxzj6eolzgwjx.py
# Topologically Sorted Source Nodes: [x_6, x_7, hidden_states_7, hidden_states_8], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_7 => add_9
#   hidden_states_8 => add_10, add_11, mul_8, mul_9, rsqrt_2, sub_4, var_mean_2
#   x_6 => add_tensor_33
#   x_7 => view_19
# Graph fragment:
#   %add_4 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0" = PlaceHolder[target=add_4]
#   %mm_default_33 : Tensor "f32[16, 768][768, 1]cuda:0" = PlaceHolder[target=mm_default_33]
#   %arg13_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg13_1]
#   %getitem_12 : Tensor "f32[1, 16, 1][16, 1, 16]cuda:0" = PlaceHolder[target=getitem_12]
#   %buf22 : Tensor "f32[1, 16, 1][16, 1, 16]cuda:0" = PlaceHolder[target=buf22]
#   %arg15_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg15_1]
#   %arg16_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg16_1]
#   %add_tensor_33 : Tensor "f32[16, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_33, %arg13_1), kwargs = {})
#   %view_19 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_33, [1, 16, 768]), kwargs = {})
#   %add_9 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %view_19), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_9, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_4 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_9, %getitem_12), kwargs = {})
#   %add_10 : Tensor "f32[1, 16, 1][16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_11, 1e-05), kwargs = {})
#   %rsqrt_2 : Tensor "f32[1, 16, 1][16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_10,), kwargs = {})
#   %mul_8 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_2), kwargs = {})
#   %mul_9 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %arg15_1), kwargs = {})
#   %add_11 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %arg16_1), kwargs = {})
#   return %getitem_12,%buf22,%add_11
triton_per_fused_add_addmm_native_layer_norm_view_5 = async_compile.triton('triton_per_fused_add_addmm_native_layer_norm_view_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_native_layer_norm_view_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 205824}}
)
@triton.jit
def triton_per_fused_add_addmm_native_layer_norm_view_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16
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


# kernel path: /tmp/cnnbench-transformers-ob_ijj07/repeat_03/a1/torchinductor/tmpuqb_59nb/m5/cm5y4qzqfsa7zrgnjswk65q2jceoci2grzwwh22it65glp6s5qcx.py
# Topologically Sorted Source Nodes: [x_6, x_7, hidden_states_7, x_10, x_11, hidden_states_9, hidden_states_10], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_10 => add_13, add_14, mul_10, mul_11, rsqrt_3, sub_5, var_mean_3
#   hidden_states_7 => add_9
#   hidden_states_9 => add_12
#   x_10 => add_tensor_32
#   x_11 => view_27
#   x_6 => add_tensor_33
#   x_7 => view_19
# Graph fragment:
#   %mm_default_32 : Tensor "f32[16, 768][768, 1]cuda:0" = PlaceHolder[target=mm_default_32]
#   %arg19_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg19_1]
#   %add_4 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0" = PlaceHolder[target=add_4]
#   %mm_default_33 : Tensor "f32[16, 768][768, 1]cuda:0" = PlaceHolder[target=mm_default_33]
#   %arg13_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg13_1]
#   %add_12 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0" = PlaceHolder[target=add_12]
#   %getitem_21 : Tensor "f32[1, 16, 1][16, 1, 16]cuda:0" = PlaceHolder[target=getitem_21]
#   %buf35 : Tensor "f32[1, 16, 1][16, 1, 16]cuda:0" = PlaceHolder[target=buf35]
#   %arg21_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg21_1]
#   %arg22_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg22_1]
#   %add_tensor_33 : Tensor "f32[16, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_33, %arg13_1), kwargs = {})
#   %view_19 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_33, [1, 16, 768]), kwargs = {})
#   %add_9 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %view_19), kwargs = {})
#   %add_tensor_32 : Tensor "f32[16, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_32, %arg19_1), kwargs = {})
#   %view_27 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_32, [1, 16, 768]), kwargs = {})
#   %add_12 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_27, %add_9), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_12, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_5 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_12, %getitem_21), kwargs = {})
#   %add_13 : Tensor "f32[1, 16, 1][16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_20, 1e-05), kwargs = {})
#   %rsqrt_3 : Tensor "f32[1, 16, 1][16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_13,), kwargs = {})
#   %mul_10 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_3), kwargs = {})
#   %mul_11 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %arg21_1), kwargs = {})
#   %add_14 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %arg22_1), kwargs = {})
#   return %add_12,%getitem_21,%buf35,%add_14
triton_per_fused_add_addmm_native_layer_norm_view_6 = async_compile.triton('triton_per_fused_add_addmm_native_layer_norm_view_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_native_layer_norm_view_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 356352}}
)
@triton.jit
def triton_per_fused_add_addmm_native_layer_norm_view_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16
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


# kernel path: /tmp/cnnbench-transformers-ob_ijj07/repeat_03/a1/torchinductor/tmpuqb_59nb/n5/cn5drgj55mabns5bc5i7nj2iwlnlly62rbfoy3roikqqnbm2fwkk.py
# Topologically Sorted Source Nodes: [cache_position, result_1, kv_arange, kv_arange_1, le, batch_arange, index, eq, index_1, result_2, batched_outputs_2, x_81, split_10, view_115, query_states_21, view_113, key_states_21, view_114, value_states_21, attn_output_40, x_89, split_11, view_126, query_states_23, view_124, key_states_23, view_125, value_states_23, attn_output_44], Original ATen: [aten.arange, aten.view, aten.add, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.split, aten.transpose, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   attn_output_40 => _scaled_dot_product_efficient_attention_10, expand_11, full_default_21, full_default_22, where_10
#   attn_output_44 => _scaled_dot_product_efficient_attention_11, expand_12, full_default_23, full_default_24, where_11
#   batch_arange => iota_2
#   batched_outputs_2 => view_7
#   cache_position => iota
#   eq => eq, view_5, view_6
#   index => index, view_3
#   index_1 => index_1, view_4
#   key_states_21 => permute_40
#   key_states_23 => permute_44
#   kv_arange => iota_1
#   kv_arange_1 => add_1
#   le => le, view_1
#   query_states_21 => permute_42
#   query_states_23 => permute_46
#   result_1 => bitwise_and, full_default
#   result_2 => bitwise_and_1
#   split_10 => split_10
#   split_11 => split_11
#   value_states_21 => permute_41
#   value_states_23 => permute_45
#   view_113 => view_130
#   view_114 => view_131
#   view_115 => view_132
#   view_124 => view_142
#   view_125 => view_143
#   view_126 => view_144
#   x_81 => view_129
#   x_89 => view_141
# Graph fragment:
#   %cumsum : Tensor "i64[1, 16][16, 1]cuda:0" = PlaceHolder[target=cumsum]
#   %iota : Tensor "i64[16][1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %full_default : Tensor "b8[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([16, 1], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %iota_1 : Tensor "i64[16][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add_1 : Tensor "i64[16][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota_1, 0), kwargs = {})
#   %view_1 : Tensor "i64[16, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%iota, [16, 1]), kwargs = {})
#   %le : Tensor "b8[16, 16][16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Tensor](args = (%add_1, %view_1), kwargs = {})
#   %bitwise_and : Tensor "b8[16, 16][16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%full_default, %le), kwargs = {})
#   %iota_2 : Tensor "i64[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %view_3 : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%iota_2, [1, 1]), kwargs = {})
#   %index : Tensor "i64[1, 16][16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%cumsum, [%view_3, %iota]), kwargs = {})
#   %view_5 : Tensor "i64[1, 16, 1][16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%index, [1, 16, 1]), kwargs = {})
#   %view_4 : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%iota_2, [1, 1]), kwargs = {})
#   %index_1 : Tensor "i64[1, 16][16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%cumsum, [%view_4, %add_1]), kwargs = {})
#   %view_6 : Tensor "i64[1, 1, 16][16, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%index_1, [1, 1, 16]), kwargs = {})
#   %eq : Tensor "b8[1, 16, 16][256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%view_5, %view_6), kwargs = {})
#   %bitwise_and_1 : Tensor "b8[1, 16, 16][256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and, %eq), kwargs = {})
#   %view_7 : Tensor "b8[1, 1, 16, 16][256, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bitwise_and_1, [1, 1, 16, 16]), kwargs = {})
#   %view_129 : Tensor "f32[1, 16, 2304][36864, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_40, [1, 16, 2304]), kwargs = {})
#   %split_10 : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_129, 768, 2), kwargs = {})
#   %view_132 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_112, [1, 16, -1, 64]), kwargs = {})
#   %permute_42 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_132, [0, 2, 1, 3]), kwargs = {})
#   %view_130 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_113, [1, 16, -1, 64]), kwargs = {})
#   %permute_40 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_130, [0, 2, 1, 3]), kwargs = {})
#   %view_131 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_114, [1, 16, -1, 64]), kwargs = {})
#   %permute_41 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_131, [0, 2, 1, 3]), kwargs = {})
#   %full_default_22 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_21 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_10 : Tensor "f32[1, 1, 16, 16][256, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand, %full_default_22, %full_default_21), kwargs = {})
#   %expand_11 : Tensor "f32[1, 12, 16, 16][256, 0, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%where_10, [1, 12, 16, 16]), kwargs = {})
#   %_scaled_dot_product_efficient_attention_10 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_42, %permute_40, %permute_41, %expand_11, False), kwargs = {})
#   %view_141 : Tensor "f32[1, 16, 2304][36864, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_44, [1, 16, 2304]), kwargs = {})
#   %split_11 : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_141, 768, 2), kwargs = {})
#   %view_144 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_123, [1, 16, -1, 64]), kwargs = {})
#   %permute_46 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_144, [0, 2, 1, 3]), kwargs = {})
#   %view_142 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_124, [1, 16, -1, 64]), kwargs = {})
#   %permute_44 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_142, [0, 2, 1, 3]), kwargs = {})
#   %view_143 : Tensor "f32[1, 16, 12, 64][36864, 2304, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_125, [1, 16, -1, 64]), kwargs = {})
#   %permute_45 : Tensor "f32[1, 12, 16, 64][36864, 64, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_143, [0, 2, 1, 3]), kwargs = {})
#   %full_default_24 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_23 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_11 : Tensor "f32[1, 1, 16, 16][256, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand, %full_default_24, %full_default_23), kwargs = {})
#   %expand_12 : Tensor "f32[1, 12, 16, 16][256, 0, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%where_11, [1, 12, 16, 16]), kwargs = {})
#   %_scaled_dot_product_efficient_attention_11 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_46, %permute_44, %permute_45, %expand_12, False), kwargs = {})
#   return %buf206,%buf226
triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_bitwise_and_eq_expand_index_le_scalar_tensor_split_transpose_view_where_7 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_bitwise_and_eq_expand_index_le_scalar_tensor_split_transpose_view_where_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_bitwise_and_eq_expand_index_le_scalar_tensor_split_transpose_view_where_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4224}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_bitwise_and_eq_expand_index_le_scalar_tensor_split_transpose_view_where_7(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp5 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = x1
    tmp2 = tmp0 <= tmp1
    tmp3 = tl.full([1], True, tl.int1)
    tmp4 = tmp3 & tmp2
    tmp7 = tmp5 == tmp6
    tmp8 = tmp4 & tmp7
    tmp9 = 0.0
    tmp10 = float("-inf")
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tl.store(out_ptr0 + (x2), tmp11, xmask)
    tl.store(out_ptr1 + (x2), tmp11, xmask)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-transformers-ob_ijj07/repeat_03/a1/torchinductor/tmpuqb_59nb/qx/cqxp4qkuvdyerf6x47rzmpv55xsjkmwal5kymfolzw5gb7gk6fal.py
# Topologically Sorted Source Nodes: [x_94, x_95, hidden_states_73, hidden_states_74], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states_73 => add_97
#   hidden_states_74 => add_98, add_99, mul_96, mul_97, rsqrt_24, sub_26, var_mean_24
#   x_94 => add_tensor
#   x_95 => view_151
# Graph fragment:
#   %add_92 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0" = PlaceHolder[target=add_92]
#   %mm_default : Tensor "f32[16, 768][768, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %arg145_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg145_1]
#   %getitem_133 : Tensor "f32[1, 16, 1][16, 1, 16]cuda:0" = PlaceHolder[target=getitem_133]
#   %buf242 : Tensor "f32[1, 16, 1][16, 1, 16]cuda:0" = PlaceHolder[target=buf242]
#   %arg147_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg147_1]
#   %arg148_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg148_1]
#   %add_tensor : Tensor "f32[16, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %arg145_1), kwargs = {})
#   %view_151 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor, [1, 16, 768]), kwargs = {})
#   %add_97 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_92, %view_151), kwargs = {})
#   %var_mean_24 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_97, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_26 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_97, %getitem_133), kwargs = {})
#   %add_98 : Tensor "f32[1, 16, 1][16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_132, 1e-05), kwargs = {})
#   %rsqrt_24 : Tensor "f32[1, 16, 1][16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_98,), kwargs = {})
#   %mul_96 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %rsqrt_24), kwargs = {})
#   %mul_97 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_96, %arg147_1), kwargs = {})
#   %add_99 : Tensor "f32[1, 16, 768][12288, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_97, %arg148_1), kwargs = {})
#   return %getitem_133,%buf242,%add_99
triton_per_fused_add_addmm_native_layer_norm_view_8 = async_compile.triton('triton_per_fused_add_addmm_native_layer_norm_view_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_native_layer_norm_view_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 205824}}
)
@triton.jit
def triton_per_fused_add_addmm_native_layer_norm_view_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1 = args
        args.clear()
        assert_size_stride(arg0_1, (1, 16), (16, 1))
        assert_size_stride(arg1_1, (50257, 768), (768, 1))
        assert_size_stride(arg2_1, (1024, 768), (768, 1))
        assert_size_stride(arg3_1, (768, ), (1, ))
        assert_size_stride(arg4_1, (768, ), (1, ))
        assert_size_stride(arg5_1, (2304, ), (1, ))
        assert_size_stride(arg6_1, (768, 2304), (2304, 1))
        assert_size_stride(arg7_1, (768, ), (1, ))
        assert_size_stride(arg8_1, (768, 768), (768, 1))
        assert_size_stride(arg9_1, (768, ), (1, ))
        assert_size_stride(arg10_1, (768, ), (1, ))
        assert_size_stride(arg11_1, (3072, ), (1, ))
        assert_size_stride(arg12_1, (768, 3072), (3072, 1))
        assert_size_stride(arg13_1, (768, ), (1, ))
        assert_size_stride(arg14_1, (3072, 768), (768, 1))
        assert_size_stride(arg15_1, (768, ), (1, ))
        assert_size_stride(arg16_1, (768, ), (1, ))
        assert_size_stride(arg17_1, (2304, ), (1, ))
        assert_size_stride(arg18_1, (768, 2304), (2304, 1))
        assert_size_stride(arg19_1, (768, ), (1, ))
        assert_size_stride(arg20_1, (768, 768), (768, 1))
        assert_size_stride(arg21_1, (768, ), (1, ))
        assert_size_stride(arg22_1, (768, ), (1, ))
        assert_size_stride(arg23_1, (3072, ), (1, ))
        assert_size_stride(arg24_1, (768, 3072), (3072, 1))
        assert_size_stride(arg25_1, (768, ), (1, ))
        assert_size_stride(arg26_1, (3072, 768), (768, 1))
        assert_size_stride(arg27_1, (768, ), (1, ))
        assert_size_stride(arg28_1, (768, ), (1, ))
        assert_size_stride(arg29_1, (2304, ), (1, ))
        assert_size_stride(arg30_1, (768, 2304), (2304, 1))
        assert_size_stride(arg31_1, (768, ), (1, ))
        assert_size_stride(arg32_1, (768, 768), (768, 1))
        assert_size_stride(arg33_1, (768, ), (1, ))
        assert_size_stride(arg34_1, (768, ), (1, ))
        assert_size_stride(arg35_1, (3072, ), (1, ))
        assert_size_stride(arg36_1, (768, 3072), (3072, 1))
        assert_size_stride(arg37_1, (768, ), (1, ))
        assert_size_stride(arg38_1, (3072, 768), (768, 1))
        assert_size_stride(arg39_1, (768, ), (1, ))
        assert_size_stride(arg40_1, (768, ), (1, ))
        assert_size_stride(arg41_1, (2304, ), (1, ))
        assert_size_stride(arg42_1, (768, 2304), (2304, 1))
        assert_size_stride(arg43_1, (768, ), (1, ))
        assert_size_stride(arg44_1, (768, 768), (768, 1))
        assert_size_stride(arg45_1, (768, ), (1, ))
        assert_size_stride(arg46_1, (768, ), (1, ))
        assert_size_stride(arg47_1, (3072, ), (1, ))
        assert_size_stride(arg48_1, (768, 3072), (3072, 1))
        assert_size_stride(arg49_1, (768, ), (1, ))
        assert_size_stride(arg50_1, (3072, 768), (768, 1))
        assert_size_stride(arg51_1, (768, ), (1, ))
        assert_size_stride(arg52_1, (768, ), (1, ))
        assert_size_stride(arg53_1, (2304, ), (1, ))
        assert_size_stride(arg54_1, (768, 2304), (2304, 1))
        assert_size_stride(arg55_1, (768, ), (1, ))
        assert_size_stride(arg56_1, (768, 768), (768, 1))
        assert_size_stride(arg57_1, (768, ), (1, ))
        assert_size_stride(arg58_1, (768, ), (1, ))
        assert_size_stride(arg59_1, (3072, ), (1, ))
        assert_size_stride(arg60_1, (768, 3072), (3072, 1))
        assert_size_stride(arg61_1, (768, ), (1, ))
        assert_size_stride(arg62_1, (3072, 768), (768, 1))
        assert_size_stride(arg63_1, (768, ), (1, ))
        assert_size_stride(arg64_1, (768, ), (1, ))
        assert_size_stride(arg65_1, (2304, ), (1, ))
        assert_size_stride(arg66_1, (768, 2304), (2304, 1))
        assert_size_stride(arg67_1, (768, ), (1, ))
        assert_size_stride(arg68_1, (768, 768), (768, 1))
        assert_size_stride(arg69_1, (768, ), (1, ))
        assert_size_stride(arg70_1, (768, ), (1, ))
        assert_size_stride(arg71_1, (3072, ), (1, ))
        assert_size_stride(arg72_1, (768, 3072), (3072, 1))
        assert_size_stride(arg73_1, (768, ), (1, ))
        assert_size_stride(arg74_1, (3072, 768), (768, 1))
        assert_size_stride(arg75_1, (768, ), (1, ))
        assert_size_stride(arg76_1, (768, ), (1, ))
        assert_size_stride(arg77_1, (2304, ), (1, ))
        assert_size_stride(arg78_1, (768, 2304), (2304, 1))
        assert_size_stride(arg79_1, (768, ), (1, ))
        assert_size_stride(arg80_1, (768, 768), (768, 1))
        assert_size_stride(arg81_1, (768, ), (1, ))
        assert_size_stride(arg82_1, (768, ), (1, ))
        assert_size_stride(arg83_1, (3072, ), (1, ))
        assert_size_stride(arg84_1, (768, 3072), (3072, 1))
        assert_size_stride(arg85_1, (768, ), (1, ))
        assert_size_stride(arg86_1, (3072, 768), (768, 1))
        assert_size_stride(arg87_1, (768, ), (1, ))
        assert_size_stride(arg88_1, (768, ), (1, ))
        assert_size_stride(arg89_1, (2304, ), (1, ))
        assert_size_stride(arg90_1, (768, 2304), (2304, 1))
        assert_size_stride(arg91_1, (768, ), (1, ))
        assert_size_stride(arg92_1, (768, 768), (768, 1))
        assert_size_stride(arg93_1, (768, ), (1, ))
        assert_size_stride(arg94_1, (768, ), (1, ))
        assert_size_stride(arg95_1, (3072, ), (1, ))
        assert_size_stride(arg96_1, (768, 3072), (3072, 1))
        assert_size_stride(arg97_1, (768, ), (1, ))
        assert_size_stride(arg98_1, (3072, 768), (768, 1))
        assert_size_stride(arg99_1, (768, ), (1, ))
        assert_size_stride(arg100_1, (768, ), (1, ))
        assert_size_stride(arg101_1, (2304, ), (1, ))
        assert_size_stride(arg102_1, (768, 2304), (2304, 1))
        assert_size_stride(arg103_1, (768, ), (1, ))
        assert_size_stride(arg104_1, (768, 768), (768, 1))
        assert_size_stride(arg105_1, (768, ), (1, ))
        assert_size_stride(arg106_1, (768, ), (1, ))
        assert_size_stride(arg107_1, (3072, ), (1, ))
        assert_size_stride(arg108_1, (768, 3072), (3072, 1))
        assert_size_stride(arg109_1, (768, ), (1, ))
        assert_size_stride(arg110_1, (3072, 768), (768, 1))
        assert_size_stride(arg111_1, (768, ), (1, ))
        assert_size_stride(arg112_1, (768, ), (1, ))
        assert_size_stride(arg113_1, (2304, ), (1, ))
        assert_size_stride(arg114_1, (768, 2304), (2304, 1))
        assert_size_stride(arg115_1, (768, ), (1, ))
        assert_size_stride(arg116_1, (768, 768), (768, 1))
        assert_size_stride(arg117_1, (768, ), (1, ))
        assert_size_stride(arg118_1, (768, ), (1, ))
        assert_size_stride(arg119_1, (3072, ), (1, ))
        assert_size_stride(arg120_1, (768, 3072), (3072, 1))
        assert_size_stride(arg121_1, (768, ), (1, ))
        assert_size_stride(arg122_1, (3072, 768), (768, 1))
        assert_size_stride(arg123_1, (768, ), (1, ))
        assert_size_stride(arg124_1, (768, ), (1, ))
        assert_size_stride(arg125_1, (2304, ), (1, ))
        assert_size_stride(arg126_1, (768, 2304), (2304, 1))
        assert_size_stride(arg127_1, (768, ), (1, ))
        assert_size_stride(arg128_1, (768, 768), (768, 1))
        assert_size_stride(arg129_1, (768, ), (1, ))
        assert_size_stride(arg130_1, (768, ), (1, ))
        assert_size_stride(arg131_1, (3072, ), (1, ))
        assert_size_stride(arg132_1, (768, 3072), (3072, 1))
        assert_size_stride(arg133_1, (768, ), (1, ))
        assert_size_stride(arg134_1, (3072, 768), (768, 1))
        assert_size_stride(arg135_1, (768, ), (1, ))
        assert_size_stride(arg136_1, (768, ), (1, ))
        assert_size_stride(arg137_1, (2304, ), (1, ))
        assert_size_stride(arg138_1, (768, 2304), (2304, 1))
        assert_size_stride(arg139_1, (768, ), (1, ))
        assert_size_stride(arg140_1, (768, 768), (768, 1))
        assert_size_stride(arg141_1, (768, ), (1, ))
        assert_size_stride(arg142_1, (768, ), (1, ))
        assert_size_stride(arg143_1, (3072, ), (1, ))
        assert_size_stride(arg144_1, (768, 3072), (3072, 1))
        assert_size_stride(arg145_1, (768, ), (1, ))
        assert_size_stride(arg146_1, (3072, 768), (768, 1))
        assert_size_stride(arg147_1, (768, ), (1, ))
        assert_size_stride(arg148_1, (768, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf3 = empty_strided_cuda((1, 16, 768), (12288, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [inputs_embeds, cache_position, position_ids, position_embeds, hidden_states, hidden_states_2], Original ATen: [aten.embedding, aten.arange, aten.unsqueeze, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_red_fused_add_arange_embedding_native_layer_norm_unsqueeze_0.run(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, buf3, 16, 768, stream=stream0)
            del arg3_1
            del arg4_1
            buf4 = empty_strided_cuda((16, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [inputs_embeds, cache_position, position_ids, position_embeds, hidden_states, hidden_states_2, view_1, x], Original ATen: [aten.embedding, aten.arange, aten.unsqueeze, aten.add, aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.addmm(arg5_1, reinterpret_tensor(buf3, (16, 768), (768, 1), 0), arg6_1, alpha=1, beta=1, out=buf4)
            del arg5_1
            del arg6_1
            buf5 = empty_strided_cuda((1, 16), (16, 1), torch.int64)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem, first_dummy_value, position_diff, ne, packed_sequence_mask], Original ATen: [aten.arange, aten.unsqueeze, aten.slice, aten.sub, aten.cat, aten.ne, aten.cumsum]
            stream0 = get_raw_stream(0)
            triton_per_fused_arange_cat_cumsum_ne_slice_sub_unsqueeze_1.run(buf5, 1, 16, stream=stream0)
            buf6 = empty_strided_cuda((1, 1, 16, 16), (256, 0, 16, 1), torch.float32)
            buf26 = empty_strided_cuda((1, 1, 16, 16), (256, 0, 16, 1), torch.float32)
            buf46 = empty_strided_cuda((1, 1, 16, 16), (256, 0, 16, 1), torch.float32)
            buf66 = empty_strided_cuda((1, 1, 16, 16), (256, 0, 16, 1), torch.float32)
            buf86 = empty_strided_cuda((1, 1, 16, 16), (256, 0, 16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [cache_position, x_1, split, view_5, query_states_1, view_3, key_states_1, view_4, value_states_1, result_1, kv_arange, kv_arange_1, le, batch_arange, index, eq, index_1, result_2, batched_outputs_2, attn_output, x_9, split_1, view_16, query_states_3, view_14, key_states_3, view_15, value_states_3, attn_output_4, x_17, split_2, view_27, query_states_5, view_25, key_states_5, view_26, value_states_5, attn_output_8, x_25, split_3, view_38, query_states_7, view_36, key_states_7, view_37, value_states_7, attn_output_12, x_33, split_4, view_49, query_states_9, view_47, key_states_9, view_48, value_states_9, attn_output_16], Original ATen: [aten.arange, aten.view, aten.split, aten.transpose, aten.add, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_bitwise_and_eq_expand_index_le_scalar_tensor_split_transpose_view_where_2.run(buf5, buf6, buf26, buf46, buf66, buf86, 256, stream=stream0)
            # Topologically Sorted Source Nodes: [cache_position, x_1, split, view_5, query_states_1, view_3, key_states_1, view_4, value_states_1, result_1, kv_arange, kv_arange_1, le, batch_arange, index, eq, index_1, result_2, batched_outputs_2, attn_output], Original ATen: [aten.arange, aten.view, aten.split, aten.transpose, aten.add, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf4, (1, 12, 16, 64), (0, 64, 2304, 1), 0), reinterpret_tensor(buf4, (1, 12, 16, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf4, (1, 12, 16, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf6, (1, 12, 16, 16), (256, 0, 16, 1), 0), False)
            del buf4
            del buf6
            buf8 = buf7[0]
            assert_size_stride(buf8, (1, 12, 16, 64), (12288, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf8, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf7
            buf12 = reinterpret_tensor(buf3, (16, 768), (768, 1), 0); del buf3  # reuse
            # Topologically Sorted Source Nodes: [transpose_3, reshape, view_6, x_2], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf8, (16, 768), (768, 1), 0), arg8_1, out=buf12)
            del arg8_1
            buf13 = reinterpret_tensor(buf12, (1, 16, 768), (12288, 768, 1), 0); del buf12  # reuse
            buf17 = reinterpret_tensor(buf8, (1, 16, 768), (12288, 768, 1), 0); del buf8  # reuse
            # Topologically Sorted Source Nodes: [inputs_embeds, cache_position, position_ids, position_embeds, hidden_states, x_2, x_3, hidden_states_3, hidden_states_4], Original ATen: [aten.embedding, aten.arange, aten.unsqueeze, aten.add, aten.addmm, aten.view, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_arange_embedding_native_layer_norm_unsqueeze_view_3.run(buf13, arg7_1, arg0_1, arg1_1, arg2_1, arg9_1, arg10_1, buf17, 16, 768, stream=stream0)
            del arg0_1
            del arg10_1
            del arg1_1
            del arg2_1
            del arg7_1
            del arg9_1
            buf18 = empty_strided_cuda((16, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_4, view_8, x_4], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf17, (16, 768), (768, 1), 0), arg12_1, out=buf18)
            del arg12_1
            del buf17
            buf19 = reinterpret_tensor(buf18, (1, 16, 3072), (49152, 3072, 1), 0); del buf18  # reuse
            # Topologically Sorted Source Nodes: [x_4, x_5, mul, pow_1, mul_1, add_2, mul_2, tanh, add_3, hidden_states_5], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_4.run(buf19, arg11_1, 49152, stream=stream0)
            del arg11_1
            buf20 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_4, x_5, mul, pow_1, mul_1, add_2, mul_2, tanh, add_3, hidden_states_5, view_10, x_6], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf19, (16, 3072), (3072, 1), 0), arg14_1, out=buf20)
            del arg14_1
            del buf19
            buf24 = empty_strided_cuda((1, 16, 768), (12288, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_6, x_7, hidden_states_7, hidden_states_8], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_5.run(buf13, buf20, arg13_1, arg15_1, arg16_1, buf24, 16, 768, stream=stream0)
            del arg15_1
            del arg16_1
            buf25 = empty_strided_cuda((16, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_6, x_7, hidden_states_7, hidden_states_8, view_12, x_8], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg17_1, reinterpret_tensor(buf24, (16, 768), (768, 1), 0), arg18_1, alpha=1, beta=1, out=buf25)
            del arg17_1
            del arg18_1
            # Topologically Sorted Source Nodes: [cache_position, result_1, kv_arange, kv_arange_1, le, batch_arange, index, eq, index_1, result_2, batched_outputs_2, x_9, split_1, view_16, query_states_3, view_14, key_states_3, view_15, value_states_3, attn_output_4], Original ATen: [aten.arange, aten.view, aten.add, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.split, aten.transpose, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf27 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf25, (1, 12, 16, 64), (0, 64, 2304, 1), 0), reinterpret_tensor(buf25, (1, 12, 16, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf25, (1, 12, 16, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf26, (1, 12, 16, 16), (256, 0, 16, 1), 0), False)
            del buf25
            buf28 = buf27[0]
            assert_size_stride(buf28, (1, 12, 16, 64), (12288, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf28, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf27
            buf32 = reinterpret_tensor(buf24, (16, 768), (768, 1), 0); del buf24  # reuse
            # Topologically Sorted Source Nodes: [transpose_7, reshape_1, view_17, x_10], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf28, (16, 768), (768, 1), 0), arg20_1, out=buf32)
            del arg20_1
            buf33 = reinterpret_tensor(buf32, (1, 16, 768), (12288, 768, 1), 0); del buf32  # reuse
            buf37 = reinterpret_tensor(buf28, (1, 16, 768), (12288, 768, 1), 0); del buf28  # reuse
            # Topologically Sorted Source Nodes: [x_6, x_7, hidden_states_7, x_10, x_11, hidden_states_9, hidden_states_10], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_6.run(buf33, arg19_1, buf13, buf20, arg13_1, arg21_1, arg22_1, buf37, 16, 768, stream=stream0)
            del arg13_1
            del arg19_1
            del arg21_1
            del arg22_1
            del buf13
            del buf20
            buf38 = empty_strided_cuda((16, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_10, view_19, x_12], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf37, (16, 768), (768, 1), 0), arg24_1, out=buf38)
            del arg24_1
            del buf37
            buf39 = reinterpret_tensor(buf38, (1, 16, 3072), (49152, 3072, 1), 0); del buf38  # reuse
            # Topologically Sorted Source Nodes: [x_12, x_13, mul_4, pow_2, mul_5, add_6, mul_6, tanh_1, add_7, hidden_states_11], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_4.run(buf39, arg23_1, 49152, stream=stream0)
            del arg23_1
            buf40 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_12, x_13, mul_4, pow_2, mul_5, add_6, mul_6, tanh_1, add_7, hidden_states_11, view_21, x_14], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf39, (16, 3072), (3072, 1), 0), arg26_1, out=buf40)
            del arg26_1
            del buf39
            buf44 = empty_strided_cuda((1, 16, 768), (12288, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_14, x_15, hidden_states_13, hidden_states_14], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_5.run(buf33, buf40, arg25_1, arg27_1, arg28_1, buf44, 16, 768, stream=stream0)
            del arg27_1
            del arg28_1
            buf45 = empty_strided_cuda((16, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_14, x_15, hidden_states_13, hidden_states_14, view_23, x_16], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg29_1, reinterpret_tensor(buf44, (16, 768), (768, 1), 0), arg30_1, alpha=1, beta=1, out=buf45)
            del arg29_1
            del arg30_1
            # Topologically Sorted Source Nodes: [cache_position, result_1, kv_arange, kv_arange_1, le, batch_arange, index, eq, index_1, result_2, batched_outputs_2, x_17, split_2, view_27, query_states_5, view_25, key_states_5, view_26, value_states_5, attn_output_8], Original ATen: [aten.arange, aten.view, aten.add, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.split, aten.transpose, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf47 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf45, (1, 12, 16, 64), (0, 64, 2304, 1), 0), reinterpret_tensor(buf45, (1, 12, 16, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf45, (1, 12, 16, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf46, (1, 12, 16, 16), (256, 0, 16, 1), 0), False)
            del buf45
            buf48 = buf47[0]
            assert_size_stride(buf48, (1, 12, 16, 64), (12288, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf48, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf47
            buf52 = reinterpret_tensor(buf44, (16, 768), (768, 1), 0); del buf44  # reuse
            # Topologically Sorted Source Nodes: [transpose_11, reshape_2, view_28, x_18], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf48, (16, 768), (768, 1), 0), arg32_1, out=buf52)
            del arg32_1
            buf53 = reinterpret_tensor(buf52, (1, 16, 768), (12288, 768, 1), 0); del buf52  # reuse
            buf57 = reinterpret_tensor(buf48, (1, 16, 768), (12288, 768, 1), 0); del buf48  # reuse
            # Topologically Sorted Source Nodes: [x_14, x_15, hidden_states_13, x_18, x_19, hidden_states_15, hidden_states_16], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_6.run(buf53, arg31_1, buf33, buf40, arg25_1, arg33_1, arg34_1, buf57, 16, 768, stream=stream0)
            del arg25_1
            del arg31_1
            del arg33_1
            del arg34_1
            del buf33
            del buf40
            buf58 = empty_strided_cuda((16, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_16, view_30, x_20], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf57, (16, 768), (768, 1), 0), arg36_1, out=buf58)
            del arg36_1
            del buf57
            buf59 = reinterpret_tensor(buf58, (1, 16, 3072), (49152, 3072, 1), 0); del buf58  # reuse
            # Topologically Sorted Source Nodes: [x_20, x_21, mul_8, pow_3, mul_9, add_10, mul_10, tanh_2, add_11, hidden_states_17], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_4.run(buf59, arg35_1, 49152, stream=stream0)
            del arg35_1
            buf60 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_20, x_21, mul_8, pow_3, mul_9, add_10, mul_10, tanh_2, add_11, hidden_states_17, view_32, x_22], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf59, (16, 3072), (3072, 1), 0), arg38_1, out=buf60)
            del arg38_1
            del buf59
            buf64 = empty_strided_cuda((1, 16, 768), (12288, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_22, x_23, hidden_states_19, hidden_states_20], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_5.run(buf53, buf60, arg37_1, arg39_1, arg40_1, buf64, 16, 768, stream=stream0)
            del arg39_1
            del arg40_1
            buf65 = empty_strided_cuda((16, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_22, x_23, hidden_states_19, hidden_states_20, view_34, x_24], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg41_1, reinterpret_tensor(buf64, (16, 768), (768, 1), 0), arg42_1, alpha=1, beta=1, out=buf65)
            del arg41_1
            del arg42_1
            # Topologically Sorted Source Nodes: [cache_position, result_1, kv_arange, kv_arange_1, le, batch_arange, index, eq, index_1, result_2, batched_outputs_2, x_25, split_3, view_38, query_states_7, view_36, key_states_7, view_37, value_states_7, attn_output_12], Original ATen: [aten.arange, aten.view, aten.add, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.split, aten.transpose, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf67 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf65, (1, 12, 16, 64), (0, 64, 2304, 1), 0), reinterpret_tensor(buf65, (1, 12, 16, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf65, (1, 12, 16, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf66, (1, 12, 16, 16), (256, 0, 16, 1), 0), False)
            del buf65
            buf68 = buf67[0]
            assert_size_stride(buf68, (1, 12, 16, 64), (12288, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf68, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf67
            buf72 = reinterpret_tensor(buf64, (16, 768), (768, 1), 0); del buf64  # reuse
            # Topologically Sorted Source Nodes: [transpose_15, reshape_3, view_39, x_26], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf68, (16, 768), (768, 1), 0), arg44_1, out=buf72)
            del arg44_1
            buf73 = reinterpret_tensor(buf72, (1, 16, 768), (12288, 768, 1), 0); del buf72  # reuse
            buf77 = reinterpret_tensor(buf68, (1, 16, 768), (12288, 768, 1), 0); del buf68  # reuse
            # Topologically Sorted Source Nodes: [x_22, x_23, hidden_states_19, x_26, x_27, hidden_states_21, hidden_states_22], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_6.run(buf73, arg43_1, buf53, buf60, arg37_1, arg45_1, arg46_1, buf77, 16, 768, stream=stream0)
            del arg37_1
            del arg43_1
            del arg45_1
            del arg46_1
            del buf53
            del buf60
            buf78 = empty_strided_cuda((16, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_22, view_41, x_28], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf77, (16, 768), (768, 1), 0), arg48_1, out=buf78)
            del arg48_1
            del buf77
            buf79 = reinterpret_tensor(buf78, (1, 16, 3072), (49152, 3072, 1), 0); del buf78  # reuse
            # Topologically Sorted Source Nodes: [x_28, x_29, mul_12, pow_4, mul_13, add_14, mul_14, tanh_3, add_15, hidden_states_23], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_4.run(buf79, arg47_1, 49152, stream=stream0)
            del arg47_1
            buf80 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_28, x_29, mul_12, pow_4, mul_13, add_14, mul_14, tanh_3, add_15, hidden_states_23, view_43, x_30], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf79, (16, 3072), (3072, 1), 0), arg50_1, out=buf80)
            del arg50_1
            del buf79
            buf84 = empty_strided_cuda((1, 16, 768), (12288, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_30, x_31, hidden_states_25, hidden_states_26], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_5.run(buf73, buf80, arg49_1, arg51_1, arg52_1, buf84, 16, 768, stream=stream0)
            del arg51_1
            del arg52_1
            buf85 = empty_strided_cuda((16, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_30, x_31, hidden_states_25, hidden_states_26, view_45, x_32], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg53_1, reinterpret_tensor(buf84, (16, 768), (768, 1), 0), arg54_1, alpha=1, beta=1, out=buf85)
            del arg53_1
            del arg54_1
            # Topologically Sorted Source Nodes: [cache_position, result_1, kv_arange, kv_arange_1, le, batch_arange, index, eq, index_1, result_2, batched_outputs_2, x_33, split_4, view_49, query_states_9, view_47, key_states_9, view_48, value_states_9, attn_output_16], Original ATen: [aten.arange, aten.view, aten.add, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.split, aten.transpose, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf87 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf85, (1, 12, 16, 64), (0, 64, 2304, 1), 0), reinterpret_tensor(buf85, (1, 12, 16, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf85, (1, 12, 16, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf86, (1, 12, 16, 16), (256, 0, 16, 1), 0), False)
            del buf85
            buf88 = buf87[0]
            assert_size_stride(buf88, (1, 12, 16, 64), (12288, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf88, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf87
            buf92 = reinterpret_tensor(buf84, (16, 768), (768, 1), 0); del buf84  # reuse
            # Topologically Sorted Source Nodes: [transpose_19, reshape_4, view_50, x_34], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf88, (16, 768), (768, 1), 0), arg56_1, out=buf92)
            del arg56_1
            buf93 = reinterpret_tensor(buf92, (1, 16, 768), (12288, 768, 1), 0); del buf92  # reuse
            buf97 = reinterpret_tensor(buf88, (1, 16, 768), (12288, 768, 1), 0); del buf88  # reuse
            # Topologically Sorted Source Nodes: [x_30, x_31, hidden_states_25, x_34, x_35, hidden_states_27, hidden_states_28], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_6.run(buf93, arg55_1, buf73, buf80, arg49_1, arg57_1, arg58_1, buf97, 16, 768, stream=stream0)
            del arg49_1
            del arg55_1
            del arg57_1
            del arg58_1
            del buf73
            del buf80
            buf98 = empty_strided_cuda((16, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_28, view_52, x_36], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf97, (16, 768), (768, 1), 0), arg60_1, out=buf98)
            del arg60_1
            del buf97
            buf99 = reinterpret_tensor(buf98, (1, 16, 3072), (49152, 3072, 1), 0); del buf98  # reuse
            # Topologically Sorted Source Nodes: [x_36, x_37, mul_16, pow_5, mul_17, add_18, mul_18, tanh_4, add_19, hidden_states_29], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_4.run(buf99, arg59_1, 49152, stream=stream0)
            del arg59_1
            buf100 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_36, x_37, mul_16, pow_5, mul_17, add_18, mul_18, tanh_4, add_19, hidden_states_29, view_54, x_38], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf99, (16, 3072), (3072, 1), 0), arg62_1, out=buf100)
            del arg62_1
            del buf99
            buf104 = empty_strided_cuda((1, 16, 768), (12288, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_38, x_39, hidden_states_31, hidden_states_32], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_5.run(buf93, buf100, arg61_1, arg63_1, arg64_1, buf104, 16, 768, stream=stream0)
            del arg63_1
            del arg64_1
            buf105 = empty_strided_cuda((16, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_38, x_39, hidden_states_31, hidden_states_32, view_56, x_40], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg65_1, reinterpret_tensor(buf104, (16, 768), (768, 1), 0), arg66_1, alpha=1, beta=1, out=buf105)
            del arg65_1
            del arg66_1
            buf106 = buf86; del buf86  # reuse
            buf126 = buf66; del buf66  # reuse
            buf146 = buf46; del buf46  # reuse
            buf166 = buf26; del buf26  # reuse
            buf186 = empty_strided_cuda((1, 1, 16, 16), (256, 0, 16, 1), torch.float32)
            # Topologically Sorted Source Nodes: [cache_position, result_1, kv_arange, kv_arange_1, le, batch_arange, index, eq, index_1, result_2, batched_outputs_2, x_41, split_5, view_60, query_states_11, view_58, key_states_11, view_59, value_states_11, attn_output_20, x_49, split_6, view_71, query_states_13, view_69, key_states_13, view_70, value_states_13, attn_output_24, x_57, split_7, view_82, query_states_15, view_80, key_states_15, view_81, value_states_15, attn_output_28, x_65, split_8, view_93, query_states_17, view_91, key_states_17, view_92, value_states_17, attn_output_32, x_73, split_9, view_104, query_states_19, view_102, key_states_19, view_103, value_states_19, attn_output_36], Original ATen: [aten.arange, aten.view, aten.add, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.split, aten.transpose, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_bitwise_and_eq_expand_index_le_scalar_tensor_split_transpose_view_where_2.run(buf5, buf106, buf126, buf146, buf166, buf186, 256, stream=stream0)
            # Topologically Sorted Source Nodes: [cache_position, result_1, kv_arange, kv_arange_1, le, batch_arange, index, eq, index_1, result_2, batched_outputs_2, x_41, split_5, view_60, query_states_11, view_58, key_states_11, view_59, value_states_11, attn_output_20], Original ATen: [aten.arange, aten.view, aten.add, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.split, aten.transpose, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf107 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf105, (1, 12, 16, 64), (0, 64, 2304, 1), 0), reinterpret_tensor(buf105, (1, 12, 16, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf105, (1, 12, 16, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf106, (1, 12, 16, 16), (256, 0, 16, 1), 0), False)
            del buf105
            del buf106
            buf108 = buf107[0]
            assert_size_stride(buf108, (1, 12, 16, 64), (12288, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf108, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf107
            buf112 = reinterpret_tensor(buf104, (16, 768), (768, 1), 0); del buf104  # reuse
            # Topologically Sorted Source Nodes: [transpose_23, reshape_5, view_61, x_42], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf108, (16, 768), (768, 1), 0), arg68_1, out=buf112)
            del arg68_1
            buf113 = reinterpret_tensor(buf112, (1, 16, 768), (12288, 768, 1), 0); del buf112  # reuse
            buf117 = reinterpret_tensor(buf108, (1, 16, 768), (12288, 768, 1), 0); del buf108  # reuse
            # Topologically Sorted Source Nodes: [x_38, x_39, hidden_states_31, x_42, x_43, hidden_states_33, hidden_states_34], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_6.run(buf113, arg67_1, buf93, buf100, arg61_1, arg69_1, arg70_1, buf117, 16, 768, stream=stream0)
            del arg61_1
            del arg67_1
            del arg69_1
            del arg70_1
            del buf100
            del buf93
            buf118 = empty_strided_cuda((16, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_34, view_63, x_44], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf117, (16, 768), (768, 1), 0), arg72_1, out=buf118)
            del arg72_1
            del buf117
            buf119 = reinterpret_tensor(buf118, (1, 16, 3072), (49152, 3072, 1), 0); del buf118  # reuse
            # Topologically Sorted Source Nodes: [x_44, x_45, mul_20, pow_6, mul_21, add_22, mul_22, tanh_5, add_23, hidden_states_35], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_4.run(buf119, arg71_1, 49152, stream=stream0)
            del arg71_1
            buf120 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_44, x_45, mul_20, pow_6, mul_21, add_22, mul_22, tanh_5, add_23, hidden_states_35, view_65, x_46], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf119, (16, 3072), (3072, 1), 0), arg74_1, out=buf120)
            del arg74_1
            del buf119
            buf124 = empty_strided_cuda((1, 16, 768), (12288, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_46, x_47, hidden_states_37, hidden_states_38], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_5.run(buf113, buf120, arg73_1, arg75_1, arg76_1, buf124, 16, 768, stream=stream0)
            del arg75_1
            del arg76_1
            buf125 = empty_strided_cuda((16, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_46, x_47, hidden_states_37, hidden_states_38, view_67, x_48], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg77_1, reinterpret_tensor(buf124, (16, 768), (768, 1), 0), arg78_1, alpha=1, beta=1, out=buf125)
            del arg77_1
            del arg78_1
            # Topologically Sorted Source Nodes: [cache_position, result_1, kv_arange, kv_arange_1, le, batch_arange, index, eq, index_1, result_2, batched_outputs_2, x_49, split_6, view_71, query_states_13, view_69, key_states_13, view_70, value_states_13, attn_output_24], Original ATen: [aten.arange, aten.view, aten.add, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.split, aten.transpose, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf127 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf125, (1, 12, 16, 64), (0, 64, 2304, 1), 0), reinterpret_tensor(buf125, (1, 12, 16, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf125, (1, 12, 16, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf126, (1, 12, 16, 16), (256, 0, 16, 1), 0), False)
            del buf125
            del buf126
            buf128 = buf127[0]
            assert_size_stride(buf128, (1, 12, 16, 64), (12288, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf128, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf127
            buf132 = reinterpret_tensor(buf124, (16, 768), (768, 1), 0); del buf124  # reuse
            # Topologically Sorted Source Nodes: [transpose_27, reshape_6, view_72, x_50], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf128, (16, 768), (768, 1), 0), arg80_1, out=buf132)
            del arg80_1
            buf133 = reinterpret_tensor(buf132, (1, 16, 768), (12288, 768, 1), 0); del buf132  # reuse
            buf137 = reinterpret_tensor(buf128, (1, 16, 768), (12288, 768, 1), 0); del buf128  # reuse
            # Topologically Sorted Source Nodes: [x_46, x_47, hidden_states_37, x_50, x_51, hidden_states_39, hidden_states_40], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_6.run(buf133, arg79_1, buf113, buf120, arg73_1, arg81_1, arg82_1, buf137, 16, 768, stream=stream0)
            del arg73_1
            del arg79_1
            del arg81_1
            del arg82_1
            del buf113
            del buf120
            buf138 = empty_strided_cuda((16, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_40, view_74, x_52], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf137, (16, 768), (768, 1), 0), arg84_1, out=buf138)
            del arg84_1
            del buf137
            buf139 = reinterpret_tensor(buf138, (1, 16, 3072), (49152, 3072, 1), 0); del buf138  # reuse
            # Topologically Sorted Source Nodes: [x_52, x_53, mul_24, pow_7, mul_25, add_26, mul_26, tanh_6, add_27, hidden_states_41], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_4.run(buf139, arg83_1, 49152, stream=stream0)
            del arg83_1
            buf140 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_52, x_53, mul_24, pow_7, mul_25, add_26, mul_26, tanh_6, add_27, hidden_states_41, view_76, x_54], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf139, (16, 3072), (3072, 1), 0), arg86_1, out=buf140)
            del arg86_1
            del buf139
            buf144 = empty_strided_cuda((1, 16, 768), (12288, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_54, x_55, hidden_states_43, hidden_states_44], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_5.run(buf133, buf140, arg85_1, arg87_1, arg88_1, buf144, 16, 768, stream=stream0)
            del arg87_1
            del arg88_1
            buf145 = empty_strided_cuda((16, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_54, x_55, hidden_states_43, hidden_states_44, view_78, x_56], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg89_1, reinterpret_tensor(buf144, (16, 768), (768, 1), 0), arg90_1, alpha=1, beta=1, out=buf145)
            del arg89_1
            del arg90_1
            # Topologically Sorted Source Nodes: [cache_position, result_1, kv_arange, kv_arange_1, le, batch_arange, index, eq, index_1, result_2, batched_outputs_2, x_57, split_7, view_82, query_states_15, view_80, key_states_15, view_81, value_states_15, attn_output_28], Original ATen: [aten.arange, aten.view, aten.add, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.split, aten.transpose, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf147 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf145, (1, 12, 16, 64), (0, 64, 2304, 1), 0), reinterpret_tensor(buf145, (1, 12, 16, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf145, (1, 12, 16, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf146, (1, 12, 16, 16), (256, 0, 16, 1), 0), False)
            del buf145
            del buf146
            buf148 = buf147[0]
            assert_size_stride(buf148, (1, 12, 16, 64), (12288, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf148, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf147
            buf152 = reinterpret_tensor(buf144, (16, 768), (768, 1), 0); del buf144  # reuse
            # Topologically Sorted Source Nodes: [transpose_31, reshape_7, view_83, x_58], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf148, (16, 768), (768, 1), 0), arg92_1, out=buf152)
            del arg92_1
            buf153 = reinterpret_tensor(buf152, (1, 16, 768), (12288, 768, 1), 0); del buf152  # reuse
            buf157 = reinterpret_tensor(buf148, (1, 16, 768), (12288, 768, 1), 0); del buf148  # reuse
            # Topologically Sorted Source Nodes: [x_54, x_55, hidden_states_43, x_58, x_59, hidden_states_45, hidden_states_46], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_6.run(buf153, arg91_1, buf133, buf140, arg85_1, arg93_1, arg94_1, buf157, 16, 768, stream=stream0)
            del arg85_1
            del arg91_1
            del arg93_1
            del arg94_1
            del buf133
            del buf140
            buf158 = empty_strided_cuda((16, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_46, view_85, x_60], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf157, (16, 768), (768, 1), 0), arg96_1, out=buf158)
            del arg96_1
            del buf157
            buf159 = reinterpret_tensor(buf158, (1, 16, 3072), (49152, 3072, 1), 0); del buf158  # reuse
            # Topologically Sorted Source Nodes: [x_60, x_61, mul_28, pow_8, mul_29, add_30, mul_30, tanh_7, add_31, hidden_states_47], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_4.run(buf159, arg95_1, 49152, stream=stream0)
            del arg95_1
            buf160 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_60, x_61, mul_28, pow_8, mul_29, add_30, mul_30, tanh_7, add_31, hidden_states_47, view_87, x_62], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf159, (16, 3072), (3072, 1), 0), arg98_1, out=buf160)
            del arg98_1
            del buf159
            buf164 = empty_strided_cuda((1, 16, 768), (12288, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_62, x_63, hidden_states_49, hidden_states_50], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_5.run(buf153, buf160, arg97_1, arg99_1, arg100_1, buf164, 16, 768, stream=stream0)
            del arg100_1
            del arg99_1
            buf165 = empty_strided_cuda((16, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_62, x_63, hidden_states_49, hidden_states_50, view_89, x_64], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg101_1, reinterpret_tensor(buf164, (16, 768), (768, 1), 0), arg102_1, alpha=1, beta=1, out=buf165)
            del arg101_1
            del arg102_1
            # Topologically Sorted Source Nodes: [cache_position, result_1, kv_arange, kv_arange_1, le, batch_arange, index, eq, index_1, result_2, batched_outputs_2, x_65, split_8, view_93, query_states_17, view_91, key_states_17, view_92, value_states_17, attn_output_32], Original ATen: [aten.arange, aten.view, aten.add, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.split, aten.transpose, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf167 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf165, (1, 12, 16, 64), (0, 64, 2304, 1), 0), reinterpret_tensor(buf165, (1, 12, 16, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf165, (1, 12, 16, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf166, (1, 12, 16, 16), (256, 0, 16, 1), 0), False)
            del buf165
            buf168 = buf167[0]
            assert_size_stride(buf168, (1, 12, 16, 64), (12288, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf168, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf167
            buf172 = reinterpret_tensor(buf164, (16, 768), (768, 1), 0); del buf164  # reuse
            # Topologically Sorted Source Nodes: [transpose_35, reshape_8, view_94, x_66], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf168, (16, 768), (768, 1), 0), arg104_1, out=buf172)
            del arg104_1
            buf173 = reinterpret_tensor(buf172, (1, 16, 768), (12288, 768, 1), 0); del buf172  # reuse
            buf177 = reinterpret_tensor(buf168, (1, 16, 768), (12288, 768, 1), 0); del buf168  # reuse
            # Topologically Sorted Source Nodes: [x_62, x_63, hidden_states_49, x_66, x_67, hidden_states_51, hidden_states_52], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_6.run(buf173, arg103_1, buf153, buf160, arg97_1, arg105_1, arg106_1, buf177, 16, 768, stream=stream0)
            del arg103_1
            del arg105_1
            del arg106_1
            del arg97_1
            del buf153
            del buf160
            buf178 = empty_strided_cuda((16, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_52, view_96, x_68], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf177, (16, 768), (768, 1), 0), arg108_1, out=buf178)
            del arg108_1
            del buf177
            buf179 = reinterpret_tensor(buf178, (1, 16, 3072), (49152, 3072, 1), 0); del buf178  # reuse
            # Topologically Sorted Source Nodes: [x_68, x_69, mul_32, pow_9, mul_33, add_34, mul_34, tanh_8, add_35, hidden_states_53], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_4.run(buf179, arg107_1, 49152, stream=stream0)
            del arg107_1
            buf180 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_68, x_69, mul_32, pow_9, mul_33, add_34, mul_34, tanh_8, add_35, hidden_states_53, view_98, x_70], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf179, (16, 3072), (3072, 1), 0), arg110_1, out=buf180)
            del arg110_1
            del buf179
            buf184 = empty_strided_cuda((1, 16, 768), (12288, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_70, x_71, hidden_states_55, hidden_states_56], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_5.run(buf173, buf180, arg109_1, arg111_1, arg112_1, buf184, 16, 768, stream=stream0)
            del arg111_1
            del arg112_1
            buf185 = empty_strided_cuda((16, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_70, x_71, hidden_states_55, hidden_states_56, view_100, x_72], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg113_1, reinterpret_tensor(buf184, (16, 768), (768, 1), 0), arg114_1, alpha=1, beta=1, out=buf185)
            del arg113_1
            del arg114_1
            # Topologically Sorted Source Nodes: [cache_position, result_1, kv_arange, kv_arange_1, le, batch_arange, index, eq, index_1, result_2, batched_outputs_2, x_73, split_9, view_104, query_states_19, view_102, key_states_19, view_103, value_states_19, attn_output_36], Original ATen: [aten.arange, aten.view, aten.add, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.split, aten.transpose, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf187 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf185, (1, 12, 16, 64), (0, 64, 2304, 1), 0), reinterpret_tensor(buf185, (1, 12, 16, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf185, (1, 12, 16, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf186, (1, 12, 16, 16), (256, 0, 16, 1), 0), False)
            del buf185
            buf188 = buf187[0]
            assert_size_stride(buf188, (1, 12, 16, 64), (12288, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf188, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf187
            buf192 = reinterpret_tensor(buf184, (16, 768), (768, 1), 0); del buf184  # reuse
            # Topologically Sorted Source Nodes: [transpose_39, reshape_9, view_105, x_74], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf188, (16, 768), (768, 1), 0), arg116_1, out=buf192)
            del arg116_1
            buf193 = reinterpret_tensor(buf192, (1, 16, 768), (12288, 768, 1), 0); del buf192  # reuse
            buf197 = reinterpret_tensor(buf188, (1, 16, 768), (12288, 768, 1), 0); del buf188  # reuse
            # Topologically Sorted Source Nodes: [x_70, x_71, hidden_states_55, x_74, x_75, hidden_states_57, hidden_states_58], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_6.run(buf193, arg115_1, buf173, buf180, arg109_1, arg117_1, arg118_1, buf197, 16, 768, stream=stream0)
            del arg109_1
            del arg115_1
            del arg117_1
            del arg118_1
            del buf173
            del buf180
            buf198 = empty_strided_cuda((16, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_58, view_107, x_76], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf197, (16, 768), (768, 1), 0), arg120_1, out=buf198)
            del arg120_1
            del buf197
            buf199 = reinterpret_tensor(buf198, (1, 16, 3072), (49152, 3072, 1), 0); del buf198  # reuse
            # Topologically Sorted Source Nodes: [x_76, x_77, mul_36, pow_10, mul_37, add_38, mul_38, tanh_9, add_39, hidden_states_59], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_4.run(buf199, arg119_1, 49152, stream=stream0)
            del arg119_1
            buf200 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_76, x_77, mul_36, pow_10, mul_37, add_38, mul_38, tanh_9, add_39, hidden_states_59, view_109, x_78], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf199, (16, 3072), (3072, 1), 0), arg122_1, out=buf200)
            del arg122_1
            del buf199
            buf204 = empty_strided_cuda((1, 16, 768), (12288, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_78, x_79, hidden_states_61, hidden_states_62], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_5.run(buf193, buf200, arg121_1, arg123_1, arg124_1, buf204, 16, 768, stream=stream0)
            del arg123_1
            del arg124_1
            buf205 = empty_strided_cuda((16, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_78, x_79, hidden_states_61, hidden_states_62, view_111, x_80], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg125_1, reinterpret_tensor(buf204, (16, 768), (768, 1), 0), arg126_1, alpha=1, beta=1, out=buf205)
            del arg125_1
            del arg126_1
            buf206 = buf186; del buf186  # reuse
            buf226 = buf166; del buf166  # reuse
            # Topologically Sorted Source Nodes: [cache_position, result_1, kv_arange, kv_arange_1, le, batch_arange, index, eq, index_1, result_2, batched_outputs_2, x_81, split_10, view_115, query_states_21, view_113, key_states_21, view_114, value_states_21, attn_output_40, x_89, split_11, view_126, query_states_23, view_124, key_states_23, view_125, value_states_23, attn_output_44], Original ATen: [aten.arange, aten.view, aten.add, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.split, aten.transpose, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_bitwise_and_eq_expand_index_le_scalar_tensor_split_transpose_view_where_7.run(buf5, buf206, buf226, 256, stream=stream0)
            del buf5
            # Topologically Sorted Source Nodes: [cache_position, result_1, kv_arange, kv_arange_1, le, batch_arange, index, eq, index_1, result_2, batched_outputs_2, x_81, split_10, view_115, query_states_21, view_113, key_states_21, view_114, value_states_21, attn_output_40], Original ATen: [aten.arange, aten.view, aten.add, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.split, aten.transpose, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf207 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf205, (1, 12, 16, 64), (0, 64, 2304, 1), 0), reinterpret_tensor(buf205, (1, 12, 16, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf205, (1, 12, 16, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf206, (1, 12, 16, 16), (256, 0, 16, 1), 0), False)
            del buf205
            del buf206
            buf208 = buf207[0]
            assert_size_stride(buf208, (1, 12, 16, 64), (12288, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf208, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf207
            buf212 = reinterpret_tensor(buf204, (16, 768), (768, 1), 0); del buf204  # reuse
            # Topologically Sorted Source Nodes: [transpose_43, reshape_10, view_116, x_82], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf208, (16, 768), (768, 1), 0), arg128_1, out=buf212)
            del arg128_1
            buf213 = reinterpret_tensor(buf212, (1, 16, 768), (12288, 768, 1), 0); del buf212  # reuse
            buf217 = reinterpret_tensor(buf208, (1, 16, 768), (12288, 768, 1), 0); del buf208  # reuse
            # Topologically Sorted Source Nodes: [x_78, x_79, hidden_states_61, x_82, x_83, hidden_states_63, hidden_states_64], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_6.run(buf213, arg127_1, buf193, buf200, arg121_1, arg129_1, arg130_1, buf217, 16, 768, stream=stream0)
            del arg121_1
            del arg127_1
            del arg129_1
            del arg130_1
            del buf193
            del buf200
            buf218 = empty_strided_cuda((16, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_64, view_118, x_84], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf217, (16, 768), (768, 1), 0), arg132_1, out=buf218)
            del arg132_1
            del buf217
            buf219 = reinterpret_tensor(buf218, (1, 16, 3072), (49152, 3072, 1), 0); del buf218  # reuse
            # Topologically Sorted Source Nodes: [x_84, x_85, mul_40, pow_11, mul_41, add_42, mul_42, tanh_10, add_43, hidden_states_65], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_4.run(buf219, arg131_1, 49152, stream=stream0)
            del arg131_1
            buf220 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_84, x_85, mul_40, pow_11, mul_41, add_42, mul_42, tanh_10, add_43, hidden_states_65, view_120, x_86], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf219, (16, 3072), (3072, 1), 0), arg134_1, out=buf220)
            del arg134_1
            del buf219
            buf224 = empty_strided_cuda((1, 16, 768), (12288, 768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_86, x_87, hidden_states_67, hidden_states_68], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_5.run(buf213, buf220, arg133_1, arg135_1, arg136_1, buf224, 16, 768, stream=stream0)
            del arg135_1
            del arg136_1
            buf225 = empty_strided_cuda((16, 2304), (2304, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_86, x_87, hidden_states_67, hidden_states_68, view_122, x_88], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            extern_kernels.addmm(arg137_1, reinterpret_tensor(buf224, (16, 768), (768, 1), 0), arg138_1, alpha=1, beta=1, out=buf225)
            del arg137_1
            del arg138_1
            # Topologically Sorted Source Nodes: [cache_position, result_1, kv_arange, kv_arange_1, le, batch_arange, index, eq, index_1, result_2, batched_outputs_2, x_89, split_11, view_126, query_states_23, view_124, key_states_23, view_125, value_states_23, attn_output_44], Original ATen: [aten.arange, aten.view, aten.add, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.split, aten.transpose, aten.scalar_tensor, aten.where, aten.expand, aten._scaled_dot_product_efficient_attention]
            buf227 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf225, (1, 12, 16, 64), (0, 64, 2304, 1), 0), reinterpret_tensor(buf225, (1, 12, 16, 64), (0, 64, 2304, 1), 768), reinterpret_tensor(buf225, (1, 12, 16, 64), (0, 64, 2304, 1), 1536), reinterpret_tensor(buf226, (1, 12, 16, 16), (256, 0, 16, 1), 0), False)
            del buf225
            del buf226
            buf228 = buf227[0]
            assert_size_stride(buf228, (1, 12, 16, 64), (12288, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf228, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf227
            buf232 = reinterpret_tensor(buf224, (16, 768), (768, 1), 0); del buf224  # reuse
            # Topologically Sorted Source Nodes: [transpose_47, reshape_11, view_127, x_90], Original ATen: [aten.transpose, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf228, (16, 768), (768, 1), 0), arg140_1, out=buf232)
            del arg140_1
            buf233 = reinterpret_tensor(buf232, (1, 16, 768), (12288, 768, 1), 0); del buf232  # reuse
            buf237 = reinterpret_tensor(buf228, (1, 16, 768), (12288, 768, 1), 0); del buf228  # reuse
            # Topologically Sorted Source Nodes: [x_86, x_87, hidden_states_67, x_90, x_91, hidden_states_69, hidden_states_70], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_6.run(buf233, arg139_1, buf213, buf220, arg133_1, arg141_1, arg142_1, buf237, 16, 768, stream=stream0)
            del arg133_1
            del arg139_1
            del arg141_1
            del arg142_1
            del buf213
            del buf220
            buf238 = empty_strided_cuda((16, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_70, view_129, x_92], Original ATen: [aten.native_layer_norm, aten.view, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf237, (16, 768), (768, 1), 0), arg144_1, out=buf238)
            del arg144_1
            del buf237
            buf239 = reinterpret_tensor(buf238, (1, 16, 3072), (49152, 3072, 1), 0); del buf238  # reuse
            # Topologically Sorted Source Nodes: [x_92, x_93, mul_44, pow_12, mul_45, add_46, mul_46, tanh_11, add_47, hidden_states_71], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_addmm_mul_pow_tanh_view_4.run(buf239, arg143_1, 49152, stream=stream0)
            del arg143_1
            buf240 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_92, x_93, mul_44, pow_12, mul_45, add_46, mul_46, tanh_11, add_47, hidden_states_71, view_131, x_94], Original ATen: [aten.addmm, aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            extern_kernels.mm(reinterpret_tensor(buf239, (16, 3072), (3072, 1), 0), arg146_1, out=buf240)
            del arg146_1
            del buf239
            buf244 = buf233; del buf233  # reuse
            # Topologically Sorted Source Nodes: [x_94, x_95, hidden_states_73, hidden_states_74], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_8.run(buf244, buf240, arg145_1, arg147_1, arg148_1, 16, 768, stream=stream0)
            del arg145_1
            del arg147_1
            del arg148_1
            del buf240
        return (buf244, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 16), (16, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((50257, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((1024, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((768, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)


# ===== inductor generated file: /tmp/cnnbench-transformers-ob_ijj07/repeat_03/a1/torchinductor/tmpuqb_59nb/2e/c2eekjtavfj2jegok4t3geyjeblhnw2aljimpfddxzj6eolzgwjx.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_native_layer_norm_view_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 205824}}
)
@triton.jit
def triton_per_fused_add_addmm_native_layer_norm_view_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16
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


# ===== inductor generated file: /tmp/cnnbench-transformers-ob_ijj07/repeat_03/a1/torchinductor/tmpuqb_59nb/dx/cdxx2tckrq2rfzdtdjpsecgjjg5rvlcjjvpv5ckasole7tffxdye.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def _triton_helper_fn_add0(arg0_0, arg1_0):
    tmp0 = arg0_0 + arg1_0
    return tmp0

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_arange_cat_cumsum_ne_slice_sub_unsqueeze_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 256}}
)
@triton.jit
def triton_per_fused_arange_cat_cumsum_ne_slice_sub_unsqueeze_1(out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 16
    R0_BLOCK: tl.constexpr = 16
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = 1 + r0_0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.broadcast_to((-1) + (1 + r0_0), [XBLOCK, R0_BLOCK])
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1, 1], 17, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp15 = r0_0
    tmp16 = tmp15 >= tmp1
    tmp17 = tmp15 < tmp3
    tmp18 = tl.broadcast_to((-1) + (r0_0), [XBLOCK, R0_BLOCK])
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp15 >= tmp3
    tmp22 = tmp15 < tmp9
    tmp23 = tl.broadcast_to((-1) + r0_0, [XBLOCK, R0_BLOCK])
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp21, tmp23, tmp24)
    tmp26 = tl.where(tmp17, tmp20, tmp25)
    tmp27 = tmp14 - tmp26
    tmp28 = tmp27 != tmp3
    tmp29 = tmp28.to(tl.int64)
    tmp30 = tmp29.to(tl.int64)
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK, R0_BLOCK])
    tmp32, = tl.associative_scan((tmp31,), 1, _triton_helper_fn_add0)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp32, None)


# ===== inductor generated file: /tmp/cnnbench-transformers-ob_ijj07/repeat_03/a1/torchinductor/tmpuqb_59nb/ja/cjaucbdxwmlrtgyrskzzyoh4pgwb47uid5kjp6f6k3rgfh2zx7km.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_bitwise_and_eq_expand_index_le_scalar_tensor_split_transpose_view_where_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 10368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_bitwise_and_eq_expand_index_le_scalar_tensor_split_transpose_view_where_2(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp5 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = x1
    tmp2 = tmp0 <= tmp1
    tmp3 = tl.full([1], True, tl.int1)
    tmp4 = tmp3 & tmp2
    tmp7 = tmp5 == tmp6
    tmp8 = tmp4 & tmp7
    tmp9 = 0.0
    tmp10 = float("-inf")
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tl.store(out_ptr0 + (x2), tmp11, xmask)
    tl.store(out_ptr1 + (x2), tmp11, xmask)
    tl.store(out_ptr2 + (x2), tmp11, xmask)
    tl.store(out_ptr3 + (x2), tmp11, xmask)
    tl.store(out_ptr4 + (x2), tmp11, xmask)


# ===== inductor generated file: /tmp/cnnbench-transformers-ob_ijj07/repeat_03/a1/torchinductor/tmpuqb_59nb/lj/cljm2yw4ttzhprxdzyni7ggrjibfnhmatq2ohteunj2nfqlnqaqr.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16, 'r0_': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_arange_embedding_native_layer_norm_unsqueeze_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_arange_embedding_native_layer_norm_unsqueeze_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 16
    r0_numel = 768
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp10_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp10_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp10_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp7 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.full([XBLOCK, R0_BLOCK], 50257, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert(((0 <= tmp4) & (tmp4 < 50257)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 50257")
        tmp6 = tl.load(in_ptr1 + (r0_1 + 768*tmp4), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
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
        r0_1 = r0_index
        tmp22 = tl.load(in_ptr2 + (r0_1 + 768*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp31 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp33 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.full([XBLOCK, R0_BLOCK], 50257, tl.int32)
        tmp17 = tmp0 + tmp16
        tmp18 = tmp0 < 0
        tmp19 = tl.where(tmp18, tmp17, tmp0)
        tl.device_assert(((0 <= tmp19) & (tmp19 < 50257)) | ~(xmask), "index out of bounds: 0 <= tmp19 < 50257")
        tmp21 = tl.load(in_ptr1 + (r0_1 + 768*tmp19), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
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
        tl.store(out_ptr2 + (r0_1 + 768*x0), tmp34, r0_mask & xmask)


# ===== inductor generated file: /tmp/cnnbench-transformers-ob_ijj07/repeat_03/a1/torchinductor/tmpuqb_59nb/m5/cm5y4qzqfsa7zrgnjswk65q2jceoci2grzwwh22it65glp6s5qcx.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_native_layer_norm_view_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 356352}}
)
@triton.jit
def triton_per_fused_add_addmm_native_layer_norm_view_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16
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


# ===== inductor generated file: /tmp/cnnbench-transformers-ob_ijj07/repeat_03/a1/torchinductor/tmpuqb_59nb/n5/cn5drgj55mabns5bc5i7nj2iwlnlly62rbfoy3roikqqnbm2fwkk.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_bitwise_and_eq_expand_index_le_scalar_tensor_split_transpose_view_where_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4224}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention_add_arange_bitwise_and_eq_expand_index_le_scalar_tensor_split_transpose_view_where_7(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp5 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = x1
    tmp2 = tmp0 <= tmp1
    tmp3 = tl.full([1], True, tl.int1)
    tmp4 = tmp3 & tmp2
    tmp7 = tmp5 == tmp6
    tmp8 = tmp4 & tmp7
    tmp9 = 0.0
    tmp10 = float("-inf")
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tl.store(out_ptr0 + (x2), tmp11, xmask)
    tl.store(out_ptr1 + (x2), tmp11, xmask)


# ===== inductor generated file: /tmp/cnnbench-transformers-ob_ijj07/repeat_03/a1/torchinductor/tmpuqb_59nb/qx/cqxp4qkuvdyerf6x47rzmpv55xsjkmwal5kymfolzw5gb7gk6fal.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_native_layer_norm_view_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 205824}}
)
@triton.jit
def triton_per_fused_add_addmm_native_layer_norm_view_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16
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


# ===== inductor generated file: /tmp/cnnbench-transformers-ob_ijj07/repeat_03/a1/torchinductor/tmpuqb_59nb/s5/cs5rl3zynqzmsjh2xshqzukk3gtxthgzyqi3b5jc7jul5gcv3uc4.py =====

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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_mul_pow_tanh_view_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 602112}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_mul_pow_tanh_view_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
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


# ===== inductor generated file: /tmp/cnnbench-transformers-ob_ijj07/repeat_03/a1/torchinductor/tmpuqb_59nb/tb/ctbeuecaa5nknwy6bvucluzl476p3mx7abciczu4tjqirt7duxrh.debug/fx_graph_readable.py =====
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "i64[1, 16]", arg1_1: "f32[50257, 768]", arg2_1: "f32[1024, 768]", arg3_1: "f32[768]", arg4_1: "f32[768]", arg5_1: "f32[2304]", arg6_1: "f32[768, 2304]", arg7_1: "f32[768]", arg8_1: "f32[768, 768]", arg9_1: "f32[768]", arg10_1: "f32[768]", arg11_1: "f32[3072]", arg12_1: "f32[768, 3072]", arg13_1: "f32[768]", arg14_1: "f32[3072, 768]", arg15_1: "f32[768]", arg16_1: "f32[768]", arg17_1: "f32[2304]", arg18_1: "f32[768, 2304]", arg19_1: "f32[768]", arg20_1: "f32[768, 768]", arg21_1: "f32[768]", arg22_1: "f32[768]", arg23_1: "f32[3072]", arg24_1: "f32[768, 3072]", arg25_1: "f32[768]", arg26_1: "f32[3072, 768]", arg27_1: "f32[768]", arg28_1: "f32[768]", arg29_1: "f32[2304]", arg30_1: "f32[768, 2304]", arg31_1: "f32[768]", arg32_1: "f32[768, 768]", arg33_1: "f32[768]", arg34_1: "f32[768]", arg35_1: "f32[3072]", arg36_1: "f32[768, 3072]", arg37_1: "f32[768]", arg38_1: "f32[3072, 768]", arg39_1: "f32[768]", arg40_1: "f32[768]", arg41_1: "f32[2304]", arg42_1: "f32[768, 2304]", arg43_1: "f32[768]", arg44_1: "f32[768, 768]", arg45_1: "f32[768]", arg46_1: "f32[768]", arg47_1: "f32[3072]", arg48_1: "f32[768, 3072]", arg49_1: "f32[768]", arg50_1: "f32[3072, 768]", arg51_1: "f32[768]", arg52_1: "f32[768]", arg53_1: "f32[2304]", arg54_1: "f32[768, 2304]", arg55_1: "f32[768]", arg56_1: "f32[768, 768]", arg57_1: "f32[768]", arg58_1: "f32[768]", arg59_1: "f32[3072]", arg60_1: "f32[768, 3072]", arg61_1: "f32[768]", arg62_1: "f32[3072, 768]", arg63_1: "f32[768]", arg64_1: "f32[768]", arg65_1: "f32[2304]", arg66_1: "f32[768, 2304]", arg67_1: "f32[768]", arg68_1: "f32[768, 768]", arg69_1: "f32[768]", arg70_1: "f32[768]", arg71_1: "f32[3072]", arg72_1: "f32[768, 3072]", arg73_1: "f32[768]", arg74_1: "f32[3072, 768]", arg75_1: "f32[768]", arg76_1: "f32[768]", arg77_1: "f32[2304]", arg78_1: "f32[768, 2304]", arg79_1: "f32[768]", arg80_1: "f32[768, 768]", arg81_1: "f32[768]", arg82_1: "f32[768]", arg83_1: "f32[3072]", arg84_1: "f32[768, 3072]", arg85_1: "f32[768]", arg86_1: "f32[3072, 768]", arg87_1: "f32[768]", arg88_1: "f32[768]", arg89_1: "f32[2304]", arg90_1: "f32[768, 2304]", arg91_1: "f32[768]", arg92_1: "f32[768, 768]", arg93_1: "f32[768]", arg94_1: "f32[768]", arg95_1: "f32[3072]", arg96_1: "f32[768, 3072]", arg97_1: "f32[768]", arg98_1: "f32[3072, 768]", arg99_1: "f32[768]", arg100_1: "f32[768]", arg101_1: "f32[2304]", arg102_1: "f32[768, 2304]", arg103_1: "f32[768]", arg104_1: "f32[768, 768]", arg105_1: "f32[768]", arg106_1: "f32[768]", arg107_1: "f32[3072]", arg108_1: "f32[768, 3072]", arg109_1: "f32[768]", arg110_1: "f32[3072, 768]", arg111_1: "f32[768]", arg112_1: "f32[768]", arg113_1: "f32[2304]", arg114_1: "f32[768, 2304]", arg115_1: "f32[768]", arg116_1: "f32[768, 768]", arg117_1: "f32[768]", arg118_1: "f32[768]", arg119_1: "f32[3072]", arg120_1: "f32[768, 3072]", arg121_1: "f32[768]", arg122_1: "f32[3072, 768]", arg123_1: "f32[768]", arg124_1: "f32[768]", arg125_1: "f32[2304]", arg126_1: "f32[768, 2304]", arg127_1: "f32[768]", arg128_1: "f32[768, 768]", arg129_1: "f32[768]", arg130_1: "f32[768]", arg131_1: "f32[3072]", arg132_1: "f32[768, 3072]", arg133_1: "f32[768]", arg134_1: "f32[3072, 768]", arg135_1: "f32[768]", arg136_1: "f32[768]", arg137_1: "f32[2304]", arg138_1: "f32[768, 2304]", arg139_1: "f32[768]", arg140_1: "f32[768, 768]", arg141_1: "f32[768]", arg142_1: "f32[768]", arg143_1: "f32[3072]", arg144_1: "f32[768, 3072]", arg145_1: "f32[768]", arg146_1: "f32[3072, 768]", arg147_1: "f32[768]", arg148_1: "f32[768]"):
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:857 in forward, code: inputs_embeds = self.wte(input_ids)
        embedding: "f32[1, 16, 768]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:861 in forward, code: cache_position = torch.arange(
        iota: "i64[16]" = torch.ops.prims.iota.default(16, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:865 in forward, code: position_ids = cache_position.unsqueeze(0)
        unsqueeze: "i64[1, 16]" = torch.ops.aten.unsqueeze.default(iota, 0)
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:867 in forward, code: position_embeds = self.wpe(position_ids)
        embedding_1: "f32[1, 16, 768]" = torch.ops.aten.embedding.default(arg2_1, unsqueeze);  arg2_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:868 in forward, code: hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)
        add: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/masking_utils.py:654 in find_packed_sequence_indices, code: first_dummy_value = position_ids[:, :1] - 1  # We just need the diff on this first value to be 1
        slice_1: "i64[1, 1]" = torch.ops.aten.slice.Tensor(unsqueeze, 1, 0, 1)
        sub: "i64[1, 1]" = torch.ops.aten.sub.Tensor(slice_1, 1);  slice_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/masking_utils.py:655 in find_packed_sequence_indices, code: position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
        cat: "i64[1, 17]" = torch.ops.aten.cat.default([sub, unsqueeze], -1);  sub = unsqueeze = None
        slice_2: "i64[1, 16]" = torch.ops.aten.slice.Tensor(cat, -1, 0, 16)
        slice_3: "i64[1, 16]" = torch.ops.aten.slice.Tensor(cat, -1, 1, 17);  cat = None
        sub_1: "i64[1, 16]" = torch.ops.aten.sub.Tensor(slice_3, slice_2);  slice_3 = slice_2 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/masking_utils.py:656 in find_packed_sequence_indices, code: packed_sequence_mask = (position_diff != 1).cumsum(-1)
        ne: "b8[1, 16]" = torch.ops.aten.ne.Scalar(sub_1, 1);  sub_1 = None
        cumsum: "i64[1, 16]" = torch.ops.aten.cumsum.default(ne, -1);  ne = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/masking_utils.py:379 in sdpa_mask_recent_torch, code: kv_arange = torch.arange(kv_length, device=cache_position.device)
        iota_1: "i64[16]" = torch.ops.prims.iota.default(16, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/masking_utils.py:380 in sdpa_mask_recent_torch, code: kv_arange += kv_offset
        add_1: "i64[16]" = torch.ops.aten.add.Tensor(iota_1, 0);  iota_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/masking_utils.py:386 in sdpa_mask_recent_torch, code: batch_arange = torch.arange(batch_size, device=cache_position.device)
        iota_2: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/torch/_dynamo/_trace_wrapped_higher_order_op.py:146 in __torch_function__, code: return func(*args, **(kwargs or {}))
        full: "b8[16]" = torch.ops.aten.full.default([16], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/torch/_dynamo/_trace_wrapped_higher_order_op.py:146 in __torch_function__, code: return func(*args, **(kwargs or {}))
        view_1: "i64[16, 1]" = torch.ops.aten.view.default(iota, [16, 1])
        le: "b8[16, 16]" = torch.ops.aten.le.Tensor(add_1, view_1);  view_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/torch/_dynamo/_trace_wrapped_higher_order_op.py:146 in __torch_function__, code: return func(*args, **(kwargs or {}))
        full_default: "b8[16, 1]" = torch.ops.aten.full.default([16, 1], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        bitwise_and: "b8[16, 16]" = torch.ops.aten.bitwise_and.Tensor(full_default, le);  full_default = le = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/torch/_dynamo/_trace_wrapped_higher_order_op.py:99 in forward, code: return torch.ops.aten.index(x, indices)
        view_3: "i64[1, 1]" = torch.ops.aten.view.default(iota_2, [1, 1])
        index: "i64[1, 16]" = torch.ops.aten.index.Tensor(cumsum, [view_3, iota]);  view_3 = iota = None
        view_4: "i64[1, 1]" = torch.ops.aten.view.default(iota_2, [1, 1]);  iota_2 = None
        index_1: "i64[1, 16]" = torch.ops.aten.index.Tensor(cumsum, [view_4, add_1]);  cumsum = view_4 = add_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/torch/_dynamo/_trace_wrapped_higher_order_op.py:146 in __torch_function__, code: return func(*args, **(kwargs or {}))
        view_5: "i64[1, 16, 1]" = torch.ops.aten.view.default(index, [1, 16, 1]);  index = None
        view_6: "i64[1, 1, 16]" = torch.ops.aten.view.default(index_1, [1, 1, 16]);  index_1 = None
        eq: "b8[1, 16, 16]" = torch.ops.aten.eq.Tensor(view_5, view_6);  view_5 = view_6 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/torch/_dynamo/_trace_wrapped_higher_order_op.py:146 in __torch_function__, code: return func(*args, **(kwargs or {}))
        bitwise_and_1: "b8[1, 16, 16]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, eq);  bitwise_and = eq = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/torch/_functorch/vmap.py:183 in _maybe_remove_batch_dim, code: return _remove_batch_dim(batched_output, vmap_level, batch_size, out_dim)
        view_7: "b8[1, 1, 16, 16]" = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 16, 16]);  bitwise_and_1 = None
        expand: "b8[1, 1, 16, 16]" = torch.ops.aten.expand.default(view_7, [1, 1, 16, 16]);  view_7 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 16, 1]" = var_mean[0]
        getitem_1: "f32[1, 16, 1]" = var_mean[1];  var_mean = None
        add_2: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub_2: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add, getitem_1);  getitem_1 = None
        mul: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt);  sub_2 = rsqrt = None
        mul_1: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = arg3_1 = None
        add_3: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_8: "f32[16, 768]" = torch.ops.aten.view.default(add_3, [-1, 768]);  add_3 = None
        addmm: "f32[16, 2304]" = torch.ops.aten.addmm.default(arg5_1, view_8, arg6_1);  arg5_1 = view_8 = arg6_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_9: "f32[1, 16, 2304]" = torch.ops.aten.view.default(addmm, [1, 16, 2304]);  addmm = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split = torch.ops.aten.split.Tensor(view_9, 768, 2);  view_9 = None
        getitem_2: "f32[1, 16, 768]" = split[0]
        getitem_3: "f32[1, 16, 768]" = split[1]
        getitem_4: "f32[1, 16, 768]" = split[2];  split = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_10: "f32[1, 16, 12, 64]" = torch.ops.aten.view.default(getitem_3, [1, 16, -1, 64]);  getitem_3 = None
        permute: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_11: "f32[1, 16, 12, 64]" = torch.ops.aten.view.default(getitem_4, [1, 16, -1, 64]);  getitem_4 = None
        permute_1: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_11, [0, 2, 1, 3]);  view_11 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_12: "f32[1, 16, 12, 64]" = torch.ops.aten.view.default(getitem_2, [1, 16, -1, 64]);  getitem_2 = None
        permute_2: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[1, 1, 16, 16]" = torch.ops.aten.where.self(expand, full_default_2, full_default_1);  full_default_2 = full_default_1 = None
        expand_1: "f32[1, 12, 16, 16]" = torch.ops.aten.expand.default(where, [1, 12, 16, 16]);  where = None
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_2, permute, permute_1, expand_1, False);  permute_2 = permute = permute_1 = expand_1 = None
        getitem_5: "f32[1, 12, 16, 64]" = _scaled_dot_product_efficient_attention[0];  _scaled_dot_product_efficient_attention = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_3: "f32[1, 16, 12, 64]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_13: "f32[1, 16, 768]" = torch.ops.aten.view.default(permute_3, [1, 16, -1]);  permute_3 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_14: "f32[16, 768]" = torch.ops.aten.view.default(view_13, [-1, 768]);  view_13 = None
        addmm_1: "f32[16, 768]" = torch.ops.aten.addmm.default(arg7_1, view_14, arg8_1);  arg7_1 = view_14 = arg8_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_15: "f32[1, 16, 768]" = torch.ops.aten.view.default(addmm_1, [1, 16, 768]);  addmm_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_4: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(view_15, add);  view_15 = add = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
        getitem_9: "f32[1, 16, 1]" = var_mean_1[0]
        getitem_10: "f32[1, 16, 1]" = var_mean_1[1];  var_mean_1 = None
        add_5: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_9, 1e-05);  getitem_9 = None
        rsqrt_1: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_3: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_4, getitem_10);  getitem_10 = None
        mul_2: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = rsqrt_1 = None
        mul_3: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_2, arg9_1);  mul_2 = arg9_1 = None
        add_6: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_3, arg10_1);  mul_3 = arg10_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_16: "f32[16, 768]" = torch.ops.aten.view.default(add_6, [-1, 768]);  add_6 = None
        addmm_2: "f32[16, 3072]" = torch.ops.aten.addmm.default(arg11_1, view_16, arg12_1);  arg11_1 = view_16 = arg12_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_17: "f32[1, 16, 3072]" = torch.ops.aten.view.default(addmm_2, [1, 16, 3072]);  addmm_2 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_4: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(view_17, 0.5)
        pow_1: "f32[1, 16, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_17, 3.0)
        mul_5: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_7: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(view_17, mul_5);  view_17 = mul_5 = None
        mul_6: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(add_7, 0.7978845608028654);  add_7 = None
        tanh: "f32[1, 16, 3072]" = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
        add_8: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_7: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(mul_4, add_8);  mul_4 = add_8 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_18: "f32[16, 3072]" = torch.ops.aten.view.default(mul_7, [-1, 3072]);  mul_7 = None
        addmm_3: "f32[16, 768]" = torch.ops.aten.addmm.default(arg13_1, view_18, arg14_1);  arg13_1 = view_18 = arg14_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_19: "f32[1, 16, 768]" = torch.ops.aten.view.default(addmm_3, [1, 16, 768]);  addmm_3 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_9: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(add_4, view_19);  add_4 = view_19 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
        getitem_11: "f32[1, 16, 1]" = var_mean_2[0]
        getitem_12: "f32[1, 16, 1]" = var_mean_2[1];  var_mean_2 = None
        add_10: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
        rsqrt_2: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_4: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_9, getitem_12);  getitem_12 = None
        mul_8: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = rsqrt_2 = None
        mul_9: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_8, arg15_1);  mul_8 = arg15_1 = None
        add_11: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_9, arg16_1);  mul_9 = arg16_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_20: "f32[16, 768]" = torch.ops.aten.view.default(add_11, [-1, 768]);  add_11 = None
        addmm_4: "f32[16, 2304]" = torch.ops.aten.addmm.default(arg17_1, view_20, arg18_1);  arg17_1 = view_20 = arg18_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_21: "f32[1, 16, 2304]" = torch.ops.aten.view.default(addmm_4, [1, 16, 2304]);  addmm_4 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_1 = torch.ops.aten.split.Tensor(view_21, 768, 2);  view_21 = None
        getitem_13: "f32[1, 16, 768]" = split_1[0]
        getitem_14: "f32[1, 16, 768]" = split_1[1]
        getitem_15: "f32[1, 16, 768]" = split_1[2];  split_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_22: "f32[1, 16, 12, 64]" = torch.ops.aten.view.default(getitem_14, [1, 16, -1, 64]);  getitem_14 = None
        permute_4: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_23: "f32[1, 16, 12, 64]" = torch.ops.aten.view.default(getitem_15, [1, 16, -1, 64]);  getitem_15 = None
        permute_5: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_23, [0, 2, 1, 3]);  view_23 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_24: "f32[1, 16, 12, 64]" = torch.ops.aten.view.default(getitem_13, [1, 16, -1, 64]);  getitem_13 = None
        permute_6: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_3: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_4: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[1, 1, 16, 16]" = torch.ops.aten.where.self(expand, full_default_4, full_default_3);  full_default_4 = full_default_3 = None
        expand_2: "f32[1, 12, 16, 16]" = torch.ops.aten.expand.default(where_1, [1, 12, 16, 16]);  where_1 = None
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_6, permute_4, permute_5, expand_2, False);  permute_6 = permute_4 = permute_5 = expand_2 = None
        getitem_16: "f32[1, 12, 16, 64]" = _scaled_dot_product_efficient_attention_1[0];  _scaled_dot_product_efficient_attention_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "f32[1, 16, 12, 64]" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3]);  getitem_16 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_25: "f32[1, 16, 768]" = torch.ops.aten.view.default(permute_7, [1, 16, -1]);  permute_7 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_26: "f32[16, 768]" = torch.ops.aten.view.default(view_25, [-1, 768]);  view_25 = None
        addmm_5: "f32[16, 768]" = torch.ops.aten.addmm.default(arg19_1, view_26, arg20_1);  arg19_1 = view_26 = arg20_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_27: "f32[1, 16, 768]" = torch.ops.aten.view.default(addmm_5, [1, 16, 768]);  addmm_5 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_12: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(view_27, add_9);  view_27 = add_9 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_3 = torch.ops.aten.var_mean.correction(add_12, [2], correction = 0, keepdim = True)
        getitem_20: "f32[1, 16, 1]" = var_mean_3[0]
        getitem_21: "f32[1, 16, 1]" = var_mean_3[1];  var_mean_3 = None
        add_13: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_3: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        sub_5: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_12, getitem_21);  getitem_21 = None
        mul_10: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_11: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_10, arg21_1);  mul_10 = arg21_1 = None
        add_14: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_11, arg22_1);  mul_11 = arg22_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_28: "f32[16, 768]" = torch.ops.aten.view.default(add_14, [-1, 768]);  add_14 = None
        addmm_6: "f32[16, 3072]" = torch.ops.aten.addmm.default(arg23_1, view_28, arg24_1);  arg23_1 = view_28 = arg24_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_29: "f32[1, 16, 3072]" = torch.ops.aten.view.default(addmm_6, [1, 16, 3072]);  addmm_6 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_12: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(view_29, 0.5)
        pow_2: "f32[1, 16, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_29, 3.0)
        mul_13: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_15: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(view_29, mul_13);  view_29 = mul_13 = None
        mul_14: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(add_15, 0.7978845608028654);  add_15 = None
        tanh_1: "f32[1, 16, 3072]" = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
        add_16: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_15: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(mul_12, add_16);  mul_12 = add_16 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_30: "f32[16, 3072]" = torch.ops.aten.view.default(mul_15, [-1, 3072]);  mul_15 = None
        addmm_7: "f32[16, 768]" = torch.ops.aten.addmm.default(arg25_1, view_30, arg26_1);  arg25_1 = view_30 = arg26_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_31: "f32[1, 16, 768]" = torch.ops.aten.view.default(addmm_7, [1, 16, 768]);  addmm_7 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_17: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(add_12, view_31);  add_12 = view_31 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_4 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_22: "f32[1, 16, 1]" = var_mean_4[0]
        getitem_23: "f32[1, 16, 1]" = var_mean_4[1];  var_mean_4 = None
        add_18: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_4: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_6: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_17, getitem_23);  getitem_23 = None
        mul_16: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
        mul_17: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_16, arg27_1);  mul_16 = arg27_1 = None
        add_19: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_17, arg28_1);  mul_17 = arg28_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_32: "f32[16, 768]" = torch.ops.aten.view.default(add_19, [-1, 768]);  add_19 = None
        addmm_8: "f32[16, 2304]" = torch.ops.aten.addmm.default(arg29_1, view_32, arg30_1);  arg29_1 = view_32 = arg30_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_33: "f32[1, 16, 2304]" = torch.ops.aten.view.default(addmm_8, [1, 16, 2304]);  addmm_8 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_2 = torch.ops.aten.split.Tensor(view_33, 768, 2);  view_33 = None
        getitem_24: "f32[1, 16, 768]" = split_2[0]
        getitem_25: "f32[1, 16, 768]" = split_2[1]
        getitem_26: "f32[1, 16, 768]" = split_2[2];  split_2 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_34: "f32[1, 16, 12, 64]" = torch.ops.aten.view.default(getitem_25, [1, 16, -1, 64]);  getitem_25 = None
        permute_8: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_35: "f32[1, 16, 12, 64]" = torch.ops.aten.view.default(getitem_26, [1, 16, -1, 64]);  getitem_26 = None
        permute_9: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_35, [0, 2, 1, 3]);  view_35 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_36: "f32[1, 16, 12, 64]" = torch.ops.aten.view.default(getitem_24, [1, 16, -1, 64]);  getitem_24 = None
        permute_10: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_5: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_6: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[1, 1, 16, 16]" = torch.ops.aten.where.self(expand, full_default_6, full_default_5);  full_default_6 = full_default_5 = None
        expand_3: "f32[1, 12, 16, 16]" = torch.ops.aten.expand.default(where_2, [1, 12, 16, 16]);  where_2 = None
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_10, permute_8, permute_9, expand_3, False);  permute_10 = permute_8 = permute_9 = expand_3 = None
        getitem_27: "f32[1, 12, 16, 64]" = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_11: "f32[1, 16, 12, 64]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3]);  getitem_27 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_37: "f32[1, 16, 768]" = torch.ops.aten.view.default(permute_11, [1, 16, -1]);  permute_11 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_38: "f32[16, 768]" = torch.ops.aten.view.default(view_37, [-1, 768]);  view_37 = None
        addmm_9: "f32[16, 768]" = torch.ops.aten.addmm.default(arg31_1, view_38, arg32_1);  arg31_1 = view_38 = arg32_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_39: "f32[1, 16, 768]" = torch.ops.aten.view.default(addmm_9, [1, 16, 768]);  addmm_9 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_20: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(view_39, add_17);  view_39 = add_17 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_5 = torch.ops.aten.var_mean.correction(add_20, [2], correction = 0, keepdim = True)
        getitem_31: "f32[1, 16, 1]" = var_mean_5[0]
        getitem_32: "f32[1, 16, 1]" = var_mean_5[1];  var_mean_5 = None
        add_21: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_31, 1e-05);  getitem_31 = None
        rsqrt_5: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_7: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_20, getitem_32);  getitem_32 = None
        mul_18: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_5);  sub_7 = rsqrt_5 = None
        mul_19: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_18, arg33_1);  mul_18 = arg33_1 = None
        add_22: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_19, arg34_1);  mul_19 = arg34_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_40: "f32[16, 768]" = torch.ops.aten.view.default(add_22, [-1, 768]);  add_22 = None
        addmm_10: "f32[16, 3072]" = torch.ops.aten.addmm.default(arg35_1, view_40, arg36_1);  arg35_1 = view_40 = arg36_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_41: "f32[1, 16, 3072]" = torch.ops.aten.view.default(addmm_10, [1, 16, 3072]);  addmm_10 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_20: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.5)
        pow_3: "f32[1, 16, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_41, 3.0)
        mul_21: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_23: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(view_41, mul_21);  view_41 = mul_21 = None
        mul_22: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(add_23, 0.7978845608028654);  add_23 = None
        tanh_2: "f32[1, 16, 3072]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
        add_24: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_23: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(mul_20, add_24);  mul_20 = add_24 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_42: "f32[16, 3072]" = torch.ops.aten.view.default(mul_23, [-1, 3072]);  mul_23 = None
        addmm_11: "f32[16, 768]" = torch.ops.aten.addmm.default(arg37_1, view_42, arg38_1);  arg37_1 = view_42 = arg38_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_43: "f32[1, 16, 768]" = torch.ops.aten.view.default(addmm_11, [1, 16, 768]);  addmm_11 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_25: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(add_20, view_43);  add_20 = view_43 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_6 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_33: "f32[1, 16, 1]" = var_mean_6[0]
        getitem_34: "f32[1, 16, 1]" = var_mean_6[1];  var_mean_6 = None
        add_26: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_33, 1e-05);  getitem_33 = None
        rsqrt_6: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_8: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_25, getitem_34);  getitem_34 = None
        mul_24: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_6);  sub_8 = rsqrt_6 = None
        mul_25: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_24, arg39_1);  mul_24 = arg39_1 = None
        add_27: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_25, arg40_1);  mul_25 = arg40_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_44: "f32[16, 768]" = torch.ops.aten.view.default(add_27, [-1, 768]);  add_27 = None
        addmm_12: "f32[16, 2304]" = torch.ops.aten.addmm.default(arg41_1, view_44, arg42_1);  arg41_1 = view_44 = arg42_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_45: "f32[1, 16, 2304]" = torch.ops.aten.view.default(addmm_12, [1, 16, 2304]);  addmm_12 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_3 = torch.ops.aten.split.Tensor(view_45, 768, 2);  view_45 = None
        getitem_35: "f32[1, 16, 768]" = split_3[0]
        getitem_36: "f32[1, 16, 768]" = split_3[1]
        getitem_37: "f32[1, 16, 768]" = split_3[2];  split_3 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_46: "f32[1, 16, 12, 64]" = torch.ops.aten.view.default(getitem_36, [1, 16, -1, 64]);  getitem_36 = None
        permute_12: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_47: "f32[1, 16, 12, 64]" = torch.ops.aten.view.default(getitem_37, [1, 16, -1, 64]);  getitem_37 = None
        permute_13: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_47, [0, 2, 1, 3]);  view_47 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_48: "f32[1, 16, 12, 64]" = torch.ops.aten.view.default(getitem_35, [1, 16, -1, 64]);  getitem_35 = None
        permute_14: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_7: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_8: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "f32[1, 1, 16, 16]" = torch.ops.aten.where.self(expand, full_default_8, full_default_7);  full_default_8 = full_default_7 = None
        expand_4: "f32[1, 12, 16, 16]" = torch.ops.aten.expand.default(where_3, [1, 12, 16, 16]);  where_3 = None
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_14, permute_12, permute_13, expand_4, False);  permute_14 = permute_12 = permute_13 = expand_4 = None
        getitem_38: "f32[1, 12, 16, 64]" = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_15: "f32[1, 16, 12, 64]" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3]);  getitem_38 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_49: "f32[1, 16, 768]" = torch.ops.aten.view.default(permute_15, [1, 16, -1]);  permute_15 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_50: "f32[16, 768]" = torch.ops.aten.view.default(view_49, [-1, 768]);  view_49 = None
        addmm_13: "f32[16, 768]" = torch.ops.aten.addmm.default(arg43_1, view_50, arg44_1);  arg43_1 = view_50 = arg44_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_51: "f32[1, 16, 768]" = torch.ops.aten.view.default(addmm_13, [1, 16, 768]);  addmm_13 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_28: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(view_51, add_25);  view_51 = add_25 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_7 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
        getitem_42: "f32[1, 16, 1]" = var_mean_7[0]
        getitem_43: "f32[1, 16, 1]" = var_mean_7[1];  var_mean_7 = None
        add_29: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_7: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        sub_9: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_28, getitem_43);  getitem_43 = None
        mul_26: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_7);  sub_9 = rsqrt_7 = None
        mul_27: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_26, arg45_1);  mul_26 = arg45_1 = None
        add_30: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_27, arg46_1);  mul_27 = arg46_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_52: "f32[16, 768]" = torch.ops.aten.view.default(add_30, [-1, 768]);  add_30 = None
        addmm_14: "f32[16, 3072]" = torch.ops.aten.addmm.default(arg47_1, view_52, arg48_1);  arg47_1 = view_52 = arg48_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_53: "f32[1, 16, 3072]" = torch.ops.aten.view.default(addmm_14, [1, 16, 3072]);  addmm_14 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_28: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(view_53, 0.5)
        pow_4: "f32[1, 16, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_53, 3.0)
        mul_29: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_31: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(view_53, mul_29);  view_53 = mul_29 = None
        mul_30: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(add_31, 0.7978845608028654);  add_31 = None
        tanh_3: "f32[1, 16, 3072]" = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
        add_32: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_31: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(mul_28, add_32);  mul_28 = add_32 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_54: "f32[16, 3072]" = torch.ops.aten.view.default(mul_31, [-1, 3072]);  mul_31 = None
        addmm_15: "f32[16, 768]" = torch.ops.aten.addmm.default(arg49_1, view_54, arg50_1);  arg49_1 = view_54 = arg50_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_55: "f32[1, 16, 768]" = torch.ops.aten.view.default(addmm_15, [1, 16, 768]);  addmm_15 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_33: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(add_28, view_55);  add_28 = view_55 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_8 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
        getitem_44: "f32[1, 16, 1]" = var_mean_8[0]
        getitem_45: "f32[1, 16, 1]" = var_mean_8[1];  var_mean_8 = None
        add_34: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_8: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        sub_10: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_33, getitem_45);  getitem_45 = None
        mul_32: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_8);  sub_10 = rsqrt_8 = None
        mul_33: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_32, arg51_1);  mul_32 = arg51_1 = None
        add_35: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_33, arg52_1);  mul_33 = arg52_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_56: "f32[16, 768]" = torch.ops.aten.view.default(add_35, [-1, 768]);  add_35 = None
        addmm_16: "f32[16, 2304]" = torch.ops.aten.addmm.default(arg53_1, view_56, arg54_1);  arg53_1 = view_56 = arg54_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_57: "f32[1, 16, 2304]" = torch.ops.aten.view.default(addmm_16, [1, 16, 2304]);  addmm_16 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_4 = torch.ops.aten.split.Tensor(view_57, 768, 2);  view_57 = None
        getitem_46: "f32[1, 16, 768]" = split_4[0]
        getitem_47: "f32[1, 16, 768]" = split_4[1]
        getitem_48: "f32[1, 16, 768]" = split_4[2];  split_4 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_58: "f32[1, 16, 12, 64]" = torch.ops.aten.view.default(getitem_47, [1, 16, -1, 64]);  getitem_47 = None
        permute_16: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_59: "f32[1, 16, 12, 64]" = torch.ops.aten.view.default(getitem_48, [1, 16, -1, 64]);  getitem_48 = None
        permute_17: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_59, [0, 2, 1, 3]);  view_59 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_60: "f32[1, 16, 12, 64]" = torch.ops.aten.view.default(getitem_46, [1, 16, -1, 64]);  getitem_46 = None
        permute_18: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_9: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_10: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "f32[1, 1, 16, 16]" = torch.ops.aten.where.self(expand, full_default_10, full_default_9);  full_default_10 = full_default_9 = None
        expand_5: "f32[1, 12, 16, 16]" = torch.ops.aten.expand.default(where_4, [1, 12, 16, 16]);  where_4 = None
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_18, permute_16, permute_17, expand_5, False);  permute_18 = permute_16 = permute_17 = expand_5 = None
        getitem_49: "f32[1, 12, 16, 64]" = _scaled_dot_product_efficient_attention_4[0];  _scaled_dot_product_efficient_attention_4 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_19: "f32[1, 16, 12, 64]" = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3]);  getitem_49 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_61: "f32[1, 16, 768]" = torch.ops.aten.view.default(permute_19, [1, 16, -1]);  permute_19 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_62: "f32[16, 768]" = torch.ops.aten.view.default(view_61, [-1, 768]);  view_61 = None
        addmm_17: "f32[16, 768]" = torch.ops.aten.addmm.default(arg55_1, view_62, arg56_1);  arg55_1 = view_62 = arg56_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_63: "f32[1, 16, 768]" = torch.ops.aten.view.default(addmm_17, [1, 16, 768]);  addmm_17 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_36: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(view_63, add_33);  view_63 = add_33 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_9 = torch.ops.aten.var_mean.correction(add_36, [2], correction = 0, keepdim = True)
        getitem_53: "f32[1, 16, 1]" = var_mean_9[0]
        getitem_54: "f32[1, 16, 1]" = var_mean_9[1];  var_mean_9 = None
        add_37: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_53, 1e-05);  getitem_53 = None
        rsqrt_9: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        sub_11: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_36, getitem_54);  getitem_54 = None
        mul_34: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_9);  sub_11 = rsqrt_9 = None
        mul_35: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_34, arg57_1);  mul_34 = arg57_1 = None
        add_38: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_35, arg58_1);  mul_35 = arg58_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_64: "f32[16, 768]" = torch.ops.aten.view.default(add_38, [-1, 768]);  add_38 = None
        addmm_18: "f32[16, 3072]" = torch.ops.aten.addmm.default(arg59_1, view_64, arg60_1);  arg59_1 = view_64 = arg60_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_65: "f32[1, 16, 3072]" = torch.ops.aten.view.default(addmm_18, [1, 16, 3072]);  addmm_18 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_36: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(view_65, 0.5)
        pow_5: "f32[1, 16, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_65, 3.0)
        mul_37: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_39: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(view_65, mul_37);  view_65 = mul_37 = None
        mul_38: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(add_39, 0.7978845608028654);  add_39 = None
        tanh_4: "f32[1, 16, 3072]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_40: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_39: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(mul_36, add_40);  mul_36 = add_40 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_66: "f32[16, 3072]" = torch.ops.aten.view.default(mul_39, [-1, 3072]);  mul_39 = None
        addmm_19: "f32[16, 768]" = torch.ops.aten.addmm.default(arg61_1, view_66, arg62_1);  arg61_1 = view_66 = arg62_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_67: "f32[1, 16, 768]" = torch.ops.aten.view.default(addmm_19, [1, 16, 768]);  addmm_19 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_41: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(add_36, view_67);  add_36 = view_67 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_10 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
        getitem_55: "f32[1, 16, 1]" = var_mean_10[0]
        getitem_56: "f32[1, 16, 1]" = var_mean_10[1];  var_mean_10 = None
        add_42: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_55, 1e-05);  getitem_55 = None
        rsqrt_10: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_12: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_41, getitem_56);  getitem_56 = None
        mul_40: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_10);  sub_12 = rsqrt_10 = None
        mul_41: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_40, arg63_1);  mul_40 = arg63_1 = None
        add_43: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_41, arg64_1);  mul_41 = arg64_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_68: "f32[16, 768]" = torch.ops.aten.view.default(add_43, [-1, 768]);  add_43 = None
        addmm_20: "f32[16, 2304]" = torch.ops.aten.addmm.default(arg65_1, view_68, arg66_1);  arg65_1 = view_68 = arg66_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_69: "f32[1, 16, 2304]" = torch.ops.aten.view.default(addmm_20, [1, 16, 2304]);  addmm_20 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_5 = torch.ops.aten.split.Tensor(view_69, 768, 2);  view_69 = None
        getitem_57: "f32[1, 16, 768]" = split_5[0]
        getitem_58: "f32[1, 16, 768]" = split_5[1]
        getitem_59: "f32[1, 16, 768]" = split_5[2];  split_5 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_70: "f32[1, 16, 12, 64]" = torch.ops.aten.view.default(getitem_58, [1, 16, -1, 64]);  getitem_58 = None
        permute_20: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_71: "f32[1, 16, 12, 64]" = torch.ops.aten.view.default(getitem_59, [1, 16, -1, 64]);  getitem_59 = None
        permute_21: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_72: "f32[1, 16, 12, 64]" = torch.ops.aten.view.default(getitem_57, [1, 16, -1, 64]);  getitem_57 = None
        permute_22: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_11: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_12: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5: "f32[1, 1, 16, 16]" = torch.ops.aten.where.self(expand, full_default_12, full_default_11);  full_default_12 = full_default_11 = None
        expand_6: "f32[1, 12, 16, 16]" = torch.ops.aten.expand.default(where_5, [1, 12, 16, 16]);  where_5 = None
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_22, permute_20, permute_21, expand_6, False);  permute_22 = permute_20 = permute_21 = expand_6 = None
        getitem_60: "f32[1, 12, 16, 64]" = _scaled_dot_product_efficient_attention_5[0];  _scaled_dot_product_efficient_attention_5 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_23: "f32[1, 16, 12, 64]" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_73: "f32[1, 16, 768]" = torch.ops.aten.view.default(permute_23, [1, 16, -1]);  permute_23 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_74: "f32[16, 768]" = torch.ops.aten.view.default(view_73, [-1, 768]);  view_73 = None
        addmm_21: "f32[16, 768]" = torch.ops.aten.addmm.default(arg67_1, view_74, arg68_1);  arg67_1 = view_74 = arg68_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_75: "f32[1, 16, 768]" = torch.ops.aten.view.default(addmm_21, [1, 16, 768]);  addmm_21 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_44: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(view_75, add_41);  view_75 = add_41 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_11 = torch.ops.aten.var_mean.correction(add_44, [2], correction = 0, keepdim = True)
        getitem_64: "f32[1, 16, 1]" = var_mean_11[0]
        getitem_65: "f32[1, 16, 1]" = var_mean_11[1];  var_mean_11 = None
        add_45: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_11: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        sub_13: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_44, getitem_65);  getitem_65 = None
        mul_42: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_11);  sub_13 = rsqrt_11 = None
        mul_43: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_42, arg69_1);  mul_42 = arg69_1 = None
        add_46: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_43, arg70_1);  mul_43 = arg70_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_76: "f32[16, 768]" = torch.ops.aten.view.default(add_46, [-1, 768]);  add_46 = None
        addmm_22: "f32[16, 3072]" = torch.ops.aten.addmm.default(arg71_1, view_76, arg72_1);  arg71_1 = view_76 = arg72_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_77: "f32[1, 16, 3072]" = torch.ops.aten.view.default(addmm_22, [1, 16, 3072]);  addmm_22 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_44: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(view_77, 0.5)
        pow_6: "f32[1, 16, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_77, 3.0)
        mul_45: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_47: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(view_77, mul_45);  view_77 = mul_45 = None
        mul_46: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(add_47, 0.7978845608028654);  add_47 = None
        tanh_5: "f32[1, 16, 3072]" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
        add_48: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_47: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(mul_44, add_48);  mul_44 = add_48 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_78: "f32[16, 3072]" = torch.ops.aten.view.default(mul_47, [-1, 3072]);  mul_47 = None
        addmm_23: "f32[16, 768]" = torch.ops.aten.addmm.default(arg73_1, view_78, arg74_1);  arg73_1 = view_78 = arg74_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_79: "f32[1, 16, 768]" = torch.ops.aten.view.default(addmm_23, [1, 16, 768]);  addmm_23 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_49: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(add_44, view_79);  add_44 = view_79 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_12 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
        getitem_66: "f32[1, 16, 1]" = var_mean_12[0]
        getitem_67: "f32[1, 16, 1]" = var_mean_12[1];  var_mean_12 = None
        add_50: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_12: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        sub_14: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_49, getitem_67);  getitem_67 = None
        mul_48: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_12);  sub_14 = rsqrt_12 = None
        mul_49: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_48, arg75_1);  mul_48 = arg75_1 = None
        add_51: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_49, arg76_1);  mul_49 = arg76_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_80: "f32[16, 768]" = torch.ops.aten.view.default(add_51, [-1, 768]);  add_51 = None
        addmm_24: "f32[16, 2304]" = torch.ops.aten.addmm.default(arg77_1, view_80, arg78_1);  arg77_1 = view_80 = arg78_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_81: "f32[1, 16, 2304]" = torch.ops.aten.view.default(addmm_24, [1, 16, 2304]);  addmm_24 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_6 = torch.ops.aten.split.Tensor(view_81, 768, 2);  view_81 = None
        getitem_68: "f32[1, 16, 768]" = split_6[0]
        getitem_69: "f32[1, 16, 768]" = split_6[1]
        getitem_70: "f32[1, 16, 768]" = split_6[2];  split_6 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_82: "f32[1, 16, 12, 64]" = torch.ops.aten.view.default(getitem_69, [1, 16, -1, 64]);  getitem_69 = None
        permute_24: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_83: "f32[1, 16, 12, 64]" = torch.ops.aten.view.default(getitem_70, [1, 16, -1, 64]);  getitem_70 = None
        permute_25: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_83, [0, 2, 1, 3]);  view_83 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_84: "f32[1, 16, 12, 64]" = torch.ops.aten.view.default(getitem_68, [1, 16, -1, 64]);  getitem_68 = None
        permute_26: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_13: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_14: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6: "f32[1, 1, 16, 16]" = torch.ops.aten.where.self(expand, full_default_14, full_default_13);  full_default_14 = full_default_13 = None
        expand_7: "f32[1, 12, 16, 16]" = torch.ops.aten.expand.default(where_6, [1, 12, 16, 16]);  where_6 = None
        _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_26, permute_24, permute_25, expand_7, False);  permute_26 = permute_24 = permute_25 = expand_7 = None
        getitem_71: "f32[1, 12, 16, 64]" = _scaled_dot_product_efficient_attention_6[0];  _scaled_dot_product_efficient_attention_6 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_27: "f32[1, 16, 12, 64]" = torch.ops.aten.permute.default(getitem_71, [0, 2, 1, 3]);  getitem_71 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_85: "f32[1, 16, 768]" = torch.ops.aten.view.default(permute_27, [1, 16, -1]);  permute_27 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_86: "f32[16, 768]" = torch.ops.aten.view.default(view_85, [-1, 768]);  view_85 = None
        addmm_25: "f32[16, 768]" = torch.ops.aten.addmm.default(arg79_1, view_86, arg80_1);  arg79_1 = view_86 = arg80_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_87: "f32[1, 16, 768]" = torch.ops.aten.view.default(addmm_25, [1, 16, 768]);  addmm_25 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_52: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(view_87, add_49);  view_87 = add_49 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_13 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
        getitem_75: "f32[1, 16, 1]" = var_mean_13[0]
        getitem_76: "f32[1, 16, 1]" = var_mean_13[1];  var_mean_13 = None
        add_53: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_75, 1e-05);  getitem_75 = None
        rsqrt_13: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_15: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_52, getitem_76);  getitem_76 = None
        mul_50: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_13);  sub_15 = rsqrt_13 = None
        mul_51: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_50, arg81_1);  mul_50 = arg81_1 = None
        add_54: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_51, arg82_1);  mul_51 = arg82_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_88: "f32[16, 768]" = torch.ops.aten.view.default(add_54, [-1, 768]);  add_54 = None
        addmm_26: "f32[16, 3072]" = torch.ops.aten.addmm.default(arg83_1, view_88, arg84_1);  arg83_1 = view_88 = arg84_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_89: "f32[1, 16, 3072]" = torch.ops.aten.view.default(addmm_26, [1, 16, 3072]);  addmm_26 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_52: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(view_89, 0.5)
        pow_7: "f32[1, 16, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_89, 3.0)
        mul_53: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_55: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(view_89, mul_53);  view_89 = mul_53 = None
        mul_54: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(add_55, 0.7978845608028654);  add_55 = None
        tanh_6: "f32[1, 16, 3072]" = torch.ops.aten.tanh.default(mul_54);  mul_54 = None
        add_56: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
        mul_55: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(mul_52, add_56);  mul_52 = add_56 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_90: "f32[16, 3072]" = torch.ops.aten.view.default(mul_55, [-1, 3072]);  mul_55 = None
        addmm_27: "f32[16, 768]" = torch.ops.aten.addmm.default(arg85_1, view_90, arg86_1);  arg85_1 = view_90 = arg86_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_91: "f32[1, 16, 768]" = torch.ops.aten.view.default(addmm_27, [1, 16, 768]);  addmm_27 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_57: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(add_52, view_91);  add_52 = view_91 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_14 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_77: "f32[1, 16, 1]" = var_mean_14[0]
        getitem_78: "f32[1, 16, 1]" = var_mean_14[1];  var_mean_14 = None
        add_58: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_77, 1e-05);  getitem_77 = None
        rsqrt_14: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_16: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_57, getitem_78);  getitem_78 = None
        mul_56: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_14);  sub_16 = rsqrt_14 = None
        mul_57: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_56, arg87_1);  mul_56 = arg87_1 = None
        add_59: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_57, arg88_1);  mul_57 = arg88_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_92: "f32[16, 768]" = torch.ops.aten.view.default(add_59, [-1, 768]);  add_59 = None
        addmm_28: "f32[16, 2304]" = torch.ops.aten.addmm.default(arg89_1, view_92, arg90_1);  arg89_1 = view_92 = arg90_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_93: "f32[1, 16, 2304]" = torch.ops.aten.view.default(addmm_28, [1, 16, 2304]);  addmm_28 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_7 = torch.ops.aten.split.Tensor(view_93, 768, 2);  view_93 = None
        getitem_79: "f32[1, 16, 768]" = split_7[0]
        getitem_80: "f32[1, 16, 768]" = split_7[1]
        getitem_81: "f32[1, 16, 768]" = split_7[2];  split_7 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_94: "f32[1, 16, 12, 64]" = torch.ops.aten.view.default(getitem_80, [1, 16, -1, 64]);  getitem_80 = None
        permute_28: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_95: "f32[1, 16, 12, 64]" = torch.ops.aten.view.default(getitem_81, [1, 16, -1, 64]);  getitem_81 = None
        permute_29: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_96: "f32[1, 16, 12, 64]" = torch.ops.aten.view.default(getitem_79, [1, 16, -1, 64]);  getitem_79 = None
        permute_30: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_15: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_16: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_7: "f32[1, 1, 16, 16]" = torch.ops.aten.where.self(expand, full_default_16, full_default_15);  full_default_16 = full_default_15 = None
        expand_8: "f32[1, 12, 16, 16]" = torch.ops.aten.expand.default(where_7, [1, 12, 16, 16]);  where_7 = None
        _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_30, permute_28, permute_29, expand_8, False);  permute_30 = permute_28 = permute_29 = expand_8 = None
        getitem_82: "f32[1, 12, 16, 64]" = _scaled_dot_product_efficient_attention_7[0];  _scaled_dot_product_efficient_attention_7 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_31: "f32[1, 16, 12, 64]" = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3]);  getitem_82 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_97: "f32[1, 16, 768]" = torch.ops.aten.view.default(permute_31, [1, 16, -1]);  permute_31 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_98: "f32[16, 768]" = torch.ops.aten.view.default(view_97, [-1, 768]);  view_97 = None
        addmm_29: "f32[16, 768]" = torch.ops.aten.addmm.default(arg91_1, view_98, arg92_1);  arg91_1 = view_98 = arg92_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_99: "f32[1, 16, 768]" = torch.ops.aten.view.default(addmm_29, [1, 16, 768]);  addmm_29 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_60: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(view_99, add_57);  view_99 = add_57 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_15 = torch.ops.aten.var_mean.correction(add_60, [2], correction = 0, keepdim = True)
        getitem_86: "f32[1, 16, 1]" = var_mean_15[0]
        getitem_87: "f32[1, 16, 1]" = var_mean_15[1];  var_mean_15 = None
        add_61: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
        rsqrt_15: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        sub_17: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_60, getitem_87);  getitem_87 = None
        mul_58: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_15);  sub_17 = rsqrt_15 = None
        mul_59: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_58, arg93_1);  mul_58 = arg93_1 = None
        add_62: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_59, arg94_1);  mul_59 = arg94_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_100: "f32[16, 768]" = torch.ops.aten.view.default(add_62, [-1, 768]);  add_62 = None
        addmm_30: "f32[16, 3072]" = torch.ops.aten.addmm.default(arg95_1, view_100, arg96_1);  arg95_1 = view_100 = arg96_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_101: "f32[1, 16, 3072]" = torch.ops.aten.view.default(addmm_30, [1, 16, 3072]);  addmm_30 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_60: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(view_101, 0.5)
        pow_8: "f32[1, 16, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_101, 3.0)
        mul_61: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_63: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(view_101, mul_61);  view_101 = mul_61 = None
        mul_62: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(add_63, 0.7978845608028654);  add_63 = None
        tanh_7: "f32[1, 16, 3072]" = torch.ops.aten.tanh.default(mul_62);  mul_62 = None
        add_64: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
        mul_63: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(mul_60, add_64);  mul_60 = add_64 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_102: "f32[16, 3072]" = torch.ops.aten.view.default(mul_63, [-1, 3072]);  mul_63 = None
        addmm_31: "f32[16, 768]" = torch.ops.aten.addmm.default(arg97_1, view_102, arg98_1);  arg97_1 = view_102 = arg98_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_103: "f32[1, 16, 768]" = torch.ops.aten.view.default(addmm_31, [1, 16, 768]);  addmm_31 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_65: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(add_60, view_103);  add_60 = view_103 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_16 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
        getitem_88: "f32[1, 16, 1]" = var_mean_16[0]
        getitem_89: "f32[1, 16, 1]" = var_mean_16[1];  var_mean_16 = None
        add_66: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_16: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        sub_18: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_65, getitem_89);  getitem_89 = None
        mul_64: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_16);  sub_18 = rsqrt_16 = None
        mul_65: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_64, arg99_1);  mul_64 = arg99_1 = None
        add_67: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_65, arg100_1);  mul_65 = arg100_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_104: "f32[16, 768]" = torch.ops.aten.view.default(add_67, [-1, 768]);  add_67 = None
        addmm_32: "f32[16, 2304]" = torch.ops.aten.addmm.default(arg101_1, view_104, arg102_1);  arg101_1 = view_104 = arg102_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_105: "f32[1, 16, 2304]" = torch.ops.aten.view.default(addmm_32, [1, 16, 2304]);  addmm_32 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_8 = torch.ops.aten.split.Tensor(view_105, 768, 2);  view_105 = None
        getitem_90: "f32[1, 16, 768]" = split_8[0]
        getitem_91: "f32[1, 16, 768]" = split_8[1]
        getitem_92: "f32[1, 16, 768]" = split_8[2];  split_8 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_106: "f32[1, 16, 12, 64]" = torch.ops.aten.view.default(getitem_91, [1, 16, -1, 64]);  getitem_91 = None
        permute_32: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_107: "f32[1, 16, 12, 64]" = torch.ops.aten.view.default(getitem_92, [1, 16, -1, 64]);  getitem_92 = None
        permute_33: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_107, [0, 2, 1, 3]);  view_107 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_108: "f32[1, 16, 12, 64]" = torch.ops.aten.view.default(getitem_90, [1, 16, -1, 64]);  getitem_90 = None
        permute_34: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_108, [0, 2, 1, 3]);  view_108 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_17: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_18: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_8: "f32[1, 1, 16, 16]" = torch.ops.aten.where.self(expand, full_default_18, full_default_17);  full_default_18 = full_default_17 = None
        expand_9: "f32[1, 12, 16, 16]" = torch.ops.aten.expand.default(where_8, [1, 12, 16, 16]);  where_8 = None
        _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_34, permute_32, permute_33, expand_9, False);  permute_34 = permute_32 = permute_33 = expand_9 = None
        getitem_93: "f32[1, 12, 16, 64]" = _scaled_dot_product_efficient_attention_8[0];  _scaled_dot_product_efficient_attention_8 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_35: "f32[1, 16, 12, 64]" = torch.ops.aten.permute.default(getitem_93, [0, 2, 1, 3]);  getitem_93 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_109: "f32[1, 16, 768]" = torch.ops.aten.view.default(permute_35, [1, 16, -1]);  permute_35 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_110: "f32[16, 768]" = torch.ops.aten.view.default(view_109, [-1, 768]);  view_109 = None
        addmm_33: "f32[16, 768]" = torch.ops.aten.addmm.default(arg103_1, view_110, arg104_1);  arg103_1 = view_110 = arg104_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_111: "f32[1, 16, 768]" = torch.ops.aten.view.default(addmm_33, [1, 16, 768]);  addmm_33 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_68: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(view_111, add_65);  view_111 = add_65 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_17 = torch.ops.aten.var_mean.correction(add_68, [2], correction = 0, keepdim = True)
        getitem_97: "f32[1, 16, 1]" = var_mean_17[0]
        getitem_98: "f32[1, 16, 1]" = var_mean_17[1];  var_mean_17 = None
        add_69: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_97, 1e-05);  getitem_97 = None
        rsqrt_17: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        sub_19: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_68, getitem_98);  getitem_98 = None
        mul_66: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_17);  sub_19 = rsqrt_17 = None
        mul_67: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_66, arg105_1);  mul_66 = arg105_1 = None
        add_70: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_67, arg106_1);  mul_67 = arg106_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_112: "f32[16, 768]" = torch.ops.aten.view.default(add_70, [-1, 768]);  add_70 = None
        addmm_34: "f32[16, 3072]" = torch.ops.aten.addmm.default(arg107_1, view_112, arg108_1);  arg107_1 = view_112 = arg108_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_113: "f32[1, 16, 3072]" = torch.ops.aten.view.default(addmm_34, [1, 16, 3072]);  addmm_34 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_68: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(view_113, 0.5)
        pow_9: "f32[1, 16, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_113, 3.0)
        mul_69: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_71: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(view_113, mul_69);  view_113 = mul_69 = None
        mul_70: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(add_71, 0.7978845608028654);  add_71 = None
        tanh_8: "f32[1, 16, 3072]" = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
        add_72: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
        mul_71: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(mul_68, add_72);  mul_68 = add_72 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_114: "f32[16, 3072]" = torch.ops.aten.view.default(mul_71, [-1, 3072]);  mul_71 = None
        addmm_35: "f32[16, 768]" = torch.ops.aten.addmm.default(arg109_1, view_114, arg110_1);  arg109_1 = view_114 = arg110_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_115: "f32[1, 16, 768]" = torch.ops.aten.view.default(addmm_35, [1, 16, 768]);  addmm_35 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_73: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(add_68, view_115);  add_68 = view_115 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_18 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
        getitem_99: "f32[1, 16, 1]" = var_mean_18[0]
        getitem_100: "f32[1, 16, 1]" = var_mean_18[1];  var_mean_18 = None
        add_74: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_99, 1e-05);  getitem_99 = None
        rsqrt_18: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_20: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_73, getitem_100);  getitem_100 = None
        mul_72: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_18);  sub_20 = rsqrt_18 = None
        mul_73: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_72, arg111_1);  mul_72 = arg111_1 = None
        add_75: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_73, arg112_1);  mul_73 = arg112_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_116: "f32[16, 768]" = torch.ops.aten.view.default(add_75, [-1, 768]);  add_75 = None
        addmm_36: "f32[16, 2304]" = torch.ops.aten.addmm.default(arg113_1, view_116, arg114_1);  arg113_1 = view_116 = arg114_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_117: "f32[1, 16, 2304]" = torch.ops.aten.view.default(addmm_36, [1, 16, 2304]);  addmm_36 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_9 = torch.ops.aten.split.Tensor(view_117, 768, 2);  view_117 = None
        getitem_101: "f32[1, 16, 768]" = split_9[0]
        getitem_102: "f32[1, 16, 768]" = split_9[1]
        getitem_103: "f32[1, 16, 768]" = split_9[2];  split_9 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_118: "f32[1, 16, 12, 64]" = torch.ops.aten.view.default(getitem_102, [1, 16, -1, 64]);  getitem_102 = None
        permute_36: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_119: "f32[1, 16, 12, 64]" = torch.ops.aten.view.default(getitem_103, [1, 16, -1, 64]);  getitem_103 = None
        permute_37: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_120: "f32[1, 16, 12, 64]" = torch.ops.aten.view.default(getitem_101, [1, 16, -1, 64]);  getitem_101 = None
        permute_38: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_19: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_20: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_9: "f32[1, 1, 16, 16]" = torch.ops.aten.where.self(expand, full_default_20, full_default_19);  full_default_20 = full_default_19 = None
        expand_10: "f32[1, 12, 16, 16]" = torch.ops.aten.expand.default(where_9, [1, 12, 16, 16]);  where_9 = None
        _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_38, permute_36, permute_37, expand_10, False);  permute_38 = permute_36 = permute_37 = expand_10 = None
        getitem_104: "f32[1, 12, 16, 64]" = _scaled_dot_product_efficient_attention_9[0];  _scaled_dot_product_efficient_attention_9 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_39: "f32[1, 16, 12, 64]" = torch.ops.aten.permute.default(getitem_104, [0, 2, 1, 3]);  getitem_104 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_121: "f32[1, 16, 768]" = torch.ops.aten.view.default(permute_39, [1, 16, -1]);  permute_39 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_122: "f32[16, 768]" = torch.ops.aten.view.default(view_121, [-1, 768]);  view_121 = None
        addmm_37: "f32[16, 768]" = torch.ops.aten.addmm.default(arg115_1, view_122, arg116_1);  arg115_1 = view_122 = arg116_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_123: "f32[1, 16, 768]" = torch.ops.aten.view.default(addmm_37, [1, 16, 768]);  addmm_37 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_76: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(view_123, add_73);  view_123 = add_73 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_19 = torch.ops.aten.var_mean.correction(add_76, [2], correction = 0, keepdim = True)
        getitem_108: "f32[1, 16, 1]" = var_mean_19[0]
        getitem_109: "f32[1, 16, 1]" = var_mean_19[1];  var_mean_19 = None
        add_77: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
        rsqrt_19: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        sub_21: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_76, getitem_109);  getitem_109 = None
        mul_74: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_19);  sub_21 = rsqrt_19 = None
        mul_75: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_74, arg117_1);  mul_74 = arg117_1 = None
        add_78: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_75, arg118_1);  mul_75 = arg118_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_124: "f32[16, 768]" = torch.ops.aten.view.default(add_78, [-1, 768]);  add_78 = None
        addmm_38: "f32[16, 3072]" = torch.ops.aten.addmm.default(arg119_1, view_124, arg120_1);  arg119_1 = view_124 = arg120_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_125: "f32[1, 16, 3072]" = torch.ops.aten.view.default(addmm_38, [1, 16, 3072]);  addmm_38 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_76: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(view_125, 0.5)
        pow_10: "f32[1, 16, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_125, 3.0)
        mul_77: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_79: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(view_125, mul_77);  view_125 = mul_77 = None
        mul_78: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(add_79, 0.7978845608028654);  add_79 = None
        tanh_9: "f32[1, 16, 3072]" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
        add_80: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
        mul_79: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(mul_76, add_80);  mul_76 = add_80 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_126: "f32[16, 3072]" = torch.ops.aten.view.default(mul_79, [-1, 3072]);  mul_79 = None
        addmm_39: "f32[16, 768]" = torch.ops.aten.addmm.default(arg121_1, view_126, arg122_1);  arg121_1 = view_126 = arg122_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_127: "f32[1, 16, 768]" = torch.ops.aten.view.default(addmm_39, [1, 16, 768]);  addmm_39 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_81: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(add_76, view_127);  add_76 = view_127 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_20 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
        getitem_110: "f32[1, 16, 1]" = var_mean_20[0]
        getitem_111: "f32[1, 16, 1]" = var_mean_20[1];  var_mean_20 = None
        add_82: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
        rsqrt_20: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        sub_22: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_81, getitem_111);  getitem_111 = None
        mul_80: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_20);  sub_22 = rsqrt_20 = None
        mul_81: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_80, arg123_1);  mul_80 = arg123_1 = None
        add_83: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_81, arg124_1);  mul_81 = arg124_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_128: "f32[16, 768]" = torch.ops.aten.view.default(add_83, [-1, 768]);  add_83 = None
        addmm_40: "f32[16, 2304]" = torch.ops.aten.addmm.default(arg125_1, view_128, arg126_1);  arg125_1 = view_128 = arg126_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_129: "f32[1, 16, 2304]" = torch.ops.aten.view.default(addmm_40, [1, 16, 2304]);  addmm_40 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_10 = torch.ops.aten.split.Tensor(view_129, 768, 2);  view_129 = None
        getitem_112: "f32[1, 16, 768]" = split_10[0]
        getitem_113: "f32[1, 16, 768]" = split_10[1]
        getitem_114: "f32[1, 16, 768]" = split_10[2];  split_10 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_130: "f32[1, 16, 12, 64]" = torch.ops.aten.view.default(getitem_113, [1, 16, -1, 64]);  getitem_113 = None
        permute_40: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_131: "f32[1, 16, 12, 64]" = torch.ops.aten.view.default(getitem_114, [1, 16, -1, 64]);  getitem_114 = None
        permute_41: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_131, [0, 2, 1, 3]);  view_131 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_132: "f32[1, 16, 12, 64]" = torch.ops.aten.view.default(getitem_112, [1, 16, -1, 64]);  getitem_112 = None
        permute_42: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_132, [0, 2, 1, 3]);  view_132 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_21: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_22: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_10: "f32[1, 1, 16, 16]" = torch.ops.aten.where.self(expand, full_default_22, full_default_21);  full_default_22 = full_default_21 = None
        expand_11: "f32[1, 12, 16, 16]" = torch.ops.aten.expand.default(where_10, [1, 12, 16, 16]);  where_10 = None
        _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_42, permute_40, permute_41, expand_11, False);  permute_42 = permute_40 = permute_41 = expand_11 = None
        getitem_115: "f32[1, 12, 16, 64]" = _scaled_dot_product_efficient_attention_10[0];  _scaled_dot_product_efficient_attention_10 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_43: "f32[1, 16, 12, 64]" = torch.ops.aten.permute.default(getitem_115, [0, 2, 1, 3]);  getitem_115 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_133: "f32[1, 16, 768]" = torch.ops.aten.view.default(permute_43, [1, 16, -1]);  permute_43 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_134: "f32[16, 768]" = torch.ops.aten.view.default(view_133, [-1, 768]);  view_133 = None
        addmm_41: "f32[16, 768]" = torch.ops.aten.addmm.default(arg127_1, view_134, arg128_1);  arg127_1 = view_134 = arg128_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_135: "f32[1, 16, 768]" = torch.ops.aten.view.default(addmm_41, [1, 16, 768]);  addmm_41 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_84: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(view_135, add_81);  view_135 = add_81 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_21 = torch.ops.aten.var_mean.correction(add_84, [2], correction = 0, keepdim = True)
        getitem_119: "f32[1, 16, 1]" = var_mean_21[0]
        getitem_120: "f32[1, 16, 1]" = var_mean_21[1];  var_mean_21 = None
        add_85: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_119, 1e-05);  getitem_119 = None
        rsqrt_21: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        sub_23: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_84, getitem_120);  getitem_120 = None
        mul_82: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_21);  sub_23 = rsqrt_21 = None
        mul_83: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_82, arg129_1);  mul_82 = arg129_1 = None
        add_86: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_83, arg130_1);  mul_83 = arg130_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_136: "f32[16, 768]" = torch.ops.aten.view.default(add_86, [-1, 768]);  add_86 = None
        addmm_42: "f32[16, 3072]" = torch.ops.aten.addmm.default(arg131_1, view_136, arg132_1);  arg131_1 = view_136 = arg132_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_137: "f32[1, 16, 3072]" = torch.ops.aten.view.default(addmm_42, [1, 16, 3072]);  addmm_42 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_84: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(view_137, 0.5)
        pow_11: "f32[1, 16, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_137, 3.0)
        mul_85: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_87: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(view_137, mul_85);  view_137 = mul_85 = None
        mul_86: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(add_87, 0.7978845608028654);  add_87 = None
        tanh_10: "f32[1, 16, 3072]" = torch.ops.aten.tanh.default(mul_86);  mul_86 = None
        add_88: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
        mul_87: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(mul_84, add_88);  mul_84 = add_88 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_138: "f32[16, 3072]" = torch.ops.aten.view.default(mul_87, [-1, 3072]);  mul_87 = None
        addmm_43: "f32[16, 768]" = torch.ops.aten.addmm.default(arg133_1, view_138, arg134_1);  arg133_1 = view_138 = arg134_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_139: "f32[1, 16, 768]" = torch.ops.aten.view.default(addmm_43, [1, 16, 768]);  addmm_43 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_89: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(add_84, view_139);  add_84 = view_139 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_22 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
        getitem_121: "f32[1, 16, 1]" = var_mean_22[0]
        getitem_122: "f32[1, 16, 1]" = var_mean_22[1];  var_mean_22 = None
        add_90: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_121, 1e-05);  getitem_121 = None
        rsqrt_22: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        sub_24: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_89, getitem_122);  getitem_122 = None
        mul_88: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_22);  sub_24 = rsqrt_22 = None
        mul_89: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_88, arg135_1);  mul_88 = arg135_1 = None
        add_91: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_89, arg136_1);  mul_89 = arg136_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_140: "f32[16, 768]" = torch.ops.aten.view.default(add_91, [-1, 768]);  add_91 = None
        addmm_44: "f32[16, 2304]" = torch.ops.aten.addmm.default(arg137_1, view_140, arg138_1);  arg137_1 = view_140 = arg138_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_141: "f32[1, 16, 2304]" = torch.ops.aten.view.default(addmm_44, [1, 16, 2304]);  addmm_44 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_11 = torch.ops.aten.split.Tensor(view_141, 768, 2);  view_141 = None
        getitem_123: "f32[1, 16, 768]" = split_11[0]
        getitem_124: "f32[1, 16, 768]" = split_11[1]
        getitem_125: "f32[1, 16, 768]" = split_11[2];  split_11 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_142: "f32[1, 16, 12, 64]" = torch.ops.aten.view.default(getitem_124, [1, 16, -1, 64]);  getitem_124 = None
        permute_44: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_143: "f32[1, 16, 12, 64]" = torch.ops.aten.view.default(getitem_125, [1, 16, -1, 64]);  getitem_125 = None
        permute_45: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_143, [0, 2, 1, 3]);  view_143 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_144: "f32[1, 16, 12, 64]" = torch.ops.aten.view.default(getitem_123, [1, 16, -1, 64]);  getitem_123 = None
        permute_46: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_144, [0, 2, 1, 3]);  view_144 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_23: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_24: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_11: "f32[1, 1, 16, 16]" = torch.ops.aten.where.self(expand, full_default_24, full_default_23);  expand = full_default_24 = full_default_23 = None
        expand_12: "f32[1, 12, 16, 16]" = torch.ops.aten.expand.default(where_11, [1, 12, 16, 16]);  where_11 = None
        _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_46, permute_44, permute_45, expand_12, False);  permute_46 = permute_44 = permute_45 = expand_12 = None
        getitem_126: "f32[1, 12, 16, 64]" = _scaled_dot_product_efficient_attention_11[0];  _scaled_dot_product_efficient_attention_11 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_47: "f32[1, 16, 12, 64]" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3]);  getitem_126 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_145: "f32[1, 16, 768]" = torch.ops.aten.view.default(permute_47, [1, 16, -1]);  permute_47 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_146: "f32[16, 768]" = torch.ops.aten.view.default(view_145, [-1, 768]);  view_145 = None
        addmm_45: "f32[16, 768]" = torch.ops.aten.addmm.default(arg139_1, view_146, arg140_1);  arg139_1 = view_146 = arg140_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_147: "f32[1, 16, 768]" = torch.ops.aten.view.default(addmm_45, [1, 16, 768]);  addmm_45 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_92: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(view_147, add_89);  view_147 = add_89 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_23 = torch.ops.aten.var_mean.correction(add_92, [2], correction = 0, keepdim = True)
        getitem_130: "f32[1, 16, 1]" = var_mean_23[0]
        getitem_131: "f32[1, 16, 1]" = var_mean_23[1];  var_mean_23 = None
        add_93: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
        rsqrt_23: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        sub_25: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_92, getitem_131);  getitem_131 = None
        mul_90: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_23);  sub_25 = rsqrt_23 = None
        mul_91: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_90, arg141_1);  mul_90 = arg141_1 = None
        add_94: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_91, arg142_1);  mul_91 = arg142_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_148: "f32[16, 768]" = torch.ops.aten.view.default(add_94, [-1, 768]);  add_94 = None
        addmm_46: "f32[16, 3072]" = torch.ops.aten.addmm.default(arg143_1, view_148, arg144_1);  arg143_1 = view_148 = arg144_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_149: "f32[1, 16, 3072]" = torch.ops.aten.view.default(addmm_46, [1, 16, 3072]);  addmm_46 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_92: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(view_149, 0.5)
        pow_12: "f32[1, 16, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_149, 3.0)
        mul_93: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_95: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(view_149, mul_93);  view_149 = mul_93 = None
        mul_94: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(add_95, 0.7978845608028654);  add_95 = None
        tanh_11: "f32[1, 16, 3072]" = torch.ops.aten.tanh.default(mul_94);  mul_94 = None
        add_96: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
        mul_95: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(mul_92, add_96);  mul_92 = add_96 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_150: "f32[16, 3072]" = torch.ops.aten.view.default(mul_95, [-1, 3072]);  mul_95 = None
        addmm_47: "f32[16, 768]" = torch.ops.aten.addmm.default(arg145_1, view_150, arg146_1);  arg145_1 = view_150 = arg146_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_151: "f32[1, 16, 768]" = torch.ops.aten.view.default(addmm_47, [1, 16, 768]);  addmm_47 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_97: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(add_92, view_151);  add_92 = view_151 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:953 in forward, code: hidden_states = self.ln_f(hidden_states)
        var_mean_24 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
        getitem_132: "f32[1, 16, 1]" = var_mean_24[0]
        getitem_133: "f32[1, 16, 1]" = var_mean_24[1];  var_mean_24 = None
        add_98: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-05);  getitem_132 = None
        rsqrt_24: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        sub_26: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_97, getitem_133);  add_97 = getitem_133 = None
        mul_96: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_24);  sub_26 = rsqrt_24 = None
        mul_97: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_96, arg147_1);  mul_96 = arg147_1 = None
        add_99: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_97, arg148_1);  mul_97 = arg148_1 = None
        return (add_99,)
        

# ===== inductor generated file: /tmp/cnnbench-transformers-ob_ijj07/repeat_03/a1/torchinductor/tmpuqb_59nb/tb/ctbeuecaa5nknwy6bvucluzl476p3mx7abciczu4tjqirt7duxrh.debug/fx_graph_runnable.py =====

import os
os.environ['TORCHINDUCTOR_FORCE_DISABLE_CACHES'] = '1'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/cnnbench-transformers-ob_ijj07/repeat_03/a1/torchinductor/tmpuqb_59nb'
os.environ['TRITON_CACHE_DIR'] = '/tmp/cnnbench-transformers-ob_ijj07/repeat_03/a1/torchinductor/tmpuqb_59nb/triton'

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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1):
        embedding = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None
        iota = torch.ops.prims.iota.default(16, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze = torch.ops.aten.unsqueeze.default(iota, 0)
        embedding_1 = torch.ops.aten.embedding.default(arg2_1, unsqueeze);  arg2_1 = None
        add = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        slice_1 = torch.ops.aten.slice.Tensor(unsqueeze, 1, 0, 1)
        sub = torch.ops.aten.sub.Tensor(slice_1, 1);  slice_1 = None
        cat = torch.ops.aten.cat.default([sub, unsqueeze], -1);  sub = unsqueeze = None
        slice_2 = torch.ops.aten.slice.Tensor(cat, -1, 0, 16)
        slice_3 = torch.ops.aten.slice.Tensor(cat, -1, 1, 17);  cat = None
        sub_1 = torch.ops.aten.sub.Tensor(slice_3, slice_2);  slice_3 = slice_2 = None
        ne = torch.ops.aten.ne.Scalar(sub_1, 1);  sub_1 = None
        cumsum = torch.ops.aten.cumsum.default(ne, -1);  ne = None
        iota_1 = torch.ops.prims.iota.default(16, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_1 = torch.ops.aten.add.Tensor(iota_1, 0);  iota_1 = None
        iota_2 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        full = torch.ops.aten.full.default([16], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full = None
        view_1 = torch.ops.aten.view.default(iota, [16, 1])
        le = torch.ops.aten.le.Tensor(add_1, view_1);  view_1 = None
        full_default = torch.ops.aten.full.default([16, 1], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        bitwise_and = torch.ops.aten.bitwise_and.Tensor(full_default, le);  full_default = le = None
        view_3 = torch.ops.aten.view.default(iota_2, [1, 1])
        index = torch.ops.aten.index.Tensor(cumsum, [view_3, iota]);  view_3 = iota = None
        view_4 = torch.ops.aten.view.default(iota_2, [1, 1]);  iota_2 = None
        index_1 = torch.ops.aten.index.Tensor(cumsum, [view_4, add_1]);  cumsum = view_4 = add_1 = None
        view_5 = torch.ops.aten.view.default(index, [1, 16, 1]);  index = None
        view_6 = torch.ops.aten.view.default(index_1, [1, 1, 16]);  index_1 = None
        eq = torch.ops.aten.eq.Tensor(view_5, view_6);  view_5 = view_6 = None
        bitwise_and_1 = torch.ops.aten.bitwise_and.Tensor(bitwise_and, eq);  bitwise_and = eq = None
        view_7 = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 16, 16]);  bitwise_and_1 = None
        expand = torch.ops.aten.expand.default(view_7, [1, 1, 16, 16]);  view_7 = None
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_2 = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub_2 = torch.ops.aten.sub.Tensor(add, getitem_1);  getitem_1 = None
        mul = torch.ops.aten.mul.Tensor(sub_2, rsqrt);  sub_2 = rsqrt = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = arg3_1 = None
        add_3 = torch.ops.aten.add.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None
        view_8 = torch.ops.aten.view.default(add_3, [-1, 768]);  add_3 = None
        addmm = torch.ops.aten.addmm.default(arg5_1, view_8, arg6_1);  arg5_1 = view_8 = arg6_1 = None
        view_9 = torch.ops.aten.view.default(addmm, [1, 16, 2304]);  addmm = None
        split = torch.ops.aten.split.Tensor(view_9, 768, 2);  view_9 = None
        getitem_2 = split[0]
        getitem_3 = split[1]
        getitem_4 = split[2];  split = None
        view_10 = torch.ops.aten.view.default(getitem_3, [1, 16, -1, 64]);  getitem_3 = None
        permute = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        view_11 = torch.ops.aten.view.default(getitem_4, [1, 16, -1, 64]);  getitem_4 = None
        permute_1 = torch.ops.aten.permute.default(view_11, [0, 2, 1, 3]);  view_11 = None
        view_12 = torch.ops.aten.view.default(getitem_2, [1, 16, -1, 64]);  getitem_2 = None
        permute_2 = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
        full_default_1 = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(expand, full_default_2, full_default_1);  full_default_2 = full_default_1 = None
        expand_1 = torch.ops.aten.expand.default(where, [1, 12, 16, 16]);  where = None
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_2, permute, permute_1, expand_1, False);  permute_2 = permute = permute_1 = expand_1 = None
        getitem_5 = _scaled_dot_product_efficient_attention[0];  _scaled_dot_product_efficient_attention = None
        permute_3 = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None
        view_13 = torch.ops.aten.view.default(permute_3, [1, 16, -1]);  permute_3 = None
        view_14 = torch.ops.aten.view.default(view_13, [-1, 768]);  view_13 = None
        addmm_1 = torch.ops.aten.addmm.default(arg7_1, view_14, arg8_1);  arg7_1 = view_14 = arg8_1 = None
        view_15 = torch.ops.aten.view.default(addmm_1, [1, 16, 768]);  addmm_1 = None
        add_4 = torch.ops.aten.add.Tensor(view_15, add);  view_15 = add = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
        getitem_9 = var_mean_1[0]
        getitem_10 = var_mean_1[1];  var_mean_1 = None
        add_5 = torch.ops.aten.add.Tensor(getitem_9, 1e-05);  getitem_9 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_3 = torch.ops.aten.sub.Tensor(add_4, getitem_10);  getitem_10 = None
        mul_2 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = rsqrt_1 = None
        mul_3 = torch.ops.aten.mul.Tensor(mul_2, arg9_1);  mul_2 = arg9_1 = None
        add_6 = torch.ops.aten.add.Tensor(mul_3, arg10_1);  mul_3 = arg10_1 = None
        view_16 = torch.ops.aten.view.default(add_6, [-1, 768]);  add_6 = None
        addmm_2 = torch.ops.aten.addmm.default(arg11_1, view_16, arg12_1);  arg11_1 = view_16 = arg12_1 = None
        view_17 = torch.ops.aten.view.default(addmm_2, [1, 16, 3072]);  addmm_2 = None
        mul_4 = torch.ops.aten.mul.Tensor(view_17, 0.5)
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(view_17, 3.0)
        mul_5 = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_7 = torch.ops.aten.add.Tensor(view_17, mul_5);  view_17 = mul_5 = None
        mul_6 = torch.ops.aten.mul.Tensor(add_7, 0.7978845608028654);  add_7 = None
        tanh = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
        add_8 = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_7 = torch.ops.aten.mul.Tensor(mul_4, add_8);  mul_4 = add_8 = None
        view_18 = torch.ops.aten.view.default(mul_7, [-1, 3072]);  mul_7 = None
        addmm_3 = torch.ops.aten.addmm.default(arg13_1, view_18, arg14_1);  arg13_1 = view_18 = arg14_1 = None
        view_19 = torch.ops.aten.view.default(addmm_3, [1, 16, 768]);  addmm_3 = None
        add_9 = torch.ops.aten.add.Tensor(add_4, view_19);  add_4 = view_19 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
        getitem_11 = var_mean_2[0]
        getitem_12 = var_mean_2[1];  var_mean_2 = None
        add_10 = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_4 = torch.ops.aten.sub.Tensor(add_9, getitem_12);  getitem_12 = None
        mul_8 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = rsqrt_2 = None
        mul_9 = torch.ops.aten.mul.Tensor(mul_8, arg15_1);  mul_8 = arg15_1 = None
        add_11 = torch.ops.aten.add.Tensor(mul_9, arg16_1);  mul_9 = arg16_1 = None
        view_20 = torch.ops.aten.view.default(add_11, [-1, 768]);  add_11 = None
        addmm_4 = torch.ops.aten.addmm.default(arg17_1, view_20, arg18_1);  arg17_1 = view_20 = arg18_1 = None
        view_21 = torch.ops.aten.view.default(addmm_4, [1, 16, 2304]);  addmm_4 = None
        split_1 = torch.ops.aten.split.Tensor(view_21, 768, 2);  view_21 = None
        getitem_13 = split_1[0]
        getitem_14 = split_1[1]
        getitem_15 = split_1[2];  split_1 = None
        view_22 = torch.ops.aten.view.default(getitem_14, [1, 16, -1, 64]);  getitem_14 = None
        permute_4 = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None
        view_23 = torch.ops.aten.view.default(getitem_15, [1, 16, -1, 64]);  getitem_15 = None
        permute_5 = torch.ops.aten.permute.default(view_23, [0, 2, 1, 3]);  view_23 = None
        view_24 = torch.ops.aten.view.default(getitem_13, [1, 16, -1, 64]);  getitem_13 = None
        permute_6 = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        full_default_3 = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_4 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(expand, full_default_4, full_default_3);  full_default_4 = full_default_3 = None
        expand_2 = torch.ops.aten.expand.default(where_1, [1, 12, 16, 16]);  where_1 = None
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_6, permute_4, permute_5, expand_2, False);  permute_6 = permute_4 = permute_5 = expand_2 = None
        getitem_16 = _scaled_dot_product_efficient_attention_1[0];  _scaled_dot_product_efficient_attention_1 = None
        permute_7 = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3]);  getitem_16 = None
        view_25 = torch.ops.aten.view.default(permute_7, [1, 16, -1]);  permute_7 = None
        view_26 = torch.ops.aten.view.default(view_25, [-1, 768]);  view_25 = None
        addmm_5 = torch.ops.aten.addmm.default(arg19_1, view_26, arg20_1);  arg19_1 = view_26 = arg20_1 = None
        view_27 = torch.ops.aten.view.default(addmm_5, [1, 16, 768]);  addmm_5 = None
        add_12 = torch.ops.aten.add.Tensor(view_27, add_9);  view_27 = add_9 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_12, [2], correction = 0, keepdim = True)
        getitem_20 = var_mean_3[0]
        getitem_21 = var_mean_3[1];  var_mean_3 = None
        add_13 = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        sub_5 = torch.ops.aten.sub.Tensor(add_12, getitem_21);  getitem_21 = None
        mul_10 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_11 = torch.ops.aten.mul.Tensor(mul_10, arg21_1);  mul_10 = arg21_1 = None
        add_14 = torch.ops.aten.add.Tensor(mul_11, arg22_1);  mul_11 = arg22_1 = None
        view_28 = torch.ops.aten.view.default(add_14, [-1, 768]);  add_14 = None
        addmm_6 = torch.ops.aten.addmm.default(arg23_1, view_28, arg24_1);  arg23_1 = view_28 = arg24_1 = None
        view_29 = torch.ops.aten.view.default(addmm_6, [1, 16, 3072]);  addmm_6 = None
        mul_12 = torch.ops.aten.mul.Tensor(view_29, 0.5)
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(view_29, 3.0)
        mul_13 = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_15 = torch.ops.aten.add.Tensor(view_29, mul_13);  view_29 = mul_13 = None
        mul_14 = torch.ops.aten.mul.Tensor(add_15, 0.7978845608028654);  add_15 = None
        tanh_1 = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
        add_16 = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_15 = torch.ops.aten.mul.Tensor(mul_12, add_16);  mul_12 = add_16 = None
        view_30 = torch.ops.aten.view.default(mul_15, [-1, 3072]);  mul_15 = None
        addmm_7 = torch.ops.aten.addmm.default(arg25_1, view_30, arg26_1);  arg25_1 = view_30 = arg26_1 = None
        view_31 = torch.ops.aten.view.default(addmm_7, [1, 16, 768]);  addmm_7 = None
        add_17 = torch.ops.aten.add.Tensor(add_12, view_31);  add_12 = view_31 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_22 = var_mean_4[0]
        getitem_23 = var_mean_4[1];  var_mean_4 = None
        add_18 = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_6 = torch.ops.aten.sub.Tensor(add_17, getitem_23);  getitem_23 = None
        mul_16 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
        mul_17 = torch.ops.aten.mul.Tensor(mul_16, arg27_1);  mul_16 = arg27_1 = None
        add_19 = torch.ops.aten.add.Tensor(mul_17, arg28_1);  mul_17 = arg28_1 = None
        view_32 = torch.ops.aten.view.default(add_19, [-1, 768]);  add_19 = None
        addmm_8 = torch.ops.aten.addmm.default(arg29_1, view_32, arg30_1);  arg29_1 = view_32 = arg30_1 = None
        view_33 = torch.ops.aten.view.default(addmm_8, [1, 16, 2304]);  addmm_8 = None
        split_2 = torch.ops.aten.split.Tensor(view_33, 768, 2);  view_33 = None
        getitem_24 = split_2[0]
        getitem_25 = split_2[1]
        getitem_26 = split_2[2];  split_2 = None
        view_34 = torch.ops.aten.view.default(getitem_25, [1, 16, -1, 64]);  getitem_25 = None
        permute_8 = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None
        view_35 = torch.ops.aten.view.default(getitem_26, [1, 16, -1, 64]);  getitem_26 = None
        permute_9 = torch.ops.aten.permute.default(view_35, [0, 2, 1, 3]);  view_35 = None
        view_36 = torch.ops.aten.view.default(getitem_24, [1, 16, -1, 64]);  getitem_24 = None
        permute_10 = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
        full_default_5 = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_6 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2 = torch.ops.aten.where.self(expand, full_default_6, full_default_5);  full_default_6 = full_default_5 = None
        expand_3 = torch.ops.aten.expand.default(where_2, [1, 12, 16, 16]);  where_2 = None
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_10, permute_8, permute_9, expand_3, False);  permute_10 = permute_8 = permute_9 = expand_3 = None
        getitem_27 = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None
        permute_11 = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3]);  getitem_27 = None
        view_37 = torch.ops.aten.view.default(permute_11, [1, 16, -1]);  permute_11 = None
        view_38 = torch.ops.aten.view.default(view_37, [-1, 768]);  view_37 = None
        addmm_9 = torch.ops.aten.addmm.default(arg31_1, view_38, arg32_1);  arg31_1 = view_38 = arg32_1 = None
        view_39 = torch.ops.aten.view.default(addmm_9, [1, 16, 768]);  addmm_9 = None
        add_20 = torch.ops.aten.add.Tensor(view_39, add_17);  view_39 = add_17 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_20, [2], correction = 0, keepdim = True)
        getitem_31 = var_mean_5[0]
        getitem_32 = var_mean_5[1];  var_mean_5 = None
        add_21 = torch.ops.aten.add.Tensor(getitem_31, 1e-05);  getitem_31 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_7 = torch.ops.aten.sub.Tensor(add_20, getitem_32);  getitem_32 = None
        mul_18 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_5);  sub_7 = rsqrt_5 = None
        mul_19 = torch.ops.aten.mul.Tensor(mul_18, arg33_1);  mul_18 = arg33_1 = None
        add_22 = torch.ops.aten.add.Tensor(mul_19, arg34_1);  mul_19 = arg34_1 = None
        view_40 = torch.ops.aten.view.default(add_22, [-1, 768]);  add_22 = None
        addmm_10 = torch.ops.aten.addmm.default(arg35_1, view_40, arg36_1);  arg35_1 = view_40 = arg36_1 = None
        view_41 = torch.ops.aten.view.default(addmm_10, [1, 16, 3072]);  addmm_10 = None
        mul_20 = torch.ops.aten.mul.Tensor(view_41, 0.5)
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(view_41, 3.0)
        mul_21 = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_23 = torch.ops.aten.add.Tensor(view_41, mul_21);  view_41 = mul_21 = None
        mul_22 = torch.ops.aten.mul.Tensor(add_23, 0.7978845608028654);  add_23 = None
        tanh_2 = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
        add_24 = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_23 = torch.ops.aten.mul.Tensor(mul_20, add_24);  mul_20 = add_24 = None
        view_42 = torch.ops.aten.view.default(mul_23, [-1, 3072]);  mul_23 = None
        addmm_11 = torch.ops.aten.addmm.default(arg37_1, view_42, arg38_1);  arg37_1 = view_42 = arg38_1 = None
        view_43 = torch.ops.aten.view.default(addmm_11, [1, 16, 768]);  addmm_11 = None
        add_25 = torch.ops.aten.add.Tensor(add_20, view_43);  add_20 = view_43 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_33 = var_mean_6[0]
        getitem_34 = var_mean_6[1];  var_mean_6 = None
        add_26 = torch.ops.aten.add.Tensor(getitem_33, 1e-05);  getitem_33 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_8 = torch.ops.aten.sub.Tensor(add_25, getitem_34);  getitem_34 = None
        mul_24 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_6);  sub_8 = rsqrt_6 = None
        mul_25 = torch.ops.aten.mul.Tensor(mul_24, arg39_1);  mul_24 = arg39_1 = None
        add_27 = torch.ops.aten.add.Tensor(mul_25, arg40_1);  mul_25 = arg40_1 = None
        view_44 = torch.ops.aten.view.default(add_27, [-1, 768]);  add_27 = None
        addmm_12 = torch.ops.aten.addmm.default(arg41_1, view_44, arg42_1);  arg41_1 = view_44 = arg42_1 = None
        view_45 = torch.ops.aten.view.default(addmm_12, [1, 16, 2304]);  addmm_12 = None
        split_3 = torch.ops.aten.split.Tensor(view_45, 768, 2);  view_45 = None
        getitem_35 = split_3[0]
        getitem_36 = split_3[1]
        getitem_37 = split_3[2];  split_3 = None
        view_46 = torch.ops.aten.view.default(getitem_36, [1, 16, -1, 64]);  getitem_36 = None
        permute_12 = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None
        view_47 = torch.ops.aten.view.default(getitem_37, [1, 16, -1, 64]);  getitem_37 = None
        permute_13 = torch.ops.aten.permute.default(view_47, [0, 2, 1, 3]);  view_47 = None
        view_48 = torch.ops.aten.view.default(getitem_35, [1, 16, -1, 64]);  getitem_35 = None
        permute_14 = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
        full_default_7 = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_8 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3 = torch.ops.aten.where.self(expand, full_default_8, full_default_7);  full_default_8 = full_default_7 = None
        expand_4 = torch.ops.aten.expand.default(where_3, [1, 12, 16, 16]);  where_3 = None
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_14, permute_12, permute_13, expand_4, False);  permute_14 = permute_12 = permute_13 = expand_4 = None
        getitem_38 = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None
        permute_15 = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3]);  getitem_38 = None
        view_49 = torch.ops.aten.view.default(permute_15, [1, 16, -1]);  permute_15 = None
        view_50 = torch.ops.aten.view.default(view_49, [-1, 768]);  view_49 = None
        addmm_13 = torch.ops.aten.addmm.default(arg43_1, view_50, arg44_1);  arg43_1 = view_50 = arg44_1 = None
        view_51 = torch.ops.aten.view.default(addmm_13, [1, 16, 768]);  addmm_13 = None
        add_28 = torch.ops.aten.add.Tensor(view_51, add_25);  view_51 = add_25 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
        getitem_42 = var_mean_7[0]
        getitem_43 = var_mean_7[1];  var_mean_7 = None
        add_29 = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        sub_9 = torch.ops.aten.sub.Tensor(add_28, getitem_43);  getitem_43 = None
        mul_26 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_7);  sub_9 = rsqrt_7 = None
        mul_27 = torch.ops.aten.mul.Tensor(mul_26, arg45_1);  mul_26 = arg45_1 = None
        add_30 = torch.ops.aten.add.Tensor(mul_27, arg46_1);  mul_27 = arg46_1 = None
        view_52 = torch.ops.aten.view.default(add_30, [-1, 768]);  add_30 = None
        addmm_14 = torch.ops.aten.addmm.default(arg47_1, view_52, arg48_1);  arg47_1 = view_52 = arg48_1 = None
        view_53 = torch.ops.aten.view.default(addmm_14, [1, 16, 3072]);  addmm_14 = None
        mul_28 = torch.ops.aten.mul.Tensor(view_53, 0.5)
        pow_4 = torch.ops.aten.pow.Tensor_Scalar(view_53, 3.0)
        mul_29 = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_31 = torch.ops.aten.add.Tensor(view_53, mul_29);  view_53 = mul_29 = None
        mul_30 = torch.ops.aten.mul.Tensor(add_31, 0.7978845608028654);  add_31 = None
        tanh_3 = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
        add_32 = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_31 = torch.ops.aten.mul.Tensor(mul_28, add_32);  mul_28 = add_32 = None
        view_54 = torch.ops.aten.view.default(mul_31, [-1, 3072]);  mul_31 = None
        addmm_15 = torch.ops.aten.addmm.default(arg49_1, view_54, arg50_1);  arg49_1 = view_54 = arg50_1 = None
        view_55 = torch.ops.aten.view.default(addmm_15, [1, 16, 768]);  addmm_15 = None
        add_33 = torch.ops.aten.add.Tensor(add_28, view_55);  add_28 = view_55 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
        getitem_44 = var_mean_8[0]
        getitem_45 = var_mean_8[1];  var_mean_8 = None
        add_34 = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        sub_10 = torch.ops.aten.sub.Tensor(add_33, getitem_45);  getitem_45 = None
        mul_32 = torch.ops.aten.mul.Tensor(sub_10, rsqrt_8);  sub_10 = rsqrt_8 = None
        mul_33 = torch.ops.aten.mul.Tensor(mul_32, arg51_1);  mul_32 = arg51_1 = None
        add_35 = torch.ops.aten.add.Tensor(mul_33, arg52_1);  mul_33 = arg52_1 = None
        view_56 = torch.ops.aten.view.default(add_35, [-1, 768]);  add_35 = None
        addmm_16 = torch.ops.aten.addmm.default(arg53_1, view_56, arg54_1);  arg53_1 = view_56 = arg54_1 = None
        view_57 = torch.ops.aten.view.default(addmm_16, [1, 16, 2304]);  addmm_16 = None
        split_4 = torch.ops.aten.split.Tensor(view_57, 768, 2);  view_57 = None
        getitem_46 = split_4[0]
        getitem_47 = split_4[1]
        getitem_48 = split_4[2];  split_4 = None
        view_58 = torch.ops.aten.view.default(getitem_47, [1, 16, -1, 64]);  getitem_47 = None
        permute_16 = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
        view_59 = torch.ops.aten.view.default(getitem_48, [1, 16, -1, 64]);  getitem_48 = None
        permute_17 = torch.ops.aten.permute.default(view_59, [0, 2, 1, 3]);  view_59 = None
        view_60 = torch.ops.aten.view.default(getitem_46, [1, 16, -1, 64]);  getitem_46 = None
        permute_18 = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
        full_default_9 = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_10 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4 = torch.ops.aten.where.self(expand, full_default_10, full_default_9);  full_default_10 = full_default_9 = None
        expand_5 = torch.ops.aten.expand.default(where_4, [1, 12, 16, 16]);  where_4 = None
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_18, permute_16, permute_17, expand_5, False);  permute_18 = permute_16 = permute_17 = expand_5 = None
        getitem_49 = _scaled_dot_product_efficient_attention_4[0];  _scaled_dot_product_efficient_attention_4 = None
        permute_19 = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3]);  getitem_49 = None
        view_61 = torch.ops.aten.view.default(permute_19, [1, 16, -1]);  permute_19 = None
        view_62 = torch.ops.aten.view.default(view_61, [-1, 768]);  view_61 = None
        addmm_17 = torch.ops.aten.addmm.default(arg55_1, view_62, arg56_1);  arg55_1 = view_62 = arg56_1 = None
        view_63 = torch.ops.aten.view.default(addmm_17, [1, 16, 768]);  addmm_17 = None
        add_36 = torch.ops.aten.add.Tensor(view_63, add_33);  view_63 = add_33 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_36, [2], correction = 0, keepdim = True)
        getitem_53 = var_mean_9[0]
        getitem_54 = var_mean_9[1];  var_mean_9 = None
        add_37 = torch.ops.aten.add.Tensor(getitem_53, 1e-05);  getitem_53 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        sub_11 = torch.ops.aten.sub.Tensor(add_36, getitem_54);  getitem_54 = None
        mul_34 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_9);  sub_11 = rsqrt_9 = None
        mul_35 = torch.ops.aten.mul.Tensor(mul_34, arg57_1);  mul_34 = arg57_1 = None
        add_38 = torch.ops.aten.add.Tensor(mul_35, arg58_1);  mul_35 = arg58_1 = None
        view_64 = torch.ops.aten.view.default(add_38, [-1, 768]);  add_38 = None
        addmm_18 = torch.ops.aten.addmm.default(arg59_1, view_64, arg60_1);  arg59_1 = view_64 = arg60_1 = None
        view_65 = torch.ops.aten.view.default(addmm_18, [1, 16, 3072]);  addmm_18 = None
        mul_36 = torch.ops.aten.mul.Tensor(view_65, 0.5)
        pow_5 = torch.ops.aten.pow.Tensor_Scalar(view_65, 3.0)
        mul_37 = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_39 = torch.ops.aten.add.Tensor(view_65, mul_37);  view_65 = mul_37 = None
        mul_38 = torch.ops.aten.mul.Tensor(add_39, 0.7978845608028654);  add_39 = None
        tanh_4 = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_40 = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_39 = torch.ops.aten.mul.Tensor(mul_36, add_40);  mul_36 = add_40 = None
        view_66 = torch.ops.aten.view.default(mul_39, [-1, 3072]);  mul_39 = None
        addmm_19 = torch.ops.aten.addmm.default(arg61_1, view_66, arg62_1);  arg61_1 = view_66 = arg62_1 = None
        view_67 = torch.ops.aten.view.default(addmm_19, [1, 16, 768]);  addmm_19 = None
        add_41 = torch.ops.aten.add.Tensor(add_36, view_67);  add_36 = view_67 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
        getitem_55 = var_mean_10[0]
        getitem_56 = var_mean_10[1];  var_mean_10 = None
        add_42 = torch.ops.aten.add.Tensor(getitem_55, 1e-05);  getitem_55 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_12 = torch.ops.aten.sub.Tensor(add_41, getitem_56);  getitem_56 = None
        mul_40 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_10);  sub_12 = rsqrt_10 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_40, arg63_1);  mul_40 = arg63_1 = None
        add_43 = torch.ops.aten.add.Tensor(mul_41, arg64_1);  mul_41 = arg64_1 = None
        view_68 = torch.ops.aten.view.default(add_43, [-1, 768]);  add_43 = None
        addmm_20 = torch.ops.aten.addmm.default(arg65_1, view_68, arg66_1);  arg65_1 = view_68 = arg66_1 = None
        view_69 = torch.ops.aten.view.default(addmm_20, [1, 16, 2304]);  addmm_20 = None
        split_5 = torch.ops.aten.split.Tensor(view_69, 768, 2);  view_69 = None
        getitem_57 = split_5[0]
        getitem_58 = split_5[1]
        getitem_59 = split_5[2];  split_5 = None
        view_70 = torch.ops.aten.view.default(getitem_58, [1, 16, -1, 64]);  getitem_58 = None
        permute_20 = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
        view_71 = torch.ops.aten.view.default(getitem_59, [1, 16, -1, 64]);  getitem_59 = None
        permute_21 = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
        view_72 = torch.ops.aten.view.default(getitem_57, [1, 16, -1, 64]);  getitem_57 = None
        permute_22 = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
        full_default_11 = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_12 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5 = torch.ops.aten.where.self(expand, full_default_12, full_default_11);  full_default_12 = full_default_11 = None
        expand_6 = torch.ops.aten.expand.default(where_5, [1, 12, 16, 16]);  where_5 = None
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_22, permute_20, permute_21, expand_6, False);  permute_22 = permute_20 = permute_21 = expand_6 = None
        getitem_60 = _scaled_dot_product_efficient_attention_5[0];  _scaled_dot_product_efficient_attention_5 = None
        permute_23 = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None
        view_73 = torch.ops.aten.view.default(permute_23, [1, 16, -1]);  permute_23 = None
        view_74 = torch.ops.aten.view.default(view_73, [-1, 768]);  view_73 = None
        addmm_21 = torch.ops.aten.addmm.default(arg67_1, view_74, arg68_1);  arg67_1 = view_74 = arg68_1 = None
        view_75 = torch.ops.aten.view.default(addmm_21, [1, 16, 768]);  addmm_21 = None
        add_44 = torch.ops.aten.add.Tensor(view_75, add_41);  view_75 = add_41 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_44, [2], correction = 0, keepdim = True)
        getitem_64 = var_mean_11[0]
        getitem_65 = var_mean_11[1];  var_mean_11 = None
        add_45 = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        sub_13 = torch.ops.aten.sub.Tensor(add_44, getitem_65);  getitem_65 = None
        mul_42 = torch.ops.aten.mul.Tensor(sub_13, rsqrt_11);  sub_13 = rsqrt_11 = None
        mul_43 = torch.ops.aten.mul.Tensor(mul_42, arg69_1);  mul_42 = arg69_1 = None
        add_46 = torch.ops.aten.add.Tensor(mul_43, arg70_1);  mul_43 = arg70_1 = None
        view_76 = torch.ops.aten.view.default(add_46, [-1, 768]);  add_46 = None
        addmm_22 = torch.ops.aten.addmm.default(arg71_1, view_76, arg72_1);  arg71_1 = view_76 = arg72_1 = None
        view_77 = torch.ops.aten.view.default(addmm_22, [1, 16, 3072]);  addmm_22 = None
        mul_44 = torch.ops.aten.mul.Tensor(view_77, 0.5)
        pow_6 = torch.ops.aten.pow.Tensor_Scalar(view_77, 3.0)
        mul_45 = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_47 = torch.ops.aten.add.Tensor(view_77, mul_45);  view_77 = mul_45 = None
        mul_46 = torch.ops.aten.mul.Tensor(add_47, 0.7978845608028654);  add_47 = None
        tanh_5 = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
        add_48 = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_47 = torch.ops.aten.mul.Tensor(mul_44, add_48);  mul_44 = add_48 = None
        view_78 = torch.ops.aten.view.default(mul_47, [-1, 3072]);  mul_47 = None
        addmm_23 = torch.ops.aten.addmm.default(arg73_1, view_78, arg74_1);  arg73_1 = view_78 = arg74_1 = None
        view_79 = torch.ops.aten.view.default(addmm_23, [1, 16, 768]);  addmm_23 = None
        add_49 = torch.ops.aten.add.Tensor(add_44, view_79);  add_44 = view_79 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
        getitem_66 = var_mean_12[0]
        getitem_67 = var_mean_12[1];  var_mean_12 = None
        add_50 = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        sub_14 = torch.ops.aten.sub.Tensor(add_49, getitem_67);  getitem_67 = None
        mul_48 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_12);  sub_14 = rsqrt_12 = None
        mul_49 = torch.ops.aten.mul.Tensor(mul_48, arg75_1);  mul_48 = arg75_1 = None
        add_51 = torch.ops.aten.add.Tensor(mul_49, arg76_1);  mul_49 = arg76_1 = None
        view_80 = torch.ops.aten.view.default(add_51, [-1, 768]);  add_51 = None
        addmm_24 = torch.ops.aten.addmm.default(arg77_1, view_80, arg78_1);  arg77_1 = view_80 = arg78_1 = None
        view_81 = torch.ops.aten.view.default(addmm_24, [1, 16, 2304]);  addmm_24 = None
        split_6 = torch.ops.aten.split.Tensor(view_81, 768, 2);  view_81 = None
        getitem_68 = split_6[0]
        getitem_69 = split_6[1]
        getitem_70 = split_6[2];  split_6 = None
        view_82 = torch.ops.aten.view.default(getitem_69, [1, 16, -1, 64]);  getitem_69 = None
        permute_24 = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
        view_83 = torch.ops.aten.view.default(getitem_70, [1, 16, -1, 64]);  getitem_70 = None
        permute_25 = torch.ops.aten.permute.default(view_83, [0, 2, 1, 3]);  view_83 = None
        view_84 = torch.ops.aten.view.default(getitem_68, [1, 16, -1, 64]);  getitem_68 = None
        permute_26 = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None
        full_default_13 = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_14 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6 = torch.ops.aten.where.self(expand, full_default_14, full_default_13);  full_default_14 = full_default_13 = None
        expand_7 = torch.ops.aten.expand.default(where_6, [1, 12, 16, 16]);  where_6 = None
        _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_26, permute_24, permute_25, expand_7, False);  permute_26 = permute_24 = permute_25 = expand_7 = None
        getitem_71 = _scaled_dot_product_efficient_attention_6[0];  _scaled_dot_product_efficient_attention_6 = None
        permute_27 = torch.ops.aten.permute.default(getitem_71, [0, 2, 1, 3]);  getitem_71 = None
        view_85 = torch.ops.aten.view.default(permute_27, [1, 16, -1]);  permute_27 = None
        view_86 = torch.ops.aten.view.default(view_85, [-1, 768]);  view_85 = None
        addmm_25 = torch.ops.aten.addmm.default(arg79_1, view_86, arg80_1);  arg79_1 = view_86 = arg80_1 = None
        view_87 = torch.ops.aten.view.default(addmm_25, [1, 16, 768]);  addmm_25 = None
        add_52 = torch.ops.aten.add.Tensor(view_87, add_49);  view_87 = add_49 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
        getitem_75 = var_mean_13[0]
        getitem_76 = var_mean_13[1];  var_mean_13 = None
        add_53 = torch.ops.aten.add.Tensor(getitem_75, 1e-05);  getitem_75 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_15 = torch.ops.aten.sub.Tensor(add_52, getitem_76);  getitem_76 = None
        mul_50 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_13);  sub_15 = rsqrt_13 = None
        mul_51 = torch.ops.aten.mul.Tensor(mul_50, arg81_1);  mul_50 = arg81_1 = None
        add_54 = torch.ops.aten.add.Tensor(mul_51, arg82_1);  mul_51 = arg82_1 = None
        view_88 = torch.ops.aten.view.default(add_54, [-1, 768]);  add_54 = None
        addmm_26 = torch.ops.aten.addmm.default(arg83_1, view_88, arg84_1);  arg83_1 = view_88 = arg84_1 = None
        view_89 = torch.ops.aten.view.default(addmm_26, [1, 16, 3072]);  addmm_26 = None
        mul_52 = torch.ops.aten.mul.Tensor(view_89, 0.5)
        pow_7 = torch.ops.aten.pow.Tensor_Scalar(view_89, 3.0)
        mul_53 = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_55 = torch.ops.aten.add.Tensor(view_89, mul_53);  view_89 = mul_53 = None
        mul_54 = torch.ops.aten.mul.Tensor(add_55, 0.7978845608028654);  add_55 = None
        tanh_6 = torch.ops.aten.tanh.default(mul_54);  mul_54 = None
        add_56 = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
        mul_55 = torch.ops.aten.mul.Tensor(mul_52, add_56);  mul_52 = add_56 = None
        view_90 = torch.ops.aten.view.default(mul_55, [-1, 3072]);  mul_55 = None
        addmm_27 = torch.ops.aten.addmm.default(arg85_1, view_90, arg86_1);  arg85_1 = view_90 = arg86_1 = None
        view_91 = torch.ops.aten.view.default(addmm_27, [1, 16, 768]);  addmm_27 = None
        add_57 = torch.ops.aten.add.Tensor(add_52, view_91);  add_52 = view_91 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_77 = var_mean_14[0]
        getitem_78 = var_mean_14[1];  var_mean_14 = None
        add_58 = torch.ops.aten.add.Tensor(getitem_77, 1e-05);  getitem_77 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_16 = torch.ops.aten.sub.Tensor(add_57, getitem_78);  getitem_78 = None
        mul_56 = torch.ops.aten.mul.Tensor(sub_16, rsqrt_14);  sub_16 = rsqrt_14 = None
        mul_57 = torch.ops.aten.mul.Tensor(mul_56, arg87_1);  mul_56 = arg87_1 = None
        add_59 = torch.ops.aten.add.Tensor(mul_57, arg88_1);  mul_57 = arg88_1 = None
        view_92 = torch.ops.aten.view.default(add_59, [-1, 768]);  add_59 = None
        addmm_28 = torch.ops.aten.addmm.default(arg89_1, view_92, arg90_1);  arg89_1 = view_92 = arg90_1 = None
        view_93 = torch.ops.aten.view.default(addmm_28, [1, 16, 2304]);  addmm_28 = None
        split_7 = torch.ops.aten.split.Tensor(view_93, 768, 2);  view_93 = None
        getitem_79 = split_7[0]
        getitem_80 = split_7[1]
        getitem_81 = split_7[2];  split_7 = None
        view_94 = torch.ops.aten.view.default(getitem_80, [1, 16, -1, 64]);  getitem_80 = None
        permute_28 = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
        view_95 = torch.ops.aten.view.default(getitem_81, [1, 16, -1, 64]);  getitem_81 = None
        permute_29 = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None
        view_96 = torch.ops.aten.view.default(getitem_79, [1, 16, -1, 64]);  getitem_79 = None
        permute_30 = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
        full_default_15 = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_16 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_7 = torch.ops.aten.where.self(expand, full_default_16, full_default_15);  full_default_16 = full_default_15 = None
        expand_8 = torch.ops.aten.expand.default(where_7, [1, 12, 16, 16]);  where_7 = None
        _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_30, permute_28, permute_29, expand_8, False);  permute_30 = permute_28 = permute_29 = expand_8 = None
        getitem_82 = _scaled_dot_product_efficient_attention_7[0];  _scaled_dot_product_efficient_attention_7 = None
        permute_31 = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3]);  getitem_82 = None
        view_97 = torch.ops.aten.view.default(permute_31, [1, 16, -1]);  permute_31 = None
        view_98 = torch.ops.aten.view.default(view_97, [-1, 768]);  view_97 = None
        addmm_29 = torch.ops.aten.addmm.default(arg91_1, view_98, arg92_1);  arg91_1 = view_98 = arg92_1 = None
        view_99 = torch.ops.aten.view.default(addmm_29, [1, 16, 768]);  addmm_29 = None
        add_60 = torch.ops.aten.add.Tensor(view_99, add_57);  view_99 = add_57 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_60, [2], correction = 0, keepdim = True)
        getitem_86 = var_mean_15[0]
        getitem_87 = var_mean_15[1];  var_mean_15 = None
        add_61 = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        sub_17 = torch.ops.aten.sub.Tensor(add_60, getitem_87);  getitem_87 = None
        mul_58 = torch.ops.aten.mul.Tensor(sub_17, rsqrt_15);  sub_17 = rsqrt_15 = None
        mul_59 = torch.ops.aten.mul.Tensor(mul_58, arg93_1);  mul_58 = arg93_1 = None
        add_62 = torch.ops.aten.add.Tensor(mul_59, arg94_1);  mul_59 = arg94_1 = None
        view_100 = torch.ops.aten.view.default(add_62, [-1, 768]);  add_62 = None
        addmm_30 = torch.ops.aten.addmm.default(arg95_1, view_100, arg96_1);  arg95_1 = view_100 = arg96_1 = None
        view_101 = torch.ops.aten.view.default(addmm_30, [1, 16, 3072]);  addmm_30 = None
        mul_60 = torch.ops.aten.mul.Tensor(view_101, 0.5)
        pow_8 = torch.ops.aten.pow.Tensor_Scalar(view_101, 3.0)
        mul_61 = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_63 = torch.ops.aten.add.Tensor(view_101, mul_61);  view_101 = mul_61 = None
        mul_62 = torch.ops.aten.mul.Tensor(add_63, 0.7978845608028654);  add_63 = None
        tanh_7 = torch.ops.aten.tanh.default(mul_62);  mul_62 = None
        add_64 = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
        mul_63 = torch.ops.aten.mul.Tensor(mul_60, add_64);  mul_60 = add_64 = None
        view_102 = torch.ops.aten.view.default(mul_63, [-1, 3072]);  mul_63 = None
        addmm_31 = torch.ops.aten.addmm.default(arg97_1, view_102, arg98_1);  arg97_1 = view_102 = arg98_1 = None
        view_103 = torch.ops.aten.view.default(addmm_31, [1, 16, 768]);  addmm_31 = None
        add_65 = torch.ops.aten.add.Tensor(add_60, view_103);  add_60 = view_103 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
        getitem_88 = var_mean_16[0]
        getitem_89 = var_mean_16[1];  var_mean_16 = None
        add_66 = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        sub_18 = torch.ops.aten.sub.Tensor(add_65, getitem_89);  getitem_89 = None
        mul_64 = torch.ops.aten.mul.Tensor(sub_18, rsqrt_16);  sub_18 = rsqrt_16 = None
        mul_65 = torch.ops.aten.mul.Tensor(mul_64, arg99_1);  mul_64 = arg99_1 = None
        add_67 = torch.ops.aten.add.Tensor(mul_65, arg100_1);  mul_65 = arg100_1 = None
        view_104 = torch.ops.aten.view.default(add_67, [-1, 768]);  add_67 = None
        addmm_32 = torch.ops.aten.addmm.default(arg101_1, view_104, arg102_1);  arg101_1 = view_104 = arg102_1 = None
        view_105 = torch.ops.aten.view.default(addmm_32, [1, 16, 2304]);  addmm_32 = None
        split_8 = torch.ops.aten.split.Tensor(view_105, 768, 2);  view_105 = None
        getitem_90 = split_8[0]
        getitem_91 = split_8[1]
        getitem_92 = split_8[2];  split_8 = None
        view_106 = torch.ops.aten.view.default(getitem_91, [1, 16, -1, 64]);  getitem_91 = None
        permute_32 = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None
        view_107 = torch.ops.aten.view.default(getitem_92, [1, 16, -1, 64]);  getitem_92 = None
        permute_33 = torch.ops.aten.permute.default(view_107, [0, 2, 1, 3]);  view_107 = None
        view_108 = torch.ops.aten.view.default(getitem_90, [1, 16, -1, 64]);  getitem_90 = None
        permute_34 = torch.ops.aten.permute.default(view_108, [0, 2, 1, 3]);  view_108 = None
        full_default_17 = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_18 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_8 = torch.ops.aten.where.self(expand, full_default_18, full_default_17);  full_default_18 = full_default_17 = None
        expand_9 = torch.ops.aten.expand.default(where_8, [1, 12, 16, 16]);  where_8 = None
        _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_34, permute_32, permute_33, expand_9, False);  permute_34 = permute_32 = permute_33 = expand_9 = None
        getitem_93 = _scaled_dot_product_efficient_attention_8[0];  _scaled_dot_product_efficient_attention_8 = None
        permute_35 = torch.ops.aten.permute.default(getitem_93, [0, 2, 1, 3]);  getitem_93 = None
        view_109 = torch.ops.aten.view.default(permute_35, [1, 16, -1]);  permute_35 = None
        view_110 = torch.ops.aten.view.default(view_109, [-1, 768]);  view_109 = None
        addmm_33 = torch.ops.aten.addmm.default(arg103_1, view_110, arg104_1);  arg103_1 = view_110 = arg104_1 = None
        view_111 = torch.ops.aten.view.default(addmm_33, [1, 16, 768]);  addmm_33 = None
        add_68 = torch.ops.aten.add.Tensor(view_111, add_65);  view_111 = add_65 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_68, [2], correction = 0, keepdim = True)
        getitem_97 = var_mean_17[0]
        getitem_98 = var_mean_17[1];  var_mean_17 = None
        add_69 = torch.ops.aten.add.Tensor(getitem_97, 1e-05);  getitem_97 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        sub_19 = torch.ops.aten.sub.Tensor(add_68, getitem_98);  getitem_98 = None
        mul_66 = torch.ops.aten.mul.Tensor(sub_19, rsqrt_17);  sub_19 = rsqrt_17 = None
        mul_67 = torch.ops.aten.mul.Tensor(mul_66, arg105_1);  mul_66 = arg105_1 = None
        add_70 = torch.ops.aten.add.Tensor(mul_67, arg106_1);  mul_67 = arg106_1 = None
        view_112 = torch.ops.aten.view.default(add_70, [-1, 768]);  add_70 = None
        addmm_34 = torch.ops.aten.addmm.default(arg107_1, view_112, arg108_1);  arg107_1 = view_112 = arg108_1 = None
        view_113 = torch.ops.aten.view.default(addmm_34, [1, 16, 3072]);  addmm_34 = None
        mul_68 = torch.ops.aten.mul.Tensor(view_113, 0.5)
        pow_9 = torch.ops.aten.pow.Tensor_Scalar(view_113, 3.0)
        mul_69 = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_71 = torch.ops.aten.add.Tensor(view_113, mul_69);  view_113 = mul_69 = None
        mul_70 = torch.ops.aten.mul.Tensor(add_71, 0.7978845608028654);  add_71 = None
        tanh_8 = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
        add_72 = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
        mul_71 = torch.ops.aten.mul.Tensor(mul_68, add_72);  mul_68 = add_72 = None
        view_114 = torch.ops.aten.view.default(mul_71, [-1, 3072]);  mul_71 = None
        addmm_35 = torch.ops.aten.addmm.default(arg109_1, view_114, arg110_1);  arg109_1 = view_114 = arg110_1 = None
        view_115 = torch.ops.aten.view.default(addmm_35, [1, 16, 768]);  addmm_35 = None
        add_73 = torch.ops.aten.add.Tensor(add_68, view_115);  add_68 = view_115 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
        getitem_99 = var_mean_18[0]
        getitem_100 = var_mean_18[1];  var_mean_18 = None
        add_74 = torch.ops.aten.add.Tensor(getitem_99, 1e-05);  getitem_99 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_20 = torch.ops.aten.sub.Tensor(add_73, getitem_100);  getitem_100 = None
        mul_72 = torch.ops.aten.mul.Tensor(sub_20, rsqrt_18);  sub_20 = rsqrt_18 = None
        mul_73 = torch.ops.aten.mul.Tensor(mul_72, arg111_1);  mul_72 = arg111_1 = None
        add_75 = torch.ops.aten.add.Tensor(mul_73, arg112_1);  mul_73 = arg112_1 = None
        view_116 = torch.ops.aten.view.default(add_75, [-1, 768]);  add_75 = None
        addmm_36 = torch.ops.aten.addmm.default(arg113_1, view_116, arg114_1);  arg113_1 = view_116 = arg114_1 = None
        view_117 = torch.ops.aten.view.default(addmm_36, [1, 16, 2304]);  addmm_36 = None
        split_9 = torch.ops.aten.split.Tensor(view_117, 768, 2);  view_117 = None
        getitem_101 = split_9[0]
        getitem_102 = split_9[1]
        getitem_103 = split_9[2];  split_9 = None
        view_118 = torch.ops.aten.view.default(getitem_102, [1, 16, -1, 64]);  getitem_102 = None
        permute_36 = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
        view_119 = torch.ops.aten.view.default(getitem_103, [1, 16, -1, 64]);  getitem_103 = None
        permute_37 = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None
        view_120 = torch.ops.aten.view.default(getitem_101, [1, 16, -1, 64]);  getitem_101 = None
        permute_38 = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
        full_default_19 = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_20 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_9 = torch.ops.aten.where.self(expand, full_default_20, full_default_19);  full_default_20 = full_default_19 = None
        expand_10 = torch.ops.aten.expand.default(where_9, [1, 12, 16, 16]);  where_9 = None
        _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_38, permute_36, permute_37, expand_10, False);  permute_38 = permute_36 = permute_37 = expand_10 = None
        getitem_104 = _scaled_dot_product_efficient_attention_9[0];  _scaled_dot_product_efficient_attention_9 = None
        permute_39 = torch.ops.aten.permute.default(getitem_104, [0, 2, 1, 3]);  getitem_104 = None
        view_121 = torch.ops.aten.view.default(permute_39, [1, 16, -1]);  permute_39 = None
        view_122 = torch.ops.aten.view.default(view_121, [-1, 768]);  view_121 = None
        addmm_37 = torch.ops.aten.addmm.default(arg115_1, view_122, arg116_1);  arg115_1 = view_122 = arg116_1 = None
        view_123 = torch.ops.aten.view.default(addmm_37, [1, 16, 768]);  addmm_37 = None
        add_76 = torch.ops.aten.add.Tensor(view_123, add_73);  view_123 = add_73 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_76, [2], correction = 0, keepdim = True)
        getitem_108 = var_mean_19[0]
        getitem_109 = var_mean_19[1];  var_mean_19 = None
        add_77 = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        sub_21 = torch.ops.aten.sub.Tensor(add_76, getitem_109);  getitem_109 = None
        mul_74 = torch.ops.aten.mul.Tensor(sub_21, rsqrt_19);  sub_21 = rsqrt_19 = None
        mul_75 = torch.ops.aten.mul.Tensor(mul_74, arg117_1);  mul_74 = arg117_1 = None
        add_78 = torch.ops.aten.add.Tensor(mul_75, arg118_1);  mul_75 = arg118_1 = None
        view_124 = torch.ops.aten.view.default(add_78, [-1, 768]);  add_78 = None
        addmm_38 = torch.ops.aten.addmm.default(arg119_1, view_124, arg120_1);  arg119_1 = view_124 = arg120_1 = None
        view_125 = torch.ops.aten.view.default(addmm_38, [1, 16, 3072]);  addmm_38 = None
        mul_76 = torch.ops.aten.mul.Tensor(view_125, 0.5)
        pow_10 = torch.ops.aten.pow.Tensor_Scalar(view_125, 3.0)
        mul_77 = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_79 = torch.ops.aten.add.Tensor(view_125, mul_77);  view_125 = mul_77 = None
        mul_78 = torch.ops.aten.mul.Tensor(add_79, 0.7978845608028654);  add_79 = None
        tanh_9 = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
        add_80 = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
        mul_79 = torch.ops.aten.mul.Tensor(mul_76, add_80);  mul_76 = add_80 = None
        view_126 = torch.ops.aten.view.default(mul_79, [-1, 3072]);  mul_79 = None
        addmm_39 = torch.ops.aten.addmm.default(arg121_1, view_126, arg122_1);  arg121_1 = view_126 = arg122_1 = None
        view_127 = torch.ops.aten.view.default(addmm_39, [1, 16, 768]);  addmm_39 = None
        add_81 = torch.ops.aten.add.Tensor(add_76, view_127);  add_76 = view_127 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
        getitem_110 = var_mean_20[0]
        getitem_111 = var_mean_20[1];  var_mean_20 = None
        add_82 = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        sub_22 = torch.ops.aten.sub.Tensor(add_81, getitem_111);  getitem_111 = None
        mul_80 = torch.ops.aten.mul.Tensor(sub_22, rsqrt_20);  sub_22 = rsqrt_20 = None
        mul_81 = torch.ops.aten.mul.Tensor(mul_80, arg123_1);  mul_80 = arg123_1 = None
        add_83 = torch.ops.aten.add.Tensor(mul_81, arg124_1);  mul_81 = arg124_1 = None
        view_128 = torch.ops.aten.view.default(add_83, [-1, 768]);  add_83 = None
        addmm_40 = torch.ops.aten.addmm.default(arg125_1, view_128, arg126_1);  arg125_1 = view_128 = arg126_1 = None
        view_129 = torch.ops.aten.view.default(addmm_40, [1, 16, 2304]);  addmm_40 = None
        split_10 = torch.ops.aten.split.Tensor(view_129, 768, 2);  view_129 = None
        getitem_112 = split_10[0]
        getitem_113 = split_10[1]
        getitem_114 = split_10[2];  split_10 = None
        view_130 = torch.ops.aten.view.default(getitem_113, [1, 16, -1, 64]);  getitem_113 = None
        permute_40 = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
        view_131 = torch.ops.aten.view.default(getitem_114, [1, 16, -1, 64]);  getitem_114 = None
        permute_41 = torch.ops.aten.permute.default(view_131, [0, 2, 1, 3]);  view_131 = None
        view_132 = torch.ops.aten.view.default(getitem_112, [1, 16, -1, 64]);  getitem_112 = None
        permute_42 = torch.ops.aten.permute.default(view_132, [0, 2, 1, 3]);  view_132 = None
        full_default_21 = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_22 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_10 = torch.ops.aten.where.self(expand, full_default_22, full_default_21);  full_default_22 = full_default_21 = None
        expand_11 = torch.ops.aten.expand.default(where_10, [1, 12, 16, 16]);  where_10 = None
        _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_42, permute_40, permute_41, expand_11, False);  permute_42 = permute_40 = permute_41 = expand_11 = None
        getitem_115 = _scaled_dot_product_efficient_attention_10[0];  _scaled_dot_product_efficient_attention_10 = None
        permute_43 = torch.ops.aten.permute.default(getitem_115, [0, 2, 1, 3]);  getitem_115 = None
        view_133 = torch.ops.aten.view.default(permute_43, [1, 16, -1]);  permute_43 = None
        view_134 = torch.ops.aten.view.default(view_133, [-1, 768]);  view_133 = None
        addmm_41 = torch.ops.aten.addmm.default(arg127_1, view_134, arg128_1);  arg127_1 = view_134 = arg128_1 = None
        view_135 = torch.ops.aten.view.default(addmm_41, [1, 16, 768]);  addmm_41 = None
        add_84 = torch.ops.aten.add.Tensor(view_135, add_81);  view_135 = add_81 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_84, [2], correction = 0, keepdim = True)
        getitem_119 = var_mean_21[0]
        getitem_120 = var_mean_21[1];  var_mean_21 = None
        add_85 = torch.ops.aten.add.Tensor(getitem_119, 1e-05);  getitem_119 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        sub_23 = torch.ops.aten.sub.Tensor(add_84, getitem_120);  getitem_120 = None
        mul_82 = torch.ops.aten.mul.Tensor(sub_23, rsqrt_21);  sub_23 = rsqrt_21 = None
        mul_83 = torch.ops.aten.mul.Tensor(mul_82, arg129_1);  mul_82 = arg129_1 = None
        add_86 = torch.ops.aten.add.Tensor(mul_83, arg130_1);  mul_83 = arg130_1 = None
        view_136 = torch.ops.aten.view.default(add_86, [-1, 768]);  add_86 = None
        addmm_42 = torch.ops.aten.addmm.default(arg131_1, view_136, arg132_1);  arg131_1 = view_136 = arg132_1 = None
        view_137 = torch.ops.aten.view.default(addmm_42, [1, 16, 3072]);  addmm_42 = None
        mul_84 = torch.ops.aten.mul.Tensor(view_137, 0.5)
        pow_11 = torch.ops.aten.pow.Tensor_Scalar(view_137, 3.0)
        mul_85 = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_87 = torch.ops.aten.add.Tensor(view_137, mul_85);  view_137 = mul_85 = None
        mul_86 = torch.ops.aten.mul.Tensor(add_87, 0.7978845608028654);  add_87 = None
        tanh_10 = torch.ops.aten.tanh.default(mul_86);  mul_86 = None
        add_88 = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
        mul_87 = torch.ops.aten.mul.Tensor(mul_84, add_88);  mul_84 = add_88 = None
        view_138 = torch.ops.aten.view.default(mul_87, [-1, 3072]);  mul_87 = None
        addmm_43 = torch.ops.aten.addmm.default(arg133_1, view_138, arg134_1);  arg133_1 = view_138 = arg134_1 = None
        view_139 = torch.ops.aten.view.default(addmm_43, [1, 16, 768]);  addmm_43 = None
        add_89 = torch.ops.aten.add.Tensor(add_84, view_139);  add_84 = view_139 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
        getitem_121 = var_mean_22[0]
        getitem_122 = var_mean_22[1];  var_mean_22 = None
        add_90 = torch.ops.aten.add.Tensor(getitem_121, 1e-05);  getitem_121 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        sub_24 = torch.ops.aten.sub.Tensor(add_89, getitem_122);  getitem_122 = None
        mul_88 = torch.ops.aten.mul.Tensor(sub_24, rsqrt_22);  sub_24 = rsqrt_22 = None
        mul_89 = torch.ops.aten.mul.Tensor(mul_88, arg135_1);  mul_88 = arg135_1 = None
        add_91 = torch.ops.aten.add.Tensor(mul_89, arg136_1);  mul_89 = arg136_1 = None
        view_140 = torch.ops.aten.view.default(add_91, [-1, 768]);  add_91 = None
        addmm_44 = torch.ops.aten.addmm.default(arg137_1, view_140, arg138_1);  arg137_1 = view_140 = arg138_1 = None
        view_141 = torch.ops.aten.view.default(addmm_44, [1, 16, 2304]);  addmm_44 = None
        split_11 = torch.ops.aten.split.Tensor(view_141, 768, 2);  view_141 = None
        getitem_123 = split_11[0]
        getitem_124 = split_11[1]
        getitem_125 = split_11[2];  split_11 = None
        view_142 = torch.ops.aten.view.default(getitem_124, [1, 16, -1, 64]);  getitem_124 = None
        permute_44 = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None
        view_143 = torch.ops.aten.view.default(getitem_125, [1, 16, -1, 64]);  getitem_125 = None
        permute_45 = torch.ops.aten.permute.default(view_143, [0, 2, 1, 3]);  view_143 = None
        view_144 = torch.ops.aten.view.default(getitem_123, [1, 16, -1, 64]);  getitem_123 = None
        permute_46 = torch.ops.aten.permute.default(view_144, [0, 2, 1, 3]);  view_144 = None
        full_default_23 = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_24 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_11 = torch.ops.aten.where.self(expand, full_default_24, full_default_23);  expand = full_default_24 = full_default_23 = None
        expand_12 = torch.ops.aten.expand.default(where_11, [1, 12, 16, 16]);  where_11 = None
        _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_46, permute_44, permute_45, expand_12, False);  permute_46 = permute_44 = permute_45 = expand_12 = None
        getitem_126 = _scaled_dot_product_efficient_attention_11[0];  _scaled_dot_product_efficient_attention_11 = None
        permute_47 = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3]);  getitem_126 = None
        view_145 = torch.ops.aten.view.default(permute_47, [1, 16, -1]);  permute_47 = None
        view_146 = torch.ops.aten.view.default(view_145, [-1, 768]);  view_145 = None
        addmm_45 = torch.ops.aten.addmm.default(arg139_1, view_146, arg140_1);  arg139_1 = view_146 = arg140_1 = None
        view_147 = torch.ops.aten.view.default(addmm_45, [1, 16, 768]);  addmm_45 = None
        add_92 = torch.ops.aten.add.Tensor(view_147, add_89);  view_147 = add_89 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_92, [2], correction = 0, keepdim = True)
        getitem_130 = var_mean_23[0]
        getitem_131 = var_mean_23[1];  var_mean_23 = None
        add_93 = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        sub_25 = torch.ops.aten.sub.Tensor(add_92, getitem_131);  getitem_131 = None
        mul_90 = torch.ops.aten.mul.Tensor(sub_25, rsqrt_23);  sub_25 = rsqrt_23 = None
        mul_91 = torch.ops.aten.mul.Tensor(mul_90, arg141_1);  mul_90 = arg141_1 = None
        add_94 = torch.ops.aten.add.Tensor(mul_91, arg142_1);  mul_91 = arg142_1 = None
        view_148 = torch.ops.aten.view.default(add_94, [-1, 768]);  add_94 = None
        addmm_46 = torch.ops.aten.addmm.default(arg143_1, view_148, arg144_1);  arg143_1 = view_148 = arg144_1 = None
        view_149 = torch.ops.aten.view.default(addmm_46, [1, 16, 3072]);  addmm_46 = None
        mul_92 = torch.ops.aten.mul.Tensor(view_149, 0.5)
        pow_12 = torch.ops.aten.pow.Tensor_Scalar(view_149, 3.0)
        mul_93 = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_95 = torch.ops.aten.add.Tensor(view_149, mul_93);  view_149 = mul_93 = None
        mul_94 = torch.ops.aten.mul.Tensor(add_95, 0.7978845608028654);  add_95 = None
        tanh_11 = torch.ops.aten.tanh.default(mul_94);  mul_94 = None
        add_96 = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
        mul_95 = torch.ops.aten.mul.Tensor(mul_92, add_96);  mul_92 = add_96 = None
        view_150 = torch.ops.aten.view.default(mul_95, [-1, 3072]);  mul_95 = None
        addmm_47 = torch.ops.aten.addmm.default(arg145_1, view_150, arg146_1);  arg145_1 = view_150 = arg146_1 = None
        view_151 = torch.ops.aten.view.default(addmm_47, [1, 16, 768]);  addmm_47 = None
        add_97 = torch.ops.aten.add.Tensor(add_92, view_151);  add_92 = view_151 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
        getitem_132 = var_mean_24[0]
        getitem_133 = var_mean_24[1];  var_mean_24 = None
        add_98 = torch.ops.aten.add.Tensor(getitem_132, 1e-05);  getitem_132 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        sub_26 = torch.ops.aten.sub.Tensor(add_97, getitem_133);  add_97 = getitem_133 = None
        mul_96 = torch.ops.aten.mul.Tensor(sub_26, rsqrt_24);  sub_26 = rsqrt_24 = None
        mul_97 = torch.ops.aten.mul.Tensor(mul_96, arg147_1);  mul_96 = arg147_1 = None
        add_99 = torch.ops.aten.add.Tensor(mul_97, arg148_1);  mul_97 = arg148_1 = None
        return (add_99,)
        
def load_args(reader):
    buf0 = reader.storage(None, 128, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (1, 16), dtype=torch.int64, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 154389504, device=device(type='cuda', index=0))
    reader.tensor(buf1, (50257, 768), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf2, (1024, 768), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf3, (768,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf4, (768,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf5, (2304,), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf6, (768, 2304), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf7, (768,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf8, (768, 768), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf9, (768,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf10, (768,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf11, (3072,), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf12, (768, 3072), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf13, (768,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf14, (3072, 768), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf15, (768,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf16, (768,), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf17, (2304,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf18, (768, 2304), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf19, (768,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf20, (768, 768), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf21, (768,), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf22, (768,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf23, (3072,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf24, (768, 3072), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf25, (768,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf26, (3072, 768), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf27, (768,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf28, (768,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf29, (2304,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf30, (768, 2304), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf31, (768,), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf32, (768, 768), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf33, (768,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf34, (768,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf35, (3072,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf36, (768, 3072), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf37, (768,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf38, (3072, 768), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf39, (768,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf40, (768,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf41, (2304,), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf42, (768, 2304), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf43, (768,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf44, (768, 768), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf45, (768,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf46, (768,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf47, (3072,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf48, (768, 3072), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf49, (768,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf50, (3072, 768), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf51, (768,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf52, (768,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf53, (2304,), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf54, (768, 2304), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf55, (768,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf56, (768, 768), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf57, (768,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf58, (768,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf59, (3072,), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf60, (768, 3072), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf61, (768,), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf62, (3072, 768), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf63, (768,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf64, (768,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf65, (2304,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf66, (768, 2304), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf67, (768,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf68, (768, 768), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf69, (768,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf70, (768,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf71, (3072,), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf72, (768, 3072), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf73, (768,), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf74, (3072, 768), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf75, (768,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf76, (768,), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf77, (2304,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf78, (768, 2304), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf79, (768,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf80, (768, 768), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf81, (768,), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf82, (768,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf83, (3072,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf84, (768, 3072), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf85, (768,), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf86, (3072, 768), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf87, (768,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf88, (768,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf89, (2304,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf90, (768, 2304), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf91, (768,), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf92, (768, 768), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf93, (768,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf94, (768,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf95, (3072,), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf96, (768, 3072), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf97, (768,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf98, (3072, 768), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf99, (768,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf100, (768,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf101, (2304,), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf102, (768, 2304), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf103, (768,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf104, (768, 768), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf105, (768,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf106, (768,), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf107, (3072,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf108, (768, 3072), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf109, (768,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf110, (3072, 768), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf111, (768,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf112, (768,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf113, (2304,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf114, (768, 2304), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf115, (768,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf116, (768, 768), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf117, (768,), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf118, (768,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf119, (3072,), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf120, (768, 3072), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf121, (768,), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf122, (3072, 768), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf123, (768,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf124, (768,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf125, (2304,), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf126, (768, 2304), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf127, (768,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf128, (768, 768), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf129, (768,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf130, (768,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf131, (3072,), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf132, (768, 3072), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf133, (768,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf134, (3072, 768), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf135, (768,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf136, (768,), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf137, (2304,), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf138, (768, 2304), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf139, (768,), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf140, (768, 768), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf141, (768,), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf142, (768,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf143, (3072,), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf144, (768, 3072), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf145, (768,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf146, (3072, 768), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf147, (768,), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf148, (768,), is_leaf=True)  # arg148_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)

# ===== inductor generated file: /tmp/cnnbench-transformers-ob_ijj07/repeat_03/a1/torchinductor/tmpuqb_59nb/tb/ctbeuecaa5nknwy6bvucluzl476p3mx7abciczu4tjqirt7duxrh.debug/fx_graph_transformed.py =====
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "i64[1, 16]", arg1_1: "f32[50257, 768]", arg2_1: "f32[1024, 768]", arg3_1: "f32[768]", arg4_1: "f32[768]", arg5_1: "f32[2304]", arg6_1: "f32[768, 2304]", arg7_1: "f32[768]", arg8_1: "f32[768, 768]", arg9_1: "f32[768]", arg10_1: "f32[768]", arg11_1: "f32[3072]", arg12_1: "f32[768, 3072]", arg13_1: "f32[768]", arg14_1: "f32[3072, 768]", arg15_1: "f32[768]", arg16_1: "f32[768]", arg17_1: "f32[2304]", arg18_1: "f32[768, 2304]", arg19_1: "f32[768]", arg20_1: "f32[768, 768]", arg21_1: "f32[768]", arg22_1: "f32[768]", arg23_1: "f32[3072]", arg24_1: "f32[768, 3072]", arg25_1: "f32[768]", arg26_1: "f32[3072, 768]", arg27_1: "f32[768]", arg28_1: "f32[768]", arg29_1: "f32[2304]", arg30_1: "f32[768, 2304]", arg31_1: "f32[768]", arg32_1: "f32[768, 768]", arg33_1: "f32[768]", arg34_1: "f32[768]", arg35_1: "f32[3072]", arg36_1: "f32[768, 3072]", arg37_1: "f32[768]", arg38_1: "f32[3072, 768]", arg39_1: "f32[768]", arg40_1: "f32[768]", arg41_1: "f32[2304]", arg42_1: "f32[768, 2304]", arg43_1: "f32[768]", arg44_1: "f32[768, 768]", arg45_1: "f32[768]", arg46_1: "f32[768]", arg47_1: "f32[3072]", arg48_1: "f32[768, 3072]", arg49_1: "f32[768]", arg50_1: "f32[3072, 768]", arg51_1: "f32[768]", arg52_1: "f32[768]", arg53_1: "f32[2304]", arg54_1: "f32[768, 2304]", arg55_1: "f32[768]", arg56_1: "f32[768, 768]", arg57_1: "f32[768]", arg58_1: "f32[768]", arg59_1: "f32[3072]", arg60_1: "f32[768, 3072]", arg61_1: "f32[768]", arg62_1: "f32[3072, 768]", arg63_1: "f32[768]", arg64_1: "f32[768]", arg65_1: "f32[2304]", arg66_1: "f32[768, 2304]", arg67_1: "f32[768]", arg68_1: "f32[768, 768]", arg69_1: "f32[768]", arg70_1: "f32[768]", arg71_1: "f32[3072]", arg72_1: "f32[768, 3072]", arg73_1: "f32[768]", arg74_1: "f32[3072, 768]", arg75_1: "f32[768]", arg76_1: "f32[768]", arg77_1: "f32[2304]", arg78_1: "f32[768, 2304]", arg79_1: "f32[768]", arg80_1: "f32[768, 768]", arg81_1: "f32[768]", arg82_1: "f32[768]", arg83_1: "f32[3072]", arg84_1: "f32[768, 3072]", arg85_1: "f32[768]", arg86_1: "f32[3072, 768]", arg87_1: "f32[768]", arg88_1: "f32[768]", arg89_1: "f32[2304]", arg90_1: "f32[768, 2304]", arg91_1: "f32[768]", arg92_1: "f32[768, 768]", arg93_1: "f32[768]", arg94_1: "f32[768]", arg95_1: "f32[3072]", arg96_1: "f32[768, 3072]", arg97_1: "f32[768]", arg98_1: "f32[3072, 768]", arg99_1: "f32[768]", arg100_1: "f32[768]", arg101_1: "f32[2304]", arg102_1: "f32[768, 2304]", arg103_1: "f32[768]", arg104_1: "f32[768, 768]", arg105_1: "f32[768]", arg106_1: "f32[768]", arg107_1: "f32[3072]", arg108_1: "f32[768, 3072]", arg109_1: "f32[768]", arg110_1: "f32[3072, 768]", arg111_1: "f32[768]", arg112_1: "f32[768]", arg113_1: "f32[2304]", arg114_1: "f32[768, 2304]", arg115_1: "f32[768]", arg116_1: "f32[768, 768]", arg117_1: "f32[768]", arg118_1: "f32[768]", arg119_1: "f32[3072]", arg120_1: "f32[768, 3072]", arg121_1: "f32[768]", arg122_1: "f32[3072, 768]", arg123_1: "f32[768]", arg124_1: "f32[768]", arg125_1: "f32[2304]", arg126_1: "f32[768, 2304]", arg127_1: "f32[768]", arg128_1: "f32[768, 768]", arg129_1: "f32[768]", arg130_1: "f32[768]", arg131_1: "f32[3072]", arg132_1: "f32[768, 3072]", arg133_1: "f32[768]", arg134_1: "f32[3072, 768]", arg135_1: "f32[768]", arg136_1: "f32[768]", arg137_1: "f32[2304]", arg138_1: "f32[768, 2304]", arg139_1: "f32[768]", arg140_1: "f32[768, 768]", arg141_1: "f32[768]", arg142_1: "f32[768]", arg143_1: "f32[3072]", arg144_1: "f32[768, 3072]", arg145_1: "f32[768]", arg146_1: "f32[3072, 768]", arg147_1: "f32[768]", arg148_1: "f32[768]"):
         # File: /opt/venvs/xla/lib/python3.12/site-packages/torch/_dynamo/_trace_wrapped_higher_order_op.py:146 in __torch_function__, code: return func(*args, **(kwargs or {}))
        full: "b8[16]" = torch.ops.aten.full.default([16], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  full = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:857 in forward, code: inputs_embeds = self.wte(input_ids)
        embedding: "f32[1, 16, 768]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:861 in forward, code: cache_position = torch.arange(
        iota: "i64[16]" = torch.ops.prims.iota.default(16, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:865 in forward, code: position_ids = cache_position.unsqueeze(0)
        unsqueeze: "i64[1, 16]" = torch.ops.aten.unsqueeze.default(iota, 0)
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:867 in forward, code: position_embeds = self.wpe(position_ids)
        embedding_1: "f32[1, 16, 768]" = torch.ops.aten.embedding.default(arg2_1, unsqueeze);  arg2_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:868 in forward, code: hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)
        add: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 16, 1]" = var_mean[0]
        getitem_1: "f32[1, 16, 1]" = var_mean[1];  var_mean = None
        sub_2: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add, getitem_1);  getitem_1 = None
        add_2: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt);  sub_2 = rsqrt = None
        mul_1: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = arg3_1 = None
        add_3: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_8: "f32[16, 768]" = torch.ops.aten.reshape.default(add_3, [-1, 768]);  add_3 = None
        addmm: "f32[16, 2304]" = torch.ops.aten.addmm.default(arg5_1, view_8, arg6_1);  arg5_1 = view_8 = arg6_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_9: "f32[1, 16, 2304]" = torch.ops.aten.reshape.default(addmm, [1, 16, 2304]);  addmm = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split = torch.ops.aten.split.Tensor(view_9, 768, 2);  view_9 = None
        getitem_2: "f32[1, 16, 768]" = split[0]
        getitem_3: "f32[1, 16, 768]" = split[1]
        getitem_4: "f32[1, 16, 768]" = split[2];  split = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_12: "f32[1, 16, 12, 64]" = torch.ops.aten.reshape.default(getitem_2, [1, 16, -1, 64]);  getitem_2 = None
        permute_2: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_10: "f32[1, 16, 12, 64]" = torch.ops.aten.reshape.default(getitem_3, [1, 16, -1, 64]);  getitem_3 = None
        permute: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_11: "f32[1, 16, 12, 64]" = torch.ops.aten.reshape.default(getitem_4, [1, 16, -1, 64]);  getitem_4 = None
        permute_1: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_11, [0, 2, 1, 3]);  view_11 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/torch/_dynamo/_trace_wrapped_higher_order_op.py:146 in __torch_function__, code: return func(*args, **(kwargs or {}))
        full_default: "b8[16, 1]" = torch.ops.aten.full.default([16, 1], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/masking_utils.py:379 in sdpa_mask_recent_torch, code: kv_arange = torch.arange(kv_length, device=cache_position.device)
        iota_1: "i64[16]" = torch.ops.prims.iota.default(16, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/masking_utils.py:380 in sdpa_mask_recent_torch, code: kv_arange += kv_offset
        add_1: "i64[16]" = torch.ops.aten.add.Tensor(iota_1, 0);  iota_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/torch/_dynamo/_trace_wrapped_higher_order_op.py:146 in __torch_function__, code: return func(*args, **(kwargs or {}))
        view_1: "i64[16, 1]" = torch.ops.aten.reshape.default(iota, [16, 1])
        le: "b8[16, 16]" = torch.ops.aten.le.Tensor(add_1, view_1);  view_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/torch/_dynamo/_trace_wrapped_higher_order_op.py:146 in __torch_function__, code: return func(*args, **(kwargs or {}))
        bitwise_and: "b8[16, 16]" = torch.ops.aten.bitwise_and.Tensor(full_default, le);  full_default = le = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/masking_utils.py:654 in find_packed_sequence_indices, code: first_dummy_value = position_ids[:, :1] - 1  # We just need the diff on this first value to be 1
        slice_1: "i64[1, 1]" = torch.ops.aten.slice.Tensor(unsqueeze, 1, 0, 1)
        sub: "i64[1, 1]" = torch.ops.aten.sub.Tensor(slice_1, 1);  slice_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/masking_utils.py:655 in find_packed_sequence_indices, code: position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
        cat: "i64[1, 17]" = torch.ops.aten.cat.default([sub, unsqueeze], -1);  sub = unsqueeze = None
        slice_3: "i64[1, 16]" = torch.ops.aten.slice.Tensor(cat, -1, 1, 17)
        slice_2: "i64[1, 16]" = torch.ops.aten.slice.Tensor(cat, -1, 0, 16);  cat = None
        sub_1: "i64[1, 16]" = torch.ops.aten.sub.Tensor(slice_3, slice_2);  slice_3 = slice_2 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/masking_utils.py:656 in find_packed_sequence_indices, code: packed_sequence_mask = (position_diff != 1).cumsum(-1)
        ne: "b8[1, 16]" = torch.ops.aten.ne.Scalar(sub_1, 1);  sub_1 = None
        cumsum: "i64[1, 16]" = torch.ops.aten.cumsum.default(ne, -1);  ne = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/masking_utils.py:386 in sdpa_mask_recent_torch, code: batch_arange = torch.arange(batch_size, device=cache_position.device)
        iota_2: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/torch/_dynamo/_trace_wrapped_higher_order_op.py:99 in forward, code: return torch.ops.aten.index(x, indices)
        view_3: "i64[1, 1]" = torch.ops.aten.reshape.default(iota_2, [1, 1])
        index: "i64[1, 16]" = torch.ops.aten.index.Tensor(cumsum, [view_3, iota]);  view_3 = iota = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/torch/_dynamo/_trace_wrapped_higher_order_op.py:146 in __torch_function__, code: return func(*args, **(kwargs or {}))
        view_5: "i64[1, 16, 1]" = torch.ops.aten.reshape.default(index, [1, 16, 1]);  index = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/torch/_dynamo/_trace_wrapped_higher_order_op.py:99 in forward, code: return torch.ops.aten.index(x, indices)
        view_4: "i64[1, 1]" = torch.ops.aten.reshape.default(iota_2, [1, 1]);  iota_2 = None
        index_1: "i64[1, 16]" = torch.ops.aten.index.Tensor(cumsum, [view_4, add_1]);  cumsum = view_4 = add_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/torch/_dynamo/_trace_wrapped_higher_order_op.py:146 in __torch_function__, code: return func(*args, **(kwargs or {}))
        view_6: "i64[1, 1, 16]" = torch.ops.aten.reshape.default(index_1, [1, 1, 16]);  index_1 = None
        eq: "b8[1, 16, 16]" = torch.ops.aten.eq.Tensor(view_5, view_6);  view_5 = view_6 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/torch/_dynamo/_trace_wrapped_higher_order_op.py:146 in __torch_function__, code: return func(*args, **(kwargs or {}))
        bitwise_and_1: "b8[1, 16, 16]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, eq);  bitwise_and = eq = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/torch/_functorch/vmap.py:183 in _maybe_remove_batch_dim, code: return _remove_batch_dim(batched_output, vmap_level, batch_size, out_dim)
        view_7: "b8[1, 1, 16, 16]" = torch.ops.aten.reshape.default(bitwise_and_1, [1, 1, 16, 16]);  bitwise_and_1 = None
        expand: "b8[1, 1, 16, 16]" = torch.ops.aten.expand.default(view_7, [1, 1, 16, 16]);  view_7 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[1, 1, 16, 16]" = torch.ops.aten.where.self(expand, full_default_2, full_default_1);  full_default_2 = full_default_1 = None
        expand_1: "f32[1, 12, 16, 16]" = torch.ops.aten.expand.default(where, [1, 12, 16, 16]);  where = None
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_2, permute, permute_1, expand_1, False);  permute_2 = permute = permute_1 = expand_1 = None
        getitem_5: "f32[1, 12, 16, 64]" = _scaled_dot_product_efficient_attention[0];  _scaled_dot_product_efficient_attention = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_3: "f32[1, 16, 12, 64]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_13: "f32[1, 16, 768]" = torch.ops.aten.reshape.default(permute_3, [1, 16, -1]);  permute_3 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_14: "f32[16, 768]" = torch.ops.aten.reshape.default(view_13, [-1, 768]);  view_13 = None
        mm_default_35: "f32[16, 768]" = torch.ops.aten.mm.default(view_14, arg8_1);  view_14 = arg8_1 = None
        add_tensor_35: "f32[16, 768]" = torch.ops.aten.add.Tensor(mm_default_35, arg7_1);  mm_default_35 = arg7_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_15: "f32[1, 16, 768]" = torch.ops.aten.reshape.default(add_tensor_35, [1, 16, 768]);  add_tensor_35 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_4: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(view_15, add);  view_15 = add = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
        getitem_9: "f32[1, 16, 1]" = var_mean_1[0]
        getitem_10: "f32[1, 16, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_3: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_4, getitem_10);  getitem_10 = None
        add_5: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_9, 1e-05);  getitem_9 = None
        rsqrt_1: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        mul_2: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = rsqrt_1 = None
        mul_3: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_2, arg9_1);  mul_2 = arg9_1 = None
        add_6: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_3, arg10_1);  mul_3 = arg10_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_16: "f32[16, 768]" = torch.ops.aten.reshape.default(add_6, [-1, 768]);  add_6 = None
        mm_default_34: "f32[16, 3072]" = torch.ops.aten.mm.default(view_16, arg12_1);  view_16 = arg12_1 = None
        add_tensor_34: "f32[16, 3072]" = torch.ops.aten.add.Tensor(mm_default_34, arg11_1);  mm_default_34 = arg11_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_17: "f32[1, 16, 3072]" = torch.ops.aten.reshape.default(add_tensor_34, [1, 16, 3072]);  add_tensor_34 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_4: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(view_17, 0.5)
        pow_1: "f32[1, 16, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_17, 3.0)
        mul_5: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_7: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(view_17, mul_5);  view_17 = mul_5 = None
        mul_6: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(add_7, 0.7978845608028654);  add_7 = None
        tanh: "f32[1, 16, 3072]" = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
        add_8: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_7: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(mul_4, add_8);  mul_4 = add_8 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_18: "f32[16, 3072]" = torch.ops.aten.reshape.default(mul_7, [-1, 3072]);  mul_7 = None
        mm_default_33: "f32[16, 768]" = torch.ops.aten.mm.default(view_18, arg14_1);  view_18 = arg14_1 = None
        add_tensor_33: "f32[16, 768]" = torch.ops.aten.add.Tensor(mm_default_33, arg13_1);  mm_default_33 = arg13_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_19: "f32[1, 16, 768]" = torch.ops.aten.reshape.default(add_tensor_33, [1, 16, 768]);  add_tensor_33 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_9: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(add_4, view_19);  add_4 = view_19 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
        getitem_11: "f32[1, 16, 1]" = var_mean_2[0]
        getitem_12: "f32[1, 16, 1]" = var_mean_2[1];  var_mean_2 = None
        sub_4: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_9, getitem_12);  getitem_12 = None
        add_10: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_11, 1e-05);  getitem_11 = None
        rsqrt_2: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_8: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = rsqrt_2 = None
        mul_9: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_8, arg15_1);  mul_8 = arg15_1 = None
        add_11: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_9, arg16_1);  mul_9 = arg16_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_20: "f32[16, 768]" = torch.ops.aten.reshape.default(add_11, [-1, 768]);  add_11 = None
        addmm_4: "f32[16, 2304]" = torch.ops.aten.addmm.default(arg17_1, view_20, arg18_1);  arg17_1 = view_20 = arg18_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_21: "f32[1, 16, 2304]" = torch.ops.aten.reshape.default(addmm_4, [1, 16, 2304]);  addmm_4 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_1 = torch.ops.aten.split.Tensor(view_21, 768, 2);  view_21 = None
        getitem_13: "f32[1, 16, 768]" = split_1[0]
        getitem_14: "f32[1, 16, 768]" = split_1[1]
        getitem_15: "f32[1, 16, 768]" = split_1[2];  split_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_24: "f32[1, 16, 12, 64]" = torch.ops.aten.reshape.default(getitem_13, [1, 16, -1, 64]);  getitem_13 = None
        permute_6: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_22: "f32[1, 16, 12, 64]" = torch.ops.aten.reshape.default(getitem_14, [1, 16, -1, 64]);  getitem_14 = None
        permute_4: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_23: "f32[1, 16, 12, 64]" = torch.ops.aten.reshape.default(getitem_15, [1, 16, -1, 64]);  getitem_15 = None
        permute_5: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_23, [0, 2, 1, 3]);  view_23 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_4: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[1, 1, 16, 16]" = torch.ops.aten.where.self(expand, full_default_4, full_default_3);  full_default_4 = full_default_3 = None
        expand_2: "f32[1, 12, 16, 16]" = torch.ops.aten.expand.default(where_1, [1, 12, 16, 16]);  where_1 = None
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_6, permute_4, permute_5, expand_2, False);  permute_6 = permute_4 = permute_5 = expand_2 = None
        getitem_16: "f32[1, 12, 16, 64]" = _scaled_dot_product_efficient_attention_1[0];  _scaled_dot_product_efficient_attention_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "f32[1, 16, 12, 64]" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3]);  getitem_16 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_25: "f32[1, 16, 768]" = torch.ops.aten.reshape.default(permute_7, [1, 16, -1]);  permute_7 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_26: "f32[16, 768]" = torch.ops.aten.reshape.default(view_25, [-1, 768]);  view_25 = None
        mm_default_32: "f32[16, 768]" = torch.ops.aten.mm.default(view_26, arg20_1);  view_26 = arg20_1 = None
        add_tensor_32: "f32[16, 768]" = torch.ops.aten.add.Tensor(mm_default_32, arg19_1);  mm_default_32 = arg19_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_27: "f32[1, 16, 768]" = torch.ops.aten.reshape.default(add_tensor_32, [1, 16, 768]);  add_tensor_32 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_12: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(view_27, add_9);  view_27 = add_9 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_3 = torch.ops.aten.var_mean.correction(add_12, [2], correction = 0, keepdim = True)
        getitem_20: "f32[1, 16, 1]" = var_mean_3[0]
        getitem_21: "f32[1, 16, 1]" = var_mean_3[1];  var_mean_3 = None
        sub_5: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_12, getitem_21);  getitem_21 = None
        add_13: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_3: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        mul_10: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_11: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_10, arg21_1);  mul_10 = arg21_1 = None
        add_14: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_11, arg22_1);  mul_11 = arg22_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_28: "f32[16, 768]" = torch.ops.aten.reshape.default(add_14, [-1, 768]);  add_14 = None
        mm_default_31: "f32[16, 3072]" = torch.ops.aten.mm.default(view_28, arg24_1);  view_28 = arg24_1 = None
        add_tensor_31: "f32[16, 3072]" = torch.ops.aten.add.Tensor(mm_default_31, arg23_1);  mm_default_31 = arg23_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_29: "f32[1, 16, 3072]" = torch.ops.aten.reshape.default(add_tensor_31, [1, 16, 3072]);  add_tensor_31 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_12: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(view_29, 0.5)
        pow_2: "f32[1, 16, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_29, 3.0)
        mul_13: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_15: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(view_29, mul_13);  view_29 = mul_13 = None
        mul_14: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(add_15, 0.7978845608028654);  add_15 = None
        tanh_1: "f32[1, 16, 3072]" = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
        add_16: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_15: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(mul_12, add_16);  mul_12 = add_16 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_30: "f32[16, 3072]" = torch.ops.aten.reshape.default(mul_15, [-1, 3072]);  mul_15 = None
        mm_default_30: "f32[16, 768]" = torch.ops.aten.mm.default(view_30, arg26_1);  view_30 = arg26_1 = None
        add_tensor_30: "f32[16, 768]" = torch.ops.aten.add.Tensor(mm_default_30, arg25_1);  mm_default_30 = arg25_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_31: "f32[1, 16, 768]" = torch.ops.aten.reshape.default(add_tensor_30, [1, 16, 768]);  add_tensor_30 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_17: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(add_12, view_31);  add_12 = view_31 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_4 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_22: "f32[1, 16, 1]" = var_mean_4[0]
        getitem_23: "f32[1, 16, 1]" = var_mean_4[1];  var_mean_4 = None
        sub_6: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_17, getitem_23);  getitem_23 = None
        add_18: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_4: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        mul_16: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
        mul_17: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_16, arg27_1);  mul_16 = arg27_1 = None
        add_19: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_17, arg28_1);  mul_17 = arg28_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_32: "f32[16, 768]" = torch.ops.aten.reshape.default(add_19, [-1, 768]);  add_19 = None
        addmm_8: "f32[16, 2304]" = torch.ops.aten.addmm.default(arg29_1, view_32, arg30_1);  arg29_1 = view_32 = arg30_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_33: "f32[1, 16, 2304]" = torch.ops.aten.reshape.default(addmm_8, [1, 16, 2304]);  addmm_8 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_2 = torch.ops.aten.split.Tensor(view_33, 768, 2);  view_33 = None
        getitem_24: "f32[1, 16, 768]" = split_2[0]
        getitem_25: "f32[1, 16, 768]" = split_2[1]
        getitem_26: "f32[1, 16, 768]" = split_2[2];  split_2 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_36: "f32[1, 16, 12, 64]" = torch.ops.aten.reshape.default(getitem_24, [1, 16, -1, 64]);  getitem_24 = None
        permute_10: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_34: "f32[1, 16, 12, 64]" = torch.ops.aten.reshape.default(getitem_25, [1, 16, -1, 64]);  getitem_25 = None
        permute_8: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_35: "f32[1, 16, 12, 64]" = torch.ops.aten.reshape.default(getitem_26, [1, 16, -1, 64]);  getitem_26 = None
        permute_9: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_35, [0, 2, 1, 3]);  view_35 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_6: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_5: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[1, 1, 16, 16]" = torch.ops.aten.where.self(expand, full_default_6, full_default_5);  full_default_6 = full_default_5 = None
        expand_3: "f32[1, 12, 16, 16]" = torch.ops.aten.expand.default(where_2, [1, 12, 16, 16]);  where_2 = None
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_10, permute_8, permute_9, expand_3, False);  permute_10 = permute_8 = permute_9 = expand_3 = None
        getitem_27: "f32[1, 12, 16, 64]" = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_11: "f32[1, 16, 12, 64]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3]);  getitem_27 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_37: "f32[1, 16, 768]" = torch.ops.aten.reshape.default(permute_11, [1, 16, -1]);  permute_11 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_38: "f32[16, 768]" = torch.ops.aten.reshape.default(view_37, [-1, 768]);  view_37 = None
        mm_default_29: "f32[16, 768]" = torch.ops.aten.mm.default(view_38, arg32_1);  view_38 = arg32_1 = None
        add_tensor_29: "f32[16, 768]" = torch.ops.aten.add.Tensor(mm_default_29, arg31_1);  mm_default_29 = arg31_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_39: "f32[1, 16, 768]" = torch.ops.aten.reshape.default(add_tensor_29, [1, 16, 768]);  add_tensor_29 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_20: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(view_39, add_17);  view_39 = add_17 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_5 = torch.ops.aten.var_mean.correction(add_20, [2], correction = 0, keepdim = True)
        getitem_31: "f32[1, 16, 1]" = var_mean_5[0]
        getitem_32: "f32[1, 16, 1]" = var_mean_5[1];  var_mean_5 = None
        sub_7: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_20, getitem_32);  getitem_32 = None
        add_21: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_31, 1e-05);  getitem_31 = None
        rsqrt_5: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        mul_18: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_5);  sub_7 = rsqrt_5 = None
        mul_19: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_18, arg33_1);  mul_18 = arg33_1 = None
        add_22: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_19, arg34_1);  mul_19 = arg34_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_40: "f32[16, 768]" = torch.ops.aten.reshape.default(add_22, [-1, 768]);  add_22 = None
        mm_default_28: "f32[16, 3072]" = torch.ops.aten.mm.default(view_40, arg36_1);  view_40 = arg36_1 = None
        add_tensor_28: "f32[16, 3072]" = torch.ops.aten.add.Tensor(mm_default_28, arg35_1);  mm_default_28 = arg35_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_41: "f32[1, 16, 3072]" = torch.ops.aten.reshape.default(add_tensor_28, [1, 16, 3072]);  add_tensor_28 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_20: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.5)
        pow_3: "f32[1, 16, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_41, 3.0)
        mul_21: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_23: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(view_41, mul_21);  view_41 = mul_21 = None
        mul_22: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(add_23, 0.7978845608028654);  add_23 = None
        tanh_2: "f32[1, 16, 3072]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
        add_24: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_23: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(mul_20, add_24);  mul_20 = add_24 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_42: "f32[16, 3072]" = torch.ops.aten.reshape.default(mul_23, [-1, 3072]);  mul_23 = None
        mm_default_27: "f32[16, 768]" = torch.ops.aten.mm.default(view_42, arg38_1);  view_42 = arg38_1 = None
        add_tensor_27: "f32[16, 768]" = torch.ops.aten.add.Tensor(mm_default_27, arg37_1);  mm_default_27 = arg37_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_43: "f32[1, 16, 768]" = torch.ops.aten.reshape.default(add_tensor_27, [1, 16, 768]);  add_tensor_27 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_25: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(add_20, view_43);  add_20 = view_43 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_6 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_33: "f32[1, 16, 1]" = var_mean_6[0]
        getitem_34: "f32[1, 16, 1]" = var_mean_6[1];  var_mean_6 = None
        sub_8: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_25, getitem_34);  getitem_34 = None
        add_26: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_33, 1e-05);  getitem_33 = None
        rsqrt_6: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        mul_24: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_6);  sub_8 = rsqrt_6 = None
        mul_25: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_24, arg39_1);  mul_24 = arg39_1 = None
        add_27: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_25, arg40_1);  mul_25 = arg40_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_44: "f32[16, 768]" = torch.ops.aten.reshape.default(add_27, [-1, 768]);  add_27 = None
        addmm_12: "f32[16, 2304]" = torch.ops.aten.addmm.default(arg41_1, view_44, arg42_1);  arg41_1 = view_44 = arg42_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_45: "f32[1, 16, 2304]" = torch.ops.aten.reshape.default(addmm_12, [1, 16, 2304]);  addmm_12 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_3 = torch.ops.aten.split.Tensor(view_45, 768, 2);  view_45 = None
        getitem_35: "f32[1, 16, 768]" = split_3[0]
        getitem_36: "f32[1, 16, 768]" = split_3[1]
        getitem_37: "f32[1, 16, 768]" = split_3[2];  split_3 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_48: "f32[1, 16, 12, 64]" = torch.ops.aten.reshape.default(getitem_35, [1, 16, -1, 64]);  getitem_35 = None
        permute_14: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_46: "f32[1, 16, 12, 64]" = torch.ops.aten.reshape.default(getitem_36, [1, 16, -1, 64]);  getitem_36 = None
        permute_12: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_47: "f32[1, 16, 12, 64]" = torch.ops.aten.reshape.default(getitem_37, [1, 16, -1, 64]);  getitem_37 = None
        permute_13: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_47, [0, 2, 1, 3]);  view_47 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_8: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_7: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "f32[1, 1, 16, 16]" = torch.ops.aten.where.self(expand, full_default_8, full_default_7);  full_default_8 = full_default_7 = None
        expand_4: "f32[1, 12, 16, 16]" = torch.ops.aten.expand.default(where_3, [1, 12, 16, 16]);  where_3 = None
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_14, permute_12, permute_13, expand_4, False);  permute_14 = permute_12 = permute_13 = expand_4 = None
        getitem_38: "f32[1, 12, 16, 64]" = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_15: "f32[1, 16, 12, 64]" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3]);  getitem_38 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_49: "f32[1, 16, 768]" = torch.ops.aten.reshape.default(permute_15, [1, 16, -1]);  permute_15 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_50: "f32[16, 768]" = torch.ops.aten.reshape.default(view_49, [-1, 768]);  view_49 = None
        mm_default_26: "f32[16, 768]" = torch.ops.aten.mm.default(view_50, arg44_1);  view_50 = arg44_1 = None
        add_tensor_26: "f32[16, 768]" = torch.ops.aten.add.Tensor(mm_default_26, arg43_1);  mm_default_26 = arg43_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_51: "f32[1, 16, 768]" = torch.ops.aten.reshape.default(add_tensor_26, [1, 16, 768]);  add_tensor_26 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_28: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(view_51, add_25);  view_51 = add_25 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_7 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
        getitem_42: "f32[1, 16, 1]" = var_mean_7[0]
        getitem_43: "f32[1, 16, 1]" = var_mean_7[1];  var_mean_7 = None
        sub_9: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_28, getitem_43);  getitem_43 = None
        add_29: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_7: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        mul_26: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_7);  sub_9 = rsqrt_7 = None
        mul_27: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_26, arg45_1);  mul_26 = arg45_1 = None
        add_30: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_27, arg46_1);  mul_27 = arg46_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_52: "f32[16, 768]" = torch.ops.aten.reshape.default(add_30, [-1, 768]);  add_30 = None
        mm_default_25: "f32[16, 3072]" = torch.ops.aten.mm.default(view_52, arg48_1);  view_52 = arg48_1 = None
        add_tensor_25: "f32[16, 3072]" = torch.ops.aten.add.Tensor(mm_default_25, arg47_1);  mm_default_25 = arg47_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_53: "f32[1, 16, 3072]" = torch.ops.aten.reshape.default(add_tensor_25, [1, 16, 3072]);  add_tensor_25 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_28: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(view_53, 0.5)
        pow_4: "f32[1, 16, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_53, 3.0)
        mul_29: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_31: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(view_53, mul_29);  view_53 = mul_29 = None
        mul_30: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(add_31, 0.7978845608028654);  add_31 = None
        tanh_3: "f32[1, 16, 3072]" = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
        add_32: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_31: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(mul_28, add_32);  mul_28 = add_32 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_54: "f32[16, 3072]" = torch.ops.aten.reshape.default(mul_31, [-1, 3072]);  mul_31 = None
        mm_default_24: "f32[16, 768]" = torch.ops.aten.mm.default(view_54, arg50_1);  view_54 = arg50_1 = None
        add_tensor_24: "f32[16, 768]" = torch.ops.aten.add.Tensor(mm_default_24, arg49_1);  mm_default_24 = arg49_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_55: "f32[1, 16, 768]" = torch.ops.aten.reshape.default(add_tensor_24, [1, 16, 768]);  add_tensor_24 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_33: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(add_28, view_55);  add_28 = view_55 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_8 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
        getitem_44: "f32[1, 16, 1]" = var_mean_8[0]
        getitem_45: "f32[1, 16, 1]" = var_mean_8[1];  var_mean_8 = None
        sub_10: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_33, getitem_45);  getitem_45 = None
        add_34: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_8: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        mul_32: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_8);  sub_10 = rsqrt_8 = None
        mul_33: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_32, arg51_1);  mul_32 = arg51_1 = None
        add_35: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_33, arg52_1);  mul_33 = arg52_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_56: "f32[16, 768]" = torch.ops.aten.reshape.default(add_35, [-1, 768]);  add_35 = None
        addmm_16: "f32[16, 2304]" = torch.ops.aten.addmm.default(arg53_1, view_56, arg54_1);  arg53_1 = view_56 = arg54_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_57: "f32[1, 16, 2304]" = torch.ops.aten.reshape.default(addmm_16, [1, 16, 2304]);  addmm_16 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_4 = torch.ops.aten.split.Tensor(view_57, 768, 2);  view_57 = None
        getitem_46: "f32[1, 16, 768]" = split_4[0]
        getitem_47: "f32[1, 16, 768]" = split_4[1]
        getitem_48: "f32[1, 16, 768]" = split_4[2];  split_4 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_60: "f32[1, 16, 12, 64]" = torch.ops.aten.reshape.default(getitem_46, [1, 16, -1, 64]);  getitem_46 = None
        permute_18: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_58: "f32[1, 16, 12, 64]" = torch.ops.aten.reshape.default(getitem_47, [1, 16, -1, 64]);  getitem_47 = None
        permute_16: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_59: "f32[1, 16, 12, 64]" = torch.ops.aten.reshape.default(getitem_48, [1, 16, -1, 64]);  getitem_48 = None
        permute_17: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_59, [0, 2, 1, 3]);  view_59 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_10: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_9: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "f32[1, 1, 16, 16]" = torch.ops.aten.where.self(expand, full_default_10, full_default_9);  full_default_10 = full_default_9 = None
        expand_5: "f32[1, 12, 16, 16]" = torch.ops.aten.expand.default(where_4, [1, 12, 16, 16]);  where_4 = None
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_18, permute_16, permute_17, expand_5, False);  permute_18 = permute_16 = permute_17 = expand_5 = None
        getitem_49: "f32[1, 12, 16, 64]" = _scaled_dot_product_efficient_attention_4[0];  _scaled_dot_product_efficient_attention_4 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_19: "f32[1, 16, 12, 64]" = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3]);  getitem_49 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_61: "f32[1, 16, 768]" = torch.ops.aten.reshape.default(permute_19, [1, 16, -1]);  permute_19 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_62: "f32[16, 768]" = torch.ops.aten.reshape.default(view_61, [-1, 768]);  view_61 = None
        mm_default_23: "f32[16, 768]" = torch.ops.aten.mm.default(view_62, arg56_1);  view_62 = arg56_1 = None
        add_tensor_23: "f32[16, 768]" = torch.ops.aten.add.Tensor(mm_default_23, arg55_1);  mm_default_23 = arg55_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_63: "f32[1, 16, 768]" = torch.ops.aten.reshape.default(add_tensor_23, [1, 16, 768]);  add_tensor_23 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_36: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(view_63, add_33);  view_63 = add_33 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_9 = torch.ops.aten.var_mean.correction(add_36, [2], correction = 0, keepdim = True)
        getitem_53: "f32[1, 16, 1]" = var_mean_9[0]
        getitem_54: "f32[1, 16, 1]" = var_mean_9[1];  var_mean_9 = None
        sub_11: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_36, getitem_54);  getitem_54 = None
        add_37: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_53, 1e-05);  getitem_53 = None
        rsqrt_9: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        mul_34: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_9);  sub_11 = rsqrt_9 = None
        mul_35: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_34, arg57_1);  mul_34 = arg57_1 = None
        add_38: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_35, arg58_1);  mul_35 = arg58_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_64: "f32[16, 768]" = torch.ops.aten.reshape.default(add_38, [-1, 768]);  add_38 = None
        mm_default_22: "f32[16, 3072]" = torch.ops.aten.mm.default(view_64, arg60_1);  view_64 = arg60_1 = None
        add_tensor_22: "f32[16, 3072]" = torch.ops.aten.add.Tensor(mm_default_22, arg59_1);  mm_default_22 = arg59_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_65: "f32[1, 16, 3072]" = torch.ops.aten.reshape.default(add_tensor_22, [1, 16, 3072]);  add_tensor_22 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_36: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(view_65, 0.5)
        pow_5: "f32[1, 16, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_65, 3.0)
        mul_37: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_39: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(view_65, mul_37);  view_65 = mul_37 = None
        mul_38: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(add_39, 0.7978845608028654);  add_39 = None
        tanh_4: "f32[1, 16, 3072]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_40: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_39: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(mul_36, add_40);  mul_36 = add_40 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_66: "f32[16, 3072]" = torch.ops.aten.reshape.default(mul_39, [-1, 3072]);  mul_39 = None
        mm_default_21: "f32[16, 768]" = torch.ops.aten.mm.default(view_66, arg62_1);  view_66 = arg62_1 = None
        add_tensor_21: "f32[16, 768]" = torch.ops.aten.add.Tensor(mm_default_21, arg61_1);  mm_default_21 = arg61_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_67: "f32[1, 16, 768]" = torch.ops.aten.reshape.default(add_tensor_21, [1, 16, 768]);  add_tensor_21 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_41: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(add_36, view_67);  add_36 = view_67 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_10 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
        getitem_55: "f32[1, 16, 1]" = var_mean_10[0]
        getitem_56: "f32[1, 16, 1]" = var_mean_10[1];  var_mean_10 = None
        sub_12: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_41, getitem_56);  getitem_56 = None
        add_42: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_55, 1e-05);  getitem_55 = None
        rsqrt_10: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        mul_40: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_10);  sub_12 = rsqrt_10 = None
        mul_41: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_40, arg63_1);  mul_40 = arg63_1 = None
        add_43: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_41, arg64_1);  mul_41 = arg64_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_68: "f32[16, 768]" = torch.ops.aten.reshape.default(add_43, [-1, 768]);  add_43 = None
        addmm_20: "f32[16, 2304]" = torch.ops.aten.addmm.default(arg65_1, view_68, arg66_1);  arg65_1 = view_68 = arg66_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_69: "f32[1, 16, 2304]" = torch.ops.aten.reshape.default(addmm_20, [1, 16, 2304]);  addmm_20 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_5 = torch.ops.aten.split.Tensor(view_69, 768, 2);  view_69 = None
        getitem_57: "f32[1, 16, 768]" = split_5[0]
        getitem_58: "f32[1, 16, 768]" = split_5[1]
        getitem_59: "f32[1, 16, 768]" = split_5[2];  split_5 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_72: "f32[1, 16, 12, 64]" = torch.ops.aten.reshape.default(getitem_57, [1, 16, -1, 64]);  getitem_57 = None
        permute_22: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_70: "f32[1, 16, 12, 64]" = torch.ops.aten.reshape.default(getitem_58, [1, 16, -1, 64]);  getitem_58 = None
        permute_20: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_71: "f32[1, 16, 12, 64]" = torch.ops.aten.reshape.default(getitem_59, [1, 16, -1, 64]);  getitem_59 = None
        permute_21: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_12: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_11: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5: "f32[1, 1, 16, 16]" = torch.ops.aten.where.self(expand, full_default_12, full_default_11);  full_default_12 = full_default_11 = None
        expand_6: "f32[1, 12, 16, 16]" = torch.ops.aten.expand.default(where_5, [1, 12, 16, 16]);  where_5 = None
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_22, permute_20, permute_21, expand_6, False);  permute_22 = permute_20 = permute_21 = expand_6 = None
        getitem_60: "f32[1, 12, 16, 64]" = _scaled_dot_product_efficient_attention_5[0];  _scaled_dot_product_efficient_attention_5 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_23: "f32[1, 16, 12, 64]" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_73: "f32[1, 16, 768]" = torch.ops.aten.reshape.default(permute_23, [1, 16, -1]);  permute_23 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_74: "f32[16, 768]" = torch.ops.aten.reshape.default(view_73, [-1, 768]);  view_73 = None
        mm_default_20: "f32[16, 768]" = torch.ops.aten.mm.default(view_74, arg68_1);  view_74 = arg68_1 = None
        add_tensor_20: "f32[16, 768]" = torch.ops.aten.add.Tensor(mm_default_20, arg67_1);  mm_default_20 = arg67_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_75: "f32[1, 16, 768]" = torch.ops.aten.reshape.default(add_tensor_20, [1, 16, 768]);  add_tensor_20 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_44: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(view_75, add_41);  view_75 = add_41 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_11 = torch.ops.aten.var_mean.correction(add_44, [2], correction = 0, keepdim = True)
        getitem_64: "f32[1, 16, 1]" = var_mean_11[0]
        getitem_65: "f32[1, 16, 1]" = var_mean_11[1];  var_mean_11 = None
        sub_13: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_44, getitem_65);  getitem_65 = None
        add_45: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_11: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        mul_42: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_11);  sub_13 = rsqrt_11 = None
        mul_43: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_42, arg69_1);  mul_42 = arg69_1 = None
        add_46: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_43, arg70_1);  mul_43 = arg70_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_76: "f32[16, 768]" = torch.ops.aten.reshape.default(add_46, [-1, 768]);  add_46 = None
        mm_default_19: "f32[16, 3072]" = torch.ops.aten.mm.default(view_76, arg72_1);  view_76 = arg72_1 = None
        add_tensor_19: "f32[16, 3072]" = torch.ops.aten.add.Tensor(mm_default_19, arg71_1);  mm_default_19 = arg71_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_77: "f32[1, 16, 3072]" = torch.ops.aten.reshape.default(add_tensor_19, [1, 16, 3072]);  add_tensor_19 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_44: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(view_77, 0.5)
        pow_6: "f32[1, 16, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_77, 3.0)
        mul_45: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_47: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(view_77, mul_45);  view_77 = mul_45 = None
        mul_46: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(add_47, 0.7978845608028654);  add_47 = None
        tanh_5: "f32[1, 16, 3072]" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
        add_48: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_47: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(mul_44, add_48);  mul_44 = add_48 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_78: "f32[16, 3072]" = torch.ops.aten.reshape.default(mul_47, [-1, 3072]);  mul_47 = None
        mm_default_18: "f32[16, 768]" = torch.ops.aten.mm.default(view_78, arg74_1);  view_78 = arg74_1 = None
        add_tensor_18: "f32[16, 768]" = torch.ops.aten.add.Tensor(mm_default_18, arg73_1);  mm_default_18 = arg73_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_79: "f32[1, 16, 768]" = torch.ops.aten.reshape.default(add_tensor_18, [1, 16, 768]);  add_tensor_18 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_49: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(add_44, view_79);  add_44 = view_79 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_12 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
        getitem_66: "f32[1, 16, 1]" = var_mean_12[0]
        getitem_67: "f32[1, 16, 1]" = var_mean_12[1];  var_mean_12 = None
        sub_14: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_49, getitem_67);  getitem_67 = None
        add_50: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_12: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        mul_48: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_12);  sub_14 = rsqrt_12 = None
        mul_49: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_48, arg75_1);  mul_48 = arg75_1 = None
        add_51: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_49, arg76_1);  mul_49 = arg76_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_80: "f32[16, 768]" = torch.ops.aten.reshape.default(add_51, [-1, 768]);  add_51 = None
        addmm_24: "f32[16, 2304]" = torch.ops.aten.addmm.default(arg77_1, view_80, arg78_1);  arg77_1 = view_80 = arg78_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_81: "f32[1, 16, 2304]" = torch.ops.aten.reshape.default(addmm_24, [1, 16, 2304]);  addmm_24 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_6 = torch.ops.aten.split.Tensor(view_81, 768, 2);  view_81 = None
        getitem_68: "f32[1, 16, 768]" = split_6[0]
        getitem_69: "f32[1, 16, 768]" = split_6[1]
        getitem_70: "f32[1, 16, 768]" = split_6[2];  split_6 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_84: "f32[1, 16, 12, 64]" = torch.ops.aten.reshape.default(getitem_68, [1, 16, -1, 64]);  getitem_68 = None
        permute_26: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_82: "f32[1, 16, 12, 64]" = torch.ops.aten.reshape.default(getitem_69, [1, 16, -1, 64]);  getitem_69 = None
        permute_24: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_83: "f32[1, 16, 12, 64]" = torch.ops.aten.reshape.default(getitem_70, [1, 16, -1, 64]);  getitem_70 = None
        permute_25: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_83, [0, 2, 1, 3]);  view_83 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_14: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_13: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6: "f32[1, 1, 16, 16]" = torch.ops.aten.where.self(expand, full_default_14, full_default_13);  full_default_14 = full_default_13 = None
        expand_7: "f32[1, 12, 16, 16]" = torch.ops.aten.expand.default(where_6, [1, 12, 16, 16]);  where_6 = None
        _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_26, permute_24, permute_25, expand_7, False);  permute_26 = permute_24 = permute_25 = expand_7 = None
        getitem_71: "f32[1, 12, 16, 64]" = _scaled_dot_product_efficient_attention_6[0];  _scaled_dot_product_efficient_attention_6 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_27: "f32[1, 16, 12, 64]" = torch.ops.aten.permute.default(getitem_71, [0, 2, 1, 3]);  getitem_71 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_85: "f32[1, 16, 768]" = torch.ops.aten.reshape.default(permute_27, [1, 16, -1]);  permute_27 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_86: "f32[16, 768]" = torch.ops.aten.reshape.default(view_85, [-1, 768]);  view_85 = None
        mm_default_17: "f32[16, 768]" = torch.ops.aten.mm.default(view_86, arg80_1);  view_86 = arg80_1 = None
        add_tensor_17: "f32[16, 768]" = torch.ops.aten.add.Tensor(mm_default_17, arg79_1);  mm_default_17 = arg79_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_87: "f32[1, 16, 768]" = torch.ops.aten.reshape.default(add_tensor_17, [1, 16, 768]);  add_tensor_17 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_52: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(view_87, add_49);  view_87 = add_49 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_13 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
        getitem_75: "f32[1, 16, 1]" = var_mean_13[0]
        getitem_76: "f32[1, 16, 1]" = var_mean_13[1];  var_mean_13 = None
        sub_15: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_52, getitem_76);  getitem_76 = None
        add_53: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_75, 1e-05);  getitem_75 = None
        rsqrt_13: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        mul_50: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_13);  sub_15 = rsqrt_13 = None
        mul_51: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_50, arg81_1);  mul_50 = arg81_1 = None
        add_54: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_51, arg82_1);  mul_51 = arg82_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_88: "f32[16, 768]" = torch.ops.aten.reshape.default(add_54, [-1, 768]);  add_54 = None
        mm_default_16: "f32[16, 3072]" = torch.ops.aten.mm.default(view_88, arg84_1);  view_88 = arg84_1 = None
        add_tensor_16: "f32[16, 3072]" = torch.ops.aten.add.Tensor(mm_default_16, arg83_1);  mm_default_16 = arg83_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_89: "f32[1, 16, 3072]" = torch.ops.aten.reshape.default(add_tensor_16, [1, 16, 3072]);  add_tensor_16 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_52: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(view_89, 0.5)
        pow_7: "f32[1, 16, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_89, 3.0)
        mul_53: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_55: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(view_89, mul_53);  view_89 = mul_53 = None
        mul_54: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(add_55, 0.7978845608028654);  add_55 = None
        tanh_6: "f32[1, 16, 3072]" = torch.ops.aten.tanh.default(mul_54);  mul_54 = None
        add_56: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
        mul_55: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(mul_52, add_56);  mul_52 = add_56 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_90: "f32[16, 3072]" = torch.ops.aten.reshape.default(mul_55, [-1, 3072]);  mul_55 = None
        mm_default_15: "f32[16, 768]" = torch.ops.aten.mm.default(view_90, arg86_1);  view_90 = arg86_1 = None
        add_tensor_15: "f32[16, 768]" = torch.ops.aten.add.Tensor(mm_default_15, arg85_1);  mm_default_15 = arg85_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_91: "f32[1, 16, 768]" = torch.ops.aten.reshape.default(add_tensor_15, [1, 16, 768]);  add_tensor_15 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_57: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(add_52, view_91);  add_52 = view_91 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_14 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_77: "f32[1, 16, 1]" = var_mean_14[0]
        getitem_78: "f32[1, 16, 1]" = var_mean_14[1];  var_mean_14 = None
        sub_16: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_57, getitem_78);  getitem_78 = None
        add_58: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_77, 1e-05);  getitem_77 = None
        rsqrt_14: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_56: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_14);  sub_16 = rsqrt_14 = None
        mul_57: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_56, arg87_1);  mul_56 = arg87_1 = None
        add_59: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_57, arg88_1);  mul_57 = arg88_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_92: "f32[16, 768]" = torch.ops.aten.reshape.default(add_59, [-1, 768]);  add_59 = None
        addmm_28: "f32[16, 2304]" = torch.ops.aten.addmm.default(arg89_1, view_92, arg90_1);  arg89_1 = view_92 = arg90_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_93: "f32[1, 16, 2304]" = torch.ops.aten.reshape.default(addmm_28, [1, 16, 2304]);  addmm_28 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_7 = torch.ops.aten.split.Tensor(view_93, 768, 2);  view_93 = None
        getitem_79: "f32[1, 16, 768]" = split_7[0]
        getitem_80: "f32[1, 16, 768]" = split_7[1]
        getitem_81: "f32[1, 16, 768]" = split_7[2];  split_7 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_96: "f32[1, 16, 12, 64]" = torch.ops.aten.reshape.default(getitem_79, [1, 16, -1, 64]);  getitem_79 = None
        permute_30: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_94: "f32[1, 16, 12, 64]" = torch.ops.aten.reshape.default(getitem_80, [1, 16, -1, 64]);  getitem_80 = None
        permute_28: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_95: "f32[1, 16, 12, 64]" = torch.ops.aten.reshape.default(getitem_81, [1, 16, -1, 64]);  getitem_81 = None
        permute_29: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_16: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_15: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_7: "f32[1, 1, 16, 16]" = torch.ops.aten.where.self(expand, full_default_16, full_default_15);  full_default_16 = full_default_15 = None
        expand_8: "f32[1, 12, 16, 16]" = torch.ops.aten.expand.default(where_7, [1, 12, 16, 16]);  where_7 = None
        _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_30, permute_28, permute_29, expand_8, False);  permute_30 = permute_28 = permute_29 = expand_8 = None
        getitem_82: "f32[1, 12, 16, 64]" = _scaled_dot_product_efficient_attention_7[0];  _scaled_dot_product_efficient_attention_7 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_31: "f32[1, 16, 12, 64]" = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3]);  getitem_82 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_97: "f32[1, 16, 768]" = torch.ops.aten.reshape.default(permute_31, [1, 16, -1]);  permute_31 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_98: "f32[16, 768]" = torch.ops.aten.reshape.default(view_97, [-1, 768]);  view_97 = None
        mm_default_14: "f32[16, 768]" = torch.ops.aten.mm.default(view_98, arg92_1);  view_98 = arg92_1 = None
        add_tensor_14: "f32[16, 768]" = torch.ops.aten.add.Tensor(mm_default_14, arg91_1);  mm_default_14 = arg91_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_99: "f32[1, 16, 768]" = torch.ops.aten.reshape.default(add_tensor_14, [1, 16, 768]);  add_tensor_14 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_60: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(view_99, add_57);  view_99 = add_57 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_15 = torch.ops.aten.var_mean.correction(add_60, [2], correction = 0, keepdim = True)
        getitem_86: "f32[1, 16, 1]" = var_mean_15[0]
        getitem_87: "f32[1, 16, 1]" = var_mean_15[1];  var_mean_15 = None
        sub_17: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_60, getitem_87);  getitem_87 = None
        add_61: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
        rsqrt_15: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        mul_58: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_15);  sub_17 = rsqrt_15 = None
        mul_59: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_58, arg93_1);  mul_58 = arg93_1 = None
        add_62: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_59, arg94_1);  mul_59 = arg94_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_100: "f32[16, 768]" = torch.ops.aten.reshape.default(add_62, [-1, 768]);  add_62 = None
        mm_default_13: "f32[16, 3072]" = torch.ops.aten.mm.default(view_100, arg96_1);  view_100 = arg96_1 = None
        add_tensor_13: "f32[16, 3072]" = torch.ops.aten.add.Tensor(mm_default_13, arg95_1);  mm_default_13 = arg95_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_101: "f32[1, 16, 3072]" = torch.ops.aten.reshape.default(add_tensor_13, [1, 16, 3072]);  add_tensor_13 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_60: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(view_101, 0.5)
        pow_8: "f32[1, 16, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_101, 3.0)
        mul_61: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_63: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(view_101, mul_61);  view_101 = mul_61 = None
        mul_62: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(add_63, 0.7978845608028654);  add_63 = None
        tanh_7: "f32[1, 16, 3072]" = torch.ops.aten.tanh.default(mul_62);  mul_62 = None
        add_64: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
        mul_63: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(mul_60, add_64);  mul_60 = add_64 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_102: "f32[16, 3072]" = torch.ops.aten.reshape.default(mul_63, [-1, 3072]);  mul_63 = None
        mm_default_12: "f32[16, 768]" = torch.ops.aten.mm.default(view_102, arg98_1);  view_102 = arg98_1 = None
        add_tensor_12: "f32[16, 768]" = torch.ops.aten.add.Tensor(mm_default_12, arg97_1);  mm_default_12 = arg97_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_103: "f32[1, 16, 768]" = torch.ops.aten.reshape.default(add_tensor_12, [1, 16, 768]);  add_tensor_12 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_65: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(add_60, view_103);  add_60 = view_103 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_16 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
        getitem_88: "f32[1, 16, 1]" = var_mean_16[0]
        getitem_89: "f32[1, 16, 1]" = var_mean_16[1];  var_mean_16 = None
        sub_18: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_65, getitem_89);  getitem_89 = None
        add_66: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_16: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        mul_64: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_16);  sub_18 = rsqrt_16 = None
        mul_65: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_64, arg99_1);  mul_64 = arg99_1 = None
        add_67: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_65, arg100_1);  mul_65 = arg100_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_104: "f32[16, 768]" = torch.ops.aten.reshape.default(add_67, [-1, 768]);  add_67 = None
        addmm_32: "f32[16, 2304]" = torch.ops.aten.addmm.default(arg101_1, view_104, arg102_1);  arg101_1 = view_104 = arg102_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_105: "f32[1, 16, 2304]" = torch.ops.aten.reshape.default(addmm_32, [1, 16, 2304]);  addmm_32 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_8 = torch.ops.aten.split.Tensor(view_105, 768, 2);  view_105 = None
        getitem_90: "f32[1, 16, 768]" = split_8[0]
        getitem_91: "f32[1, 16, 768]" = split_8[1]
        getitem_92: "f32[1, 16, 768]" = split_8[2];  split_8 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_108: "f32[1, 16, 12, 64]" = torch.ops.aten.reshape.default(getitem_90, [1, 16, -1, 64]);  getitem_90 = None
        permute_34: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_108, [0, 2, 1, 3]);  view_108 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_106: "f32[1, 16, 12, 64]" = torch.ops.aten.reshape.default(getitem_91, [1, 16, -1, 64]);  getitem_91 = None
        permute_32: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_107: "f32[1, 16, 12, 64]" = torch.ops.aten.reshape.default(getitem_92, [1, 16, -1, 64]);  getitem_92 = None
        permute_33: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_107, [0, 2, 1, 3]);  view_107 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_18: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_17: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_8: "f32[1, 1, 16, 16]" = torch.ops.aten.where.self(expand, full_default_18, full_default_17);  full_default_18 = full_default_17 = None
        expand_9: "f32[1, 12, 16, 16]" = torch.ops.aten.expand.default(where_8, [1, 12, 16, 16]);  where_8 = None
        _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_34, permute_32, permute_33, expand_9, False);  permute_34 = permute_32 = permute_33 = expand_9 = None
        getitem_93: "f32[1, 12, 16, 64]" = _scaled_dot_product_efficient_attention_8[0];  _scaled_dot_product_efficient_attention_8 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_35: "f32[1, 16, 12, 64]" = torch.ops.aten.permute.default(getitem_93, [0, 2, 1, 3]);  getitem_93 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_109: "f32[1, 16, 768]" = torch.ops.aten.reshape.default(permute_35, [1, 16, -1]);  permute_35 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_110: "f32[16, 768]" = torch.ops.aten.reshape.default(view_109, [-1, 768]);  view_109 = None
        mm_default_11: "f32[16, 768]" = torch.ops.aten.mm.default(view_110, arg104_1);  view_110 = arg104_1 = None
        add_tensor_11: "f32[16, 768]" = torch.ops.aten.add.Tensor(mm_default_11, arg103_1);  mm_default_11 = arg103_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_111: "f32[1, 16, 768]" = torch.ops.aten.reshape.default(add_tensor_11, [1, 16, 768]);  add_tensor_11 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_68: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(view_111, add_65);  view_111 = add_65 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_17 = torch.ops.aten.var_mean.correction(add_68, [2], correction = 0, keepdim = True)
        getitem_97: "f32[1, 16, 1]" = var_mean_17[0]
        getitem_98: "f32[1, 16, 1]" = var_mean_17[1];  var_mean_17 = None
        sub_19: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_68, getitem_98);  getitem_98 = None
        add_69: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_97, 1e-05);  getitem_97 = None
        rsqrt_17: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        mul_66: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_17);  sub_19 = rsqrt_17 = None
        mul_67: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_66, arg105_1);  mul_66 = arg105_1 = None
        add_70: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_67, arg106_1);  mul_67 = arg106_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_112: "f32[16, 768]" = torch.ops.aten.reshape.default(add_70, [-1, 768]);  add_70 = None
        mm_default_10: "f32[16, 3072]" = torch.ops.aten.mm.default(view_112, arg108_1);  view_112 = arg108_1 = None
        add_tensor_10: "f32[16, 3072]" = torch.ops.aten.add.Tensor(mm_default_10, arg107_1);  mm_default_10 = arg107_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_113: "f32[1, 16, 3072]" = torch.ops.aten.reshape.default(add_tensor_10, [1, 16, 3072]);  add_tensor_10 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_68: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(view_113, 0.5)
        pow_9: "f32[1, 16, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_113, 3.0)
        mul_69: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_71: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(view_113, mul_69);  view_113 = mul_69 = None
        mul_70: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(add_71, 0.7978845608028654);  add_71 = None
        tanh_8: "f32[1, 16, 3072]" = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
        add_72: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
        mul_71: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(mul_68, add_72);  mul_68 = add_72 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_114: "f32[16, 3072]" = torch.ops.aten.reshape.default(mul_71, [-1, 3072]);  mul_71 = None
        mm_default_9: "f32[16, 768]" = torch.ops.aten.mm.default(view_114, arg110_1);  view_114 = arg110_1 = None
        add_tensor_9: "f32[16, 768]" = torch.ops.aten.add.Tensor(mm_default_9, arg109_1);  mm_default_9 = arg109_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_115: "f32[1, 16, 768]" = torch.ops.aten.reshape.default(add_tensor_9, [1, 16, 768]);  add_tensor_9 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_73: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(add_68, view_115);  add_68 = view_115 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_18 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
        getitem_99: "f32[1, 16, 1]" = var_mean_18[0]
        getitem_100: "f32[1, 16, 1]" = var_mean_18[1];  var_mean_18 = None
        sub_20: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_73, getitem_100);  getitem_100 = None
        add_74: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_99, 1e-05);  getitem_99 = None
        rsqrt_18: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        mul_72: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_18);  sub_20 = rsqrt_18 = None
        mul_73: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_72, arg111_1);  mul_72 = arg111_1 = None
        add_75: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_73, arg112_1);  mul_73 = arg112_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_116: "f32[16, 768]" = torch.ops.aten.reshape.default(add_75, [-1, 768]);  add_75 = None
        addmm_36: "f32[16, 2304]" = torch.ops.aten.addmm.default(arg113_1, view_116, arg114_1);  arg113_1 = view_116 = arg114_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_117: "f32[1, 16, 2304]" = torch.ops.aten.reshape.default(addmm_36, [1, 16, 2304]);  addmm_36 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_9 = torch.ops.aten.split.Tensor(view_117, 768, 2);  view_117 = None
        getitem_101: "f32[1, 16, 768]" = split_9[0]
        getitem_102: "f32[1, 16, 768]" = split_9[1]
        getitem_103: "f32[1, 16, 768]" = split_9[2];  split_9 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_120: "f32[1, 16, 12, 64]" = torch.ops.aten.reshape.default(getitem_101, [1, 16, -1, 64]);  getitem_101 = None
        permute_38: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_118: "f32[1, 16, 12, 64]" = torch.ops.aten.reshape.default(getitem_102, [1, 16, -1, 64]);  getitem_102 = None
        permute_36: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_119: "f32[1, 16, 12, 64]" = torch.ops.aten.reshape.default(getitem_103, [1, 16, -1, 64]);  getitem_103 = None
        permute_37: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_20: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_19: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_9: "f32[1, 1, 16, 16]" = torch.ops.aten.where.self(expand, full_default_20, full_default_19);  full_default_20 = full_default_19 = None
        expand_10: "f32[1, 12, 16, 16]" = torch.ops.aten.expand.default(where_9, [1, 12, 16, 16]);  where_9 = None
        _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_38, permute_36, permute_37, expand_10, False);  permute_38 = permute_36 = permute_37 = expand_10 = None
        getitem_104: "f32[1, 12, 16, 64]" = _scaled_dot_product_efficient_attention_9[0];  _scaled_dot_product_efficient_attention_9 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_39: "f32[1, 16, 12, 64]" = torch.ops.aten.permute.default(getitem_104, [0, 2, 1, 3]);  getitem_104 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_121: "f32[1, 16, 768]" = torch.ops.aten.reshape.default(permute_39, [1, 16, -1]);  permute_39 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_122: "f32[16, 768]" = torch.ops.aten.reshape.default(view_121, [-1, 768]);  view_121 = None
        mm_default_8: "f32[16, 768]" = torch.ops.aten.mm.default(view_122, arg116_1);  view_122 = arg116_1 = None
        add_tensor_8: "f32[16, 768]" = torch.ops.aten.add.Tensor(mm_default_8, arg115_1);  mm_default_8 = arg115_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_123: "f32[1, 16, 768]" = torch.ops.aten.reshape.default(add_tensor_8, [1, 16, 768]);  add_tensor_8 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_76: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(view_123, add_73);  view_123 = add_73 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_19 = torch.ops.aten.var_mean.correction(add_76, [2], correction = 0, keepdim = True)
        getitem_108: "f32[1, 16, 1]" = var_mean_19[0]
        getitem_109: "f32[1, 16, 1]" = var_mean_19[1];  var_mean_19 = None
        sub_21: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_76, getitem_109);  getitem_109 = None
        add_77: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
        rsqrt_19: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        mul_74: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_19);  sub_21 = rsqrt_19 = None
        mul_75: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_74, arg117_1);  mul_74 = arg117_1 = None
        add_78: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_75, arg118_1);  mul_75 = arg118_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_124: "f32[16, 768]" = torch.ops.aten.reshape.default(add_78, [-1, 768]);  add_78 = None
        mm_default_7: "f32[16, 3072]" = torch.ops.aten.mm.default(view_124, arg120_1);  view_124 = arg120_1 = None
        add_tensor_7: "f32[16, 3072]" = torch.ops.aten.add.Tensor(mm_default_7, arg119_1);  mm_default_7 = arg119_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_125: "f32[1, 16, 3072]" = torch.ops.aten.reshape.default(add_tensor_7, [1, 16, 3072]);  add_tensor_7 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_76: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(view_125, 0.5)
        pow_10: "f32[1, 16, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_125, 3.0)
        mul_77: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_79: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(view_125, mul_77);  view_125 = mul_77 = None
        mul_78: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(add_79, 0.7978845608028654);  add_79 = None
        tanh_9: "f32[1, 16, 3072]" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
        add_80: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
        mul_79: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(mul_76, add_80);  mul_76 = add_80 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_126: "f32[16, 3072]" = torch.ops.aten.reshape.default(mul_79, [-1, 3072]);  mul_79 = None
        mm_default_6: "f32[16, 768]" = torch.ops.aten.mm.default(view_126, arg122_1);  view_126 = arg122_1 = None
        add_tensor_6: "f32[16, 768]" = torch.ops.aten.add.Tensor(mm_default_6, arg121_1);  mm_default_6 = arg121_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_127: "f32[1, 16, 768]" = torch.ops.aten.reshape.default(add_tensor_6, [1, 16, 768]);  add_tensor_6 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_81: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(add_76, view_127);  add_76 = view_127 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_20 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
        getitem_110: "f32[1, 16, 1]" = var_mean_20[0]
        getitem_111: "f32[1, 16, 1]" = var_mean_20[1];  var_mean_20 = None
        sub_22: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_81, getitem_111);  getitem_111 = None
        add_82: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
        rsqrt_20: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        mul_80: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_20);  sub_22 = rsqrt_20 = None
        mul_81: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_80, arg123_1);  mul_80 = arg123_1 = None
        add_83: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_81, arg124_1);  mul_81 = arg124_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_128: "f32[16, 768]" = torch.ops.aten.reshape.default(add_83, [-1, 768]);  add_83 = None
        addmm_40: "f32[16, 2304]" = torch.ops.aten.addmm.default(arg125_1, view_128, arg126_1);  arg125_1 = view_128 = arg126_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_129: "f32[1, 16, 2304]" = torch.ops.aten.reshape.default(addmm_40, [1, 16, 2304]);  addmm_40 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_10 = torch.ops.aten.split.Tensor(view_129, 768, 2);  view_129 = None
        getitem_112: "f32[1, 16, 768]" = split_10[0]
        getitem_113: "f32[1, 16, 768]" = split_10[1]
        getitem_114: "f32[1, 16, 768]" = split_10[2];  split_10 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_132: "f32[1, 16, 12, 64]" = torch.ops.aten.reshape.default(getitem_112, [1, 16, -1, 64]);  getitem_112 = None
        permute_42: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_132, [0, 2, 1, 3]);  view_132 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_130: "f32[1, 16, 12, 64]" = torch.ops.aten.reshape.default(getitem_113, [1, 16, -1, 64]);  getitem_113 = None
        permute_40: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_131: "f32[1, 16, 12, 64]" = torch.ops.aten.reshape.default(getitem_114, [1, 16, -1, 64]);  getitem_114 = None
        permute_41: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_131, [0, 2, 1, 3]);  view_131 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_22: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_21: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_10: "f32[1, 1, 16, 16]" = torch.ops.aten.where.self(expand, full_default_22, full_default_21);  full_default_22 = full_default_21 = None
        expand_11: "f32[1, 12, 16, 16]" = torch.ops.aten.expand.default(where_10, [1, 12, 16, 16]);  where_10 = None
        _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_42, permute_40, permute_41, expand_11, False);  permute_42 = permute_40 = permute_41 = expand_11 = None
        getitem_115: "f32[1, 12, 16, 64]" = _scaled_dot_product_efficient_attention_10[0];  _scaled_dot_product_efficient_attention_10 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_43: "f32[1, 16, 12, 64]" = torch.ops.aten.permute.default(getitem_115, [0, 2, 1, 3]);  getitem_115 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_133: "f32[1, 16, 768]" = torch.ops.aten.reshape.default(permute_43, [1, 16, -1]);  permute_43 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_134: "f32[16, 768]" = torch.ops.aten.reshape.default(view_133, [-1, 768]);  view_133 = None
        mm_default_5: "f32[16, 768]" = torch.ops.aten.mm.default(view_134, arg128_1);  view_134 = arg128_1 = None
        add_tensor_5: "f32[16, 768]" = torch.ops.aten.add.Tensor(mm_default_5, arg127_1);  mm_default_5 = arg127_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_135: "f32[1, 16, 768]" = torch.ops.aten.reshape.default(add_tensor_5, [1, 16, 768]);  add_tensor_5 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_84: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(view_135, add_81);  view_135 = add_81 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_21 = torch.ops.aten.var_mean.correction(add_84, [2], correction = 0, keepdim = True)
        getitem_119: "f32[1, 16, 1]" = var_mean_21[0]
        getitem_120: "f32[1, 16, 1]" = var_mean_21[1];  var_mean_21 = None
        sub_23: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_84, getitem_120);  getitem_120 = None
        add_85: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_119, 1e-05);  getitem_119 = None
        rsqrt_21: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        mul_82: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_21);  sub_23 = rsqrt_21 = None
        mul_83: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_82, arg129_1);  mul_82 = arg129_1 = None
        add_86: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_83, arg130_1);  mul_83 = arg130_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_136: "f32[16, 768]" = torch.ops.aten.reshape.default(add_86, [-1, 768]);  add_86 = None
        mm_default_4: "f32[16, 3072]" = torch.ops.aten.mm.default(view_136, arg132_1);  view_136 = arg132_1 = None
        add_tensor_4: "f32[16, 3072]" = torch.ops.aten.add.Tensor(mm_default_4, arg131_1);  mm_default_4 = arg131_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_137: "f32[1, 16, 3072]" = torch.ops.aten.reshape.default(add_tensor_4, [1, 16, 3072]);  add_tensor_4 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_84: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(view_137, 0.5)
        pow_11: "f32[1, 16, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_137, 3.0)
        mul_85: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_87: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(view_137, mul_85);  view_137 = mul_85 = None
        mul_86: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(add_87, 0.7978845608028654);  add_87 = None
        tanh_10: "f32[1, 16, 3072]" = torch.ops.aten.tanh.default(mul_86);  mul_86 = None
        add_88: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
        mul_87: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(mul_84, add_88);  mul_84 = add_88 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_138: "f32[16, 3072]" = torch.ops.aten.reshape.default(mul_87, [-1, 3072]);  mul_87 = None
        mm_default_3: "f32[16, 768]" = torch.ops.aten.mm.default(view_138, arg134_1);  view_138 = arg134_1 = None
        add_tensor_3: "f32[16, 768]" = torch.ops.aten.add.Tensor(mm_default_3, arg133_1);  mm_default_3 = arg133_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_139: "f32[1, 16, 768]" = torch.ops.aten.reshape.default(add_tensor_3, [1, 16, 768]);  add_tensor_3 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_89: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(add_84, view_139);  add_84 = view_139 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:413 in forward, code: hidden_states = self.ln_1(hidden_states)
        var_mean_22 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
        getitem_121: "f32[1, 16, 1]" = var_mean_22[0]
        getitem_122: "f32[1, 16, 1]" = var_mean_22[1];  var_mean_22 = None
        sub_24: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_89, getitem_122);  getitem_122 = None
        add_90: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_121, 1e-05);  getitem_121 = None
        rsqrt_22: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        mul_88: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_22);  sub_24 = rsqrt_22 = None
        mul_89: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_88, arg135_1);  mul_88 = arg135_1 = None
        add_91: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_89, arg136_1);  mul_89 = arg136_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_140: "f32[16, 768]" = torch.ops.aten.reshape.default(add_91, [-1, 768]);  add_91 = None
        addmm_44: "f32[16, 2304]" = torch.ops.aten.addmm.default(arg137_1, view_140, arg138_1);  arg137_1 = view_140 = arg138_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_141: "f32[1, 16, 2304]" = torch.ops.aten.reshape.default(addmm_44, [1, 16, 2304]);  addmm_44 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:314 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        split_11 = torch.ops.aten.split.Tensor(view_141, 768, 2);  view_141 = None
        getitem_123: "f32[1, 16, 768]" = split_11[0]
        getitem_124: "f32[1, 16, 768]" = split_11[1]
        getitem_125: "f32[1, 16, 768]" = split_11[2];  split_11 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:320 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        view_144: "f32[1, 16, 12, 64]" = torch.ops.aten.reshape.default(getitem_123, [1, 16, -1, 64]);  getitem_123 = None
        permute_46: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_144, [0, 2, 1, 3]);  view_144 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:316 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        view_142: "f32[1, 16, 12, 64]" = torch.ops.aten.reshape.default(getitem_124, [1, 16, -1, 64]);  getitem_124 = None
        permute_44: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:317 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        view_143: "f32[1, 16, 12, 64]" = torch.ops.aten.reshape.default(getitem_125, [1, 16, -1, 64]);  getitem_125 = None
        permute_45: "f32[1, 12, 16, 64]" = torch.ops.aten.permute.default(view_143, [0, 2, 1, 3]);  view_143 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:83 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_24: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_23: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_11: "f32[1, 1, 16, 16]" = torch.ops.aten.where.self(expand, full_default_24, full_default_23);  expand = full_default_24 = full_default_23 = None
        expand_12: "f32[1, 12, 16, 16]" = torch.ops.aten.expand.default(where_11, [1, 12, 16, 16]);  where_11 = None
        _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_46, permute_44, permute_45, expand_12, False);  permute_46 = permute_44 = permute_45 = expand_12 = None
        getitem_126: "f32[1, 12, 16, 64]" = _scaled_dot_product_efficient_attention_11[0];  _scaled_dot_product_efficient_attention_11 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:93 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_47: "f32[1, 16, 12, 64]" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3]);  getitem_126 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:358 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_145: "f32[1, 16, 768]" = torch.ops.aten.reshape.default(permute_47, [1, 16, -1]);  permute_47 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_146: "f32[16, 768]" = torch.ops.aten.reshape.default(view_145, [-1, 768]);  view_145 = None
        mm_default_2: "f32[16, 768]" = torch.ops.aten.mm.default(view_146, arg140_1);  view_146 = arg140_1 = None
        add_tensor_2: "f32[16, 768]" = torch.ops.aten.add.Tensor(mm_default_2, arg139_1);  mm_default_2 = arg139_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_147: "f32[1, 16, 768]" = torch.ops.aten.reshape.default(add_tensor_2, [1, 16, 768]);  add_tensor_2 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:425 in forward, code: hidden_states = attn_output + residual
        add_92: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(view_147, add_89);  view_147 = add_89 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:449 in forward, code: hidden_states = self.ln_2(hidden_states)
        var_mean_23 = torch.ops.aten.var_mean.correction(add_92, [2], correction = 0, keepdim = True)
        getitem_130: "f32[1, 16, 1]" = var_mean_23[0]
        getitem_131: "f32[1, 16, 1]" = var_mean_23[1];  var_mean_23 = None
        sub_25: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_92, getitem_131);  getitem_131 = None
        add_93: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
        rsqrt_23: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        mul_90: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_23);  sub_25 = rsqrt_23 = None
        mul_91: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_90, arg141_1);  mul_90 = arg141_1 = None
        add_94: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_91, arg142_1);  mul_91 = arg142_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_148: "f32[16, 768]" = torch.ops.aten.reshape.default(add_94, [-1, 768]);  add_94 = None
        mm_default_1: "f32[16, 3072]" = torch.ops.aten.mm.default(view_148, arg144_1);  view_148 = arg144_1 = None
        add_tensor_1: "f32[16, 3072]" = torch.ops.aten.add.Tensor(mm_default_1, arg143_1);  mm_default_1 = arg143_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_149: "f32[1, 16, 3072]" = torch.ops.aten.reshape.default(add_tensor_1, [1, 16, 3072]);  add_tensor_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:48 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_92: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(view_149, 0.5)
        pow_12: "f32[1, 16, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_149, 3.0)
        mul_93: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_95: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(view_149, mul_93);  view_149 = mul_93 = None
        mul_94: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(add_95, 0.7978845608028654);  add_95 = None
        tanh_11: "f32[1, 16, 3072]" = torch.ops.aten.tanh.default(mul_94);  mul_94 = None
        add_96: "f32[1, 16, 3072]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
        mul_95: "f32[1, 16, 3072]" = torch.ops.aten.mul.Tensor(mul_92, add_96);  mul_92 = add_96 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_150: "f32[16, 3072]" = torch.ops.aten.reshape.default(mul_95, [-1, 3072]);  mul_95 = None
        mm_default: "f32[16, 768]" = torch.ops.aten.mm.default(view_150, arg146_1);  view_150 = arg146_1 = None
        add_tensor: "f32[16, 768]" = torch.ops.aten.add.Tensor(mm_default, arg145_1);  mm_default = arg145_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/pytorch_utils.py:123 in forward, code: x = x.view(size_out)
        view_151: "f32[1, 16, 768]" = torch.ops.aten.reshape.default(add_tensor, [1, 16, 768]);  add_tensor = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:452 in forward, code: hidden_states = residual + feed_forward_hidden_states
        add_97: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(add_92, view_151);  add_92 = view_151 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:953 in forward, code: hidden_states = self.ln_f(hidden_states)
        var_mean_24 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
        getitem_132: "f32[1, 16, 1]" = var_mean_24[0]
        getitem_133: "f32[1, 16, 1]" = var_mean_24[1];  var_mean_24 = None
        sub_26: "f32[1, 16, 768]" = torch.ops.aten.sub.Tensor(add_97, getitem_133);  add_97 = getitem_133 = None
        add_98: "f32[1, 16, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-05);  getitem_132 = None
        rsqrt_24: "f32[1, 16, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        mul_96: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_24);  sub_26 = rsqrt_24 = None
        mul_97: "f32[1, 16, 768]" = torch.ops.aten.mul.Tensor(mul_96, arg147_1);  mul_96 = arg147_1 = None
        add_99: "f32[1, 16, 768]" = torch.ops.aten.add.Tensor(mul_97, arg148_1);  mul_97 = arg148_1 = None
        return (add_99,)
        

# ===== inductor generated file: /tmp/cnnbench-transformers-ob_ijj07/repeat_03/a1/torchinductor/tmpuqb_59nb/tg/ctgkujgbvk26wf25wgqfcccy3meikawb6tsgv2vykji5s3geakfs.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_arange_embedding_native_layer_norm_unsqueeze_view_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 6, 'num_reduction': 4, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_addmm_arange_embedding_native_layer_norm_unsqueeze_view_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16
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
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (r0_1 + 768*x0), r0_mask & xmask, other=0.0)
    tmp36 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tl.full([XBLOCK, R0_BLOCK], 50257, tl.int32)
    tmp5 = tmp3 + tmp4
    tmp6 = tmp3 < 0
    tmp7 = tl.where(tmp6, tmp5, tmp3)
    tl.device_assert(((0 <= tmp7) & (tmp7 < 50257)) | ~(xmask), "index out of bounds: 0 <= tmp7 < 50257")
    tmp9 = tl.load(in_ptr2 + (r0_1 + 768*tmp7), r0_mask & xmask, other=0.0)
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
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp12, r0_mask & xmask)
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp39, r0_mask & xmask)
