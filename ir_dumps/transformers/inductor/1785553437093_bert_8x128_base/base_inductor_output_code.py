# ===== inductor generated file: /tmp/cnnbench-bert-fold-weaagmnf/repeat_02/base/attempt_01/torchinductor/tmp5mgaaf9a/m5/cm5rpuin46rzdtknz5aqqo3766n3gtq6mrme23rqyrl6joqahvar.debug/output_code.py =====
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


# kernel path: /tmp/cnnbench-bert-fold-weaagmnf/repeat_02/base/attempt_01/torchinductor/tmp5mgaaf9a/js/cjsxh5oszam3v7uzmmlndzn7wzr6xe7iv2kei52srdd625ijoylt.py
# Topologically Sorted Source Nodes: [inputs_embeds, buffered_token_type_ids, buffered_token_type_ids_expanded, token_type_embeddings, embeddings, position_ids, position_embeddings, embeddings_1, embeddings_2], Original ATen: [aten.embedding, aten.slice, aten.expand, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   buffered_token_type_ids => slice_1
#   buffered_token_type_ids_expanded => expand
#   embeddings => add
#   embeddings_1 => add_1
#   embeddings_2 => add_2, add_3, mul, mul_1, rsqrt, sub, var_mean
#   inputs_embeds => embedding
#   position_embeddings => embedding_2
#   position_ids => slice_2
#   token_type_embeddings => embedding_1
# Graph fragment:
#   %arg0_1 : Tensor "i64[8, 128][128, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg3_1 : Tensor "f32[30522, 768][768, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg1_1 : Tensor "i64[1, 512][512, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg4_1 : Tensor "f32[2, 768][768, 1]cuda:0" = PlaceHolder[target=arg4_1]
#   %arg2_1 : Tensor "i64[1, 512][512, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg5_1 : Tensor "f32[512, 768][768, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %add_1 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0" = PlaceHolder[target=add_1]
#   %getitem_1 : Tensor "f32[8, 128, 1][128, 1, 1024]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf2 : Tensor "f32[8, 128, 1][128, 1, 1024]cuda:0" = PlaceHolder[target=buf2]
#   %arg6_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %arg7_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg7_1]
#   %embedding : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %arg0_1, 0), kwargs = {})
#   %slice_1 : Tensor "i64[1, 128][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 0, 128), kwargs = {})
#   %expand : Tensor "i64[8, 128][0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%slice_1, [8, 128]), kwargs = {})
#   %embedding_1 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg4_1, %expand), kwargs = {})
#   %add : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %slice_2 : Tensor "i64[1, 128][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg2_1, 1, 0, 128), kwargs = {})
#   %embedding_2 : Tensor "f32[1, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg5_1, %slice_2), kwargs = {})
#   %add_1 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %embedding_2), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_1, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %getitem_1), kwargs = {})
#   %add_2 : Tensor "f32[8, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-12), kwargs = {})
#   %rsqrt : Tensor "f32[8, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg6_1), kwargs = {})
#   %add_3 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %arg7_1), kwargs = {})
#   return %add_1,%getitem_1,%buf2,%add_3
triton_per_fused_add_embedding_expand_native_layer_norm_slice_0 = async_compile.triton('triton_per_fused_add_embedding_expand_native_layer_norm_slice_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_expand_native_layer_norm_slice_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_embedding_expand_native_layer_norm_slice_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    x3 = xindex
    r0_2 = r0_index
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr6 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.load(in_ptr7 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.full([XBLOCK, R0_BLOCK], 30522, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 30522)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 30522")
    tmp6 = tl.load(in_ptr1 + (r0_2 + 768*tmp4), r0_mask & xmask, other=0.0)
    tmp8 = tl.full([XBLOCK, R0_BLOCK], 2, tl.int32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp7 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp7)
    tl.device_assert(((0 <= tmp11) & (tmp11 < 2)) | ~(xmask), "index out of bounds: 0 <= tmp11 < 2")
    tmp13 = tl.load(in_ptr3 + (r0_2 + 768*tmp11), r0_mask & xmask, other=0.0)
    tmp14 = tmp6 + tmp13
    tmp16 = tl.full([XBLOCK, R0_BLOCK], 512, tl.int32)
    tmp17 = tmp15 + tmp16
    tmp18 = tmp15 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp15)
    tl.device_assert(((0 <= tmp19) & (tmp19 < 512)) | ~(xmask), "index out of bounds: 0 <= tmp19 < 512")
    tmp21 = tl.load(in_ptr5 + (r0_2 + 768*tmp19), r0_mask & xmask, other=0.0)
    tmp22 = tmp14 + tmp21
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, R0_BLOCK])
    tmp25 = tl.where(r0_mask & xmask, tmp23, 0)
    tmp26 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
    tmp28 = tl.where(r0_mask & xmask, tmp26, 0)
    tmp29 = tl.sum(tmp28, 1)[:, None].to(tl.float32)
    tmp30 = tl.full([XBLOCK, 1], 768, tl.int32)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = (tmp29 / tmp31)
    tmp33 = tmp23 - tmp32
    tmp34 = tmp33 * tmp33
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK, R0_BLOCK])
    tmp37 = tl.where(r0_mask & xmask, tmp35, 0)
    tmp38 = tl.sum(tmp37, 1)[:, None].to(tl.float32)
    tmp39 = tmp22 - tmp32
    tmp40 = 768.0
    tmp41 = (tmp38 / tmp40)
    tmp42 = 1e-12
    tmp43 = tmp41 + tmp42
    tmp44 = libdevice.rsqrt(tmp43)
    tmp45 = tmp39 * tmp44
    tmp47 = tmp45 * tmp46
    tmp49 = tmp47 + tmp48
    tl.store(in_out_ptr0 + (r0_2 + 768*x3), tmp49, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-bert-fold-weaagmnf/repeat_02/base/attempt_01/torchinductor/tmp5mgaaf9a/bc/cbcg7yrdae5gcculs4k4kh75h33f4hiihrpuew2o7doyzs2jpeqy.py
# Topologically Sorted Source Nodes: [linear, view, query_layer, linear_1, view_1, key_layer, linear_2, view_2, value_layer, tensor, getitem_2, expand_1, expanded_mask, inverted_mask, to_1, extended_attention_mask, attn_output, linear_6, view_3, query_layer_1, linear_7, view_4, key_layer_1, linear_8, view_5, value_layer_1, attn_output_3, linear_12, view_6, query_layer_2, linear_13, view_7, key_layer_2, linear_14, view_8, value_layer_2, attn_output_6, linear_18, view_9, query_layer_3, linear_19, view_10, key_layer_3, linear_20, view_11, value_layer_3, attn_output_9, linear_24, view_12, query_layer_4, linear_25, view_13, key_layer_4, linear_26, view_14, value_layer_4, attn_output_12], Original ATen: [aten.view, aten.transpose, aten.lift_fresh, aten.unsqueeze, aten.expand, aten._to_copy, aten.sub, aten.masked_fill, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   attn_output => _scaled_dot_product_efficient_attention, expand_2
#   attn_output_12 => _scaled_dot_product_efficient_attention_4, expand_6
#   attn_output_3 => _scaled_dot_product_efficient_attention_1, expand_3
#   attn_output_6 => _scaled_dot_product_efficient_attention_2, expand_4
#   attn_output_9 => _scaled_dot_product_efficient_attention_3, expand_5
#   expand_1 => expand_1
#   expanded_mask => convert_element_type
#   extended_attention_mask => full_default_1, where
#   getitem_2 => unsqueeze, unsqueeze_1
#   inverted_mask => sub_1
#   key_layer => permute_3
#   key_layer_1 => permute_13
#   key_layer_2 => permute_23
#   key_layer_3 => permute_33
#   key_layer_4 => permute_43
#   linear => view_1
#   linear_1 => view_4
#   linear_12 => view_33
#   linear_13 => view_36
#   linear_14 => view_39
#   linear_18 => view_49
#   linear_19 => view_52
#   linear_2 => view_7
#   linear_20 => view_55
#   linear_24 => view_65
#   linear_25 => view_68
#   linear_26 => view_71
#   linear_6 => view_17
#   linear_7 => view_20
#   linear_8 => view_23
#   query_layer => permute_1
#   query_layer_1 => permute_11
#   query_layer_2 => permute_21
#   query_layer_3 => permute_31
#   query_layer_4 => permute_41
#   tensor => full_default
#   to_1 => convert_element_type_1
#   value_layer => permute_5
#   value_layer_1 => permute_15
#   value_layer_2 => permute_25
#   value_layer_3 => permute_35
#   value_layer_4 => permute_45
#   view => view_2
#   view_1 => view_5
#   view_10 => view_53
#   view_11 => view_56
#   view_12 => view_66
#   view_13 => view_69
#   view_14 => view_72
#   view_2 => view_8
#   view_3 => view_18
#   view_4 => view_21
#   view_5 => view_24
#   view_6 => view_34
#   view_7 => view_37
#   view_8 => view_40
#   view_9 => view_50
# Graph fragment:
#   %arg8_1 : Tensor "i64[8, 128][128, 1]cuda:0" = PlaceHolder[target=arg8_1]
#   %view_1 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [8, 128, 768]), kwargs = {})
#   %view_2 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [8, -1, 12, 64]), kwargs = {})
#   %permute_1 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_2, [0, 2, 1, 3]), kwargs = {})
#   %view_4 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_1, [8, 128, 768]), kwargs = {})
#   %view_5 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_4, [8, -1, 12, 64]), kwargs = {})
#   %permute_3 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_5, [0, 2, 1, 3]), kwargs = {})
#   %view_7 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_2, [8, 128, 768]), kwargs = {})
#   %view_8 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_7, [8, -1, 12, 64]), kwargs = {})
#   %permute_5 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_8, [0, 2, 1, 3]), kwargs = {})
#   %full_default : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cpu, pin_memory: False})
#   %unsqueeze : Tensor "i64[8, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg8_1, 1), kwargs = {})
#   %unsqueeze_1 : Tensor "i64[8, 1, 1, 128][128, 128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 2), kwargs = {})
#   %expand_1 : Tensor "i64[8, 1, 128, 128][128, 128, 0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_1, [8, 1, 128, 128]), kwargs = {})
#   %convert_element_type : Tensor "f32[8, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%expand_1, torch.float32), kwargs = {})
#   %sub_1 : Tensor "f32[8, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default, %convert_element_type), kwargs = {})
#   %convert_element_type_1 : Tensor "b8[8, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub_1, torch.bool), kwargs = {})
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -3.4028234663852886e+38), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "f32[8, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=12] = call_function[target=torch.ops.aten.where.self](args = (%convert_element_type_1, %full_default_1, %sub_1), kwargs = {})
#   %expand_2 : Tensor "f32[8, 12, 128, 128][16384, 0, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%where, [8, 12, 128, 128]), kwargs = {})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_1, %permute_3, %permute_5, %expand_2, False), kwargs = {})
#   %view_17 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_6, [8, 128, 768]), kwargs = {})
#   %view_18 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_17, [8, -1, 12, 64]), kwargs = {})
#   %permute_11 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_18, [0, 2, 1, 3]), kwargs = {})
#   %view_20 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_7, [8, 128, 768]), kwargs = {})
#   %view_21 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_20, [8, -1, 12, 64]), kwargs = {})
#   %permute_13 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_21, [0, 2, 1, 3]), kwargs = {})
#   %view_23 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_8, [8, 128, 768]), kwargs = {})
#   %view_24 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_23, [8, -1, 12, 64]), kwargs = {})
#   %permute_15 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_24, [0, 2, 1, 3]), kwargs = {})
#   %expand_3 : Tensor "f32[8, 12, 128, 128][16384, 0, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%where, [8, 12, 128, 128]), kwargs = {})
#   %_scaled_dot_product_efficient_attention_1 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_11, %permute_13, %permute_15, %expand_3, False), kwargs = {})
#   %view_33 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_12, [8, 128, 768]), kwargs = {})
#   %view_34 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_33, [8, -1, 12, 64]), kwargs = {})
#   %permute_21 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_34, [0, 2, 1, 3]), kwargs = {})
#   %view_36 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_13, [8, 128, 768]), kwargs = {})
#   %view_37 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_36, [8, -1, 12, 64]), kwargs = {})
#   %permute_23 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_37, [0, 2, 1, 3]), kwargs = {})
#   %view_39 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_14, [8, 128, 768]), kwargs = {})
#   %view_40 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_39, [8, -1, 12, 64]), kwargs = {})
#   %permute_25 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_40, [0, 2, 1, 3]), kwargs = {})
#   %expand_4 : Tensor "f32[8, 12, 128, 128][16384, 0, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%where, [8, 12, 128, 128]), kwargs = {})
#   %_scaled_dot_product_efficient_attention_2 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_21, %permute_23, %permute_25, %expand_4, False), kwargs = {})
#   %view_49 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_18, [8, 128, 768]), kwargs = {})
#   %view_50 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_49, [8, -1, 12, 64]), kwargs = {})
#   %permute_31 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_50, [0, 2, 1, 3]), kwargs = {})
#   %view_52 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_19, [8, 128, 768]), kwargs = {})
#   %view_53 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_52, [8, -1, 12, 64]), kwargs = {})
#   %permute_33 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_53, [0, 2, 1, 3]), kwargs = {})
#   %view_55 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_20, [8, 128, 768]), kwargs = {})
#   %view_56 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_55, [8, -1, 12, 64]), kwargs = {})
#   %permute_35 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_56, [0, 2, 1, 3]), kwargs = {})
#   %expand_5 : Tensor "f32[8, 12, 128, 128][16384, 0, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%where, [8, 12, 128, 128]), kwargs = {})
#   %_scaled_dot_product_efficient_attention_3 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_31, %permute_33, %permute_35, %expand_5, False), kwargs = {})
#   %view_65 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_24, [8, 128, 768]), kwargs = {})
#   %view_66 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_65, [8, -1, 12, 64]), kwargs = {})
#   %permute_41 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_66, [0, 2, 1, 3]), kwargs = {})
#   %view_68 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_25, [8, 128, 768]), kwargs = {})
#   %view_69 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_68, [8, -1, 12, 64]), kwargs = {})
#   %permute_43 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_69, [0, 2, 1, 3]), kwargs = {})
#   %view_71 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_26, [8, 128, 768]), kwargs = {})
#   %view_72 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_71, [8, -1, 12, 64]), kwargs = {})
#   %permute_45 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_72, [0, 2, 1, 3]), kwargs = {})
#   %expand_6 : Tensor "f32[8, 12, 128, 128][16384, 0, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%where, [8, 12, 128, 128]), kwargs = {})
#   %_scaled_dot_product_efficient_attention_4 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_41, %permute_43, %permute_45, %expand_6, False), kwargs = {})
#   return %buf8,%buf29,%buf50,%buf71,%buf92
triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_lift_fresh_masked_fill_sub_transpose_unsqueeze_view_1 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_lift_fresh_masked_fill_sub_transpose_unsqueeze_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_lift_fresh_masked_fill_sub_transpose_unsqueeze_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 5251072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_lift_fresh_masked_fill_sub_transpose_unsqueeze_view_1(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 128)
    x2 = xindex // 16384
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x2), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp2 - tmp1
    tmp4 = (tmp3 != 0)
    tmp5 = -3.4028234663852886e+38
    tmp6 = tl.where(tmp4, tmp5, tmp3)
    tl.store(out_ptr0 + (x3), tmp6, None)
    tl.store(out_ptr1 + (x3), tmp6, None)
    tl.store(out_ptr2 + (x3), tmp6, None)
    tl.store(out_ptr3 + (x3), tmp6, None)
    tl.store(out_ptr4 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-bert-fold-weaagmnf/repeat_02/base/attempt_01/torchinductor/tmp5mgaaf9a/3o/c3o3cagdzec6vdr53idsfidv322vhrsgvavfqz4humgbetninzuc.py
# Topologically Sorted Source Nodes: [hidden_states, add_1, hidden_states_2], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_1 => add_4
#   hidden_states => add_tensor_35, view_11
#   hidden_states_2 => add_5, add_6, mul_2, mul_3, rsqrt_1, sub_2, var_mean_1
# Graph fragment:
#   %mm_default_35 : Tensor "f32[1024, 768][768, 1]cuda:0" = PlaceHolder[target=mm_default_35]
#   %arg16_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg16_1]
#   %add_3 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0" = PlaceHolder[target=add_3]
#   %getitem_7 : Tensor "f32[8, 128, 1][128, 1, 1024]cuda:0" = PlaceHolder[target=getitem_7]
#   %buf16 : Tensor "f32[8, 128, 1][128, 1, 1024]cuda:0" = PlaceHolder[target=buf16]
#   %arg17_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg17_1]
#   %arg18_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg18_1]
#   %add_tensor_35 : Tensor "f32[1024, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_35, %arg16_1), kwargs = {})
#   %view_11 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_35, [8, 128, 768]), kwargs = {})
#   %add_4 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_11, %add_3), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_4, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_4, %getitem_7), kwargs = {})
#   %add_5 : Tensor "f32[8, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-12), kwargs = {})
#   %rsqrt_1 : Tensor "f32[8, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
#   %mul_2 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_1), kwargs = {})
#   %mul_3 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %arg17_1), kwargs = {})
#   %add_6 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %arg18_1), kwargs = {})
#   return %getitem_7,%buf16,%add_6
triton_per_fused_add_addmm_native_layer_norm_view_2 = async_compile.triton('triton_per_fused_add_addmm_native_layer_norm_view_2', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_native_layer_norm_view_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 12592128}}
)
@triton.jit
def triton_per_fused_add_addmm_native_layer_norm_view_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp28 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
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
    tmp24 = 1e-12
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp31, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-bert-fold-weaagmnf/repeat_02/base/attempt_01/torchinductor/tmp5mgaaf9a/r6/cr6dvhidsakzazbjgcr5uj2e5qory6i6pdmdzxcxjo4omejryiae.py
# Topologically Sorted Source Nodes: [hidden_states_3, hidden_states_4], Original ATen: [aten.addmm, aten.view, aten.gelu]
# Source node to ATen node mapping:
#   hidden_states_3 => add_tensor_34, view_13
#   hidden_states_4 => add_7, erf, mul_4, mul_5, mul_6
# Graph fragment:
#   %mm_default_34 : Tensor "f32[1024, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_default_34]
#   %arg20_1 : Tensor "f32[3072][1]cuda:0" = PlaceHolder[target=arg20_1]
#   %add_tensor_34 : Tensor "f32[1024, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_34, %arg20_1), kwargs = {})
#   %view_13 : Tensor "f32[8, 128, 3072][393216, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_34, [8, 128, 3072]), kwargs = {})
#   %mul_4 : Tensor "f32[8, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_13, 0.5), kwargs = {})
#   %mul_5 : Tensor "f32[8, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_13, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[8, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_5,), kwargs = {})
#   %add_7 : Tensor "f32[8, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_6 : Tensor "f32[8, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %add_7), kwargs = {})
#   return %mul_6
triton_poi_fused_addmm_gelu_view_3 = async_compile.triton('triton_poi_fused_addmm_gelu_view_3', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_gelu_view_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 37761024}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_gelu_view_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-bert-fold-weaagmnf/repeat_02/base/attempt_01/torchinductor/tmp5mgaaf9a/sq/csqnegz2evgpsehsmnndtg3xifvb72hammko7tbs3n2yijtya4yw.py
# Topologically Sorted Source Nodes: [tensor, getitem_2, expand_1, expanded_mask, inverted_mask, to_1, extended_attention_mask, linear_60, view_30, query_layer_10, linear_61, view_31, key_layer_10, linear_62, view_32, value_layer_10, attn_output_30, linear_66, view_33, query_layer_11, linear_67, view_34, key_layer_11, linear_68, view_35, value_layer_11, attn_output_33], Original ATen: [aten.lift_fresh, aten.unsqueeze, aten.expand, aten._to_copy, aten.sub, aten.masked_fill, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   attn_output_30 => _scaled_dot_product_efficient_attention_10, expand_12
#   attn_output_33 => _scaled_dot_product_efficient_attention_11, expand_13
#   expand_1 => expand_1
#   expanded_mask => convert_element_type
#   extended_attention_mask => full_default_1, where
#   getitem_2 => unsqueeze, unsqueeze_1
#   inverted_mask => sub_1
#   key_layer_10 => permute_103
#   key_layer_11 => permute_113
#   linear_60 => view_161
#   linear_61 => view_164
#   linear_62 => view_167
#   linear_66 => view_177
#   linear_67 => view_180
#   linear_68 => view_183
#   query_layer_10 => permute_101
#   query_layer_11 => permute_111
#   tensor => full_default
#   to_1 => convert_element_type_1
#   value_layer_10 => permute_105
#   value_layer_11 => permute_115
#   view_30 => view_162
#   view_31 => view_165
#   view_32 => view_168
#   view_33 => view_178
#   view_34 => view_181
#   view_35 => view_184
# Graph fragment:
#   %arg8_1 : Tensor "i64[8, 128][128, 1]cuda:0" = PlaceHolder[target=arg8_1]
#   %full_default : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cpu, pin_memory: False})
#   %unsqueeze : Tensor "i64[8, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg8_1, 1), kwargs = {})
#   %unsqueeze_1 : Tensor "i64[8, 1, 1, 128][128, 128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 2), kwargs = {})
#   %expand_1 : Tensor "i64[8, 1, 128, 128][128, 128, 0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_1, [8, 1, 128, 128]), kwargs = {})
#   %convert_element_type : Tensor "f32[8, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%expand_1, torch.float32), kwargs = {})
#   %sub_1 : Tensor "f32[8, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default, %convert_element_type), kwargs = {})
#   %convert_element_type_1 : Tensor "b8[8, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub_1, torch.bool), kwargs = {})
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -3.4028234663852886e+38), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "f32[8, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=12] = call_function[target=torch.ops.aten.where.self](args = (%convert_element_type_1, %full_default_1, %sub_1), kwargs = {})
#   %view_161 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_60, [8, 128, 768]), kwargs = {})
#   %view_162 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_161, [8, -1, 12, 64]), kwargs = {})
#   %permute_101 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_162, [0, 2, 1, 3]), kwargs = {})
#   %view_164 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_61, [8, 128, 768]), kwargs = {})
#   %view_165 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_164, [8, -1, 12, 64]), kwargs = {})
#   %permute_103 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_165, [0, 2, 1, 3]), kwargs = {})
#   %view_167 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_62, [8, 128, 768]), kwargs = {})
#   %view_168 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_167, [8, -1, 12, 64]), kwargs = {})
#   %permute_105 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_168, [0, 2, 1, 3]), kwargs = {})
#   %expand_12 : Tensor "f32[8, 12, 128, 128][16384, 0, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%where, [8, 12, 128, 128]), kwargs = {})
#   %_scaled_dot_product_efficient_attention_10 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_101, %permute_103, %permute_105, %expand_12, False), kwargs = {})
#   %view_177 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_66, [8, 128, 768]), kwargs = {})
#   %view_178 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_177, [8, -1, 12, 64]), kwargs = {})
#   %permute_111 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_178, [0, 2, 1, 3]), kwargs = {})
#   %view_180 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_67, [8, 128, 768]), kwargs = {})
#   %view_181 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_180, [8, -1, 12, 64]), kwargs = {})
#   %permute_113 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_181, [0, 2, 1, 3]), kwargs = {})
#   %view_183 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_68, [8, 128, 768]), kwargs = {})
#   %view_184 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_183, [8, -1, 12, 64]), kwargs = {})
#   %permute_115 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_184, [0, 2, 1, 3]), kwargs = {})
#   %expand_13 : Tensor "f32[8, 12, 128, 128][16384, 0, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%where, [8, 12, 128, 128]), kwargs = {})
#   %_scaled_dot_product_efficient_attention_11 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_111, %permute_113, %permute_115, %expand_13, False), kwargs = {})
#   return %buf218,%buf239
triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_lift_fresh_masked_fill_sub_transpose_unsqueeze_view_4 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_lift_fresh_masked_fill_sub_transpose_unsqueeze_view_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_lift_fresh_masked_fill_sub_transpose_unsqueeze_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2105344}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_lift_fresh_masked_fill_sub_transpose_unsqueeze_view_4(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 128)
    x2 = xindex // 16384
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x2), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp2 - tmp1
    tmp4 = (tmp3 != 0)
    tmp5 = -3.4028234663852886e+38
    tmp6 = tl.where(tmp4, tmp5, tmp3)
    tl.store(out_ptr0 + (x3), tmp6, None)
    tl.store(out_ptr1 + (x3), tmp6, None)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1 = args
        args.clear()
        assert_size_stride(arg0_1, (8, 128), (128, 1))
        assert_size_stride(arg1_1, (1, 512), (512, 1))
        assert_size_stride(arg2_1, (1, 512), (512, 1))
        assert_size_stride(arg3_1, (30522, 768), (768, 1))
        assert_size_stride(arg4_1, (2, 768), (768, 1))
        assert_size_stride(arg5_1, (512, 768), (768, 1))
        assert_size_stride(arg6_1, (768, ), (1, ))
        assert_size_stride(arg7_1, (768, ), (1, ))
        assert_size_stride(arg8_1, (8, 128), (128, 1))
        assert_size_stride(arg9_1, (768, 768), (768, 1))
        assert_size_stride(arg10_1, (768, ), (1, ))
        assert_size_stride(arg11_1, (768, 768), (768, 1))
        assert_size_stride(arg12_1, (768, ), (1, ))
        assert_size_stride(arg13_1, (768, 768), (768, 1))
        assert_size_stride(arg14_1, (768, ), (1, ))
        assert_size_stride(arg15_1, (768, 768), (768, 1))
        assert_size_stride(arg16_1, (768, ), (1, ))
        assert_size_stride(arg17_1, (768, ), (1, ))
        assert_size_stride(arg18_1, (768, ), (1, ))
        assert_size_stride(arg19_1, (3072, 768), (768, 1))
        assert_size_stride(arg20_1, (3072, ), (1, ))
        assert_size_stride(arg21_1, (768, 3072), (3072, 1))
        assert_size_stride(arg22_1, (768, ), (1, ))
        assert_size_stride(arg23_1, (768, ), (1, ))
        assert_size_stride(arg24_1, (768, ), (1, ))
        assert_size_stride(arg25_1, (768, 768), (768, 1))
        assert_size_stride(arg26_1, (768, ), (1, ))
        assert_size_stride(arg27_1, (768, 768), (768, 1))
        assert_size_stride(arg28_1, (768, ), (1, ))
        assert_size_stride(arg29_1, (768, 768), (768, 1))
        assert_size_stride(arg30_1, (768, ), (1, ))
        assert_size_stride(arg31_1, (768, 768), (768, 1))
        assert_size_stride(arg32_1, (768, ), (1, ))
        assert_size_stride(arg33_1, (768, ), (1, ))
        assert_size_stride(arg34_1, (768, ), (1, ))
        assert_size_stride(arg35_1, (3072, 768), (768, 1))
        assert_size_stride(arg36_1, (3072, ), (1, ))
        assert_size_stride(arg37_1, (768, 3072), (3072, 1))
        assert_size_stride(arg38_1, (768, ), (1, ))
        assert_size_stride(arg39_1, (768, ), (1, ))
        assert_size_stride(arg40_1, (768, ), (1, ))
        assert_size_stride(arg41_1, (768, 768), (768, 1))
        assert_size_stride(arg42_1, (768, ), (1, ))
        assert_size_stride(arg43_1, (768, 768), (768, 1))
        assert_size_stride(arg44_1, (768, ), (1, ))
        assert_size_stride(arg45_1, (768, 768), (768, 1))
        assert_size_stride(arg46_1, (768, ), (1, ))
        assert_size_stride(arg47_1, (768, 768), (768, 1))
        assert_size_stride(arg48_1, (768, ), (1, ))
        assert_size_stride(arg49_1, (768, ), (1, ))
        assert_size_stride(arg50_1, (768, ), (1, ))
        assert_size_stride(arg51_1, (3072, 768), (768, 1))
        assert_size_stride(arg52_1, (3072, ), (1, ))
        assert_size_stride(arg53_1, (768, 3072), (3072, 1))
        assert_size_stride(arg54_1, (768, ), (1, ))
        assert_size_stride(arg55_1, (768, ), (1, ))
        assert_size_stride(arg56_1, (768, ), (1, ))
        assert_size_stride(arg57_1, (768, 768), (768, 1))
        assert_size_stride(arg58_1, (768, ), (1, ))
        assert_size_stride(arg59_1, (768, 768), (768, 1))
        assert_size_stride(arg60_1, (768, ), (1, ))
        assert_size_stride(arg61_1, (768, 768), (768, 1))
        assert_size_stride(arg62_1, (768, ), (1, ))
        assert_size_stride(arg63_1, (768, 768), (768, 1))
        assert_size_stride(arg64_1, (768, ), (1, ))
        assert_size_stride(arg65_1, (768, ), (1, ))
        assert_size_stride(arg66_1, (768, ), (1, ))
        assert_size_stride(arg67_1, (3072, 768), (768, 1))
        assert_size_stride(arg68_1, (3072, ), (1, ))
        assert_size_stride(arg69_1, (768, 3072), (3072, 1))
        assert_size_stride(arg70_1, (768, ), (1, ))
        assert_size_stride(arg71_1, (768, ), (1, ))
        assert_size_stride(arg72_1, (768, ), (1, ))
        assert_size_stride(arg73_1, (768, 768), (768, 1))
        assert_size_stride(arg74_1, (768, ), (1, ))
        assert_size_stride(arg75_1, (768, 768), (768, 1))
        assert_size_stride(arg76_1, (768, ), (1, ))
        assert_size_stride(arg77_1, (768, 768), (768, 1))
        assert_size_stride(arg78_1, (768, ), (1, ))
        assert_size_stride(arg79_1, (768, 768), (768, 1))
        assert_size_stride(arg80_1, (768, ), (1, ))
        assert_size_stride(arg81_1, (768, ), (1, ))
        assert_size_stride(arg82_1, (768, ), (1, ))
        assert_size_stride(arg83_1, (3072, 768), (768, 1))
        assert_size_stride(arg84_1, (3072, ), (1, ))
        assert_size_stride(arg85_1, (768, 3072), (3072, 1))
        assert_size_stride(arg86_1, (768, ), (1, ))
        assert_size_stride(arg87_1, (768, ), (1, ))
        assert_size_stride(arg88_1, (768, ), (1, ))
        assert_size_stride(arg89_1, (768, 768), (768, 1))
        assert_size_stride(arg90_1, (768, ), (1, ))
        assert_size_stride(arg91_1, (768, 768), (768, 1))
        assert_size_stride(arg92_1, (768, ), (1, ))
        assert_size_stride(arg93_1, (768, 768), (768, 1))
        assert_size_stride(arg94_1, (768, ), (1, ))
        assert_size_stride(arg95_1, (768, 768), (768, 1))
        assert_size_stride(arg96_1, (768, ), (1, ))
        assert_size_stride(arg97_1, (768, ), (1, ))
        assert_size_stride(arg98_1, (768, ), (1, ))
        assert_size_stride(arg99_1, (3072, 768), (768, 1))
        assert_size_stride(arg100_1, (3072, ), (1, ))
        assert_size_stride(arg101_1, (768, 3072), (3072, 1))
        assert_size_stride(arg102_1, (768, ), (1, ))
        assert_size_stride(arg103_1, (768, ), (1, ))
        assert_size_stride(arg104_1, (768, ), (1, ))
        assert_size_stride(arg105_1, (768, 768), (768, 1))
        assert_size_stride(arg106_1, (768, ), (1, ))
        assert_size_stride(arg107_1, (768, 768), (768, 1))
        assert_size_stride(arg108_1, (768, ), (1, ))
        assert_size_stride(arg109_1, (768, 768), (768, 1))
        assert_size_stride(arg110_1, (768, ), (1, ))
        assert_size_stride(arg111_1, (768, 768), (768, 1))
        assert_size_stride(arg112_1, (768, ), (1, ))
        assert_size_stride(arg113_1, (768, ), (1, ))
        assert_size_stride(arg114_1, (768, ), (1, ))
        assert_size_stride(arg115_1, (3072, 768), (768, 1))
        assert_size_stride(arg116_1, (3072, ), (1, ))
        assert_size_stride(arg117_1, (768, 3072), (3072, 1))
        assert_size_stride(arg118_1, (768, ), (1, ))
        assert_size_stride(arg119_1, (768, ), (1, ))
        assert_size_stride(arg120_1, (768, ), (1, ))
        assert_size_stride(arg121_1, (768, 768), (768, 1))
        assert_size_stride(arg122_1, (768, ), (1, ))
        assert_size_stride(arg123_1, (768, 768), (768, 1))
        assert_size_stride(arg124_1, (768, ), (1, ))
        assert_size_stride(arg125_1, (768, 768), (768, 1))
        assert_size_stride(arg126_1, (768, ), (1, ))
        assert_size_stride(arg127_1, (768, 768), (768, 1))
        assert_size_stride(arg128_1, (768, ), (1, ))
        assert_size_stride(arg129_1, (768, ), (1, ))
        assert_size_stride(arg130_1, (768, ), (1, ))
        assert_size_stride(arg131_1, (3072, 768), (768, 1))
        assert_size_stride(arg132_1, (3072, ), (1, ))
        assert_size_stride(arg133_1, (768, 3072), (3072, 1))
        assert_size_stride(arg134_1, (768, ), (1, ))
        assert_size_stride(arg135_1, (768, ), (1, ))
        assert_size_stride(arg136_1, (768, ), (1, ))
        assert_size_stride(arg137_1, (768, 768), (768, 1))
        assert_size_stride(arg138_1, (768, ), (1, ))
        assert_size_stride(arg139_1, (768, 768), (768, 1))
        assert_size_stride(arg140_1, (768, ), (1, ))
        assert_size_stride(arg141_1, (768, 768), (768, 1))
        assert_size_stride(arg142_1, (768, ), (1, ))
        assert_size_stride(arg143_1, (768, 768), (768, 1))
        assert_size_stride(arg144_1, (768, ), (1, ))
        assert_size_stride(arg145_1, (768, ), (1, ))
        assert_size_stride(arg146_1, (768, ), (1, ))
        assert_size_stride(arg147_1, (3072, 768), (768, 1))
        assert_size_stride(arg148_1, (3072, ), (1, ))
        assert_size_stride(arg149_1, (768, 3072), (3072, 1))
        assert_size_stride(arg150_1, (768, ), (1, ))
        assert_size_stride(arg151_1, (768, ), (1, ))
        assert_size_stride(arg152_1, (768, ), (1, ))
        assert_size_stride(arg153_1, (768, 768), (768, 1))
        assert_size_stride(arg154_1, (768, ), (1, ))
        assert_size_stride(arg155_1, (768, 768), (768, 1))
        assert_size_stride(arg156_1, (768, ), (1, ))
        assert_size_stride(arg157_1, (768, 768), (768, 1))
        assert_size_stride(arg158_1, (768, ), (1, ))
        assert_size_stride(arg159_1, (768, 768), (768, 1))
        assert_size_stride(arg160_1, (768, ), (1, ))
        assert_size_stride(arg161_1, (768, ), (1, ))
        assert_size_stride(arg162_1, (768, ), (1, ))
        assert_size_stride(arg163_1, (3072, 768), (768, 1))
        assert_size_stride(arg164_1, (3072, ), (1, ))
        assert_size_stride(arg165_1, (768, 3072), (3072, 1))
        assert_size_stride(arg166_1, (768, ), (1, ))
        assert_size_stride(arg167_1, (768, ), (1, ))
        assert_size_stride(arg168_1, (768, ), (1, ))
        assert_size_stride(arg169_1, (768, 768), (768, 1))
        assert_size_stride(arg170_1, (768, ), (1, ))
        assert_size_stride(arg171_1, (768, 768), (768, 1))
        assert_size_stride(arg172_1, (768, ), (1, ))
        assert_size_stride(arg173_1, (768, 768), (768, 1))
        assert_size_stride(arg174_1, (768, ), (1, ))
        assert_size_stride(arg175_1, (768, 768), (768, 1))
        assert_size_stride(arg176_1, (768, ), (1, ))
        assert_size_stride(arg177_1, (768, ), (1, ))
        assert_size_stride(arg178_1, (768, ), (1, ))
        assert_size_stride(arg179_1, (3072, 768), (768, 1))
        assert_size_stride(arg180_1, (3072, ), (1, ))
        assert_size_stride(arg181_1, (768, 3072), (3072, 1))
        assert_size_stride(arg182_1, (768, ), (1, ))
        assert_size_stride(arg183_1, (768, ), (1, ))
        assert_size_stride(arg184_1, (768, ), (1, ))
        assert_size_stride(arg185_1, (768, 768), (768, 1))
        assert_size_stride(arg186_1, (768, ), (1, ))
        assert_size_stride(arg187_1, (768, 768), (768, 1))
        assert_size_stride(arg188_1, (768, ), (1, ))
        assert_size_stride(arg189_1, (768, 768), (768, 1))
        assert_size_stride(arg190_1, (768, ), (1, ))
        assert_size_stride(arg191_1, (768, 768), (768, 1))
        assert_size_stride(arg192_1, (768, ), (1, ))
        assert_size_stride(arg193_1, (768, ), (1, ))
        assert_size_stride(arg194_1, (768, ), (1, ))
        assert_size_stride(arg195_1, (3072, 768), (768, 1))
        assert_size_stride(arg196_1, (3072, ), (1, ))
        assert_size_stride(arg197_1, (768, 3072), (3072, 1))
        assert_size_stride(arg198_1, (768, ), (1, ))
        assert_size_stride(arg199_1, (768, ), (1, ))
        assert_size_stride(arg200_1, (768, ), (1, ))
        assert_size_stride(arg201_1, (768, 768), (768, 1))
        assert_size_stride(arg202_1, (768, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((8, 128, 768), (98304, 768, 1), torch.float32)
            buf4 = buf0; del buf0  # reuse
            # Topologically Sorted Source Nodes: [inputs_embeds, buffered_token_type_ids, buffered_token_type_ids_expanded, token_type_embeddings, embeddings, position_ids, position_embeddings, embeddings_1, embeddings_2], Original ATen: [aten.embedding, aten.slice, aten.expand, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_embedding_expand_native_layer_norm_slice_0.run(buf4, arg0_1, arg3_1, arg1_1, arg4_1, arg2_1, arg5_1, arg6_1, arg7_1, 1024, 768, stream=stream0)
            del arg0_1
            del arg1_1
            del arg2_1
            del arg3_1
            del arg4_1
            del arg5_1
            del arg6_1
            del arg7_1
            buf5 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg10_1, reinterpret_tensor(buf4, (1024, 768), (768, 1), 0), reinterpret_tensor(arg9_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf5)
            del arg10_1
            del arg9_1
            buf6 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg12_1, reinterpret_tensor(buf4, (1024, 768), (768, 1), 0), reinterpret_tensor(arg11_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf6)
            del arg11_1
            del arg12_1
            buf7 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg14_1, reinterpret_tensor(buf4, (1024, 768), (768, 1), 0), reinterpret_tensor(arg13_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf7)
            del arg13_1
            del arg14_1
            buf8 = empty_strided_cuda((8, 1, 128, 128), (16384, 0, 128, 1), torch.float32)
            buf29 = empty_strided_cuda((8, 1, 128, 128), (16384, 0, 128, 1), torch.float32)
            buf50 = empty_strided_cuda((8, 1, 128, 128), (16384, 0, 128, 1), torch.float32)
            buf71 = empty_strided_cuda((8, 1, 128, 128), (16384, 0, 128, 1), torch.float32)
            buf92 = empty_strided_cuda((8, 1, 128, 128), (16384, 0, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear, view, query_layer, linear_1, view_1, key_layer, linear_2, view_2, value_layer, tensor, getitem_2, expand_1, expanded_mask, inverted_mask, to_1, extended_attention_mask, attn_output, linear_6, view_3, query_layer_1, linear_7, view_4, key_layer_1, linear_8, view_5, value_layer_1, attn_output_3, linear_12, view_6, query_layer_2, linear_13, view_7, key_layer_2, linear_14, view_8, value_layer_2, attn_output_6, linear_18, view_9, query_layer_3, linear_19, view_10, key_layer_3, linear_20, view_11, value_layer_3, attn_output_9, linear_24, view_12, query_layer_4, linear_25, view_13, key_layer_4, linear_26, view_14, value_layer_4, attn_output_12], Original ATen: [aten.view, aten.transpose, aten.lift_fresh, aten.unsqueeze, aten.expand, aten._to_copy, aten.sub, aten.masked_fill, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_lift_fresh_masked_fill_sub_transpose_unsqueeze_view_1.run(arg8_1, buf8, buf29, buf50, buf71, buf92, 131072, stream=stream0)
            # Topologically Sorted Source Nodes: [linear, view, query_layer, linear_1, view_1, key_layer, linear_2, view_2, value_layer, tensor, getitem_2, expand_1, expanded_mask, inverted_mask, to_1, extended_attention_mask, attn_output], Original ATen: [aten.view, aten.transpose, aten.lift_fresh, aten.unsqueeze, aten.expand, aten._to_copy, aten.sub, aten.masked_fill, aten._scaled_dot_product_efficient_attention]
            buf9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf5, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf6, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf7, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf8, (8, 12, 128, 128), (16384, 0, 128, 1), 0), False)
            del buf5
            del buf6
            del buf8
            buf10 = buf9[0]
            assert_size_stride(buf10, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf10, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf9
            buf14 = buf7; del buf7  # reuse
            # Topologically Sorted Source Nodes: [attn_output_1, attn_output_2, hidden_states], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf10, (1024, 768), (768, 1), 0), reinterpret_tensor(arg15_1, (768, 768), (1, 768), 0), out=buf14)
            del arg15_1
            del buf10
            buf18 = reinterpret_tensor(buf14, (8, 128, 768), (98304, 768, 1), 0); del buf14  # reuse
            # Topologically Sorted Source Nodes: [hidden_states, add_1, hidden_states_2], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf18, arg16_1, buf4, arg17_1, arg18_1, 1024, 768, stream=stream0)
            del arg16_1
            del arg17_1
            del arg18_1
            del buf4
            buf19 = empty_strided_cuda((1024, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_3], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf18, (1024, 768), (768, 1), 0), reinterpret_tensor(arg19_1, (768, 3072), (1, 768), 0), out=buf19)
            del arg19_1
            buf20 = reinterpret_tensor(buf19, (8, 128, 3072), (393216, 3072, 1), 0); del buf19  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_3, hidden_states_4], Original ATen: [aten.addmm, aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_3.run(buf20, arg20_1, 3145728, stream=stream0)
            del arg20_1
            buf21 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_3, hidden_states_4, hidden_states_5], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            extern_kernels.mm(reinterpret_tensor(buf20, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg21_1, (3072, 768), (1, 3072), 0), out=buf21)
            del arg21_1
            buf25 = reinterpret_tensor(buf21, (8, 128, 768), (98304, 768, 1), 0); del buf21  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_5, add_2, hidden_states_7], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf25, arg22_1, buf18, arg23_1, arg24_1, 1024, 768, stream=stream0)
            del arg22_1
            del arg23_1
            del arg24_1
            buf26 = reinterpret_tensor(buf18, (1024, 768), (768, 1), 0); del buf18  # reuse
            # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg26_1, reinterpret_tensor(buf25, (1024, 768), (768, 1), 0), reinterpret_tensor(arg25_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf26)
            del arg25_1
            del arg26_1
            buf27 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg28_1, reinterpret_tensor(buf25, (1024, 768), (768, 1), 0), reinterpret_tensor(arg27_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf27)
            del arg27_1
            del arg28_1
            buf28 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg30_1, reinterpret_tensor(buf25, (1024, 768), (768, 1), 0), reinterpret_tensor(arg29_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf28)
            del arg29_1
            del arg30_1
            # Topologically Sorted Source Nodes: [tensor, getitem_2, expand_1, expanded_mask, inverted_mask, to_1, extended_attention_mask, linear_6, view_3, query_layer_1, linear_7, view_4, key_layer_1, linear_8, view_5, value_layer_1, attn_output_3], Original ATen: [aten.lift_fresh, aten.unsqueeze, aten.expand, aten._to_copy, aten.sub, aten.masked_fill, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
            buf30 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf26, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf27, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf28, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf29, (8, 12, 128, 128), (16384, 0, 128, 1), 0), False)
            del buf26
            del buf27
            buf31 = buf30[0]
            assert_size_stride(buf31, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf31, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf30
            buf35 = buf28; del buf28  # reuse
            # Topologically Sorted Source Nodes: [attn_output_4, attn_output_5, hidden_states_8], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf31, (1024, 768), (768, 1), 0), reinterpret_tensor(arg31_1, (768, 768), (1, 768), 0), out=buf35)
            del arg31_1
            del buf31
            buf39 = reinterpret_tensor(buf35, (8, 128, 768), (98304, 768, 1), 0); del buf35  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_8, add_3, hidden_states_10], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf39, arg32_1, buf25, arg33_1, arg34_1, 1024, 768, stream=stream0)
            del arg32_1
            del arg33_1
            del arg34_1
            del buf25
            buf40 = reinterpret_tensor(buf20, (1024, 3072), (3072, 1), 0); del buf20  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_11], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf39, (1024, 768), (768, 1), 0), reinterpret_tensor(arg35_1, (768, 3072), (1, 768), 0), out=buf40)
            del arg35_1
            buf41 = reinterpret_tensor(buf40, (8, 128, 3072), (393216, 3072, 1), 0); del buf40  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_11, hidden_states_12], Original ATen: [aten.addmm, aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_3.run(buf41, arg36_1, 3145728, stream=stream0)
            del arg36_1
            buf42 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_11, hidden_states_12, hidden_states_13], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            extern_kernels.mm(reinterpret_tensor(buf41, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg37_1, (3072, 768), (1, 3072), 0), out=buf42)
            del arg37_1
            buf46 = reinterpret_tensor(buf42, (8, 128, 768), (98304, 768, 1), 0); del buf42  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_13, add_4, hidden_states_15], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf46, arg38_1, buf39, arg39_1, arg40_1, 1024, 768, stream=stream0)
            del arg38_1
            del arg39_1
            del arg40_1
            buf47 = reinterpret_tensor(buf39, (1024, 768), (768, 1), 0); del buf39  # reuse
            # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg42_1, reinterpret_tensor(buf46, (1024, 768), (768, 1), 0), reinterpret_tensor(arg41_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf47)
            del arg41_1
            del arg42_1
            buf48 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg44_1, reinterpret_tensor(buf46, (1024, 768), (768, 1), 0), reinterpret_tensor(arg43_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf48)
            del arg43_1
            del arg44_1
            buf49 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg46_1, reinterpret_tensor(buf46, (1024, 768), (768, 1), 0), reinterpret_tensor(arg45_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf49)
            del arg45_1
            del arg46_1
            # Topologically Sorted Source Nodes: [tensor, getitem_2, expand_1, expanded_mask, inverted_mask, to_1, extended_attention_mask, linear_12, view_6, query_layer_2, linear_13, view_7, key_layer_2, linear_14, view_8, value_layer_2, attn_output_6], Original ATen: [aten.lift_fresh, aten.unsqueeze, aten.expand, aten._to_copy, aten.sub, aten.masked_fill, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
            buf51 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf47, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf48, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf49, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf50, (8, 12, 128, 128), (16384, 0, 128, 1), 0), False)
            del buf47
            del buf48
            buf52 = buf51[0]
            assert_size_stride(buf52, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf52, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf51
            buf56 = buf49; del buf49  # reuse
            # Topologically Sorted Source Nodes: [attn_output_7, attn_output_8, hidden_states_16], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf52, (1024, 768), (768, 1), 0), reinterpret_tensor(arg47_1, (768, 768), (1, 768), 0), out=buf56)
            del arg47_1
            del buf52
            buf60 = reinterpret_tensor(buf56, (8, 128, 768), (98304, 768, 1), 0); del buf56  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_16, add_5, hidden_states_18], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf60, arg48_1, buf46, arg49_1, arg50_1, 1024, 768, stream=stream0)
            del arg48_1
            del arg49_1
            del arg50_1
            del buf46
            buf61 = reinterpret_tensor(buf41, (1024, 3072), (3072, 1), 0); del buf41  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_19], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf60, (1024, 768), (768, 1), 0), reinterpret_tensor(arg51_1, (768, 3072), (1, 768), 0), out=buf61)
            del arg51_1
            buf62 = reinterpret_tensor(buf61, (8, 128, 3072), (393216, 3072, 1), 0); del buf61  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_19, hidden_states_20], Original ATen: [aten.addmm, aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_3.run(buf62, arg52_1, 3145728, stream=stream0)
            del arg52_1
            buf63 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_19, hidden_states_20, hidden_states_21], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            extern_kernels.mm(reinterpret_tensor(buf62, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg53_1, (3072, 768), (1, 3072), 0), out=buf63)
            del arg53_1
            buf67 = reinterpret_tensor(buf63, (8, 128, 768), (98304, 768, 1), 0); del buf63  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_21, add_6, hidden_states_23], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf67, arg54_1, buf60, arg55_1, arg56_1, 1024, 768, stream=stream0)
            del arg54_1
            del arg55_1
            del arg56_1
            buf68 = reinterpret_tensor(buf60, (1024, 768), (768, 1), 0); del buf60  # reuse
            # Topologically Sorted Source Nodes: [linear_18], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg58_1, reinterpret_tensor(buf67, (1024, 768), (768, 1), 0), reinterpret_tensor(arg57_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf68)
            del arg57_1
            del arg58_1
            buf69 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg60_1, reinterpret_tensor(buf67, (1024, 768), (768, 1), 0), reinterpret_tensor(arg59_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf69)
            del arg59_1
            del arg60_1
            buf70 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg62_1, reinterpret_tensor(buf67, (1024, 768), (768, 1), 0), reinterpret_tensor(arg61_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf70)
            del arg61_1
            del arg62_1
            # Topologically Sorted Source Nodes: [tensor, getitem_2, expand_1, expanded_mask, inverted_mask, to_1, extended_attention_mask, linear_18, view_9, query_layer_3, linear_19, view_10, key_layer_3, linear_20, view_11, value_layer_3, attn_output_9], Original ATen: [aten.lift_fresh, aten.unsqueeze, aten.expand, aten._to_copy, aten.sub, aten.masked_fill, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
            buf72 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf68, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf69, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf70, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf71, (8, 12, 128, 128), (16384, 0, 128, 1), 0), False)
            del buf68
            del buf69
            buf73 = buf72[0]
            assert_size_stride(buf73, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf73, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf72
            buf77 = buf70; del buf70  # reuse
            # Topologically Sorted Source Nodes: [attn_output_10, attn_output_11, hidden_states_24], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf73, (1024, 768), (768, 1), 0), reinterpret_tensor(arg63_1, (768, 768), (1, 768), 0), out=buf77)
            del arg63_1
            del buf73
            buf81 = reinterpret_tensor(buf77, (8, 128, 768), (98304, 768, 1), 0); del buf77  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_24, add_7, hidden_states_26], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf81, arg64_1, buf67, arg65_1, arg66_1, 1024, 768, stream=stream0)
            del arg64_1
            del arg65_1
            del arg66_1
            del buf67
            buf82 = reinterpret_tensor(buf62, (1024, 3072), (3072, 1), 0); del buf62  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_27], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf81, (1024, 768), (768, 1), 0), reinterpret_tensor(arg67_1, (768, 3072), (1, 768), 0), out=buf82)
            del arg67_1
            buf83 = reinterpret_tensor(buf82, (8, 128, 3072), (393216, 3072, 1), 0); del buf82  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_27, hidden_states_28], Original ATen: [aten.addmm, aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_3.run(buf83, arg68_1, 3145728, stream=stream0)
            del arg68_1
            buf84 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_27, hidden_states_28, hidden_states_29], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            extern_kernels.mm(reinterpret_tensor(buf83, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg69_1, (3072, 768), (1, 3072), 0), out=buf84)
            del arg69_1
            buf88 = reinterpret_tensor(buf84, (8, 128, 768), (98304, 768, 1), 0); del buf84  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_29, add_8, hidden_states_31], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf88, arg70_1, buf81, arg71_1, arg72_1, 1024, 768, stream=stream0)
            del arg70_1
            del arg71_1
            del arg72_1
            buf89 = reinterpret_tensor(buf81, (1024, 768), (768, 1), 0); del buf81  # reuse
            # Topologically Sorted Source Nodes: [linear_24], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg74_1, reinterpret_tensor(buf88, (1024, 768), (768, 1), 0), reinterpret_tensor(arg73_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf89)
            del arg73_1
            del arg74_1
            buf90 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_25], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg76_1, reinterpret_tensor(buf88, (1024, 768), (768, 1), 0), reinterpret_tensor(arg75_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf90)
            del arg75_1
            del arg76_1
            buf91 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg78_1, reinterpret_tensor(buf88, (1024, 768), (768, 1), 0), reinterpret_tensor(arg77_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf91)
            del arg77_1
            del arg78_1
            # Topologically Sorted Source Nodes: [tensor, getitem_2, expand_1, expanded_mask, inverted_mask, to_1, extended_attention_mask, linear_24, view_12, query_layer_4, linear_25, view_13, key_layer_4, linear_26, view_14, value_layer_4, attn_output_12], Original ATen: [aten.lift_fresh, aten.unsqueeze, aten.expand, aten._to_copy, aten.sub, aten.masked_fill, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
            buf93 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf89, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf90, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf91, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf92, (8, 12, 128, 128), (16384, 0, 128, 1), 0), False)
            del buf89
            del buf90
            buf94 = buf93[0]
            assert_size_stride(buf94, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf94, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf93
            buf98 = buf91; del buf91  # reuse
            # Topologically Sorted Source Nodes: [attn_output_13, attn_output_14, hidden_states_32], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf94, (1024, 768), (768, 1), 0), reinterpret_tensor(arg79_1, (768, 768), (1, 768), 0), out=buf98)
            del arg79_1
            del buf94
            buf102 = reinterpret_tensor(buf98, (8, 128, 768), (98304, 768, 1), 0); del buf98  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_32, add_9, hidden_states_34], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf102, arg80_1, buf88, arg81_1, arg82_1, 1024, 768, stream=stream0)
            del arg80_1
            del arg81_1
            del arg82_1
            del buf88
            buf103 = reinterpret_tensor(buf83, (1024, 3072), (3072, 1), 0); del buf83  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_35], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf102, (1024, 768), (768, 1), 0), reinterpret_tensor(arg83_1, (768, 3072), (1, 768), 0), out=buf103)
            del arg83_1
            buf104 = reinterpret_tensor(buf103, (8, 128, 3072), (393216, 3072, 1), 0); del buf103  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_36], Original ATen: [aten.addmm, aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_3.run(buf104, arg84_1, 3145728, stream=stream0)
            del arg84_1
            buf105 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_36, hidden_states_37], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            extern_kernels.mm(reinterpret_tensor(buf104, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg85_1, (3072, 768), (1, 3072), 0), out=buf105)
            del arg85_1
            del buf104
            buf109 = reinterpret_tensor(buf105, (8, 128, 768), (98304, 768, 1), 0); del buf105  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_37, add_10, hidden_states_39], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf109, arg86_1, buf102, arg87_1, arg88_1, 1024, 768, stream=stream0)
            del arg86_1
            del arg87_1
            del arg88_1
            buf110 = reinterpret_tensor(buf102, (1024, 768), (768, 1), 0); del buf102  # reuse
            # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg90_1, reinterpret_tensor(buf109, (1024, 768), (768, 1), 0), reinterpret_tensor(arg89_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf110)
            del arg89_1
            del arg90_1
            buf111 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_31], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg92_1, reinterpret_tensor(buf109, (1024, 768), (768, 1), 0), reinterpret_tensor(arg91_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf111)
            del arg91_1
            del arg92_1
            buf112 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg94_1, reinterpret_tensor(buf109, (1024, 768), (768, 1), 0), reinterpret_tensor(arg93_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf112)
            del arg93_1
            del arg94_1
            buf113 = buf92; del buf92  # reuse
            buf134 = buf71; del buf71  # reuse
            buf155 = buf50; del buf50  # reuse
            buf176 = buf29; del buf29  # reuse
            buf197 = empty_strided_cuda((8, 1, 128, 128), (16384, 0, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [tensor, getitem_2, expand_1, expanded_mask, inverted_mask, to_1, extended_attention_mask, linear_30, view_15, query_layer_5, linear_31, view_16, key_layer_5, linear_32, view_17, value_layer_5, attn_output_15, linear_36, view_18, query_layer_6, linear_37, view_19, key_layer_6, linear_38, view_20, value_layer_6, attn_output_18, linear_42, view_21, query_layer_7, linear_43, view_22, key_layer_7, linear_44, view_23, value_layer_7, attn_output_21, linear_48, view_24, query_layer_8, linear_49, view_25, key_layer_8, linear_50, view_26, value_layer_8, attn_output_24, linear_54, view_27, query_layer_9, linear_55, view_28, key_layer_9, linear_56, view_29, value_layer_9, attn_output_27], Original ATen: [aten.lift_fresh, aten.unsqueeze, aten.expand, aten._to_copy, aten.sub, aten.masked_fill, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_lift_fresh_masked_fill_sub_transpose_unsqueeze_view_1.run(arg8_1, buf113, buf134, buf155, buf176, buf197, 131072, stream=stream0)
            # Topologically Sorted Source Nodes: [tensor, getitem_2, expand_1, expanded_mask, inverted_mask, to_1, extended_attention_mask, linear_30, view_15, query_layer_5, linear_31, view_16, key_layer_5, linear_32, view_17, value_layer_5, attn_output_15], Original ATen: [aten.lift_fresh, aten.unsqueeze, aten.expand, aten._to_copy, aten.sub, aten.masked_fill, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
            buf114 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf110, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf111, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf112, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf113, (8, 12, 128, 128), (16384, 0, 128, 1), 0), False)
            del buf110
            del buf111
            del buf113
            buf115 = buf114[0]
            assert_size_stride(buf115, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf115, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf114
            buf119 = buf112; del buf112  # reuse
            # Topologically Sorted Source Nodes: [attn_output_16, attn_output_17, hidden_states_40], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf115, (1024, 768), (768, 1), 0), reinterpret_tensor(arg95_1, (768, 768), (1, 768), 0), out=buf119)
            del arg95_1
            del buf115
            buf123 = reinterpret_tensor(buf119, (8, 128, 768), (98304, 768, 1), 0); del buf119  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_40, add_11, hidden_states_42], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf123, arg96_1, buf109, arg97_1, arg98_1, 1024, 768, stream=stream0)
            del arg96_1
            del arg97_1
            del arg98_1
            del buf109
            buf124 = empty_strided_cuda((1024, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_43], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf123, (1024, 768), (768, 1), 0), reinterpret_tensor(arg99_1, (768, 3072), (1, 768), 0), out=buf124)
            del arg99_1
            buf125 = reinterpret_tensor(buf124, (8, 128, 3072), (393216, 3072, 1), 0); del buf124  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_43, hidden_states_44], Original ATen: [aten.addmm, aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_3.run(buf125, arg100_1, 3145728, stream=stream0)
            del arg100_1
            buf126 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_43, hidden_states_44, hidden_states_45], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            extern_kernels.mm(reinterpret_tensor(buf125, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg101_1, (3072, 768), (1, 3072), 0), out=buf126)
            del arg101_1
            buf130 = reinterpret_tensor(buf126, (8, 128, 768), (98304, 768, 1), 0); del buf126  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_45, add_12, hidden_states_47], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf130, arg102_1, buf123, arg103_1, arg104_1, 1024, 768, stream=stream0)
            del arg102_1
            del arg103_1
            del arg104_1
            buf131 = reinterpret_tensor(buf123, (1024, 768), (768, 1), 0); del buf123  # reuse
            # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg106_1, reinterpret_tensor(buf130, (1024, 768), (768, 1), 0), reinterpret_tensor(arg105_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf131)
            del arg105_1
            del arg106_1
            buf132 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_37], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg108_1, reinterpret_tensor(buf130, (1024, 768), (768, 1), 0), reinterpret_tensor(arg107_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf132)
            del arg107_1
            del arg108_1
            buf133 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_38], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg110_1, reinterpret_tensor(buf130, (1024, 768), (768, 1), 0), reinterpret_tensor(arg109_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf133)
            del arg109_1
            del arg110_1
            # Topologically Sorted Source Nodes: [tensor, getitem_2, expand_1, expanded_mask, inverted_mask, to_1, extended_attention_mask, linear_36, view_18, query_layer_6, linear_37, view_19, key_layer_6, linear_38, view_20, value_layer_6, attn_output_18], Original ATen: [aten.lift_fresh, aten.unsqueeze, aten.expand, aten._to_copy, aten.sub, aten.masked_fill, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
            buf135 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf131, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf132, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf133, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf134, (8, 12, 128, 128), (16384, 0, 128, 1), 0), False)
            del buf131
            del buf132
            del buf134
            buf136 = buf135[0]
            assert_size_stride(buf136, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf136, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf135
            buf140 = buf133; del buf133  # reuse
            # Topologically Sorted Source Nodes: [attn_output_19, attn_output_20, hidden_states_48], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf136, (1024, 768), (768, 1), 0), reinterpret_tensor(arg111_1, (768, 768), (1, 768), 0), out=buf140)
            del arg111_1
            del buf136
            buf144 = reinterpret_tensor(buf140, (8, 128, 768), (98304, 768, 1), 0); del buf140  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_48, add_13, hidden_states_50], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf144, arg112_1, buf130, arg113_1, arg114_1, 1024, 768, stream=stream0)
            del arg112_1
            del arg113_1
            del arg114_1
            del buf130
            buf145 = reinterpret_tensor(buf125, (1024, 3072), (3072, 1), 0); del buf125  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_51], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf144, (1024, 768), (768, 1), 0), reinterpret_tensor(arg115_1, (768, 3072), (1, 768), 0), out=buf145)
            del arg115_1
            buf146 = reinterpret_tensor(buf145, (8, 128, 3072), (393216, 3072, 1), 0); del buf145  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_51, hidden_states_52], Original ATen: [aten.addmm, aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_3.run(buf146, arg116_1, 3145728, stream=stream0)
            del arg116_1
            buf147 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_51, hidden_states_52, hidden_states_53], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            extern_kernels.mm(reinterpret_tensor(buf146, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg117_1, (3072, 768), (1, 3072), 0), out=buf147)
            del arg117_1
            buf151 = reinterpret_tensor(buf147, (8, 128, 768), (98304, 768, 1), 0); del buf147  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_53, add_14, hidden_states_55], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf151, arg118_1, buf144, arg119_1, arg120_1, 1024, 768, stream=stream0)
            del arg118_1
            del arg119_1
            del arg120_1
            buf152 = reinterpret_tensor(buf144, (1024, 768), (768, 1), 0); del buf144  # reuse
            # Topologically Sorted Source Nodes: [linear_42], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg122_1, reinterpret_tensor(buf151, (1024, 768), (768, 1), 0), reinterpret_tensor(arg121_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf152)
            del arg121_1
            del arg122_1
            buf153 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_43], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg124_1, reinterpret_tensor(buf151, (1024, 768), (768, 1), 0), reinterpret_tensor(arg123_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf153)
            del arg123_1
            del arg124_1
            buf154 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg126_1, reinterpret_tensor(buf151, (1024, 768), (768, 1), 0), reinterpret_tensor(arg125_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf154)
            del arg125_1
            del arg126_1
            # Topologically Sorted Source Nodes: [tensor, getitem_2, expand_1, expanded_mask, inverted_mask, to_1, extended_attention_mask, linear_42, view_21, query_layer_7, linear_43, view_22, key_layer_7, linear_44, view_23, value_layer_7, attn_output_21], Original ATen: [aten.lift_fresh, aten.unsqueeze, aten.expand, aten._to_copy, aten.sub, aten.masked_fill, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
            buf156 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf152, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf153, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf154, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf155, (8, 12, 128, 128), (16384, 0, 128, 1), 0), False)
            del buf152
            del buf153
            del buf155
            buf157 = buf156[0]
            assert_size_stride(buf157, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf157, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf156
            buf161 = buf154; del buf154  # reuse
            # Topologically Sorted Source Nodes: [attn_output_22, attn_output_23, hidden_states_56], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf157, (1024, 768), (768, 1), 0), reinterpret_tensor(arg127_1, (768, 768), (1, 768), 0), out=buf161)
            del arg127_1
            del buf157
            buf165 = reinterpret_tensor(buf161, (8, 128, 768), (98304, 768, 1), 0); del buf161  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_56, add_15, hidden_states_58], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf165, arg128_1, buf151, arg129_1, arg130_1, 1024, 768, stream=stream0)
            del arg128_1
            del arg129_1
            del arg130_1
            del buf151
            buf166 = reinterpret_tensor(buf146, (1024, 3072), (3072, 1), 0); del buf146  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_59], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf165, (1024, 768), (768, 1), 0), reinterpret_tensor(arg131_1, (768, 3072), (1, 768), 0), out=buf166)
            del arg131_1
            buf167 = reinterpret_tensor(buf166, (8, 128, 3072), (393216, 3072, 1), 0); del buf166  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_59, hidden_states_60], Original ATen: [aten.addmm, aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_3.run(buf167, arg132_1, 3145728, stream=stream0)
            del arg132_1
            buf168 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_59, hidden_states_60, hidden_states_61], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            extern_kernels.mm(reinterpret_tensor(buf167, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg133_1, (3072, 768), (1, 3072), 0), out=buf168)
            del arg133_1
            buf172 = reinterpret_tensor(buf168, (8, 128, 768), (98304, 768, 1), 0); del buf168  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_61, add_16, hidden_states_63], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf172, arg134_1, buf165, arg135_1, arg136_1, 1024, 768, stream=stream0)
            del arg134_1
            del arg135_1
            del arg136_1
            buf173 = reinterpret_tensor(buf165, (1024, 768), (768, 1), 0); del buf165  # reuse
            # Topologically Sorted Source Nodes: [linear_48], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg138_1, reinterpret_tensor(buf172, (1024, 768), (768, 1), 0), reinterpret_tensor(arg137_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf173)
            del arg137_1
            del arg138_1
            buf174 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_49], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg140_1, reinterpret_tensor(buf172, (1024, 768), (768, 1), 0), reinterpret_tensor(arg139_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf174)
            del arg139_1
            del arg140_1
            buf175 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_50], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg142_1, reinterpret_tensor(buf172, (1024, 768), (768, 1), 0), reinterpret_tensor(arg141_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf175)
            del arg141_1
            del arg142_1
            # Topologically Sorted Source Nodes: [tensor, getitem_2, expand_1, expanded_mask, inverted_mask, to_1, extended_attention_mask, linear_48, view_24, query_layer_8, linear_49, view_25, key_layer_8, linear_50, view_26, value_layer_8, attn_output_24], Original ATen: [aten.lift_fresh, aten.unsqueeze, aten.expand, aten._to_copy, aten.sub, aten.masked_fill, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
            buf177 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf173, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf174, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf175, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf176, (8, 12, 128, 128), (16384, 0, 128, 1), 0), False)
            del buf173
            del buf174
            buf178 = buf177[0]
            assert_size_stride(buf178, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf178, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf177
            buf182 = buf175; del buf175  # reuse
            # Topologically Sorted Source Nodes: [attn_output_25, attn_output_26, hidden_states_64], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf178, (1024, 768), (768, 1), 0), reinterpret_tensor(arg143_1, (768, 768), (1, 768), 0), out=buf182)
            del arg143_1
            del buf178
            buf186 = reinterpret_tensor(buf182, (8, 128, 768), (98304, 768, 1), 0); del buf182  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_64, add_17, hidden_states_66], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf186, arg144_1, buf172, arg145_1, arg146_1, 1024, 768, stream=stream0)
            del arg144_1
            del arg145_1
            del arg146_1
            del buf172
            buf187 = reinterpret_tensor(buf167, (1024, 3072), (3072, 1), 0); del buf167  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_67], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf186, (1024, 768), (768, 1), 0), reinterpret_tensor(arg147_1, (768, 3072), (1, 768), 0), out=buf187)
            del arg147_1
            buf188 = reinterpret_tensor(buf187, (8, 128, 3072), (393216, 3072, 1), 0); del buf187  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_67, hidden_states_68], Original ATen: [aten.addmm, aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_3.run(buf188, arg148_1, 3145728, stream=stream0)
            del arg148_1
            buf189 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_67, hidden_states_68, hidden_states_69], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            extern_kernels.mm(reinterpret_tensor(buf188, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg149_1, (3072, 768), (1, 3072), 0), out=buf189)
            del arg149_1
            buf193 = reinterpret_tensor(buf189, (8, 128, 768), (98304, 768, 1), 0); del buf189  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_69, add_18, hidden_states_71], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf193, arg150_1, buf186, arg151_1, arg152_1, 1024, 768, stream=stream0)
            del arg150_1
            del arg151_1
            del arg152_1
            buf194 = reinterpret_tensor(buf186, (1024, 768), (768, 1), 0); del buf186  # reuse
            # Topologically Sorted Source Nodes: [linear_54], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg154_1, reinterpret_tensor(buf193, (1024, 768), (768, 1), 0), reinterpret_tensor(arg153_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf194)
            del arg153_1
            del arg154_1
            buf195 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_55], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg156_1, reinterpret_tensor(buf193, (1024, 768), (768, 1), 0), reinterpret_tensor(arg155_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf195)
            del arg155_1
            del arg156_1
            buf196 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_56], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg158_1, reinterpret_tensor(buf193, (1024, 768), (768, 1), 0), reinterpret_tensor(arg157_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf196)
            del arg157_1
            del arg158_1
            # Topologically Sorted Source Nodes: [tensor, getitem_2, expand_1, expanded_mask, inverted_mask, to_1, extended_attention_mask, linear_54, view_27, query_layer_9, linear_55, view_28, key_layer_9, linear_56, view_29, value_layer_9, attn_output_27], Original ATen: [aten.lift_fresh, aten.unsqueeze, aten.expand, aten._to_copy, aten.sub, aten.masked_fill, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
            buf198 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf194, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf195, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf196, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf197, (8, 12, 128, 128), (16384, 0, 128, 1), 0), False)
            del buf194
            del buf195
            buf199 = buf198[0]
            assert_size_stride(buf199, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf199, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf198
            buf203 = buf196; del buf196  # reuse
            # Topologically Sorted Source Nodes: [attn_output_28, attn_output_29, hidden_states_72], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf199, (1024, 768), (768, 1), 0), reinterpret_tensor(arg159_1, (768, 768), (1, 768), 0), out=buf203)
            del arg159_1
            del buf199
            buf207 = reinterpret_tensor(buf203, (8, 128, 768), (98304, 768, 1), 0); del buf203  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_72, add_19, hidden_states_74], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf207, arg160_1, buf193, arg161_1, arg162_1, 1024, 768, stream=stream0)
            del arg160_1
            del arg161_1
            del arg162_1
            del buf193
            buf208 = reinterpret_tensor(buf188, (1024, 3072), (3072, 1), 0); del buf188  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_75], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf207, (1024, 768), (768, 1), 0), reinterpret_tensor(arg163_1, (768, 3072), (1, 768), 0), out=buf208)
            del arg163_1
            buf209 = reinterpret_tensor(buf208, (8, 128, 3072), (393216, 3072, 1), 0); del buf208  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_75, hidden_states_76], Original ATen: [aten.addmm, aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_3.run(buf209, arg164_1, 3145728, stream=stream0)
            del arg164_1
            buf210 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_75, hidden_states_76, hidden_states_77], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            extern_kernels.mm(reinterpret_tensor(buf209, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg165_1, (3072, 768), (1, 3072), 0), out=buf210)
            del arg165_1
            buf214 = reinterpret_tensor(buf210, (8, 128, 768), (98304, 768, 1), 0); del buf210  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_77, add_20, hidden_states_79], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf214, arg166_1, buf207, arg167_1, arg168_1, 1024, 768, stream=stream0)
            del arg166_1
            del arg167_1
            del arg168_1
            buf215 = reinterpret_tensor(buf207, (1024, 768), (768, 1), 0); del buf207  # reuse
            # Topologically Sorted Source Nodes: [linear_60], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg170_1, reinterpret_tensor(buf214, (1024, 768), (768, 1), 0), reinterpret_tensor(arg169_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf215)
            del arg169_1
            del arg170_1
            buf216 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_61], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg172_1, reinterpret_tensor(buf214, (1024, 768), (768, 1), 0), reinterpret_tensor(arg171_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf216)
            del arg171_1
            del arg172_1
            buf217 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_62], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg174_1, reinterpret_tensor(buf214, (1024, 768), (768, 1), 0), reinterpret_tensor(arg173_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf217)
            del arg173_1
            del arg174_1
            buf218 = buf197; del buf197  # reuse
            buf239 = buf176; del buf176  # reuse
            # Topologically Sorted Source Nodes: [tensor, getitem_2, expand_1, expanded_mask, inverted_mask, to_1, extended_attention_mask, linear_60, view_30, query_layer_10, linear_61, view_31, key_layer_10, linear_62, view_32, value_layer_10, attn_output_30, linear_66, view_33, query_layer_11, linear_67, view_34, key_layer_11, linear_68, view_35, value_layer_11, attn_output_33], Original ATen: [aten.lift_fresh, aten.unsqueeze, aten.expand, aten._to_copy, aten.sub, aten.masked_fill, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_lift_fresh_masked_fill_sub_transpose_unsqueeze_view_4.run(arg8_1, buf218, buf239, 131072, stream=stream0)
            del arg8_1
            # Topologically Sorted Source Nodes: [tensor, getitem_2, expand_1, expanded_mask, inverted_mask, to_1, extended_attention_mask, linear_60, view_30, query_layer_10, linear_61, view_31, key_layer_10, linear_62, view_32, value_layer_10, attn_output_30], Original ATen: [aten.lift_fresh, aten.unsqueeze, aten.expand, aten._to_copy, aten.sub, aten.masked_fill, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
            buf219 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf215, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf216, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf217, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf218, (8, 12, 128, 128), (16384, 0, 128, 1), 0), False)
            del buf215
            del buf216
            del buf218
            buf220 = buf219[0]
            assert_size_stride(buf220, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf220, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf219
            buf224 = buf217; del buf217  # reuse
            # Topologically Sorted Source Nodes: [attn_output_31, attn_output_32, hidden_states_80], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf220, (1024, 768), (768, 1), 0), reinterpret_tensor(arg175_1, (768, 768), (1, 768), 0), out=buf224)
            del arg175_1
            del buf220
            buf228 = reinterpret_tensor(buf224, (8, 128, 768), (98304, 768, 1), 0); del buf224  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_80, add_21, hidden_states_82], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf228, arg176_1, buf214, arg177_1, arg178_1, 1024, 768, stream=stream0)
            del arg176_1
            del arg177_1
            del arg178_1
            del buf214
            buf229 = reinterpret_tensor(buf209, (1024, 3072), (3072, 1), 0); del buf209  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_83], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf228, (1024, 768), (768, 1), 0), reinterpret_tensor(arg179_1, (768, 3072), (1, 768), 0), out=buf229)
            del arg179_1
            buf230 = reinterpret_tensor(buf229, (8, 128, 3072), (393216, 3072, 1), 0); del buf229  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_83, hidden_states_84], Original ATen: [aten.addmm, aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_3.run(buf230, arg180_1, 3145728, stream=stream0)
            del arg180_1
            buf231 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_83, hidden_states_84, hidden_states_85], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            extern_kernels.mm(reinterpret_tensor(buf230, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg181_1, (3072, 768), (1, 3072), 0), out=buf231)
            del arg181_1
            buf235 = reinterpret_tensor(buf231, (8, 128, 768), (98304, 768, 1), 0); del buf231  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_85, add_22, hidden_states_87], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf235, arg182_1, buf228, arg183_1, arg184_1, 1024, 768, stream=stream0)
            del arg182_1
            del arg183_1
            del arg184_1
            buf236 = reinterpret_tensor(buf228, (1024, 768), (768, 1), 0); del buf228  # reuse
            # Topologically Sorted Source Nodes: [linear_66], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg186_1, reinterpret_tensor(buf235, (1024, 768), (768, 1), 0), reinterpret_tensor(arg185_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf236)
            del arg185_1
            del arg186_1
            buf237 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_67], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg188_1, reinterpret_tensor(buf235, (1024, 768), (768, 1), 0), reinterpret_tensor(arg187_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf237)
            del arg187_1
            del arg188_1
            buf238 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_68], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg190_1, reinterpret_tensor(buf235, (1024, 768), (768, 1), 0), reinterpret_tensor(arg189_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf238)
            del arg189_1
            del arg190_1
            # Topologically Sorted Source Nodes: [tensor, getitem_2, expand_1, expanded_mask, inverted_mask, to_1, extended_attention_mask, linear_66, view_33, query_layer_11, linear_67, view_34, key_layer_11, linear_68, view_35, value_layer_11, attn_output_33], Original ATen: [aten.lift_fresh, aten.unsqueeze, aten.expand, aten._to_copy, aten.sub, aten.masked_fill, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
            buf240 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf236, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf237, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf238, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf239, (8, 12, 128, 128), (16384, 0, 128, 1), 0), False)
            del buf236
            del buf237
            del buf239
            buf241 = buf240[0]
            assert_size_stride(buf241, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf241, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf240
            buf245 = buf238; del buf238  # reuse
            # Topologically Sorted Source Nodes: [attn_output_34, attn_output_35, hidden_states_88], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf241, (1024, 768), (768, 1), 0), reinterpret_tensor(arg191_1, (768, 768), (1, 768), 0), out=buf245)
            del arg191_1
            del buf241
            buf249 = reinterpret_tensor(buf245, (8, 128, 768), (98304, 768, 1), 0); del buf245  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_88, add_23, hidden_states_90], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf249, arg192_1, buf235, arg193_1, arg194_1, 1024, 768, stream=stream0)
            del arg192_1
            del arg193_1
            del arg194_1
            del buf235
            buf250 = reinterpret_tensor(buf230, (1024, 3072), (3072, 1), 0); del buf230  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_91], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf249, (1024, 768), (768, 1), 0), reinterpret_tensor(arg195_1, (768, 3072), (1, 768), 0), out=buf250)
            del arg195_1
            buf251 = reinterpret_tensor(buf250, (8, 128, 3072), (393216, 3072, 1), 0); del buf250  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_91, hidden_states_92], Original ATen: [aten.addmm, aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_3.run(buf251, arg196_1, 3145728, stream=stream0)
            del arg196_1
            buf252 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_91, hidden_states_92, hidden_states_93], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            extern_kernels.mm(reinterpret_tensor(buf251, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg197_1, (3072, 768), (1, 3072), 0), out=buf252)
            del arg197_1
            del buf251
            buf256 = reinterpret_tensor(buf252, (8, 128, 768), (98304, 768, 1), 0); del buf252  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_93, add_24, hidden_states_95], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf256, arg198_1, buf249, arg199_1, arg200_1, 1024, 768, stream=stream0)
            del arg198_1
            del arg199_1
            del arg200_1
            del buf249
        return (buf256, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg2_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg3_1 = rand_strided((30522, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((2, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    arg9_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)


# ===== inductor generated file: /tmp/cnnbench-bert-fold-weaagmnf/repeat_02/base/attempt_01/torchinductor/tmp5mgaaf9a/m5/cm5rpuin46rzdtknz5aqqo3766n3gtq6mrme23rqyrl6joqahvar.py =====
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


# kernel path: /tmp/cnnbench-bert-fold-weaagmnf/repeat_02/base/attempt_01/torchinductor/tmp5mgaaf9a/js/cjsxh5oszam3v7uzmmlndzn7wzr6xe7iv2kei52srdd625ijoylt.py
# Topologically Sorted Source Nodes: [inputs_embeds, buffered_token_type_ids, buffered_token_type_ids_expanded, token_type_embeddings, embeddings, position_ids, position_embeddings, embeddings_1, embeddings_2], Original ATen: [aten.embedding, aten.slice, aten.expand, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   buffered_token_type_ids => slice_1
#   buffered_token_type_ids_expanded => expand
#   embeddings => add
#   embeddings_1 => add_1
#   embeddings_2 => add_2, add_3, mul, mul_1, rsqrt, sub, var_mean
#   inputs_embeds => embedding
#   position_embeddings => embedding_2
#   position_ids => slice_2
#   token_type_embeddings => embedding_1
# Graph fragment:
#   %arg0_1 : Tensor "i64[8, 128][128, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg3_1 : Tensor "f32[30522, 768][768, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg1_1 : Tensor "i64[1, 512][512, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg4_1 : Tensor "f32[2, 768][768, 1]cuda:0" = PlaceHolder[target=arg4_1]
#   %arg2_1 : Tensor "i64[1, 512][512, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg5_1 : Tensor "f32[512, 768][768, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %add_1 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0" = PlaceHolder[target=add_1]
#   %getitem_1 : Tensor "f32[8, 128, 1][128, 1, 1024]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf2 : Tensor "f32[8, 128, 1][128, 1, 1024]cuda:0" = PlaceHolder[target=buf2]
#   %arg6_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg6_1]
#   %arg7_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg7_1]
#   %embedding : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg3_1, %arg0_1, 0), kwargs = {})
#   %slice_1 : Tensor "i64[1, 128][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg1_1, 1, 0, 128), kwargs = {})
#   %expand : Tensor "i64[8, 128][0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%slice_1, [8, 128]), kwargs = {})
#   %embedding_1 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg4_1, %expand), kwargs = {})
#   %add : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %embedding_1), kwargs = {})
#   %slice_2 : Tensor "i64[1, 128][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg2_1, 1, 0, 128), kwargs = {})
#   %embedding_2 : Tensor "f32[1, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%arg5_1, %slice_2), kwargs = {})
#   %add_1 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %embedding_2), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_1, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %getitem_1), kwargs = {})
#   %add_2 : Tensor "f32[8, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-12), kwargs = {})
#   %rsqrt : Tensor "f32[8, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg6_1), kwargs = {})
#   %add_3 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %arg7_1), kwargs = {})
#   return %add_1,%getitem_1,%buf2,%add_3
triton_per_fused_add_embedding_expand_native_layer_norm_slice_0 = async_compile.triton('triton_per_fused_add_embedding_expand_native_layer_norm_slice_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_expand_native_layer_norm_slice_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_embedding_expand_native_layer_norm_slice_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    x3 = xindex
    r0_2 = r0_index
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr6 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.load(in_ptr7 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.full([XBLOCK, R0_BLOCK], 30522, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 30522)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 30522")
    tmp6 = tl.load(in_ptr1 + (r0_2 + 768*tmp4), r0_mask & xmask, other=0.0)
    tmp8 = tl.full([XBLOCK, R0_BLOCK], 2, tl.int32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp7 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp7)
    tl.device_assert(((0 <= tmp11) & (tmp11 < 2)) | ~(xmask), "index out of bounds: 0 <= tmp11 < 2")
    tmp13 = tl.load(in_ptr3 + (r0_2 + 768*tmp11), r0_mask & xmask, other=0.0)
    tmp14 = tmp6 + tmp13
    tmp16 = tl.full([XBLOCK, R0_BLOCK], 512, tl.int32)
    tmp17 = tmp15 + tmp16
    tmp18 = tmp15 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp15)
    tl.device_assert(((0 <= tmp19) & (tmp19 < 512)) | ~(xmask), "index out of bounds: 0 <= tmp19 < 512")
    tmp21 = tl.load(in_ptr5 + (r0_2 + 768*tmp19), r0_mask & xmask, other=0.0)
    tmp22 = tmp14 + tmp21
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, R0_BLOCK])
    tmp25 = tl.where(r0_mask & xmask, tmp23, 0)
    tmp26 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
    tmp28 = tl.where(r0_mask & xmask, tmp26, 0)
    tmp29 = tl.sum(tmp28, 1)[:, None].to(tl.float32)
    tmp30 = tl.full([XBLOCK, 1], 768, tl.int32)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = (tmp29 / tmp31)
    tmp33 = tmp23 - tmp32
    tmp34 = tmp33 * tmp33
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK, R0_BLOCK])
    tmp37 = tl.where(r0_mask & xmask, tmp35, 0)
    tmp38 = tl.sum(tmp37, 1)[:, None].to(tl.float32)
    tmp39 = tmp22 - tmp32
    tmp40 = 768.0
    tmp41 = (tmp38 / tmp40)
    tmp42 = 1e-12
    tmp43 = tmp41 + tmp42
    tmp44 = libdevice.rsqrt(tmp43)
    tmp45 = tmp39 * tmp44
    tmp47 = tmp45 * tmp46
    tmp49 = tmp47 + tmp48
    tl.store(in_out_ptr0 + (r0_2 + 768*x3), tmp49, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-bert-fold-weaagmnf/repeat_02/base/attempt_01/torchinductor/tmp5mgaaf9a/bc/cbcg7yrdae5gcculs4k4kh75h33f4hiihrpuew2o7doyzs2jpeqy.py
# Topologically Sorted Source Nodes: [linear, view, query_layer, linear_1, view_1, key_layer, linear_2, view_2, value_layer, tensor, getitem_2, expand_1, expanded_mask, inverted_mask, to_1, extended_attention_mask, attn_output, linear_6, view_3, query_layer_1, linear_7, view_4, key_layer_1, linear_8, view_5, value_layer_1, attn_output_3, linear_12, view_6, query_layer_2, linear_13, view_7, key_layer_2, linear_14, view_8, value_layer_2, attn_output_6, linear_18, view_9, query_layer_3, linear_19, view_10, key_layer_3, linear_20, view_11, value_layer_3, attn_output_9, linear_24, view_12, query_layer_4, linear_25, view_13, key_layer_4, linear_26, view_14, value_layer_4, attn_output_12], Original ATen: [aten.view, aten.transpose, aten.lift_fresh, aten.unsqueeze, aten.expand, aten._to_copy, aten.sub, aten.masked_fill, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   attn_output => _scaled_dot_product_efficient_attention, expand_2
#   attn_output_12 => _scaled_dot_product_efficient_attention_4, expand_6
#   attn_output_3 => _scaled_dot_product_efficient_attention_1, expand_3
#   attn_output_6 => _scaled_dot_product_efficient_attention_2, expand_4
#   attn_output_9 => _scaled_dot_product_efficient_attention_3, expand_5
#   expand_1 => expand_1
#   expanded_mask => convert_element_type
#   extended_attention_mask => full_default_1, where
#   getitem_2 => unsqueeze, unsqueeze_1
#   inverted_mask => sub_1
#   key_layer => permute_3
#   key_layer_1 => permute_13
#   key_layer_2 => permute_23
#   key_layer_3 => permute_33
#   key_layer_4 => permute_43
#   linear => view_1
#   linear_1 => view_4
#   linear_12 => view_33
#   linear_13 => view_36
#   linear_14 => view_39
#   linear_18 => view_49
#   linear_19 => view_52
#   linear_2 => view_7
#   linear_20 => view_55
#   linear_24 => view_65
#   linear_25 => view_68
#   linear_26 => view_71
#   linear_6 => view_17
#   linear_7 => view_20
#   linear_8 => view_23
#   query_layer => permute_1
#   query_layer_1 => permute_11
#   query_layer_2 => permute_21
#   query_layer_3 => permute_31
#   query_layer_4 => permute_41
#   tensor => full_default
#   to_1 => convert_element_type_1
#   value_layer => permute_5
#   value_layer_1 => permute_15
#   value_layer_2 => permute_25
#   value_layer_3 => permute_35
#   value_layer_4 => permute_45
#   view => view_2
#   view_1 => view_5
#   view_10 => view_53
#   view_11 => view_56
#   view_12 => view_66
#   view_13 => view_69
#   view_14 => view_72
#   view_2 => view_8
#   view_3 => view_18
#   view_4 => view_21
#   view_5 => view_24
#   view_6 => view_34
#   view_7 => view_37
#   view_8 => view_40
#   view_9 => view_50
# Graph fragment:
#   %arg8_1 : Tensor "i64[8, 128][128, 1]cuda:0" = PlaceHolder[target=arg8_1]
#   %view_1 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [8, 128, 768]), kwargs = {})
#   %view_2 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_1, [8, -1, 12, 64]), kwargs = {})
#   %permute_1 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_2, [0, 2, 1, 3]), kwargs = {})
#   %view_4 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_1, [8, 128, 768]), kwargs = {})
#   %view_5 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_4, [8, -1, 12, 64]), kwargs = {})
#   %permute_3 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_5, [0, 2, 1, 3]), kwargs = {})
#   %view_7 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_2, [8, 128, 768]), kwargs = {})
#   %view_8 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_7, [8, -1, 12, 64]), kwargs = {})
#   %permute_5 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_8, [0, 2, 1, 3]), kwargs = {})
#   %full_default : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cpu, pin_memory: False})
#   %unsqueeze : Tensor "i64[8, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg8_1, 1), kwargs = {})
#   %unsqueeze_1 : Tensor "i64[8, 1, 1, 128][128, 128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 2), kwargs = {})
#   %expand_1 : Tensor "i64[8, 1, 128, 128][128, 128, 0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_1, [8, 1, 128, 128]), kwargs = {})
#   %convert_element_type : Tensor "f32[8, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%expand_1, torch.float32), kwargs = {})
#   %sub_1 : Tensor "f32[8, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default, %convert_element_type), kwargs = {})
#   %convert_element_type_1 : Tensor "b8[8, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub_1, torch.bool), kwargs = {})
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -3.4028234663852886e+38), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "f32[8, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=12] = call_function[target=torch.ops.aten.where.self](args = (%convert_element_type_1, %full_default_1, %sub_1), kwargs = {})
#   %expand_2 : Tensor "f32[8, 12, 128, 128][16384, 0, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%where, [8, 12, 128, 128]), kwargs = {})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_1, %permute_3, %permute_5, %expand_2, False), kwargs = {})
#   %view_17 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_6, [8, 128, 768]), kwargs = {})
#   %view_18 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_17, [8, -1, 12, 64]), kwargs = {})
#   %permute_11 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_18, [0, 2, 1, 3]), kwargs = {})
#   %view_20 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_7, [8, 128, 768]), kwargs = {})
#   %view_21 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_20, [8, -1, 12, 64]), kwargs = {})
#   %permute_13 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_21, [0, 2, 1, 3]), kwargs = {})
#   %view_23 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_8, [8, 128, 768]), kwargs = {})
#   %view_24 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_23, [8, -1, 12, 64]), kwargs = {})
#   %permute_15 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_24, [0, 2, 1, 3]), kwargs = {})
#   %expand_3 : Tensor "f32[8, 12, 128, 128][16384, 0, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%where, [8, 12, 128, 128]), kwargs = {})
#   %_scaled_dot_product_efficient_attention_1 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_11, %permute_13, %permute_15, %expand_3, False), kwargs = {})
#   %view_33 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_12, [8, 128, 768]), kwargs = {})
#   %view_34 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_33, [8, -1, 12, 64]), kwargs = {})
#   %permute_21 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_34, [0, 2, 1, 3]), kwargs = {})
#   %view_36 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_13, [8, 128, 768]), kwargs = {})
#   %view_37 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_36, [8, -1, 12, 64]), kwargs = {})
#   %permute_23 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_37, [0, 2, 1, 3]), kwargs = {})
#   %view_39 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_14, [8, 128, 768]), kwargs = {})
#   %view_40 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_39, [8, -1, 12, 64]), kwargs = {})
#   %permute_25 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_40, [0, 2, 1, 3]), kwargs = {})
#   %expand_4 : Tensor "f32[8, 12, 128, 128][16384, 0, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%where, [8, 12, 128, 128]), kwargs = {})
#   %_scaled_dot_product_efficient_attention_2 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_21, %permute_23, %permute_25, %expand_4, False), kwargs = {})
#   %view_49 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_18, [8, 128, 768]), kwargs = {})
#   %view_50 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_49, [8, -1, 12, 64]), kwargs = {})
#   %permute_31 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_50, [0, 2, 1, 3]), kwargs = {})
#   %view_52 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_19, [8, 128, 768]), kwargs = {})
#   %view_53 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_52, [8, -1, 12, 64]), kwargs = {})
#   %permute_33 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_53, [0, 2, 1, 3]), kwargs = {})
#   %view_55 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_20, [8, 128, 768]), kwargs = {})
#   %view_56 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_55, [8, -1, 12, 64]), kwargs = {})
#   %permute_35 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_56, [0, 2, 1, 3]), kwargs = {})
#   %expand_5 : Tensor "f32[8, 12, 128, 128][16384, 0, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%where, [8, 12, 128, 128]), kwargs = {})
#   %_scaled_dot_product_efficient_attention_3 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_31, %permute_33, %permute_35, %expand_5, False), kwargs = {})
#   %view_65 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_24, [8, 128, 768]), kwargs = {})
#   %view_66 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_65, [8, -1, 12, 64]), kwargs = {})
#   %permute_41 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_66, [0, 2, 1, 3]), kwargs = {})
#   %view_68 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_25, [8, 128, 768]), kwargs = {})
#   %view_69 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_68, [8, -1, 12, 64]), kwargs = {})
#   %permute_43 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_69, [0, 2, 1, 3]), kwargs = {})
#   %view_71 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_26, [8, 128, 768]), kwargs = {})
#   %view_72 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_71, [8, -1, 12, 64]), kwargs = {})
#   %permute_45 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_72, [0, 2, 1, 3]), kwargs = {})
#   %expand_6 : Tensor "f32[8, 12, 128, 128][16384, 0, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%where, [8, 12, 128, 128]), kwargs = {})
#   %_scaled_dot_product_efficient_attention_4 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_41, %permute_43, %permute_45, %expand_6, False), kwargs = {})
#   return %buf8,%buf29,%buf50,%buf71,%buf92
triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_lift_fresh_masked_fill_sub_transpose_unsqueeze_view_1 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_lift_fresh_masked_fill_sub_transpose_unsqueeze_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_lift_fresh_masked_fill_sub_transpose_unsqueeze_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 5251072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_lift_fresh_masked_fill_sub_transpose_unsqueeze_view_1(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 128)
    x2 = xindex // 16384
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x2), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp2 - tmp1
    tmp4 = (tmp3 != 0)
    tmp5 = -3.4028234663852886e+38
    tmp6 = tl.where(tmp4, tmp5, tmp3)
    tl.store(out_ptr0 + (x3), tmp6, None)
    tl.store(out_ptr1 + (x3), tmp6, None)
    tl.store(out_ptr2 + (x3), tmp6, None)
    tl.store(out_ptr3 + (x3), tmp6, None)
    tl.store(out_ptr4 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-bert-fold-weaagmnf/repeat_02/base/attempt_01/torchinductor/tmp5mgaaf9a/3o/c3o3cagdzec6vdr53idsfidv322vhrsgvavfqz4humgbetninzuc.py
# Topologically Sorted Source Nodes: [hidden_states, add_1, hidden_states_2], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_1 => add_4
#   hidden_states => add_tensor_35, view_11
#   hidden_states_2 => add_5, add_6, mul_2, mul_3, rsqrt_1, sub_2, var_mean_1
# Graph fragment:
#   %mm_default_35 : Tensor "f32[1024, 768][768, 1]cuda:0" = PlaceHolder[target=mm_default_35]
#   %arg16_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg16_1]
#   %add_3 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0" = PlaceHolder[target=add_3]
#   %getitem_7 : Tensor "f32[8, 128, 1][128, 1, 1024]cuda:0" = PlaceHolder[target=getitem_7]
#   %buf16 : Tensor "f32[8, 128, 1][128, 1, 1024]cuda:0" = PlaceHolder[target=buf16]
#   %arg17_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg17_1]
#   %arg18_1 : Tensor "f32[768][1]cuda:0" = PlaceHolder[target=arg18_1]
#   %add_tensor_35 : Tensor "f32[1024, 768][768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_35, %arg16_1), kwargs = {})
#   %view_11 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_35, [8, 128, 768]), kwargs = {})
#   %add_4 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_11, %add_3), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_4, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_4, %getitem_7), kwargs = {})
#   %add_5 : Tensor "f32[8, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-12), kwargs = {})
#   %rsqrt_1 : Tensor "f32[8, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
#   %mul_2 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_1), kwargs = {})
#   %mul_3 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %arg17_1), kwargs = {})
#   %add_6 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %arg18_1), kwargs = {})
#   return %getitem_7,%buf16,%add_6
triton_per_fused_add_addmm_native_layer_norm_view_2 = async_compile.triton('triton_per_fused_add_addmm_native_layer_norm_view_2', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_native_layer_norm_view_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 12592128}}
)
@triton.jit
def triton_per_fused_add_addmm_native_layer_norm_view_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp28 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
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
    tmp24 = 1e-12
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp31, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-bert-fold-weaagmnf/repeat_02/base/attempt_01/torchinductor/tmp5mgaaf9a/r6/cr6dvhidsakzazbjgcr5uj2e5qory6i6pdmdzxcxjo4omejryiae.py
# Topologically Sorted Source Nodes: [hidden_states_3, hidden_states_4], Original ATen: [aten.addmm, aten.view, aten.gelu]
# Source node to ATen node mapping:
#   hidden_states_3 => add_tensor_34, view_13
#   hidden_states_4 => add_7, erf, mul_4, mul_5, mul_6
# Graph fragment:
#   %mm_default_34 : Tensor "f32[1024, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_default_34]
#   %arg20_1 : Tensor "f32[3072][1]cuda:0" = PlaceHolder[target=arg20_1]
#   %add_tensor_34 : Tensor "f32[1024, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_34, %arg20_1), kwargs = {})
#   %view_13 : Tensor "f32[8, 128, 3072][393216, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_34, [8, 128, 3072]), kwargs = {})
#   %mul_4 : Tensor "f32[8, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_13, 0.5), kwargs = {})
#   %mul_5 : Tensor "f32[8, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_13, 0.7071067811865476), kwargs = {})
#   %erf : Tensor "f32[8, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_5,), kwargs = {})
#   %add_7 : Tensor "f32[8, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_6 : Tensor "f32[8, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %add_7), kwargs = {})
#   return %mul_6
triton_poi_fused_addmm_gelu_view_3 = async_compile.triton('triton_poi_fused_addmm_gelu_view_3', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_gelu_view_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 37761024}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_gelu_view_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: /tmp/cnnbench-bert-fold-weaagmnf/repeat_02/base/attempt_01/torchinductor/tmp5mgaaf9a/sq/csqnegz2evgpsehsmnndtg3xifvb72hammko7tbs3n2yijtya4yw.py
# Topologically Sorted Source Nodes: [tensor, getitem_2, expand_1, expanded_mask, inverted_mask, to_1, extended_attention_mask, linear_60, view_30, query_layer_10, linear_61, view_31, key_layer_10, linear_62, view_32, value_layer_10, attn_output_30, linear_66, view_33, query_layer_11, linear_67, view_34, key_layer_11, linear_68, view_35, value_layer_11, attn_output_33], Original ATen: [aten.lift_fresh, aten.unsqueeze, aten.expand, aten._to_copy, aten.sub, aten.masked_fill, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   attn_output_30 => _scaled_dot_product_efficient_attention_10, expand_12
#   attn_output_33 => _scaled_dot_product_efficient_attention_11, expand_13
#   expand_1 => expand_1
#   expanded_mask => convert_element_type
#   extended_attention_mask => full_default_1, where
#   getitem_2 => unsqueeze, unsqueeze_1
#   inverted_mask => sub_1
#   key_layer_10 => permute_103
#   key_layer_11 => permute_113
#   linear_60 => view_161
#   linear_61 => view_164
#   linear_62 => view_167
#   linear_66 => view_177
#   linear_67 => view_180
#   linear_68 => view_183
#   query_layer_10 => permute_101
#   query_layer_11 => permute_111
#   tensor => full_default
#   to_1 => convert_element_type_1
#   value_layer_10 => permute_105
#   value_layer_11 => permute_115
#   view_30 => view_162
#   view_31 => view_165
#   view_32 => view_168
#   view_33 => view_178
#   view_34 => view_181
#   view_35 => view_184
# Graph fragment:
#   %arg8_1 : Tensor "i64[8, 128][128, 1]cuda:0" = PlaceHolder[target=arg8_1]
#   %full_default : Tensor "f32[][]cpu"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cpu, pin_memory: False})
#   %unsqueeze : Tensor "i64[8, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg8_1, 1), kwargs = {})
#   %unsqueeze_1 : Tensor "i64[8, 1, 1, 128][128, 128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 2), kwargs = {})
#   %expand_1 : Tensor "i64[8, 1, 128, 128][128, 128, 0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_1, [8, 1, 128, 128]), kwargs = {})
#   %convert_element_type : Tensor "f32[8, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%expand_1, torch.float32), kwargs = {})
#   %sub_1 : Tensor "f32[8, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default, %convert_element_type), kwargs = {})
#   %convert_element_type_1 : Tensor "b8[8, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub_1, torch.bool), kwargs = {})
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -3.4028234663852886e+38), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "f32[8, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=12] = call_function[target=torch.ops.aten.where.self](args = (%convert_element_type_1, %full_default_1, %sub_1), kwargs = {})
#   %view_161 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_60, [8, 128, 768]), kwargs = {})
#   %view_162 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_161, [8, -1, 12, 64]), kwargs = {})
#   %permute_101 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_162, [0, 2, 1, 3]), kwargs = {})
#   %view_164 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_61, [8, 128, 768]), kwargs = {})
#   %view_165 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_164, [8, -1, 12, 64]), kwargs = {})
#   %permute_103 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_165, [0, 2, 1, 3]), kwargs = {})
#   %view_167 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_62, [8, 128, 768]), kwargs = {})
#   %view_168 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_167, [8, -1, 12, 64]), kwargs = {})
#   %permute_105 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_168, [0, 2, 1, 3]), kwargs = {})
#   %expand_12 : Tensor "f32[8, 12, 128, 128][16384, 0, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%where, [8, 12, 128, 128]), kwargs = {})
#   %_scaled_dot_product_efficient_attention_10 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_101, %permute_103, %permute_105, %expand_12, False), kwargs = {})
#   %view_177 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_66, [8, 128, 768]), kwargs = {})
#   %view_178 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_177, [8, -1, 12, 64]), kwargs = {})
#   %permute_111 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_178, [0, 2, 1, 3]), kwargs = {})
#   %view_180 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_67, [8, 128, 768]), kwargs = {})
#   %view_181 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_180, [8, -1, 12, 64]), kwargs = {})
#   %permute_113 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_181, [0, 2, 1, 3]), kwargs = {})
#   %view_183 : Tensor "f32[8, 128, 768][98304, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_68, [8, 128, 768]), kwargs = {})
#   %view_184 : Tensor "f32[8, 128, 12, 64][98304, 768, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_183, [8, -1, 12, 64]), kwargs = {})
#   %permute_115 : Tensor "f32[8, 12, 128, 64][98304, 64, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_184, [0, 2, 1, 3]), kwargs = {})
#   %expand_13 : Tensor "f32[8, 12, 128, 128][16384, 0, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%where, [8, 12, 128, 128]), kwargs = {})
#   %_scaled_dot_product_efficient_attention_11 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%permute_111, %permute_113, %permute_115, %expand_13, False), kwargs = {})
#   return %buf218,%buf239
triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_lift_fresh_masked_fill_sub_transpose_unsqueeze_view_4 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_lift_fresh_masked_fill_sub_transpose_unsqueeze_view_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_lift_fresh_masked_fill_sub_transpose_unsqueeze_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2105344}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_lift_fresh_masked_fill_sub_transpose_unsqueeze_view_4(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 128)
    x2 = xindex // 16384
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x2), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp2 - tmp1
    tmp4 = (tmp3 != 0)
    tmp5 = -3.4028234663852886e+38
    tmp6 = tl.where(tmp4, tmp5, tmp3)
    tl.store(out_ptr0 + (x3), tmp6, None)
    tl.store(out_ptr1 + (x3), tmp6, None)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1 = args
        args.clear()
        assert_size_stride(arg0_1, (8, 128), (128, 1))
        assert_size_stride(arg1_1, (1, 512), (512, 1))
        assert_size_stride(arg2_1, (1, 512), (512, 1))
        assert_size_stride(arg3_1, (30522, 768), (768, 1))
        assert_size_stride(arg4_1, (2, 768), (768, 1))
        assert_size_stride(arg5_1, (512, 768), (768, 1))
        assert_size_stride(arg6_1, (768, ), (1, ))
        assert_size_stride(arg7_1, (768, ), (1, ))
        assert_size_stride(arg8_1, (8, 128), (128, 1))
        assert_size_stride(arg9_1, (768, 768), (768, 1))
        assert_size_stride(arg10_1, (768, ), (1, ))
        assert_size_stride(arg11_1, (768, 768), (768, 1))
        assert_size_stride(arg12_1, (768, ), (1, ))
        assert_size_stride(arg13_1, (768, 768), (768, 1))
        assert_size_stride(arg14_1, (768, ), (1, ))
        assert_size_stride(arg15_1, (768, 768), (768, 1))
        assert_size_stride(arg16_1, (768, ), (1, ))
        assert_size_stride(arg17_1, (768, ), (1, ))
        assert_size_stride(arg18_1, (768, ), (1, ))
        assert_size_stride(arg19_1, (3072, 768), (768, 1))
        assert_size_stride(arg20_1, (3072, ), (1, ))
        assert_size_stride(arg21_1, (768, 3072), (3072, 1))
        assert_size_stride(arg22_1, (768, ), (1, ))
        assert_size_stride(arg23_1, (768, ), (1, ))
        assert_size_stride(arg24_1, (768, ), (1, ))
        assert_size_stride(arg25_1, (768, 768), (768, 1))
        assert_size_stride(arg26_1, (768, ), (1, ))
        assert_size_stride(arg27_1, (768, 768), (768, 1))
        assert_size_stride(arg28_1, (768, ), (1, ))
        assert_size_stride(arg29_1, (768, 768), (768, 1))
        assert_size_stride(arg30_1, (768, ), (1, ))
        assert_size_stride(arg31_1, (768, 768), (768, 1))
        assert_size_stride(arg32_1, (768, ), (1, ))
        assert_size_stride(arg33_1, (768, ), (1, ))
        assert_size_stride(arg34_1, (768, ), (1, ))
        assert_size_stride(arg35_1, (3072, 768), (768, 1))
        assert_size_stride(arg36_1, (3072, ), (1, ))
        assert_size_stride(arg37_1, (768, 3072), (3072, 1))
        assert_size_stride(arg38_1, (768, ), (1, ))
        assert_size_stride(arg39_1, (768, ), (1, ))
        assert_size_stride(arg40_1, (768, ), (1, ))
        assert_size_stride(arg41_1, (768, 768), (768, 1))
        assert_size_stride(arg42_1, (768, ), (1, ))
        assert_size_stride(arg43_1, (768, 768), (768, 1))
        assert_size_stride(arg44_1, (768, ), (1, ))
        assert_size_stride(arg45_1, (768, 768), (768, 1))
        assert_size_stride(arg46_1, (768, ), (1, ))
        assert_size_stride(arg47_1, (768, 768), (768, 1))
        assert_size_stride(arg48_1, (768, ), (1, ))
        assert_size_stride(arg49_1, (768, ), (1, ))
        assert_size_stride(arg50_1, (768, ), (1, ))
        assert_size_stride(arg51_1, (3072, 768), (768, 1))
        assert_size_stride(arg52_1, (3072, ), (1, ))
        assert_size_stride(arg53_1, (768, 3072), (3072, 1))
        assert_size_stride(arg54_1, (768, ), (1, ))
        assert_size_stride(arg55_1, (768, ), (1, ))
        assert_size_stride(arg56_1, (768, ), (1, ))
        assert_size_stride(arg57_1, (768, 768), (768, 1))
        assert_size_stride(arg58_1, (768, ), (1, ))
        assert_size_stride(arg59_1, (768, 768), (768, 1))
        assert_size_stride(arg60_1, (768, ), (1, ))
        assert_size_stride(arg61_1, (768, 768), (768, 1))
        assert_size_stride(arg62_1, (768, ), (1, ))
        assert_size_stride(arg63_1, (768, 768), (768, 1))
        assert_size_stride(arg64_1, (768, ), (1, ))
        assert_size_stride(arg65_1, (768, ), (1, ))
        assert_size_stride(arg66_1, (768, ), (1, ))
        assert_size_stride(arg67_1, (3072, 768), (768, 1))
        assert_size_stride(arg68_1, (3072, ), (1, ))
        assert_size_stride(arg69_1, (768, 3072), (3072, 1))
        assert_size_stride(arg70_1, (768, ), (1, ))
        assert_size_stride(arg71_1, (768, ), (1, ))
        assert_size_stride(arg72_1, (768, ), (1, ))
        assert_size_stride(arg73_1, (768, 768), (768, 1))
        assert_size_stride(arg74_1, (768, ), (1, ))
        assert_size_stride(arg75_1, (768, 768), (768, 1))
        assert_size_stride(arg76_1, (768, ), (1, ))
        assert_size_stride(arg77_1, (768, 768), (768, 1))
        assert_size_stride(arg78_1, (768, ), (1, ))
        assert_size_stride(arg79_1, (768, 768), (768, 1))
        assert_size_stride(arg80_1, (768, ), (1, ))
        assert_size_stride(arg81_1, (768, ), (1, ))
        assert_size_stride(arg82_1, (768, ), (1, ))
        assert_size_stride(arg83_1, (3072, 768), (768, 1))
        assert_size_stride(arg84_1, (3072, ), (1, ))
        assert_size_stride(arg85_1, (768, 3072), (3072, 1))
        assert_size_stride(arg86_1, (768, ), (1, ))
        assert_size_stride(arg87_1, (768, ), (1, ))
        assert_size_stride(arg88_1, (768, ), (1, ))
        assert_size_stride(arg89_1, (768, 768), (768, 1))
        assert_size_stride(arg90_1, (768, ), (1, ))
        assert_size_stride(arg91_1, (768, 768), (768, 1))
        assert_size_stride(arg92_1, (768, ), (1, ))
        assert_size_stride(arg93_1, (768, 768), (768, 1))
        assert_size_stride(arg94_1, (768, ), (1, ))
        assert_size_stride(arg95_1, (768, 768), (768, 1))
        assert_size_stride(arg96_1, (768, ), (1, ))
        assert_size_stride(arg97_1, (768, ), (1, ))
        assert_size_stride(arg98_1, (768, ), (1, ))
        assert_size_stride(arg99_1, (3072, 768), (768, 1))
        assert_size_stride(arg100_1, (3072, ), (1, ))
        assert_size_stride(arg101_1, (768, 3072), (3072, 1))
        assert_size_stride(arg102_1, (768, ), (1, ))
        assert_size_stride(arg103_1, (768, ), (1, ))
        assert_size_stride(arg104_1, (768, ), (1, ))
        assert_size_stride(arg105_1, (768, 768), (768, 1))
        assert_size_stride(arg106_1, (768, ), (1, ))
        assert_size_stride(arg107_1, (768, 768), (768, 1))
        assert_size_stride(arg108_1, (768, ), (1, ))
        assert_size_stride(arg109_1, (768, 768), (768, 1))
        assert_size_stride(arg110_1, (768, ), (1, ))
        assert_size_stride(arg111_1, (768, 768), (768, 1))
        assert_size_stride(arg112_1, (768, ), (1, ))
        assert_size_stride(arg113_1, (768, ), (1, ))
        assert_size_stride(arg114_1, (768, ), (1, ))
        assert_size_stride(arg115_1, (3072, 768), (768, 1))
        assert_size_stride(arg116_1, (3072, ), (1, ))
        assert_size_stride(arg117_1, (768, 3072), (3072, 1))
        assert_size_stride(arg118_1, (768, ), (1, ))
        assert_size_stride(arg119_1, (768, ), (1, ))
        assert_size_stride(arg120_1, (768, ), (1, ))
        assert_size_stride(arg121_1, (768, 768), (768, 1))
        assert_size_stride(arg122_1, (768, ), (1, ))
        assert_size_stride(arg123_1, (768, 768), (768, 1))
        assert_size_stride(arg124_1, (768, ), (1, ))
        assert_size_stride(arg125_1, (768, 768), (768, 1))
        assert_size_stride(arg126_1, (768, ), (1, ))
        assert_size_stride(arg127_1, (768, 768), (768, 1))
        assert_size_stride(arg128_1, (768, ), (1, ))
        assert_size_stride(arg129_1, (768, ), (1, ))
        assert_size_stride(arg130_1, (768, ), (1, ))
        assert_size_stride(arg131_1, (3072, 768), (768, 1))
        assert_size_stride(arg132_1, (3072, ), (1, ))
        assert_size_stride(arg133_1, (768, 3072), (3072, 1))
        assert_size_stride(arg134_1, (768, ), (1, ))
        assert_size_stride(arg135_1, (768, ), (1, ))
        assert_size_stride(arg136_1, (768, ), (1, ))
        assert_size_stride(arg137_1, (768, 768), (768, 1))
        assert_size_stride(arg138_1, (768, ), (1, ))
        assert_size_stride(arg139_1, (768, 768), (768, 1))
        assert_size_stride(arg140_1, (768, ), (1, ))
        assert_size_stride(arg141_1, (768, 768), (768, 1))
        assert_size_stride(arg142_1, (768, ), (1, ))
        assert_size_stride(arg143_1, (768, 768), (768, 1))
        assert_size_stride(arg144_1, (768, ), (1, ))
        assert_size_stride(arg145_1, (768, ), (1, ))
        assert_size_stride(arg146_1, (768, ), (1, ))
        assert_size_stride(arg147_1, (3072, 768), (768, 1))
        assert_size_stride(arg148_1, (3072, ), (1, ))
        assert_size_stride(arg149_1, (768, 3072), (3072, 1))
        assert_size_stride(arg150_1, (768, ), (1, ))
        assert_size_stride(arg151_1, (768, ), (1, ))
        assert_size_stride(arg152_1, (768, ), (1, ))
        assert_size_stride(arg153_1, (768, 768), (768, 1))
        assert_size_stride(arg154_1, (768, ), (1, ))
        assert_size_stride(arg155_1, (768, 768), (768, 1))
        assert_size_stride(arg156_1, (768, ), (1, ))
        assert_size_stride(arg157_1, (768, 768), (768, 1))
        assert_size_stride(arg158_1, (768, ), (1, ))
        assert_size_stride(arg159_1, (768, 768), (768, 1))
        assert_size_stride(arg160_1, (768, ), (1, ))
        assert_size_stride(arg161_1, (768, ), (1, ))
        assert_size_stride(arg162_1, (768, ), (1, ))
        assert_size_stride(arg163_1, (3072, 768), (768, 1))
        assert_size_stride(arg164_1, (3072, ), (1, ))
        assert_size_stride(arg165_1, (768, 3072), (3072, 1))
        assert_size_stride(arg166_1, (768, ), (1, ))
        assert_size_stride(arg167_1, (768, ), (1, ))
        assert_size_stride(arg168_1, (768, ), (1, ))
        assert_size_stride(arg169_1, (768, 768), (768, 1))
        assert_size_stride(arg170_1, (768, ), (1, ))
        assert_size_stride(arg171_1, (768, 768), (768, 1))
        assert_size_stride(arg172_1, (768, ), (1, ))
        assert_size_stride(arg173_1, (768, 768), (768, 1))
        assert_size_stride(arg174_1, (768, ), (1, ))
        assert_size_stride(arg175_1, (768, 768), (768, 1))
        assert_size_stride(arg176_1, (768, ), (1, ))
        assert_size_stride(arg177_1, (768, ), (1, ))
        assert_size_stride(arg178_1, (768, ), (1, ))
        assert_size_stride(arg179_1, (3072, 768), (768, 1))
        assert_size_stride(arg180_1, (3072, ), (1, ))
        assert_size_stride(arg181_1, (768, 3072), (3072, 1))
        assert_size_stride(arg182_1, (768, ), (1, ))
        assert_size_stride(arg183_1, (768, ), (1, ))
        assert_size_stride(arg184_1, (768, ), (1, ))
        assert_size_stride(arg185_1, (768, 768), (768, 1))
        assert_size_stride(arg186_1, (768, ), (1, ))
        assert_size_stride(arg187_1, (768, 768), (768, 1))
        assert_size_stride(arg188_1, (768, ), (1, ))
        assert_size_stride(arg189_1, (768, 768), (768, 1))
        assert_size_stride(arg190_1, (768, ), (1, ))
        assert_size_stride(arg191_1, (768, 768), (768, 1))
        assert_size_stride(arg192_1, (768, ), (1, ))
        assert_size_stride(arg193_1, (768, ), (1, ))
        assert_size_stride(arg194_1, (768, ), (1, ))
        assert_size_stride(arg195_1, (3072, 768), (768, 1))
        assert_size_stride(arg196_1, (3072, ), (1, ))
        assert_size_stride(arg197_1, (768, 3072), (3072, 1))
        assert_size_stride(arg198_1, (768, ), (1, ))
        assert_size_stride(arg199_1, (768, ), (1, ))
        assert_size_stride(arg200_1, (768, ), (1, ))
        assert_size_stride(arg201_1, (768, 768), (768, 1))
        assert_size_stride(arg202_1, (768, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((8, 128, 768), (98304, 768, 1), torch.float32)
            buf4 = buf0; del buf0  # reuse
            # Topologically Sorted Source Nodes: [inputs_embeds, buffered_token_type_ids, buffered_token_type_ids_expanded, token_type_embeddings, embeddings, position_ids, position_embeddings, embeddings_1, embeddings_2], Original ATen: [aten.embedding, aten.slice, aten.expand, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_embedding_expand_native_layer_norm_slice_0.run(buf4, arg0_1, arg3_1, arg1_1, arg4_1, arg2_1, arg5_1, arg6_1, arg7_1, 1024, 768, stream=stream0)
            del arg0_1
            del arg1_1
            del arg2_1
            del arg3_1
            del arg4_1
            del arg5_1
            del arg6_1
            del arg7_1
            buf5 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg10_1, reinterpret_tensor(buf4, (1024, 768), (768, 1), 0), reinterpret_tensor(arg9_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf5)
            del arg10_1
            del arg9_1
            buf6 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg12_1, reinterpret_tensor(buf4, (1024, 768), (768, 1), 0), reinterpret_tensor(arg11_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf6)
            del arg11_1
            del arg12_1
            buf7 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg14_1, reinterpret_tensor(buf4, (1024, 768), (768, 1), 0), reinterpret_tensor(arg13_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf7)
            del arg13_1
            del arg14_1
            buf8 = empty_strided_cuda((8, 1, 128, 128), (16384, 0, 128, 1), torch.float32)
            buf29 = empty_strided_cuda((8, 1, 128, 128), (16384, 0, 128, 1), torch.float32)
            buf50 = empty_strided_cuda((8, 1, 128, 128), (16384, 0, 128, 1), torch.float32)
            buf71 = empty_strided_cuda((8, 1, 128, 128), (16384, 0, 128, 1), torch.float32)
            buf92 = empty_strided_cuda((8, 1, 128, 128), (16384, 0, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear, view, query_layer, linear_1, view_1, key_layer, linear_2, view_2, value_layer, tensor, getitem_2, expand_1, expanded_mask, inverted_mask, to_1, extended_attention_mask, attn_output, linear_6, view_3, query_layer_1, linear_7, view_4, key_layer_1, linear_8, view_5, value_layer_1, attn_output_3, linear_12, view_6, query_layer_2, linear_13, view_7, key_layer_2, linear_14, view_8, value_layer_2, attn_output_6, linear_18, view_9, query_layer_3, linear_19, view_10, key_layer_3, linear_20, view_11, value_layer_3, attn_output_9, linear_24, view_12, query_layer_4, linear_25, view_13, key_layer_4, linear_26, view_14, value_layer_4, attn_output_12], Original ATen: [aten.view, aten.transpose, aten.lift_fresh, aten.unsqueeze, aten.expand, aten._to_copy, aten.sub, aten.masked_fill, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_lift_fresh_masked_fill_sub_transpose_unsqueeze_view_1.run(arg8_1, buf8, buf29, buf50, buf71, buf92, 131072, stream=stream0)
            # Topologically Sorted Source Nodes: [linear, view, query_layer, linear_1, view_1, key_layer, linear_2, view_2, value_layer, tensor, getitem_2, expand_1, expanded_mask, inverted_mask, to_1, extended_attention_mask, attn_output], Original ATen: [aten.view, aten.transpose, aten.lift_fresh, aten.unsqueeze, aten.expand, aten._to_copy, aten.sub, aten.masked_fill, aten._scaled_dot_product_efficient_attention]
            buf9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf5, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf6, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf7, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf8, (8, 12, 128, 128), (16384, 0, 128, 1), 0), False)
            del buf5
            del buf6
            del buf8
            buf10 = buf9[0]
            assert_size_stride(buf10, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf10, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf9
            buf14 = buf7; del buf7  # reuse
            # Topologically Sorted Source Nodes: [attn_output_1, attn_output_2, hidden_states], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf10, (1024, 768), (768, 1), 0), reinterpret_tensor(arg15_1, (768, 768), (1, 768), 0), out=buf14)
            del arg15_1
            del buf10
            buf18 = reinterpret_tensor(buf14, (8, 128, 768), (98304, 768, 1), 0); del buf14  # reuse
            # Topologically Sorted Source Nodes: [hidden_states, add_1, hidden_states_2], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf18, arg16_1, buf4, arg17_1, arg18_1, 1024, 768, stream=stream0)
            del arg16_1
            del arg17_1
            del arg18_1
            del buf4
            buf19 = empty_strided_cuda((1024, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_3], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf18, (1024, 768), (768, 1), 0), reinterpret_tensor(arg19_1, (768, 3072), (1, 768), 0), out=buf19)
            del arg19_1
            buf20 = reinterpret_tensor(buf19, (8, 128, 3072), (393216, 3072, 1), 0); del buf19  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_3, hidden_states_4], Original ATen: [aten.addmm, aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_3.run(buf20, arg20_1, 3145728, stream=stream0)
            del arg20_1
            buf21 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_3, hidden_states_4, hidden_states_5], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            extern_kernels.mm(reinterpret_tensor(buf20, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg21_1, (3072, 768), (1, 3072), 0), out=buf21)
            del arg21_1
            buf25 = reinterpret_tensor(buf21, (8, 128, 768), (98304, 768, 1), 0); del buf21  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_5, add_2, hidden_states_7], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf25, arg22_1, buf18, arg23_1, arg24_1, 1024, 768, stream=stream0)
            del arg22_1
            del arg23_1
            del arg24_1
            buf26 = reinterpret_tensor(buf18, (1024, 768), (768, 1), 0); del buf18  # reuse
            # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg26_1, reinterpret_tensor(buf25, (1024, 768), (768, 1), 0), reinterpret_tensor(arg25_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf26)
            del arg25_1
            del arg26_1
            buf27 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg28_1, reinterpret_tensor(buf25, (1024, 768), (768, 1), 0), reinterpret_tensor(arg27_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf27)
            del arg27_1
            del arg28_1
            buf28 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg30_1, reinterpret_tensor(buf25, (1024, 768), (768, 1), 0), reinterpret_tensor(arg29_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf28)
            del arg29_1
            del arg30_1
            # Topologically Sorted Source Nodes: [tensor, getitem_2, expand_1, expanded_mask, inverted_mask, to_1, extended_attention_mask, linear_6, view_3, query_layer_1, linear_7, view_4, key_layer_1, linear_8, view_5, value_layer_1, attn_output_3], Original ATen: [aten.lift_fresh, aten.unsqueeze, aten.expand, aten._to_copy, aten.sub, aten.masked_fill, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
            buf30 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf26, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf27, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf28, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf29, (8, 12, 128, 128), (16384, 0, 128, 1), 0), False)
            del buf26
            del buf27
            buf31 = buf30[0]
            assert_size_stride(buf31, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf31, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf30
            buf35 = buf28; del buf28  # reuse
            # Topologically Sorted Source Nodes: [attn_output_4, attn_output_5, hidden_states_8], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf31, (1024, 768), (768, 1), 0), reinterpret_tensor(arg31_1, (768, 768), (1, 768), 0), out=buf35)
            del arg31_1
            del buf31
            buf39 = reinterpret_tensor(buf35, (8, 128, 768), (98304, 768, 1), 0); del buf35  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_8, add_3, hidden_states_10], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf39, arg32_1, buf25, arg33_1, arg34_1, 1024, 768, stream=stream0)
            del arg32_1
            del arg33_1
            del arg34_1
            del buf25
            buf40 = reinterpret_tensor(buf20, (1024, 3072), (3072, 1), 0); del buf20  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_11], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf39, (1024, 768), (768, 1), 0), reinterpret_tensor(arg35_1, (768, 3072), (1, 768), 0), out=buf40)
            del arg35_1
            buf41 = reinterpret_tensor(buf40, (8, 128, 3072), (393216, 3072, 1), 0); del buf40  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_11, hidden_states_12], Original ATen: [aten.addmm, aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_3.run(buf41, arg36_1, 3145728, stream=stream0)
            del arg36_1
            buf42 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_11, hidden_states_12, hidden_states_13], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            extern_kernels.mm(reinterpret_tensor(buf41, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg37_1, (3072, 768), (1, 3072), 0), out=buf42)
            del arg37_1
            buf46 = reinterpret_tensor(buf42, (8, 128, 768), (98304, 768, 1), 0); del buf42  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_13, add_4, hidden_states_15], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf46, arg38_1, buf39, arg39_1, arg40_1, 1024, 768, stream=stream0)
            del arg38_1
            del arg39_1
            del arg40_1
            buf47 = reinterpret_tensor(buf39, (1024, 768), (768, 1), 0); del buf39  # reuse
            # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg42_1, reinterpret_tensor(buf46, (1024, 768), (768, 1), 0), reinterpret_tensor(arg41_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf47)
            del arg41_1
            del arg42_1
            buf48 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg44_1, reinterpret_tensor(buf46, (1024, 768), (768, 1), 0), reinterpret_tensor(arg43_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf48)
            del arg43_1
            del arg44_1
            buf49 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg46_1, reinterpret_tensor(buf46, (1024, 768), (768, 1), 0), reinterpret_tensor(arg45_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf49)
            del arg45_1
            del arg46_1
            # Topologically Sorted Source Nodes: [tensor, getitem_2, expand_1, expanded_mask, inverted_mask, to_1, extended_attention_mask, linear_12, view_6, query_layer_2, linear_13, view_7, key_layer_2, linear_14, view_8, value_layer_2, attn_output_6], Original ATen: [aten.lift_fresh, aten.unsqueeze, aten.expand, aten._to_copy, aten.sub, aten.masked_fill, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
            buf51 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf47, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf48, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf49, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf50, (8, 12, 128, 128), (16384, 0, 128, 1), 0), False)
            del buf47
            del buf48
            buf52 = buf51[0]
            assert_size_stride(buf52, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf52, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf51
            buf56 = buf49; del buf49  # reuse
            # Topologically Sorted Source Nodes: [attn_output_7, attn_output_8, hidden_states_16], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf52, (1024, 768), (768, 1), 0), reinterpret_tensor(arg47_1, (768, 768), (1, 768), 0), out=buf56)
            del arg47_1
            del buf52
            buf60 = reinterpret_tensor(buf56, (8, 128, 768), (98304, 768, 1), 0); del buf56  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_16, add_5, hidden_states_18], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf60, arg48_1, buf46, arg49_1, arg50_1, 1024, 768, stream=stream0)
            del arg48_1
            del arg49_1
            del arg50_1
            del buf46
            buf61 = reinterpret_tensor(buf41, (1024, 3072), (3072, 1), 0); del buf41  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_19], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf60, (1024, 768), (768, 1), 0), reinterpret_tensor(arg51_1, (768, 3072), (1, 768), 0), out=buf61)
            del arg51_1
            buf62 = reinterpret_tensor(buf61, (8, 128, 3072), (393216, 3072, 1), 0); del buf61  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_19, hidden_states_20], Original ATen: [aten.addmm, aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_3.run(buf62, arg52_1, 3145728, stream=stream0)
            del arg52_1
            buf63 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_19, hidden_states_20, hidden_states_21], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            extern_kernels.mm(reinterpret_tensor(buf62, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg53_1, (3072, 768), (1, 3072), 0), out=buf63)
            del arg53_1
            buf67 = reinterpret_tensor(buf63, (8, 128, 768), (98304, 768, 1), 0); del buf63  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_21, add_6, hidden_states_23], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf67, arg54_1, buf60, arg55_1, arg56_1, 1024, 768, stream=stream0)
            del arg54_1
            del arg55_1
            del arg56_1
            buf68 = reinterpret_tensor(buf60, (1024, 768), (768, 1), 0); del buf60  # reuse
            # Topologically Sorted Source Nodes: [linear_18], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg58_1, reinterpret_tensor(buf67, (1024, 768), (768, 1), 0), reinterpret_tensor(arg57_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf68)
            del arg57_1
            del arg58_1
            buf69 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg60_1, reinterpret_tensor(buf67, (1024, 768), (768, 1), 0), reinterpret_tensor(arg59_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf69)
            del arg59_1
            del arg60_1
            buf70 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg62_1, reinterpret_tensor(buf67, (1024, 768), (768, 1), 0), reinterpret_tensor(arg61_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf70)
            del arg61_1
            del arg62_1
            # Topologically Sorted Source Nodes: [tensor, getitem_2, expand_1, expanded_mask, inverted_mask, to_1, extended_attention_mask, linear_18, view_9, query_layer_3, linear_19, view_10, key_layer_3, linear_20, view_11, value_layer_3, attn_output_9], Original ATen: [aten.lift_fresh, aten.unsqueeze, aten.expand, aten._to_copy, aten.sub, aten.masked_fill, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
            buf72 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf68, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf69, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf70, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf71, (8, 12, 128, 128), (16384, 0, 128, 1), 0), False)
            del buf68
            del buf69
            buf73 = buf72[0]
            assert_size_stride(buf73, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf73, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf72
            buf77 = buf70; del buf70  # reuse
            # Topologically Sorted Source Nodes: [attn_output_10, attn_output_11, hidden_states_24], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf73, (1024, 768), (768, 1), 0), reinterpret_tensor(arg63_1, (768, 768), (1, 768), 0), out=buf77)
            del arg63_1
            del buf73
            buf81 = reinterpret_tensor(buf77, (8, 128, 768), (98304, 768, 1), 0); del buf77  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_24, add_7, hidden_states_26], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf81, arg64_1, buf67, arg65_1, arg66_1, 1024, 768, stream=stream0)
            del arg64_1
            del arg65_1
            del arg66_1
            del buf67
            buf82 = reinterpret_tensor(buf62, (1024, 3072), (3072, 1), 0); del buf62  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_27], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf81, (1024, 768), (768, 1), 0), reinterpret_tensor(arg67_1, (768, 3072), (1, 768), 0), out=buf82)
            del arg67_1
            buf83 = reinterpret_tensor(buf82, (8, 128, 3072), (393216, 3072, 1), 0); del buf82  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_27, hidden_states_28], Original ATen: [aten.addmm, aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_3.run(buf83, arg68_1, 3145728, stream=stream0)
            del arg68_1
            buf84 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_27, hidden_states_28, hidden_states_29], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            extern_kernels.mm(reinterpret_tensor(buf83, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg69_1, (3072, 768), (1, 3072), 0), out=buf84)
            del arg69_1
            buf88 = reinterpret_tensor(buf84, (8, 128, 768), (98304, 768, 1), 0); del buf84  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_29, add_8, hidden_states_31], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf88, arg70_1, buf81, arg71_1, arg72_1, 1024, 768, stream=stream0)
            del arg70_1
            del arg71_1
            del arg72_1
            buf89 = reinterpret_tensor(buf81, (1024, 768), (768, 1), 0); del buf81  # reuse
            # Topologically Sorted Source Nodes: [linear_24], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg74_1, reinterpret_tensor(buf88, (1024, 768), (768, 1), 0), reinterpret_tensor(arg73_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf89)
            del arg73_1
            del arg74_1
            buf90 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_25], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg76_1, reinterpret_tensor(buf88, (1024, 768), (768, 1), 0), reinterpret_tensor(arg75_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf90)
            del arg75_1
            del arg76_1
            buf91 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg78_1, reinterpret_tensor(buf88, (1024, 768), (768, 1), 0), reinterpret_tensor(arg77_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf91)
            del arg77_1
            del arg78_1
            # Topologically Sorted Source Nodes: [tensor, getitem_2, expand_1, expanded_mask, inverted_mask, to_1, extended_attention_mask, linear_24, view_12, query_layer_4, linear_25, view_13, key_layer_4, linear_26, view_14, value_layer_4, attn_output_12], Original ATen: [aten.lift_fresh, aten.unsqueeze, aten.expand, aten._to_copy, aten.sub, aten.masked_fill, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
            buf93 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf89, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf90, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf91, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf92, (8, 12, 128, 128), (16384, 0, 128, 1), 0), False)
            del buf89
            del buf90
            buf94 = buf93[0]
            assert_size_stride(buf94, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf94, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf93
            buf98 = buf91; del buf91  # reuse
            # Topologically Sorted Source Nodes: [attn_output_13, attn_output_14, hidden_states_32], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf94, (1024, 768), (768, 1), 0), reinterpret_tensor(arg79_1, (768, 768), (1, 768), 0), out=buf98)
            del arg79_1
            del buf94
            buf102 = reinterpret_tensor(buf98, (8, 128, 768), (98304, 768, 1), 0); del buf98  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_32, add_9, hidden_states_34], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf102, arg80_1, buf88, arg81_1, arg82_1, 1024, 768, stream=stream0)
            del arg80_1
            del arg81_1
            del arg82_1
            del buf88
            buf103 = reinterpret_tensor(buf83, (1024, 3072), (3072, 1), 0); del buf83  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_35], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf102, (1024, 768), (768, 1), 0), reinterpret_tensor(arg83_1, (768, 3072), (1, 768), 0), out=buf103)
            del arg83_1
            buf104 = reinterpret_tensor(buf103, (8, 128, 3072), (393216, 3072, 1), 0); del buf103  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_36], Original ATen: [aten.addmm, aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_3.run(buf104, arg84_1, 3145728, stream=stream0)
            del arg84_1
            buf105 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_36, hidden_states_37], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            extern_kernels.mm(reinterpret_tensor(buf104, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg85_1, (3072, 768), (1, 3072), 0), out=buf105)
            del arg85_1
            del buf104
            buf109 = reinterpret_tensor(buf105, (8, 128, 768), (98304, 768, 1), 0); del buf105  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_37, add_10, hidden_states_39], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf109, arg86_1, buf102, arg87_1, arg88_1, 1024, 768, stream=stream0)
            del arg86_1
            del arg87_1
            del arg88_1
            buf110 = reinterpret_tensor(buf102, (1024, 768), (768, 1), 0); del buf102  # reuse
            # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg90_1, reinterpret_tensor(buf109, (1024, 768), (768, 1), 0), reinterpret_tensor(arg89_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf110)
            del arg89_1
            del arg90_1
            buf111 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_31], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg92_1, reinterpret_tensor(buf109, (1024, 768), (768, 1), 0), reinterpret_tensor(arg91_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf111)
            del arg91_1
            del arg92_1
            buf112 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg94_1, reinterpret_tensor(buf109, (1024, 768), (768, 1), 0), reinterpret_tensor(arg93_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf112)
            del arg93_1
            del arg94_1
            buf113 = buf92; del buf92  # reuse
            buf134 = buf71; del buf71  # reuse
            buf155 = buf50; del buf50  # reuse
            buf176 = buf29; del buf29  # reuse
            buf197 = empty_strided_cuda((8, 1, 128, 128), (16384, 0, 128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [tensor, getitem_2, expand_1, expanded_mask, inverted_mask, to_1, extended_attention_mask, linear_30, view_15, query_layer_5, linear_31, view_16, key_layer_5, linear_32, view_17, value_layer_5, attn_output_15, linear_36, view_18, query_layer_6, linear_37, view_19, key_layer_6, linear_38, view_20, value_layer_6, attn_output_18, linear_42, view_21, query_layer_7, linear_43, view_22, key_layer_7, linear_44, view_23, value_layer_7, attn_output_21, linear_48, view_24, query_layer_8, linear_49, view_25, key_layer_8, linear_50, view_26, value_layer_8, attn_output_24, linear_54, view_27, query_layer_9, linear_55, view_28, key_layer_9, linear_56, view_29, value_layer_9, attn_output_27], Original ATen: [aten.lift_fresh, aten.unsqueeze, aten.expand, aten._to_copy, aten.sub, aten.masked_fill, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_lift_fresh_masked_fill_sub_transpose_unsqueeze_view_1.run(arg8_1, buf113, buf134, buf155, buf176, buf197, 131072, stream=stream0)
            # Topologically Sorted Source Nodes: [tensor, getitem_2, expand_1, expanded_mask, inverted_mask, to_1, extended_attention_mask, linear_30, view_15, query_layer_5, linear_31, view_16, key_layer_5, linear_32, view_17, value_layer_5, attn_output_15], Original ATen: [aten.lift_fresh, aten.unsqueeze, aten.expand, aten._to_copy, aten.sub, aten.masked_fill, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
            buf114 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf110, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf111, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf112, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf113, (8, 12, 128, 128), (16384, 0, 128, 1), 0), False)
            del buf110
            del buf111
            del buf113
            buf115 = buf114[0]
            assert_size_stride(buf115, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf115, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf114
            buf119 = buf112; del buf112  # reuse
            # Topologically Sorted Source Nodes: [attn_output_16, attn_output_17, hidden_states_40], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf115, (1024, 768), (768, 1), 0), reinterpret_tensor(arg95_1, (768, 768), (1, 768), 0), out=buf119)
            del arg95_1
            del buf115
            buf123 = reinterpret_tensor(buf119, (8, 128, 768), (98304, 768, 1), 0); del buf119  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_40, add_11, hidden_states_42], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf123, arg96_1, buf109, arg97_1, arg98_1, 1024, 768, stream=stream0)
            del arg96_1
            del arg97_1
            del arg98_1
            del buf109
            buf124 = empty_strided_cuda((1024, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_43], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf123, (1024, 768), (768, 1), 0), reinterpret_tensor(arg99_1, (768, 3072), (1, 768), 0), out=buf124)
            del arg99_1
            buf125 = reinterpret_tensor(buf124, (8, 128, 3072), (393216, 3072, 1), 0); del buf124  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_43, hidden_states_44], Original ATen: [aten.addmm, aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_3.run(buf125, arg100_1, 3145728, stream=stream0)
            del arg100_1
            buf126 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_43, hidden_states_44, hidden_states_45], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            extern_kernels.mm(reinterpret_tensor(buf125, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg101_1, (3072, 768), (1, 3072), 0), out=buf126)
            del arg101_1
            buf130 = reinterpret_tensor(buf126, (8, 128, 768), (98304, 768, 1), 0); del buf126  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_45, add_12, hidden_states_47], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf130, arg102_1, buf123, arg103_1, arg104_1, 1024, 768, stream=stream0)
            del arg102_1
            del arg103_1
            del arg104_1
            buf131 = reinterpret_tensor(buf123, (1024, 768), (768, 1), 0); del buf123  # reuse
            # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg106_1, reinterpret_tensor(buf130, (1024, 768), (768, 1), 0), reinterpret_tensor(arg105_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf131)
            del arg105_1
            del arg106_1
            buf132 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_37], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg108_1, reinterpret_tensor(buf130, (1024, 768), (768, 1), 0), reinterpret_tensor(arg107_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf132)
            del arg107_1
            del arg108_1
            buf133 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_38], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg110_1, reinterpret_tensor(buf130, (1024, 768), (768, 1), 0), reinterpret_tensor(arg109_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf133)
            del arg109_1
            del arg110_1
            # Topologically Sorted Source Nodes: [tensor, getitem_2, expand_1, expanded_mask, inverted_mask, to_1, extended_attention_mask, linear_36, view_18, query_layer_6, linear_37, view_19, key_layer_6, linear_38, view_20, value_layer_6, attn_output_18], Original ATen: [aten.lift_fresh, aten.unsqueeze, aten.expand, aten._to_copy, aten.sub, aten.masked_fill, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
            buf135 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf131, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf132, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf133, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf134, (8, 12, 128, 128), (16384, 0, 128, 1), 0), False)
            del buf131
            del buf132
            del buf134
            buf136 = buf135[0]
            assert_size_stride(buf136, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf136, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf135
            buf140 = buf133; del buf133  # reuse
            # Topologically Sorted Source Nodes: [attn_output_19, attn_output_20, hidden_states_48], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf136, (1024, 768), (768, 1), 0), reinterpret_tensor(arg111_1, (768, 768), (1, 768), 0), out=buf140)
            del arg111_1
            del buf136
            buf144 = reinterpret_tensor(buf140, (8, 128, 768), (98304, 768, 1), 0); del buf140  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_48, add_13, hidden_states_50], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf144, arg112_1, buf130, arg113_1, arg114_1, 1024, 768, stream=stream0)
            del arg112_1
            del arg113_1
            del arg114_1
            del buf130
            buf145 = reinterpret_tensor(buf125, (1024, 3072), (3072, 1), 0); del buf125  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_51], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf144, (1024, 768), (768, 1), 0), reinterpret_tensor(arg115_1, (768, 3072), (1, 768), 0), out=buf145)
            del arg115_1
            buf146 = reinterpret_tensor(buf145, (8, 128, 3072), (393216, 3072, 1), 0); del buf145  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_51, hidden_states_52], Original ATen: [aten.addmm, aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_3.run(buf146, arg116_1, 3145728, stream=stream0)
            del arg116_1
            buf147 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_51, hidden_states_52, hidden_states_53], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            extern_kernels.mm(reinterpret_tensor(buf146, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg117_1, (3072, 768), (1, 3072), 0), out=buf147)
            del arg117_1
            buf151 = reinterpret_tensor(buf147, (8, 128, 768), (98304, 768, 1), 0); del buf147  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_53, add_14, hidden_states_55], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf151, arg118_1, buf144, arg119_1, arg120_1, 1024, 768, stream=stream0)
            del arg118_1
            del arg119_1
            del arg120_1
            buf152 = reinterpret_tensor(buf144, (1024, 768), (768, 1), 0); del buf144  # reuse
            # Topologically Sorted Source Nodes: [linear_42], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg122_1, reinterpret_tensor(buf151, (1024, 768), (768, 1), 0), reinterpret_tensor(arg121_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf152)
            del arg121_1
            del arg122_1
            buf153 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_43], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg124_1, reinterpret_tensor(buf151, (1024, 768), (768, 1), 0), reinterpret_tensor(arg123_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf153)
            del arg123_1
            del arg124_1
            buf154 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg126_1, reinterpret_tensor(buf151, (1024, 768), (768, 1), 0), reinterpret_tensor(arg125_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf154)
            del arg125_1
            del arg126_1
            # Topologically Sorted Source Nodes: [tensor, getitem_2, expand_1, expanded_mask, inverted_mask, to_1, extended_attention_mask, linear_42, view_21, query_layer_7, linear_43, view_22, key_layer_7, linear_44, view_23, value_layer_7, attn_output_21], Original ATen: [aten.lift_fresh, aten.unsqueeze, aten.expand, aten._to_copy, aten.sub, aten.masked_fill, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
            buf156 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf152, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf153, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf154, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf155, (8, 12, 128, 128), (16384, 0, 128, 1), 0), False)
            del buf152
            del buf153
            del buf155
            buf157 = buf156[0]
            assert_size_stride(buf157, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf157, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf156
            buf161 = buf154; del buf154  # reuse
            # Topologically Sorted Source Nodes: [attn_output_22, attn_output_23, hidden_states_56], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf157, (1024, 768), (768, 1), 0), reinterpret_tensor(arg127_1, (768, 768), (1, 768), 0), out=buf161)
            del arg127_1
            del buf157
            buf165 = reinterpret_tensor(buf161, (8, 128, 768), (98304, 768, 1), 0); del buf161  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_56, add_15, hidden_states_58], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf165, arg128_1, buf151, arg129_1, arg130_1, 1024, 768, stream=stream0)
            del arg128_1
            del arg129_1
            del arg130_1
            del buf151
            buf166 = reinterpret_tensor(buf146, (1024, 3072), (3072, 1), 0); del buf146  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_59], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf165, (1024, 768), (768, 1), 0), reinterpret_tensor(arg131_1, (768, 3072), (1, 768), 0), out=buf166)
            del arg131_1
            buf167 = reinterpret_tensor(buf166, (8, 128, 3072), (393216, 3072, 1), 0); del buf166  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_59, hidden_states_60], Original ATen: [aten.addmm, aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_3.run(buf167, arg132_1, 3145728, stream=stream0)
            del arg132_1
            buf168 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_59, hidden_states_60, hidden_states_61], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            extern_kernels.mm(reinterpret_tensor(buf167, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg133_1, (3072, 768), (1, 3072), 0), out=buf168)
            del arg133_1
            buf172 = reinterpret_tensor(buf168, (8, 128, 768), (98304, 768, 1), 0); del buf168  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_61, add_16, hidden_states_63], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf172, arg134_1, buf165, arg135_1, arg136_1, 1024, 768, stream=stream0)
            del arg134_1
            del arg135_1
            del arg136_1
            buf173 = reinterpret_tensor(buf165, (1024, 768), (768, 1), 0); del buf165  # reuse
            # Topologically Sorted Source Nodes: [linear_48], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg138_1, reinterpret_tensor(buf172, (1024, 768), (768, 1), 0), reinterpret_tensor(arg137_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf173)
            del arg137_1
            del arg138_1
            buf174 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_49], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg140_1, reinterpret_tensor(buf172, (1024, 768), (768, 1), 0), reinterpret_tensor(arg139_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf174)
            del arg139_1
            del arg140_1
            buf175 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_50], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg142_1, reinterpret_tensor(buf172, (1024, 768), (768, 1), 0), reinterpret_tensor(arg141_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf175)
            del arg141_1
            del arg142_1
            # Topologically Sorted Source Nodes: [tensor, getitem_2, expand_1, expanded_mask, inverted_mask, to_1, extended_attention_mask, linear_48, view_24, query_layer_8, linear_49, view_25, key_layer_8, linear_50, view_26, value_layer_8, attn_output_24], Original ATen: [aten.lift_fresh, aten.unsqueeze, aten.expand, aten._to_copy, aten.sub, aten.masked_fill, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
            buf177 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf173, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf174, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf175, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf176, (8, 12, 128, 128), (16384, 0, 128, 1), 0), False)
            del buf173
            del buf174
            buf178 = buf177[0]
            assert_size_stride(buf178, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf178, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf177
            buf182 = buf175; del buf175  # reuse
            # Topologically Sorted Source Nodes: [attn_output_25, attn_output_26, hidden_states_64], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf178, (1024, 768), (768, 1), 0), reinterpret_tensor(arg143_1, (768, 768), (1, 768), 0), out=buf182)
            del arg143_1
            del buf178
            buf186 = reinterpret_tensor(buf182, (8, 128, 768), (98304, 768, 1), 0); del buf182  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_64, add_17, hidden_states_66], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf186, arg144_1, buf172, arg145_1, arg146_1, 1024, 768, stream=stream0)
            del arg144_1
            del arg145_1
            del arg146_1
            del buf172
            buf187 = reinterpret_tensor(buf167, (1024, 3072), (3072, 1), 0); del buf167  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_67], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf186, (1024, 768), (768, 1), 0), reinterpret_tensor(arg147_1, (768, 3072), (1, 768), 0), out=buf187)
            del arg147_1
            buf188 = reinterpret_tensor(buf187, (8, 128, 3072), (393216, 3072, 1), 0); del buf187  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_67, hidden_states_68], Original ATen: [aten.addmm, aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_3.run(buf188, arg148_1, 3145728, stream=stream0)
            del arg148_1
            buf189 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_67, hidden_states_68, hidden_states_69], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            extern_kernels.mm(reinterpret_tensor(buf188, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg149_1, (3072, 768), (1, 3072), 0), out=buf189)
            del arg149_1
            buf193 = reinterpret_tensor(buf189, (8, 128, 768), (98304, 768, 1), 0); del buf189  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_69, add_18, hidden_states_71], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf193, arg150_1, buf186, arg151_1, arg152_1, 1024, 768, stream=stream0)
            del arg150_1
            del arg151_1
            del arg152_1
            buf194 = reinterpret_tensor(buf186, (1024, 768), (768, 1), 0); del buf186  # reuse
            # Topologically Sorted Source Nodes: [linear_54], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg154_1, reinterpret_tensor(buf193, (1024, 768), (768, 1), 0), reinterpret_tensor(arg153_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf194)
            del arg153_1
            del arg154_1
            buf195 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_55], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg156_1, reinterpret_tensor(buf193, (1024, 768), (768, 1), 0), reinterpret_tensor(arg155_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf195)
            del arg155_1
            del arg156_1
            buf196 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_56], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg158_1, reinterpret_tensor(buf193, (1024, 768), (768, 1), 0), reinterpret_tensor(arg157_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf196)
            del arg157_1
            del arg158_1
            # Topologically Sorted Source Nodes: [tensor, getitem_2, expand_1, expanded_mask, inverted_mask, to_1, extended_attention_mask, linear_54, view_27, query_layer_9, linear_55, view_28, key_layer_9, linear_56, view_29, value_layer_9, attn_output_27], Original ATen: [aten.lift_fresh, aten.unsqueeze, aten.expand, aten._to_copy, aten.sub, aten.masked_fill, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
            buf198 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf194, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf195, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf196, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf197, (8, 12, 128, 128), (16384, 0, 128, 1), 0), False)
            del buf194
            del buf195
            buf199 = buf198[0]
            assert_size_stride(buf199, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf199, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf198
            buf203 = buf196; del buf196  # reuse
            # Topologically Sorted Source Nodes: [attn_output_28, attn_output_29, hidden_states_72], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf199, (1024, 768), (768, 1), 0), reinterpret_tensor(arg159_1, (768, 768), (1, 768), 0), out=buf203)
            del arg159_1
            del buf199
            buf207 = reinterpret_tensor(buf203, (8, 128, 768), (98304, 768, 1), 0); del buf203  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_72, add_19, hidden_states_74], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf207, arg160_1, buf193, arg161_1, arg162_1, 1024, 768, stream=stream0)
            del arg160_1
            del arg161_1
            del arg162_1
            del buf193
            buf208 = reinterpret_tensor(buf188, (1024, 3072), (3072, 1), 0); del buf188  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_75], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf207, (1024, 768), (768, 1), 0), reinterpret_tensor(arg163_1, (768, 3072), (1, 768), 0), out=buf208)
            del arg163_1
            buf209 = reinterpret_tensor(buf208, (8, 128, 3072), (393216, 3072, 1), 0); del buf208  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_75, hidden_states_76], Original ATen: [aten.addmm, aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_3.run(buf209, arg164_1, 3145728, stream=stream0)
            del arg164_1
            buf210 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_75, hidden_states_76, hidden_states_77], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            extern_kernels.mm(reinterpret_tensor(buf209, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg165_1, (3072, 768), (1, 3072), 0), out=buf210)
            del arg165_1
            buf214 = reinterpret_tensor(buf210, (8, 128, 768), (98304, 768, 1), 0); del buf210  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_77, add_20, hidden_states_79], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf214, arg166_1, buf207, arg167_1, arg168_1, 1024, 768, stream=stream0)
            del arg166_1
            del arg167_1
            del arg168_1
            buf215 = reinterpret_tensor(buf207, (1024, 768), (768, 1), 0); del buf207  # reuse
            # Topologically Sorted Source Nodes: [linear_60], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg170_1, reinterpret_tensor(buf214, (1024, 768), (768, 1), 0), reinterpret_tensor(arg169_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf215)
            del arg169_1
            del arg170_1
            buf216 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_61], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg172_1, reinterpret_tensor(buf214, (1024, 768), (768, 1), 0), reinterpret_tensor(arg171_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf216)
            del arg171_1
            del arg172_1
            buf217 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_62], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg174_1, reinterpret_tensor(buf214, (1024, 768), (768, 1), 0), reinterpret_tensor(arg173_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf217)
            del arg173_1
            del arg174_1
            buf218 = buf197; del buf197  # reuse
            buf239 = buf176; del buf176  # reuse
            # Topologically Sorted Source Nodes: [tensor, getitem_2, expand_1, expanded_mask, inverted_mask, to_1, extended_attention_mask, linear_60, view_30, query_layer_10, linear_61, view_31, key_layer_10, linear_62, view_32, value_layer_10, attn_output_30, linear_66, view_33, query_layer_11, linear_67, view_34, key_layer_11, linear_68, view_35, value_layer_11, attn_output_33], Original ATen: [aten.lift_fresh, aten.unsqueeze, aten.expand, aten._to_copy, aten.sub, aten.masked_fill, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_lift_fresh_masked_fill_sub_transpose_unsqueeze_view_4.run(arg8_1, buf218, buf239, 131072, stream=stream0)
            del arg8_1
            # Topologically Sorted Source Nodes: [tensor, getitem_2, expand_1, expanded_mask, inverted_mask, to_1, extended_attention_mask, linear_60, view_30, query_layer_10, linear_61, view_31, key_layer_10, linear_62, view_32, value_layer_10, attn_output_30], Original ATen: [aten.lift_fresh, aten.unsqueeze, aten.expand, aten._to_copy, aten.sub, aten.masked_fill, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
            buf219 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf215, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf216, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf217, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf218, (8, 12, 128, 128), (16384, 0, 128, 1), 0), False)
            del buf215
            del buf216
            del buf218
            buf220 = buf219[0]
            assert_size_stride(buf220, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf220, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf219
            buf224 = buf217; del buf217  # reuse
            # Topologically Sorted Source Nodes: [attn_output_31, attn_output_32, hidden_states_80], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf220, (1024, 768), (768, 1), 0), reinterpret_tensor(arg175_1, (768, 768), (1, 768), 0), out=buf224)
            del arg175_1
            del buf220
            buf228 = reinterpret_tensor(buf224, (8, 128, 768), (98304, 768, 1), 0); del buf224  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_80, add_21, hidden_states_82], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf228, arg176_1, buf214, arg177_1, arg178_1, 1024, 768, stream=stream0)
            del arg176_1
            del arg177_1
            del arg178_1
            del buf214
            buf229 = reinterpret_tensor(buf209, (1024, 3072), (3072, 1), 0); del buf209  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_83], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf228, (1024, 768), (768, 1), 0), reinterpret_tensor(arg179_1, (768, 3072), (1, 768), 0), out=buf229)
            del arg179_1
            buf230 = reinterpret_tensor(buf229, (8, 128, 3072), (393216, 3072, 1), 0); del buf229  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_83, hidden_states_84], Original ATen: [aten.addmm, aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_3.run(buf230, arg180_1, 3145728, stream=stream0)
            del arg180_1
            buf231 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_83, hidden_states_84, hidden_states_85], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            extern_kernels.mm(reinterpret_tensor(buf230, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg181_1, (3072, 768), (1, 3072), 0), out=buf231)
            del arg181_1
            buf235 = reinterpret_tensor(buf231, (8, 128, 768), (98304, 768, 1), 0); del buf231  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_85, add_22, hidden_states_87], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf235, arg182_1, buf228, arg183_1, arg184_1, 1024, 768, stream=stream0)
            del arg182_1
            del arg183_1
            del arg184_1
            buf236 = reinterpret_tensor(buf228, (1024, 768), (768, 1), 0); del buf228  # reuse
            # Topologically Sorted Source Nodes: [linear_66], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg186_1, reinterpret_tensor(buf235, (1024, 768), (768, 1), 0), reinterpret_tensor(arg185_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf236)
            del arg185_1
            del arg186_1
            buf237 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_67], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg188_1, reinterpret_tensor(buf235, (1024, 768), (768, 1), 0), reinterpret_tensor(arg187_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf237)
            del arg187_1
            del arg188_1
            buf238 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [linear_68], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg190_1, reinterpret_tensor(buf235, (1024, 768), (768, 1), 0), reinterpret_tensor(arg189_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf238)
            del arg189_1
            del arg190_1
            # Topologically Sorted Source Nodes: [tensor, getitem_2, expand_1, expanded_mask, inverted_mask, to_1, extended_attention_mask, linear_66, view_33, query_layer_11, linear_67, view_34, key_layer_11, linear_68, view_35, value_layer_11, attn_output_33], Original ATen: [aten.lift_fresh, aten.unsqueeze, aten.expand, aten._to_copy, aten.sub, aten.masked_fill, aten.view, aten.transpose, aten._scaled_dot_product_efficient_attention]
            buf240 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf236, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf237, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf238, (8, 12, 128, 64), (98304, 64, 768, 1), 0), reinterpret_tensor(buf239, (8, 12, 128, 128), (16384, 0, 128, 1), 0), False)
            del buf236
            del buf237
            del buf239
            buf241 = buf240[0]
            assert_size_stride(buf241, (8, 12, 128, 64), (98304, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf241, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf240
            buf245 = buf238; del buf238  # reuse
            # Topologically Sorted Source Nodes: [attn_output_34, attn_output_35, hidden_states_88], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf241, (1024, 768), (768, 1), 0), reinterpret_tensor(arg191_1, (768, 768), (1, 768), 0), out=buf245)
            del arg191_1
            del buf241
            buf249 = reinterpret_tensor(buf245, (8, 128, 768), (98304, 768, 1), 0); del buf245  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_88, add_23, hidden_states_90], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf249, arg192_1, buf235, arg193_1, arg194_1, 1024, 768, stream=stream0)
            del arg192_1
            del arg193_1
            del arg194_1
            del buf235
            buf250 = reinterpret_tensor(buf230, (1024, 3072), (3072, 1), 0); del buf230  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_91], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf249, (1024, 768), (768, 1), 0), reinterpret_tensor(arg195_1, (768, 3072), (1, 768), 0), out=buf250)
            del arg195_1
            buf251 = reinterpret_tensor(buf250, (8, 128, 3072), (393216, 3072, 1), 0); del buf250  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_91, hidden_states_92], Original ATen: [aten.addmm, aten.view, aten.gelu]
            stream0 = get_raw_stream(0)
            triton_poi_fused_addmm_gelu_view_3.run(buf251, arg196_1, 3145728, stream=stream0)
            del arg196_1
            buf252 = empty_strided_cuda((1024, 768), (768, 1), torch.float32)
            # Topologically Sorted Source Nodes: [hidden_states_91, hidden_states_92, hidden_states_93], Original ATen: [aten.addmm, aten.view, aten.gelu, aten.t]
            extern_kernels.mm(reinterpret_tensor(buf251, (1024, 3072), (3072, 1), 0), reinterpret_tensor(arg197_1, (3072, 768), (1, 3072), 0), out=buf252)
            del arg197_1
            del buf251
            buf256 = reinterpret_tensor(buf252, (8, 128, 768), (98304, 768, 1), 0); del buf252  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_93, add_24, hidden_states_95], Original ATen: [aten.addmm, aten.view, aten.add, aten.native_layer_norm]
            stream0 = get_raw_stream(0)
            triton_per_fused_add_addmm_native_layer_norm_view_2.run(buf256, arg198_1, buf249, arg199_1, arg200_1, 1024, 768, stream=stream0)
            del arg198_1
            del arg199_1
            del arg200_1
            del buf249
        return (buf256, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg2_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg3_1 = rand_strided((30522, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((2, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((512, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    arg9_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)


# ===== inductor generated file: /tmp/cnnbench-bert-fold-weaagmnf/repeat_02/base/attempt_01/torchinductor/tmp5mgaaf9a/3o/c3o3cagdzec6vdr53idsfidv322vhrsgvavfqz4humgbetninzuc.py =====

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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_addmm_native_layer_norm_view_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 12592128}}
)
@triton.jit
def triton_per_fused_add_addmm_native_layer_norm_view_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp28 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
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
    tmp24 = 1e-12
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(in_out_ptr0 + (r0_1 + 768*x0), tmp31, r0_mask & xmask)


# ===== inductor generated file: /tmp/cnnbench-bert-fold-weaagmnf/repeat_02/base/attempt_01/torchinductor/tmp5mgaaf9a/bc/cbcg7yrdae5gcculs4k4kh75h33f4hiihrpuew2o7doyzs2jpeqy.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_lift_fresh_masked_fill_sub_transpose_unsqueeze_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 5251072}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_lift_fresh_masked_fill_sub_transpose_unsqueeze_view_1(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 128)
    x2 = xindex // 16384
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x2), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp2 - tmp1
    tmp4 = (tmp3 != 0)
    tmp5 = -3.4028234663852886e+38
    tmp6 = tl.where(tmp4, tmp5, tmp3)
    tl.store(out_ptr0 + (x3), tmp6, None)
    tl.store(out_ptr1 + (x3), tmp6, None)
    tl.store(out_ptr2 + (x3), tmp6, None)
    tl.store(out_ptr3 + (x3), tmp6, None)
    tl.store(out_ptr4 + (x3), tmp6, None)


# ===== inductor generated file: /tmp/cnnbench-bert-fold-weaagmnf/repeat_02/base/attempt_01/torchinductor/tmp5mgaaf9a/js/cjsxh5oszam3v7uzmmlndzn7wzr6xe7iv2kei52srdd625ijoylt.py =====

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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_embedding_expand_native_layer_norm_slice_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_embedding_expand_native_layer_norm_slice_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    x3 = xindex
    r0_2 = r0_index
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr6 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.load(in_ptr7 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.full([XBLOCK, R0_BLOCK], 30522, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 30522)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 30522")
    tmp6 = tl.load(in_ptr1 + (r0_2 + 768*tmp4), r0_mask & xmask, other=0.0)
    tmp8 = tl.full([XBLOCK, R0_BLOCK], 2, tl.int32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp7 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp7)
    tl.device_assert(((0 <= tmp11) & (tmp11 < 2)) | ~(xmask), "index out of bounds: 0 <= tmp11 < 2")
    tmp13 = tl.load(in_ptr3 + (r0_2 + 768*tmp11), r0_mask & xmask, other=0.0)
    tmp14 = tmp6 + tmp13
    tmp16 = tl.full([XBLOCK, R0_BLOCK], 512, tl.int32)
    tmp17 = tmp15 + tmp16
    tmp18 = tmp15 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp15)
    tl.device_assert(((0 <= tmp19) & (tmp19 < 512)) | ~(xmask), "index out of bounds: 0 <= tmp19 < 512")
    tmp21 = tl.load(in_ptr5 + (r0_2 + 768*tmp19), r0_mask & xmask, other=0.0)
    tmp22 = tmp14 + tmp21
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, R0_BLOCK])
    tmp25 = tl.where(r0_mask & xmask, tmp23, 0)
    tmp26 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
    tmp28 = tl.where(r0_mask & xmask, tmp26, 0)
    tmp29 = tl.sum(tmp28, 1)[:, None].to(tl.float32)
    tmp30 = tl.full([XBLOCK, 1], 768, tl.int32)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = (tmp29 / tmp31)
    tmp33 = tmp23 - tmp32
    tmp34 = tmp33 * tmp33
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK, R0_BLOCK])
    tmp37 = tl.where(r0_mask & xmask, tmp35, 0)
    tmp38 = tl.sum(tmp37, 1)[:, None].to(tl.float32)
    tmp39 = tmp22 - tmp32
    tmp40 = 768.0
    tmp41 = (tmp38 / tmp40)
    tmp42 = 1e-12
    tmp43 = tmp41 + tmp42
    tmp44 = libdevice.rsqrt(tmp43)
    tmp45 = tmp39 * tmp44
    tmp47 = tmp45 * tmp46
    tmp49 = tmp47 + tmp48
    tl.store(in_out_ptr0 + (r0_2 + 768*x3), tmp49, r0_mask & xmask)


# ===== inductor generated file: /tmp/cnnbench-bert-fold-weaagmnf/repeat_02/base/attempt_01/torchinductor/tmp5mgaaf9a/m5/cm5rpuin46rzdtknz5aqqo3766n3gtq6mrme23rqyrl6joqahvar.debug/fx_graph_readable.py =====
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "i64[8, 128]", arg1_1: "i64[1, 512]", arg2_1: "i64[1, 512]", arg3_1: "f32[30522, 768]", arg4_1: "f32[2, 768]", arg5_1: "f32[512, 768]", arg6_1: "f32[768]", arg7_1: "f32[768]", arg8_1: "i64[8, 128]", arg9_1: "f32[768, 768]", arg10_1: "f32[768]", arg11_1: "f32[768, 768]", arg12_1: "f32[768]", arg13_1: "f32[768, 768]", arg14_1: "f32[768]", arg15_1: "f32[768, 768]", arg16_1: "f32[768]", arg17_1: "f32[768]", arg18_1: "f32[768]", arg19_1: "f32[3072, 768]", arg20_1: "f32[3072]", arg21_1: "f32[768, 3072]", arg22_1: "f32[768]", arg23_1: "f32[768]", arg24_1: "f32[768]", arg25_1: "f32[768, 768]", arg26_1: "f32[768]", arg27_1: "f32[768, 768]", arg28_1: "f32[768]", arg29_1: "f32[768, 768]", arg30_1: "f32[768]", arg31_1: "f32[768, 768]", arg32_1: "f32[768]", arg33_1: "f32[768]", arg34_1: "f32[768]", arg35_1: "f32[3072, 768]", arg36_1: "f32[3072]", arg37_1: "f32[768, 3072]", arg38_1: "f32[768]", arg39_1: "f32[768]", arg40_1: "f32[768]", arg41_1: "f32[768, 768]", arg42_1: "f32[768]", arg43_1: "f32[768, 768]", arg44_1: "f32[768]", arg45_1: "f32[768, 768]", arg46_1: "f32[768]", arg47_1: "f32[768, 768]", arg48_1: "f32[768]", arg49_1: "f32[768]", arg50_1: "f32[768]", arg51_1: "f32[3072, 768]", arg52_1: "f32[3072]", arg53_1: "f32[768, 3072]", arg54_1: "f32[768]", arg55_1: "f32[768]", arg56_1: "f32[768]", arg57_1: "f32[768, 768]", arg58_1: "f32[768]", arg59_1: "f32[768, 768]", arg60_1: "f32[768]", arg61_1: "f32[768, 768]", arg62_1: "f32[768]", arg63_1: "f32[768, 768]", arg64_1: "f32[768]", arg65_1: "f32[768]", arg66_1: "f32[768]", arg67_1: "f32[3072, 768]", arg68_1: "f32[3072]", arg69_1: "f32[768, 3072]", arg70_1: "f32[768]", arg71_1: "f32[768]", arg72_1: "f32[768]", arg73_1: "f32[768, 768]", arg74_1: "f32[768]", arg75_1: "f32[768, 768]", arg76_1: "f32[768]", arg77_1: "f32[768, 768]", arg78_1: "f32[768]", arg79_1: "f32[768, 768]", arg80_1: "f32[768]", arg81_1: "f32[768]", arg82_1: "f32[768]", arg83_1: "f32[3072, 768]", arg84_1: "f32[3072]", arg85_1: "f32[768, 3072]", arg86_1: "f32[768]", arg87_1: "f32[768]", arg88_1: "f32[768]", arg89_1: "f32[768, 768]", arg90_1: "f32[768]", arg91_1: "f32[768, 768]", arg92_1: "f32[768]", arg93_1: "f32[768, 768]", arg94_1: "f32[768]", arg95_1: "f32[768, 768]", arg96_1: "f32[768]", arg97_1: "f32[768]", arg98_1: "f32[768]", arg99_1: "f32[3072, 768]", arg100_1: "f32[3072]", arg101_1: "f32[768, 3072]", arg102_1: "f32[768]", arg103_1: "f32[768]", arg104_1: "f32[768]", arg105_1: "f32[768, 768]", arg106_1: "f32[768]", arg107_1: "f32[768, 768]", arg108_1: "f32[768]", arg109_1: "f32[768, 768]", arg110_1: "f32[768]", arg111_1: "f32[768, 768]", arg112_1: "f32[768]", arg113_1: "f32[768]", arg114_1: "f32[768]", arg115_1: "f32[3072, 768]", arg116_1: "f32[3072]", arg117_1: "f32[768, 3072]", arg118_1: "f32[768]", arg119_1: "f32[768]", arg120_1: "f32[768]", arg121_1: "f32[768, 768]", arg122_1: "f32[768]", arg123_1: "f32[768, 768]", arg124_1: "f32[768]", arg125_1: "f32[768, 768]", arg126_1: "f32[768]", arg127_1: "f32[768, 768]", arg128_1: "f32[768]", arg129_1: "f32[768]", arg130_1: "f32[768]", arg131_1: "f32[3072, 768]", arg132_1: "f32[3072]", arg133_1: "f32[768, 3072]", arg134_1: "f32[768]", arg135_1: "f32[768]", arg136_1: "f32[768]", arg137_1: "f32[768, 768]", arg138_1: "f32[768]", arg139_1: "f32[768, 768]", arg140_1: "f32[768]", arg141_1: "f32[768, 768]", arg142_1: "f32[768]", arg143_1: "f32[768, 768]", arg144_1: "f32[768]", arg145_1: "f32[768]", arg146_1: "f32[768]", arg147_1: "f32[3072, 768]", arg148_1: "f32[3072]", arg149_1: "f32[768, 3072]", arg150_1: "f32[768]", arg151_1: "f32[768]", arg152_1: "f32[768]", arg153_1: "f32[768, 768]", arg154_1: "f32[768]", arg155_1: "f32[768, 768]", arg156_1: "f32[768]", arg157_1: "f32[768, 768]", arg158_1: "f32[768]", arg159_1: "f32[768, 768]", arg160_1: "f32[768]", arg161_1: "f32[768]", arg162_1: "f32[768]", arg163_1: "f32[3072, 768]", arg164_1: "f32[3072]", arg165_1: "f32[768, 3072]", arg166_1: "f32[768]", arg167_1: "f32[768]", arg168_1: "f32[768]", arg169_1: "f32[768, 768]", arg170_1: "f32[768]", arg171_1: "f32[768, 768]", arg172_1: "f32[768]", arg173_1: "f32[768, 768]", arg174_1: "f32[768]", arg175_1: "f32[768, 768]", arg176_1: "f32[768]", arg177_1: "f32[768]", arg178_1: "f32[768]", arg179_1: "f32[3072, 768]", arg180_1: "f32[3072]", arg181_1: "f32[768, 3072]", arg182_1: "f32[768]", arg183_1: "f32[768]", arg184_1: "f32[768]", arg185_1: "f32[768, 768]", arg186_1: "f32[768]", arg187_1: "f32[768, 768]", arg188_1: "f32[768]", arg189_1: "f32[768, 768]", arg190_1: "f32[768]", arg191_1: "f32[768, 768]", arg192_1: "f32[768]", arg193_1: "f32[768]", arg194_1: "f32[768]", arg195_1: "f32[3072, 768]", arg196_1: "f32[3072]", arg197_1: "f32[768, 3072]", arg198_1: "f32[768]", arg199_1: "f32[768]", arg200_1: "f32[768]", arg201_1: "f32[768, 768]", arg202_1: "f32[768]"):
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:929 in forward, code: buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
        slice_1: "i64[1, 128]" = torch.ops.aten.slice.Tensor(arg1_1, 1, 0, 128);  arg1_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:930 in forward, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
        expand: "i64[8, 128]" = torch.ops.aten.expand.default(slice_1, [8, 128]);  slice_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:166 in forward, code: position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
        slice_2: "i64[1, 128]" = torch.ops.aten.slice.Tensor(arg2_1, 1, 0, 128);  arg2_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding: "f32[8, 128, 768]" = torch.ops.aten.embedding.default(arg3_1, arg0_1, 0);  arg3_1 = arg0_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_1: "f32[8, 128, 768]" = torch.ops.aten.embedding.default(arg4_1, expand);  arg4_1 = expand = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:183 in forward, code: embeddings = inputs_embeds + token_type_embeddings
        add: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:185 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_2: "f32[1, 128, 768]" = torch.ops.aten.embedding.default(arg5_1, slice_2);  arg5_1 = slice_2 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:186 in forward, code: embeddings += position_embeddings
        add_1: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:187 in forward, code: embeddings = self.LayerNorm(embeddings)
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 128, 1]" = var_mean[0]
        getitem_1: "f32[8, 128, 1]" = var_mean[1];  var_mean = None
        add_2: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
        mul: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul, arg6_1);  mul = arg6_1 = None
        add_3: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_1, arg7_1);  mul_1 = arg7_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/modeling_attn_mask_utils.py:194 in _expand_mask, code: expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        unsqueeze: "i64[8, 1, 128]" = torch.ops.aten.unsqueeze.default(arg8_1, 1);  arg8_1 = None
        unsqueeze_1: "i64[8, 1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        expand_1: "i64[8, 1, 128, 128]" = torch.ops.aten.expand.default(unsqueeze_1, [8, 1, 128, 128]);  unsqueeze_1 = None
        convert_element_type: "f32[8, 1, 128, 128]" = torch.ops.prims.convert_element_type.default(expand_1, torch.float32);  expand_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/modeling_attn_mask_utils.py:196 in _expand_mask, code: inverted_mask = torch.tensor(1.0, dtype=dtype) - expanded_mask
        _tensor_constant0: "f32[]" = self._tensor_constant0;  _tensor_constant0 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        sub_1: "f32[8, 1, 128, 128]" = torch.ops.aten.sub.Tensor(full_default, convert_element_type);  full_default = convert_element_type = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/modeling_attn_mask_utils.py:198 in _expand_mask, code: return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
        convert_element_type_1: "b8[8, 1, 128, 128]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bool)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[8, 1, 128, 128]" = torch.ops.aten.where.self(convert_element_type_1, full_default_1, sub_1);  convert_element_type_1 = full_default_1 = sub_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:363 in forward, code: self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        view: "f32[1024, 768]" = torch.ops.aten.view.default(add_3, [1024, 768])
        permute: "f32[768, 768]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg10_1, view, permute);  arg10_1 = view = permute = None
        view_1: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm, [8, 128, 768]);  addmm = None
        view_2: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(view_1, [8, -1, 12, 64]);  view_1 = None
        permute_1: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:386 in forward, code: self.key(current_states)
        view_3: "f32[1024, 768]" = torch.ops.aten.view.default(add_3, [1024, 768])
        permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_1: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg12_1, view_3, permute_2);  arg12_1 = view_3 = permute_2 = None
        view_4: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_1, [8, 128, 768]);  addmm_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:387 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_5: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(view_4, [8, -1, 12, 64]);  view_4 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:388 in forward, code: .transpose(1, 2)
        permute_3: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:391 in forward, code: self.value(current_states)
        view_6: "f32[1024, 768]" = torch.ops.aten.view.default(add_3, [1024, 768])
        permute_4: "f32[768, 768]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_2: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg14_1, view_6, permute_4);  arg14_1 = view_6 = permute_4 = None
        view_7: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_2, [8, 128, 768]);  addmm_2 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:392 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_8: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(view_7, [8, -1, 12, 64]);  view_7 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:393 in forward, code: .transpose(1, 2)
        permute_5: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:412 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_2: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(where, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_1, permute_3, permute_5, expand_2, False);  permute_1 = permute_3 = permute_5 = expand_2 = None
        getitem_2: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention[0];  _scaled_dot_product_efficient_attention = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:421 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_6: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3]);  getitem_2 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:422 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_9: "f32[8, 128, 768]" = torch.ops.aten.view.default(permute_6, [8, 128, 768]);  permute_6 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:435 in forward, code: hidden_states = self.dense(hidden_states)
        view_10: "f32[1024, 768]" = torch.ops.aten.view.default(view_9, [1024, 768]);  view_9 = None
        permute_7: "f32[768, 768]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_3: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg16_1, view_10, permute_7);  arg16_1 = view_10 = permute_7 = None
        view_11: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_3, [8, 128, 768]);  addmm_3 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:437 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_4: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_11, add_3);  view_11 = add_3 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
        getitem_6: "f32[8, 128, 1]" = var_mean_1[0]
        getitem_7: "f32[8, 128, 1]" = var_mean_1[1];  var_mean_1 = None
        add_5: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_1: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_2: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_4, getitem_7);  add_4 = getitem_7 = None
        mul_2: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_3: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_2, arg17_1);  mul_2 = arg17_1 = None
        add_6: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_3, arg18_1);  mul_3 = arg18_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:511 in forward, code: hidden_states = self.dense(hidden_states)
        view_12: "f32[1024, 768]" = torch.ops.aten.view.default(add_6, [1024, 768])
        permute_8: "f32[768, 3072]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        addmm_4: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg20_1, view_12, permute_8);  arg20_1 = view_12 = permute_8 = None
        view_13: "f32[8, 128, 3072]" = torch.ops.aten.view.default(addmm_4, [8, 128, 3072]);  addmm_4 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:70 in forward, code: return self.act(input)
        mul_4: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_13, 0.5)
        mul_5: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_13, 0.7071067811865476);  view_13 = None
        erf: "f32[8, 128, 3072]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_7: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_4, add_7);  mul_4 = add_7 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:524 in forward, code: hidden_states = self.dense(hidden_states)
        view_14: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_6, [1024, 3072]);  mul_6 = None
        permute_9: "f32[3072, 768]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_5: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg22_1, view_14, permute_9);  arg22_1 = view_14 = permute_9 = None
        view_15: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_5, [8, 128, 768]);  addmm_5 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:526 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_8: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_15, add_6);  view_15 = add_6 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
        getitem_8: "f32[8, 128, 1]" = var_mean_2[0]
        getitem_9: "f32[8, 128, 1]" = var_mean_2[1];  var_mean_2 = None
        add_9: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_2: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        sub_3: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_8, getitem_9);  add_8 = getitem_9 = None
        mul_7: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_8: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_7, arg23_1);  mul_7 = arg23_1 = None
        add_10: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_8, arg24_1);  mul_8 = arg24_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:363 in forward, code: self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        view_16: "f32[1024, 768]" = torch.ops.aten.view.default(add_10, [1024, 768])
        permute_10: "f32[768, 768]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_6: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg26_1, view_16, permute_10);  arg26_1 = view_16 = permute_10 = None
        view_17: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_6, [8, 128, 768]);  addmm_6 = None
        view_18: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(view_17, [8, -1, 12, 64]);  view_17 = None
        permute_11: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:386 in forward, code: self.key(current_states)
        view_19: "f32[1024, 768]" = torch.ops.aten.view.default(add_10, [1024, 768])
        permute_12: "f32[768, 768]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_7: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg28_1, view_19, permute_12);  arg28_1 = view_19 = permute_12 = None
        view_20: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_7, [8, 128, 768]);  addmm_7 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:387 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_21: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(view_20, [8, -1, 12, 64]);  view_20 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:388 in forward, code: .transpose(1, 2)
        permute_13: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:391 in forward, code: self.value(current_states)
        view_22: "f32[1024, 768]" = torch.ops.aten.view.default(add_10, [1024, 768])
        permute_14: "f32[768, 768]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        addmm_8: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg30_1, view_22, permute_14);  arg30_1 = view_22 = permute_14 = None
        view_23: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_8, [8, 128, 768]);  addmm_8 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:392 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_24: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(view_23, [8, -1, 12, 64]);  view_23 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:393 in forward, code: .transpose(1, 2)
        permute_15: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:412 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_3: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(where, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_11, permute_13, permute_15, expand_3, False);  permute_11 = permute_13 = permute_15 = expand_3 = None
        getitem_10: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_1[0];  _scaled_dot_product_efficient_attention_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:421 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_16: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_10, [0, 2, 1, 3]);  getitem_10 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:422 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_25: "f32[8, 128, 768]" = torch.ops.aten.view.default(permute_16, [8, 128, 768]);  permute_16 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:435 in forward, code: hidden_states = self.dense(hidden_states)
        view_26: "f32[1024, 768]" = torch.ops.aten.view.default(view_25, [1024, 768]);  view_25 = None
        permute_17: "f32[768, 768]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_9: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg32_1, view_26, permute_17);  arg32_1 = view_26 = permute_17 = None
        view_27: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_9, [8, 128, 768]);  addmm_9 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:437 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_11: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_27, add_10);  view_27 = add_10 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
        getitem_14: "f32[8, 128, 1]" = var_mean_3[0]
        getitem_15: "f32[8, 128, 1]" = var_mean_3[1];  var_mean_3 = None
        add_12: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_3: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        sub_4: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_15);  add_11 = getitem_15 = None
        mul_9: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = rsqrt_3 = None
        mul_10: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_9, arg33_1);  mul_9 = arg33_1 = None
        add_13: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_10, arg34_1);  mul_10 = arg34_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:511 in forward, code: hidden_states = self.dense(hidden_states)
        view_28: "f32[1024, 768]" = torch.ops.aten.view.default(add_13, [1024, 768])
        permute_18: "f32[768, 3072]" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        addmm_10: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg36_1, view_28, permute_18);  arg36_1 = view_28 = permute_18 = None
        view_29: "f32[8, 128, 3072]" = torch.ops.aten.view.default(addmm_10, [8, 128, 3072]);  addmm_10 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:70 in forward, code: return self.act(input)
        mul_11: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_29, 0.5)
        mul_12: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_29, 0.7071067811865476);  view_29 = None
        erf_1: "f32[8, 128, 3072]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_14: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_13: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_11, add_14);  mul_11 = add_14 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:524 in forward, code: hidden_states = self.dense(hidden_states)
        view_30: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_13, [1024, 3072]);  mul_13 = None
        permute_19: "f32[3072, 768]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        addmm_11: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg38_1, view_30, permute_19);  arg38_1 = view_30 = permute_19 = None
        view_31: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_11, [8, 128, 768]);  addmm_11 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:526 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_15: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_31, add_13);  view_31 = add_13 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
        getitem_16: "f32[8, 128, 1]" = var_mean_4[0]
        getitem_17: "f32[8, 128, 1]" = var_mean_4[1];  var_mean_4 = None
        add_16: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_4: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_5: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_15, getitem_17);  add_15 = getitem_17 = None
        mul_14: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_4);  sub_5 = rsqrt_4 = None
        mul_15: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_14, arg39_1);  mul_14 = arg39_1 = None
        add_17: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_15, arg40_1);  mul_15 = arg40_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:363 in forward, code: self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        view_32: "f32[1024, 768]" = torch.ops.aten.view.default(add_17, [1024, 768])
        permute_20: "f32[768, 768]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_12: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg42_1, view_32, permute_20);  arg42_1 = view_32 = permute_20 = None
        view_33: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_12, [8, 128, 768]);  addmm_12 = None
        view_34: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(view_33, [8, -1, 12, 64]);  view_33 = None
        permute_21: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:386 in forward, code: self.key(current_states)
        view_35: "f32[1024, 768]" = torch.ops.aten.view.default(add_17, [1024, 768])
        permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_13: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg44_1, view_35, permute_22);  arg44_1 = view_35 = permute_22 = None
        view_36: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_13, [8, 128, 768]);  addmm_13 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:387 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_37: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(view_36, [8, -1, 12, 64]);  view_36 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:388 in forward, code: .transpose(1, 2)
        permute_23: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_37, [0, 2, 1, 3]);  view_37 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:391 in forward, code: self.value(current_states)
        view_38: "f32[1024, 768]" = torch.ops.aten.view.default(add_17, [1024, 768])
        permute_24: "f32[768, 768]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        addmm_14: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg46_1, view_38, permute_24);  arg46_1 = view_38 = permute_24 = None
        view_39: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_14, [8, 128, 768]);  addmm_14 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:392 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_40: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(view_39, [8, -1, 12, 64]);  view_39 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:393 in forward, code: .transpose(1, 2)
        permute_25: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:412 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_4: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(where, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_21, permute_23, permute_25, expand_4, False);  permute_21 = permute_23 = permute_25 = expand_4 = None
        getitem_18: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:421 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_26: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_18, [0, 2, 1, 3]);  getitem_18 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:422 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_41: "f32[8, 128, 768]" = torch.ops.aten.view.default(permute_26, [8, 128, 768]);  permute_26 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:435 in forward, code: hidden_states = self.dense(hidden_states)
        view_42: "f32[1024, 768]" = torch.ops.aten.view.default(view_41, [1024, 768]);  view_41 = None
        permute_27: "f32[768, 768]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_15: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg48_1, view_42, permute_27);  arg48_1 = view_42 = permute_27 = None
        view_43: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_15, [8, 128, 768]);  addmm_15 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:437 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_18: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_43, add_17);  view_43 = add_17 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_18, [2], correction = 0, keepdim = True)
        getitem_22: "f32[8, 128, 1]" = var_mean_5[0]
        getitem_23: "f32[8, 128, 1]" = var_mean_5[1];  var_mean_5 = None
        add_19: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_5: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        sub_6: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_18, getitem_23);  add_18 = getitem_23 = None
        mul_16: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_5);  sub_6 = rsqrt_5 = None
        mul_17: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_16, arg49_1);  mul_16 = arg49_1 = None
        add_20: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_17, arg50_1);  mul_17 = arg50_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:511 in forward, code: hidden_states = self.dense(hidden_states)
        view_44: "f32[1024, 768]" = torch.ops.aten.view.default(add_20, [1024, 768])
        permute_28: "f32[768, 3072]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        addmm_16: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg52_1, view_44, permute_28);  arg52_1 = view_44 = permute_28 = None
        view_45: "f32[8, 128, 3072]" = torch.ops.aten.view.default(addmm_16, [8, 128, 3072]);  addmm_16 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:70 in forward, code: return self.act(input)
        mul_18: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_45, 0.5)
        mul_19: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_45, 0.7071067811865476);  view_45 = None
        erf_2: "f32[8, 128, 3072]" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
        add_21: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_20: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_18, add_21);  mul_18 = add_21 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:524 in forward, code: hidden_states = self.dense(hidden_states)
        view_46: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_20, [1024, 3072]);  mul_20 = None
        permute_29: "f32[3072, 768]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_17: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg54_1, view_46, permute_29);  arg54_1 = view_46 = permute_29 = None
        view_47: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_17, [8, 128, 768]);  addmm_17 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:526 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_22: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_47, add_20);  view_47 = add_20 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_22, [2], correction = 0, keepdim = True)
        getitem_24: "f32[8, 128, 1]" = var_mean_6[0]
        getitem_25: "f32[8, 128, 1]" = var_mean_6[1];  var_mean_6 = None
        add_23: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_6: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        sub_7: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_22, getitem_25);  add_22 = getitem_25 = None
        mul_21: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_6);  sub_7 = rsqrt_6 = None
        mul_22: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_21, arg55_1);  mul_21 = arg55_1 = None
        add_24: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_22, arg56_1);  mul_22 = arg56_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:363 in forward, code: self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        view_48: "f32[1024, 768]" = torch.ops.aten.view.default(add_24, [1024, 768])
        permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        addmm_18: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg58_1, view_48, permute_30);  arg58_1 = view_48 = permute_30 = None
        view_49: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_18, [8, 128, 768]);  addmm_18 = None
        view_50: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(view_49, [8, -1, 12, 64]);  view_49 = None
        permute_31: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:386 in forward, code: self.key(current_states)
        view_51: "f32[1024, 768]" = torch.ops.aten.view.default(add_24, [1024, 768])
        permute_32: "f32[768, 768]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        addmm_19: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg60_1, view_51, permute_32);  arg60_1 = view_51 = permute_32 = None
        view_52: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_19, [8, 128, 768]);  addmm_19 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:387 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_53: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(view_52, [8, -1, 12, 64]);  view_52 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:388 in forward, code: .transpose(1, 2)
        permute_33: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:391 in forward, code: self.value(current_states)
        view_54: "f32[1024, 768]" = torch.ops.aten.view.default(add_24, [1024, 768])
        permute_34: "f32[768, 768]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm_20: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg62_1, view_54, permute_34);  arg62_1 = view_54 = permute_34 = None
        view_55: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_20, [8, 128, 768]);  addmm_20 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:392 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_56: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(view_55, [8, -1, 12, 64]);  view_55 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:393 in forward, code: .transpose(1, 2)
        permute_35: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:412 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_5: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(where, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_31, permute_33, permute_35, expand_5, False);  permute_31 = permute_33 = permute_35 = expand_5 = None
        getitem_26: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:421 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_36: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_26, [0, 2, 1, 3]);  getitem_26 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:422 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_57: "f32[8, 128, 768]" = torch.ops.aten.view.default(permute_36, [8, 128, 768]);  permute_36 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:435 in forward, code: hidden_states = self.dense(hidden_states)
        view_58: "f32[1024, 768]" = torch.ops.aten.view.default(view_57, [1024, 768]);  view_57 = None
        permute_37: "f32[768, 768]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        addmm_21: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg64_1, view_58, permute_37);  arg64_1 = view_58 = permute_37 = None
        view_59: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_21, [8, 128, 768]);  addmm_21 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:437 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_25: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_59, add_24);  view_59 = add_24 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_30: "f32[8, 128, 1]" = var_mean_7[0]
        getitem_31: "f32[8, 128, 1]" = var_mean_7[1];  var_mean_7 = None
        add_26: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_7: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_8: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_25, getitem_31);  add_25 = getitem_31 = None
        mul_23: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_7);  sub_8 = rsqrt_7 = None
        mul_24: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_23, arg65_1);  mul_23 = arg65_1 = None
        add_27: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_24, arg66_1);  mul_24 = arg66_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:511 in forward, code: hidden_states = self.dense(hidden_states)
        view_60: "f32[1024, 768]" = torch.ops.aten.view.default(add_27, [1024, 768])
        permute_38: "f32[768, 3072]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        addmm_22: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg68_1, view_60, permute_38);  arg68_1 = view_60 = permute_38 = None
        view_61: "f32[8, 128, 3072]" = torch.ops.aten.view.default(addmm_22, [8, 128, 3072]);  addmm_22 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:70 in forward, code: return self.act(input)
        mul_25: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_61, 0.5)
        mul_26: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_61, 0.7071067811865476);  view_61 = None
        erf_3: "f32[8, 128, 3072]" = torch.ops.aten.erf.default(mul_26);  mul_26 = None
        add_28: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_27: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_25, add_28);  mul_25 = add_28 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:524 in forward, code: hidden_states = self.dense(hidden_states)
        view_62: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_27, [1024, 3072]);  mul_27 = None
        permute_39: "f32[3072, 768]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        addmm_23: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg70_1, view_62, permute_39);  arg70_1 = view_62 = permute_39 = None
        view_63: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_23, [8, 128, 768]);  addmm_23 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:526 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_29: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_63, add_27);  view_63 = add_27 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
        getitem_32: "f32[8, 128, 1]" = var_mean_8[0]
        getitem_33: "f32[8, 128, 1]" = var_mean_8[1];  var_mean_8 = None
        add_30: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_8: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        sub_9: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_29, getitem_33);  add_29 = getitem_33 = None
        mul_28: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_8);  sub_9 = rsqrt_8 = None
        mul_29: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_28, arg71_1);  mul_28 = arg71_1 = None
        add_31: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_29, arg72_1);  mul_29 = arg72_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:363 in forward, code: self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        view_64: "f32[1024, 768]" = torch.ops.aten.view.default(add_31, [1024, 768])
        permute_40: "f32[768, 768]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_24: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg74_1, view_64, permute_40);  arg74_1 = view_64 = permute_40 = None
        view_65: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_24, [8, 128, 768]);  addmm_24 = None
        view_66: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(view_65, [8, -1, 12, 64]);  view_65 = None
        permute_41: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:386 in forward, code: self.key(current_states)
        view_67: "f32[1024, 768]" = torch.ops.aten.view.default(add_31, [1024, 768])
        permute_42: "f32[768, 768]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        addmm_25: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg76_1, view_67, permute_42);  arg76_1 = view_67 = permute_42 = None
        view_68: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_25, [8, 128, 768]);  addmm_25 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:387 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_69: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(view_68, [8, -1, 12, 64]);  view_68 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:388 in forward, code: .transpose(1, 2)
        permute_43: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:391 in forward, code: self.value(current_states)
        view_70: "f32[1024, 768]" = torch.ops.aten.view.default(add_31, [1024, 768])
        permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        addmm_26: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg78_1, view_70, permute_44);  arg78_1 = view_70 = permute_44 = None
        view_71: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_26, [8, 128, 768]);  addmm_26 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:392 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_72: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(view_71, [8, -1, 12, 64]);  view_71 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:393 in forward, code: .transpose(1, 2)
        permute_45: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:412 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_6: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(where, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_41, permute_43, permute_45, expand_6, False);  permute_41 = permute_43 = permute_45 = expand_6 = None
        getitem_34: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_4[0];  _scaled_dot_product_efficient_attention_4 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:421 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_46: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_34, [0, 2, 1, 3]);  getitem_34 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:422 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_73: "f32[8, 128, 768]" = torch.ops.aten.view.default(permute_46, [8, 128, 768]);  permute_46 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:435 in forward, code: hidden_states = self.dense(hidden_states)
        view_74: "f32[1024, 768]" = torch.ops.aten.view.default(view_73, [1024, 768]);  view_73 = None
        permute_47: "f32[768, 768]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_27: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg80_1, view_74, permute_47);  arg80_1 = view_74 = permute_47 = None
        view_75: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_27, [8, 128, 768]);  addmm_27 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:437 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_32: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_75, add_31);  view_75 = add_31 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
        getitem_38: "f32[8, 128, 1]" = var_mean_9[0]
        getitem_39: "f32[8, 128, 1]" = var_mean_9[1];  var_mean_9 = None
        add_33: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_9: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        sub_10: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_32, getitem_39);  add_32 = getitem_39 = None
        mul_30: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_9);  sub_10 = rsqrt_9 = None
        mul_31: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_30, arg81_1);  mul_30 = arg81_1 = None
        add_34: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_31, arg82_1);  mul_31 = arg82_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:511 in forward, code: hidden_states = self.dense(hidden_states)
        view_76: "f32[1024, 768]" = torch.ops.aten.view.default(add_34, [1024, 768])
        permute_48: "f32[768, 3072]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        addmm_28: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg84_1, view_76, permute_48);  arg84_1 = view_76 = permute_48 = None
        view_77: "f32[8, 128, 3072]" = torch.ops.aten.view.default(addmm_28, [8, 128, 3072]);  addmm_28 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:70 in forward, code: return self.act(input)
        mul_32: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_77, 0.5)
        mul_33: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_77, 0.7071067811865476);  view_77 = None
        erf_4: "f32[8, 128, 3072]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
        add_35: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_34: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_32, add_35);  mul_32 = add_35 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:524 in forward, code: hidden_states = self.dense(hidden_states)
        view_78: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_34, [1024, 3072]);  mul_34 = None
        permute_49: "f32[3072, 768]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_29: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg86_1, view_78, permute_49);  arg86_1 = view_78 = permute_49 = None
        view_79: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_29, [8, 128, 768]);  addmm_29 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:526 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_36: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_79, add_34);  view_79 = add_34 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_36, [2], correction = 0, keepdim = True)
        getitem_40: "f32[8, 128, 1]" = var_mean_10[0]
        getitem_41: "f32[8, 128, 1]" = var_mean_10[1];  var_mean_10 = None
        add_37: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_10: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        sub_11: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_36, getitem_41);  add_36 = getitem_41 = None
        mul_35: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_10);  sub_11 = rsqrt_10 = None
        mul_36: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_35, arg87_1);  mul_35 = arg87_1 = None
        add_38: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_36, arg88_1);  mul_36 = arg88_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:363 in forward, code: self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        view_80: "f32[1024, 768]" = torch.ops.aten.view.default(add_38, [1024, 768])
        permute_50: "f32[768, 768]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        addmm_30: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg90_1, view_80, permute_50);  arg90_1 = view_80 = permute_50 = None
        view_81: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_30, [8, 128, 768]);  addmm_30 = None
        view_82: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(view_81, [8, -1, 12, 64]);  view_81 = None
        permute_51: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:386 in forward, code: self.key(current_states)
        view_83: "f32[1024, 768]" = torch.ops.aten.view.default(add_38, [1024, 768])
        permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_31: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg92_1, view_83, permute_52);  arg92_1 = view_83 = permute_52 = None
        view_84: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_31, [8, 128, 768]);  addmm_31 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:387 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_85: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(view_84, [8, -1, 12, 64]);  view_84 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:388 in forward, code: .transpose(1, 2)
        permute_53: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_85, [0, 2, 1, 3]);  view_85 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:391 in forward, code: self.value(current_states)
        view_86: "f32[1024, 768]" = torch.ops.aten.view.default(add_38, [1024, 768])
        permute_54: "f32[768, 768]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        addmm_32: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg94_1, view_86, permute_54);  arg94_1 = view_86 = permute_54 = None
        view_87: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_32, [8, 128, 768]);  addmm_32 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:392 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_88: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(view_87, [8, -1, 12, 64]);  view_87 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:393 in forward, code: .transpose(1, 2)
        permute_55: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:412 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_7: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(where, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_51, permute_53, permute_55, expand_7, False);  permute_51 = permute_53 = permute_55 = expand_7 = None
        getitem_42: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_5[0];  _scaled_dot_product_efficient_attention_5 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:421 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_56: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_42, [0, 2, 1, 3]);  getitem_42 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:422 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_89: "f32[8, 128, 768]" = torch.ops.aten.view.default(permute_56, [8, 128, 768]);  permute_56 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:435 in forward, code: hidden_states = self.dense(hidden_states)
        view_90: "f32[1024, 768]" = torch.ops.aten.view.default(view_89, [1024, 768]);  view_89 = None
        permute_57: "f32[768, 768]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_33: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg96_1, view_90, permute_57);  arg96_1 = view_90 = permute_57 = None
        view_91: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_33, [8, 128, 768]);  addmm_33 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:437 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_39: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_91, add_38);  view_91 = add_38 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_39, [2], correction = 0, keepdim = True)
        getitem_46: "f32[8, 128, 1]" = var_mean_11[0]
        getitem_47: "f32[8, 128, 1]" = var_mean_11[1];  var_mean_11 = None
        add_40: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_11: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        sub_12: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_39, getitem_47);  add_39 = getitem_47 = None
        mul_37: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_11);  sub_12 = rsqrt_11 = None
        mul_38: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_37, arg97_1);  mul_37 = arg97_1 = None
        add_41: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_38, arg98_1);  mul_38 = arg98_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:511 in forward, code: hidden_states = self.dense(hidden_states)
        view_92: "f32[1024, 768]" = torch.ops.aten.view.default(add_41, [1024, 768])
        permute_58: "f32[768, 3072]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        addmm_34: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg100_1, view_92, permute_58);  arg100_1 = view_92 = permute_58 = None
        view_93: "f32[8, 128, 3072]" = torch.ops.aten.view.default(addmm_34, [8, 128, 3072]);  addmm_34 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:70 in forward, code: return self.act(input)
        mul_39: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_93, 0.5)
        mul_40: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_93, 0.7071067811865476);  view_93 = None
        erf_5: "f32[8, 128, 3072]" = torch.ops.aten.erf.default(mul_40);  mul_40 = None
        add_42: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_41: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_39, add_42);  mul_39 = add_42 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:524 in forward, code: hidden_states = self.dense(hidden_states)
        view_94: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_41, [1024, 3072]);  mul_41 = None
        permute_59: "f32[3072, 768]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        addmm_35: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg102_1, view_94, permute_59);  arg102_1 = view_94 = permute_59 = None
        view_95: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_35, [8, 128, 768]);  addmm_35 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:526 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_43: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_95, add_41);  view_95 = add_41 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
        getitem_48: "f32[8, 128, 1]" = var_mean_12[0]
        getitem_49: "f32[8, 128, 1]" = var_mean_12[1];  var_mean_12 = None
        add_44: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_12: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        sub_13: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_49);  add_43 = getitem_49 = None
        mul_42: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_12);  sub_13 = rsqrt_12 = None
        mul_43: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_42, arg103_1);  mul_42 = arg103_1 = None
        add_45: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_43, arg104_1);  mul_43 = arg104_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:363 in forward, code: self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        view_96: "f32[1024, 768]" = torch.ops.aten.view.default(add_45, [1024, 768])
        permute_60: "f32[768, 768]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        addmm_36: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg106_1, view_96, permute_60);  arg106_1 = view_96 = permute_60 = None
        view_97: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_36, [8, 128, 768]);  addmm_36 = None
        view_98: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(view_97, [8, -1, 12, 64]);  view_97 = None
        permute_61: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:386 in forward, code: self.key(current_states)
        view_99: "f32[1024, 768]" = torch.ops.aten.view.default(add_45, [1024, 768])
        permute_62: "f32[768, 768]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_37: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg108_1, view_99, permute_62);  arg108_1 = view_99 = permute_62 = None
        view_100: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_37, [8, 128, 768]);  addmm_37 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:387 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_101: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(view_100, [8, -1, 12, 64]);  view_100 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:388 in forward, code: .transpose(1, 2)
        permute_63: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_101, [0, 2, 1, 3]);  view_101 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:391 in forward, code: self.value(current_states)
        view_102: "f32[1024, 768]" = torch.ops.aten.view.default(add_45, [1024, 768])
        permute_64: "f32[768, 768]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_38: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg110_1, view_102, permute_64);  arg110_1 = view_102 = permute_64 = None
        view_103: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_38, [8, 128, 768]);  addmm_38 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:392 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_104: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(view_103, [8, -1, 12, 64]);  view_103 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:393 in forward, code: .transpose(1, 2)
        permute_65: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:412 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_8: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(where, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_61, permute_63, permute_65, expand_8, False);  permute_61 = permute_63 = permute_65 = expand_8 = None
        getitem_50: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_6[0];  _scaled_dot_product_efficient_attention_6 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:421 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_66: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_50, [0, 2, 1, 3]);  getitem_50 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:422 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_105: "f32[8, 128, 768]" = torch.ops.aten.view.default(permute_66, [8, 128, 768]);  permute_66 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:435 in forward, code: hidden_states = self.dense(hidden_states)
        view_106: "f32[1024, 768]" = torch.ops.aten.view.default(view_105, [1024, 768]);  view_105 = None
        permute_67: "f32[768, 768]" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        addmm_39: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg112_1, view_106, permute_67);  arg112_1 = view_106 = permute_67 = None
        view_107: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_39, [8, 128, 768]);  addmm_39 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:437 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_46: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_107, add_45);  view_107 = add_45 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_46, [2], correction = 0, keepdim = True)
        getitem_54: "f32[8, 128, 1]" = var_mean_13[0]
        getitem_55: "f32[8, 128, 1]" = var_mean_13[1];  var_mean_13 = None
        add_47: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-12);  getitem_54 = None
        rsqrt_13: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        sub_14: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_46, getitem_55);  add_46 = getitem_55 = None
        mul_44: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_13);  sub_14 = rsqrt_13 = None
        mul_45: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_44, arg113_1);  mul_44 = arg113_1 = None
        add_48: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_45, arg114_1);  mul_45 = arg114_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:511 in forward, code: hidden_states = self.dense(hidden_states)
        view_108: "f32[1024, 768]" = torch.ops.aten.view.default(add_48, [1024, 768])
        permute_68: "f32[768, 3072]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        addmm_40: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg116_1, view_108, permute_68);  arg116_1 = view_108 = permute_68 = None
        view_109: "f32[8, 128, 3072]" = torch.ops.aten.view.default(addmm_40, [8, 128, 3072]);  addmm_40 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:70 in forward, code: return self.act(input)
        mul_46: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_109, 0.5)
        mul_47: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_109, 0.7071067811865476);  view_109 = None
        erf_6: "f32[8, 128, 3072]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
        add_49: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_48: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_46, add_49);  mul_46 = add_49 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:524 in forward, code: hidden_states = self.dense(hidden_states)
        view_110: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_48, [1024, 3072]);  mul_48 = None
        permute_69: "f32[3072, 768]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        addmm_41: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg118_1, view_110, permute_69);  arg118_1 = view_110 = permute_69 = None
        view_111: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_41, [8, 128, 768]);  addmm_41 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:526 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_50: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_111, add_48);  view_111 = add_48 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_50, [2], correction = 0, keepdim = True)
        getitem_56: "f32[8, 128, 1]" = var_mean_14[0]
        getitem_57: "f32[8, 128, 1]" = var_mean_14[1];  var_mean_14 = None
        add_51: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-12);  getitem_56 = None
        rsqrt_14: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        sub_15: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_50, getitem_57);  add_50 = getitem_57 = None
        mul_49: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_14);  sub_15 = rsqrt_14 = None
        mul_50: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_49, arg119_1);  mul_49 = arg119_1 = None
        add_52: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_50, arg120_1);  mul_50 = arg120_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:363 in forward, code: self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        view_112: "f32[1024, 768]" = torch.ops.aten.view.default(add_52, [1024, 768])
        permute_70: "f32[768, 768]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_42: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg122_1, view_112, permute_70);  arg122_1 = view_112 = permute_70 = None
        view_113: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_42, [8, 128, 768]);  addmm_42 = None
        view_114: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(view_113, [8, -1, 12, 64]);  view_113 = None
        permute_71: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:386 in forward, code: self.key(current_states)
        view_115: "f32[1024, 768]" = torch.ops.aten.view.default(add_52, [1024, 768])
        permute_72: "f32[768, 768]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_43: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg124_1, view_115, permute_72);  arg124_1 = view_115 = permute_72 = None
        view_116: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_43, [8, 128, 768]);  addmm_43 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:387 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_117: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(view_116, [8, -1, 12, 64]);  view_116 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:388 in forward, code: .transpose(1, 2)
        permute_73: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:391 in forward, code: self.value(current_states)
        view_118: "f32[1024, 768]" = torch.ops.aten.view.default(add_52, [1024, 768])
        permute_74: "f32[768, 768]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        addmm_44: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg126_1, view_118, permute_74);  arg126_1 = view_118 = permute_74 = None
        view_119: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_44, [8, 128, 768]);  addmm_44 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:392 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_120: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(view_119, [8, -1, 12, 64]);  view_119 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:393 in forward, code: .transpose(1, 2)
        permute_75: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:412 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_9: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(where, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_71, permute_73, permute_75, expand_9, False);  permute_71 = permute_73 = permute_75 = expand_9 = None
        getitem_58: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_7[0];  _scaled_dot_product_efficient_attention_7 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:421 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_76: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_58, [0, 2, 1, 3]);  getitem_58 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:422 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_121: "f32[8, 128, 768]" = torch.ops.aten.view.default(permute_76, [8, 128, 768]);  permute_76 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:435 in forward, code: hidden_states = self.dense(hidden_states)
        view_122: "f32[1024, 768]" = torch.ops.aten.view.default(view_121, [1024, 768]);  view_121 = None
        permute_77: "f32[768, 768]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        addmm_45: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg128_1, view_122, permute_77);  arg128_1 = view_122 = permute_77 = None
        view_123: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_45, [8, 128, 768]);  addmm_45 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:437 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_53: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_123, add_52);  view_123 = add_52 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
        getitem_62: "f32[8, 128, 1]" = var_mean_15[0]
        getitem_63: "f32[8, 128, 1]" = var_mean_15[1];  var_mean_15 = None
        add_54: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-12);  getitem_62 = None
        rsqrt_15: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        sub_16: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_53, getitem_63);  add_53 = getitem_63 = None
        mul_51: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_15);  sub_16 = rsqrt_15 = None
        mul_52: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_51, arg129_1);  mul_51 = arg129_1 = None
        add_55: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_52, arg130_1);  mul_52 = arg130_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:511 in forward, code: hidden_states = self.dense(hidden_states)
        view_124: "f32[1024, 768]" = torch.ops.aten.view.default(add_55, [1024, 768])
        permute_78: "f32[768, 3072]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        addmm_46: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg132_1, view_124, permute_78);  arg132_1 = view_124 = permute_78 = None
        view_125: "f32[8, 128, 3072]" = torch.ops.aten.view.default(addmm_46, [8, 128, 3072]);  addmm_46 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:70 in forward, code: return self.act(input)
        mul_53: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_125, 0.5)
        mul_54: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_125, 0.7071067811865476);  view_125 = None
        erf_7: "f32[8, 128, 3072]" = torch.ops.aten.erf.default(mul_54);  mul_54 = None
        add_56: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_55: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_53, add_56);  mul_53 = add_56 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:524 in forward, code: hidden_states = self.dense(hidden_states)
        view_126: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_55, [1024, 3072]);  mul_55 = None
        permute_79: "f32[3072, 768]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_47: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg134_1, view_126, permute_79);  arg134_1 = view_126 = permute_79 = None
        view_127: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_47, [8, 128, 768]);  addmm_47 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:526 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_57: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_127, add_55);  view_127 = add_55 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_64: "f32[8, 128, 1]" = var_mean_16[0]
        getitem_65: "f32[8, 128, 1]" = var_mean_16[1];  var_mean_16 = None
        add_58: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-12);  getitem_64 = None
        rsqrt_16: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_17: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_57, getitem_65);  add_57 = getitem_65 = None
        mul_56: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_16);  sub_17 = rsqrt_16 = None
        mul_57: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_56, arg135_1);  mul_56 = arg135_1 = None
        add_59: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_57, arg136_1);  mul_57 = arg136_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:363 in forward, code: self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        view_128: "f32[1024, 768]" = torch.ops.aten.view.default(add_59, [1024, 768])
        permute_80: "f32[768, 768]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        addmm_48: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg138_1, view_128, permute_80);  arg138_1 = view_128 = permute_80 = None
        view_129: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_48, [8, 128, 768]);  addmm_48 = None
        view_130: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(view_129, [8, -1, 12, 64]);  view_129 = None
        permute_81: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:386 in forward, code: self.key(current_states)
        view_131: "f32[1024, 768]" = torch.ops.aten.view.default(add_59, [1024, 768])
        permute_82: "f32[768, 768]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_49: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg140_1, view_131, permute_82);  arg140_1 = view_131 = permute_82 = None
        view_132: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_49, [8, 128, 768]);  addmm_49 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:387 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_133: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(view_132, [8, -1, 12, 64]);  view_132 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:388 in forward, code: .transpose(1, 2)
        permute_83: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_133, [0, 2, 1, 3]);  view_133 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:391 in forward, code: self.value(current_states)
        view_134: "f32[1024, 768]" = torch.ops.aten.view.default(add_59, [1024, 768])
        permute_84: "f32[768, 768]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        addmm_50: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg142_1, view_134, permute_84);  arg142_1 = view_134 = permute_84 = None
        view_135: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_50, [8, 128, 768]);  addmm_50 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:392 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_136: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(view_135, [8, -1, 12, 64]);  view_135 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:393 in forward, code: .transpose(1, 2)
        permute_85: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:412 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_10: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(where, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_81, permute_83, permute_85, expand_10, False);  permute_81 = permute_83 = permute_85 = expand_10 = None
        getitem_66: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_8[0];  _scaled_dot_product_efficient_attention_8 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:421 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_86: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_66, [0, 2, 1, 3]);  getitem_66 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:422 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_137: "f32[8, 128, 768]" = torch.ops.aten.view.default(permute_86, [8, 128, 768]);  permute_86 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:435 in forward, code: hidden_states = self.dense(hidden_states)
        view_138: "f32[1024, 768]" = torch.ops.aten.view.default(view_137, [1024, 768]);  view_137 = None
        permute_87: "f32[768, 768]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        addmm_51: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg144_1, view_138, permute_87);  arg144_1 = view_138 = permute_87 = None
        view_139: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_51, [8, 128, 768]);  addmm_51 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:437 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_60: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_139, add_59);  view_139 = add_59 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_60, [2], correction = 0, keepdim = True)
        getitem_70: "f32[8, 128, 1]" = var_mean_17[0]
        getitem_71: "f32[8, 128, 1]" = var_mean_17[1];  var_mean_17 = None
        add_61: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-12);  getitem_70 = None
        rsqrt_17: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        sub_18: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_60, getitem_71);  add_60 = getitem_71 = None
        mul_58: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_17);  sub_18 = rsqrt_17 = None
        mul_59: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_58, arg145_1);  mul_58 = arg145_1 = None
        add_62: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_59, arg146_1);  mul_59 = arg146_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:511 in forward, code: hidden_states = self.dense(hidden_states)
        view_140: "f32[1024, 768]" = torch.ops.aten.view.default(add_62, [1024, 768])
        permute_88: "f32[768, 3072]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        addmm_52: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg148_1, view_140, permute_88);  arg148_1 = view_140 = permute_88 = None
        view_141: "f32[8, 128, 3072]" = torch.ops.aten.view.default(addmm_52, [8, 128, 3072]);  addmm_52 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:70 in forward, code: return self.act(input)
        mul_60: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_141, 0.5)
        mul_61: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_141, 0.7071067811865476);  view_141 = None
        erf_8: "f32[8, 128, 3072]" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
        add_63: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_62: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_60, add_63);  mul_60 = add_63 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:524 in forward, code: hidden_states = self.dense(hidden_states)
        view_142: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_62, [1024, 3072]);  mul_62 = None
        permute_89: "f32[3072, 768]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        addmm_53: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg150_1, view_142, permute_89);  arg150_1 = view_142 = permute_89 = None
        view_143: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_53, [8, 128, 768]);  addmm_53 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:526 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_64: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_143, add_62);  view_143 = add_62 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_64, [2], correction = 0, keepdim = True)
        getitem_72: "f32[8, 128, 1]" = var_mean_18[0]
        getitem_73: "f32[8, 128, 1]" = var_mean_18[1];  var_mean_18 = None
        add_65: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-12);  getitem_72 = None
        rsqrt_18: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        sub_19: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_64, getitem_73);  add_64 = getitem_73 = None
        mul_63: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_18);  sub_19 = rsqrt_18 = None
        mul_64: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_63, arg151_1);  mul_63 = arg151_1 = None
        add_66: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_64, arg152_1);  mul_64 = arg152_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:363 in forward, code: self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        view_144: "f32[1024, 768]" = torch.ops.aten.view.default(add_66, [1024, 768])
        permute_90: "f32[768, 768]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        addmm_54: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg154_1, view_144, permute_90);  arg154_1 = view_144 = permute_90 = None
        view_145: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_54, [8, 128, 768]);  addmm_54 = None
        view_146: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(view_145, [8, -1, 12, 64]);  view_145 = None
        permute_91: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:386 in forward, code: self.key(current_states)
        view_147: "f32[1024, 768]" = torch.ops.aten.view.default(add_66, [1024, 768])
        permute_92: "f32[768, 768]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        addmm_55: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg156_1, view_147, permute_92);  arg156_1 = view_147 = permute_92 = None
        view_148: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_55, [8, 128, 768]);  addmm_55 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:387 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_149: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(view_148, [8, -1, 12, 64]);  view_148 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:388 in forward, code: .transpose(1, 2)
        permute_93: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:391 in forward, code: self.value(current_states)
        view_150: "f32[1024, 768]" = torch.ops.aten.view.default(add_66, [1024, 768])
        permute_94: "f32[768, 768]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        addmm_56: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg158_1, view_150, permute_94);  arg158_1 = view_150 = permute_94 = None
        view_151: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_56, [8, 128, 768]);  addmm_56 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:392 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_152: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(view_151, [8, -1, 12, 64]);  view_151 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:393 in forward, code: .transpose(1, 2)
        permute_95: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_152, [0, 2, 1, 3]);  view_152 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:412 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_11: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(where, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_91, permute_93, permute_95, expand_11, False);  permute_91 = permute_93 = permute_95 = expand_11 = None
        getitem_74: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_9[0];  _scaled_dot_product_efficient_attention_9 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:421 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_96: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_74, [0, 2, 1, 3]);  getitem_74 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:422 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_153: "f32[8, 128, 768]" = torch.ops.aten.view.default(permute_96, [8, 128, 768]);  permute_96 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:435 in forward, code: hidden_states = self.dense(hidden_states)
        view_154: "f32[1024, 768]" = torch.ops.aten.view.default(view_153, [1024, 768]);  view_153 = None
        permute_97: "f32[768, 768]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        addmm_57: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg160_1, view_154, permute_97);  arg160_1 = view_154 = permute_97 = None
        view_155: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_57, [8, 128, 768]);  addmm_57 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:437 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_67: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_155, add_66);  view_155 = add_66 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
        getitem_78: "f32[8, 128, 1]" = var_mean_19[0]
        getitem_79: "f32[8, 128, 1]" = var_mean_19[1];  var_mean_19 = None
        add_68: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-12);  getitem_78 = None
        rsqrt_19: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        sub_20: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_67, getitem_79);  add_67 = getitem_79 = None
        mul_65: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_19);  sub_20 = rsqrt_19 = None
        mul_66: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_65, arg161_1);  mul_65 = arg161_1 = None
        add_69: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_66, arg162_1);  mul_66 = arg162_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:511 in forward, code: hidden_states = self.dense(hidden_states)
        view_156: "f32[1024, 768]" = torch.ops.aten.view.default(add_69, [1024, 768])
        permute_98: "f32[768, 3072]" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        addmm_58: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg164_1, view_156, permute_98);  arg164_1 = view_156 = permute_98 = None
        view_157: "f32[8, 128, 3072]" = torch.ops.aten.view.default(addmm_58, [8, 128, 3072]);  addmm_58 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:70 in forward, code: return self.act(input)
        mul_67: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_157, 0.5)
        mul_68: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_157, 0.7071067811865476);  view_157 = None
        erf_9: "f32[8, 128, 3072]" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
        add_70: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_69: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_67, add_70);  mul_67 = add_70 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:524 in forward, code: hidden_states = self.dense(hidden_states)
        view_158: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_69, [1024, 3072]);  mul_69 = None
        permute_99: "f32[3072, 768]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        addmm_59: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg166_1, view_158, permute_99);  arg166_1 = view_158 = permute_99 = None
        view_159: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_59, [8, 128, 768]);  addmm_59 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:526 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_71: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_159, add_69);  view_159 = add_69 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
        getitem_80: "f32[8, 128, 1]" = var_mean_20[0]
        getitem_81: "f32[8, 128, 1]" = var_mean_20[1];  var_mean_20 = None
        add_72: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-12);  getitem_80 = None
        rsqrt_20: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        sub_21: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_71, getitem_81);  add_71 = getitem_81 = None
        mul_70: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_20);  sub_21 = rsqrt_20 = None
        mul_71: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_70, arg167_1);  mul_70 = arg167_1 = None
        add_73: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_71, arg168_1);  mul_71 = arg168_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:363 in forward, code: self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        view_160: "f32[1024, 768]" = torch.ops.aten.view.default(add_73, [1024, 768])
        permute_100: "f32[768, 768]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        addmm_60: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg170_1, view_160, permute_100);  arg170_1 = view_160 = permute_100 = None
        view_161: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_60, [8, 128, 768]);  addmm_60 = None
        view_162: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(view_161, [8, -1, 12, 64]);  view_161 = None
        permute_101: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:386 in forward, code: self.key(current_states)
        view_163: "f32[1024, 768]" = torch.ops.aten.view.default(add_73, [1024, 768])
        permute_102: "f32[768, 768]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        addmm_61: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg172_1, view_163, permute_102);  arg172_1 = view_163 = permute_102 = None
        view_164: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_61, [8, 128, 768]);  addmm_61 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:387 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_165: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(view_164, [8, -1, 12, 64]);  view_164 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:388 in forward, code: .transpose(1, 2)
        permute_103: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_165, [0, 2, 1, 3]);  view_165 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:391 in forward, code: self.value(current_states)
        view_166: "f32[1024, 768]" = torch.ops.aten.view.default(add_73, [1024, 768])
        permute_104: "f32[768, 768]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        addmm_62: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg174_1, view_166, permute_104);  arg174_1 = view_166 = permute_104 = None
        view_167: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_62, [8, 128, 768]);  addmm_62 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:392 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_168: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(view_167, [8, -1, 12, 64]);  view_167 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:393 in forward, code: .transpose(1, 2)
        permute_105: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:412 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_12: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(where, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_101, permute_103, permute_105, expand_12, False);  permute_101 = permute_103 = permute_105 = expand_12 = None
        getitem_82: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_10[0];  _scaled_dot_product_efficient_attention_10 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:421 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_106: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3]);  getitem_82 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:422 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_169: "f32[8, 128, 768]" = torch.ops.aten.view.default(permute_106, [8, 128, 768]);  permute_106 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:435 in forward, code: hidden_states = self.dense(hidden_states)
        view_170: "f32[1024, 768]" = torch.ops.aten.view.default(view_169, [1024, 768]);  view_169 = None
        permute_107: "f32[768, 768]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        addmm_63: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg176_1, view_170, permute_107);  arg176_1 = view_170 = permute_107 = None
        view_171: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_63, [8, 128, 768]);  addmm_63 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:437 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_74: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_171, add_73);  view_171 = add_73 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_74, [2], correction = 0, keepdim = True)
        getitem_86: "f32[8, 128, 1]" = var_mean_21[0]
        getitem_87: "f32[8, 128, 1]" = var_mean_21[1];  var_mean_21 = None
        add_75: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-12);  getitem_86 = None
        rsqrt_21: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        sub_22: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_74, getitem_87);  add_74 = getitem_87 = None
        mul_72: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_21);  sub_22 = rsqrt_21 = None
        mul_73: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_72, arg177_1);  mul_72 = arg177_1 = None
        add_76: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_73, arg178_1);  mul_73 = arg178_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:511 in forward, code: hidden_states = self.dense(hidden_states)
        view_172: "f32[1024, 768]" = torch.ops.aten.view.default(add_76, [1024, 768])
        permute_108: "f32[768, 3072]" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        addmm_64: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg180_1, view_172, permute_108);  arg180_1 = view_172 = permute_108 = None
        view_173: "f32[8, 128, 3072]" = torch.ops.aten.view.default(addmm_64, [8, 128, 3072]);  addmm_64 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:70 in forward, code: return self.act(input)
        mul_74: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.5)
        mul_75: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476);  view_173 = None
        erf_10: "f32[8, 128, 3072]" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
        add_77: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_76: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_74, add_77);  mul_74 = add_77 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:524 in forward, code: hidden_states = self.dense(hidden_states)
        view_174: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_76, [1024, 3072]);  mul_76 = None
        permute_109: "f32[3072, 768]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        addmm_65: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg182_1, view_174, permute_109);  arg182_1 = view_174 = permute_109 = None
        view_175: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_65, [8, 128, 768]);  addmm_65 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:526 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_78: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_175, add_76);  view_175 = add_76 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_78, [2], correction = 0, keepdim = True)
        getitem_88: "f32[8, 128, 1]" = var_mean_22[0]
        getitem_89: "f32[8, 128, 1]" = var_mean_22[1];  var_mean_22 = None
        add_79: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-12);  getitem_88 = None
        rsqrt_22: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
        sub_23: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_78, getitem_89);  add_78 = getitem_89 = None
        mul_77: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_22);  sub_23 = rsqrt_22 = None
        mul_78: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_77, arg183_1);  mul_77 = arg183_1 = None
        add_80: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_78, arg184_1);  mul_78 = arg184_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:363 in forward, code: self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        view_176: "f32[1024, 768]" = torch.ops.aten.view.default(add_80, [1024, 768])
        permute_110: "f32[768, 768]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        addmm_66: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg186_1, view_176, permute_110);  arg186_1 = view_176 = permute_110 = None
        view_177: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_66, [8, 128, 768]);  addmm_66 = None
        view_178: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(view_177, [8, -1, 12, 64]);  view_177 = None
        permute_111: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:386 in forward, code: self.key(current_states)
        view_179: "f32[1024, 768]" = torch.ops.aten.view.default(add_80, [1024, 768])
        permute_112: "f32[768, 768]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        addmm_67: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg188_1, view_179, permute_112);  arg188_1 = view_179 = permute_112 = None
        view_180: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_67, [8, 128, 768]);  addmm_67 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:387 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_181: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(view_180, [8, -1, 12, 64]);  view_180 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:388 in forward, code: .transpose(1, 2)
        permute_113: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:391 in forward, code: self.value(current_states)
        view_182: "f32[1024, 768]" = torch.ops.aten.view.default(add_80, [1024, 768])
        permute_114: "f32[768, 768]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        addmm_68: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg190_1, view_182, permute_114);  arg190_1 = view_182 = permute_114 = None
        view_183: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_68, [8, 128, 768]);  addmm_68 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:392 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_184: "f32[8, 128, 12, 64]" = torch.ops.aten.view.default(view_183, [8, -1, 12, 64]);  view_183 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:393 in forward, code: .transpose(1, 2)
        permute_115: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:412 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_13: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(where, [8, 12, 128, 128]);  where = None
        _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_111, permute_113, permute_115, expand_13, False);  permute_111 = permute_113 = permute_115 = expand_13 = None
        getitem_90: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_11[0];  _scaled_dot_product_efficient_attention_11 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:421 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_116: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_90, [0, 2, 1, 3]);  getitem_90 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:422 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_185: "f32[8, 128, 768]" = torch.ops.aten.view.default(permute_116, [8, 128, 768]);  permute_116 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:435 in forward, code: hidden_states = self.dense(hidden_states)
        view_186: "f32[1024, 768]" = torch.ops.aten.view.default(view_185, [1024, 768]);  view_185 = None
        permute_117: "f32[768, 768]" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        addmm_69: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg192_1, view_186, permute_117);  arg192_1 = view_186 = permute_117 = None
        view_187: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_69, [8, 128, 768]);  addmm_69 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:437 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_81: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_187, add_80);  view_187 = add_80 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
        getitem_94: "f32[8, 128, 1]" = var_mean_23[0]
        getitem_95: "f32[8, 128, 1]" = var_mean_23[1];  var_mean_23 = None
        add_82: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-12);  getitem_94 = None
        rsqrt_23: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        sub_24: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_81, getitem_95);  add_81 = getitem_95 = None
        mul_79: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_23);  sub_24 = rsqrt_23 = None
        mul_80: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_79, arg193_1);  mul_79 = arg193_1 = None
        add_83: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_80, arg194_1);  mul_80 = arg194_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:511 in forward, code: hidden_states = self.dense(hidden_states)
        view_188: "f32[1024, 768]" = torch.ops.aten.view.default(add_83, [1024, 768])
        permute_118: "f32[768, 3072]" = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
        addmm_70: "f32[1024, 3072]" = torch.ops.aten.addmm.default(arg196_1, view_188, permute_118);  arg196_1 = view_188 = permute_118 = None
        view_189: "f32[8, 128, 3072]" = torch.ops.aten.view.default(addmm_70, [8, 128, 3072]);  addmm_70 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:70 in forward, code: return self.act(input)
        mul_81: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_189, 0.5)
        mul_82: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_189, 0.7071067811865476);  view_189 = None
        erf_11: "f32[8, 128, 3072]" = torch.ops.aten.erf.default(mul_82);  mul_82 = None
        add_84: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_83: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_81, add_84);  mul_81 = add_84 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:524 in forward, code: hidden_states = self.dense(hidden_states)
        view_190: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_83, [1024, 3072]);  mul_83 = None
        permute_119: "f32[3072, 768]" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
        addmm_71: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg198_1, view_190, permute_119);  arg198_1 = view_190 = permute_119 = None
        view_191: "f32[8, 128, 768]" = torch.ops.aten.view.default(addmm_71, [8, 128, 768]);  addmm_71 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:526 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_85: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_191, add_83);  view_191 = add_83 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
        getitem_96: "f32[8, 128, 1]" = var_mean_24[0]
        getitem_97: "f32[8, 128, 1]" = var_mean_24[1];  var_mean_24 = None
        add_86: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-12);  getitem_96 = None
        rsqrt_24: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        sub_25: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_85, getitem_97);  add_85 = getitem_97 = None
        mul_84: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_24);  sub_25 = rsqrt_24 = None
        mul_85: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_84, arg199_1);  mul_84 = arg199_1 = None
        add_87: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_85, arg200_1);  mul_85 = arg200_1 = None
        return (add_87,)
        

# ===== inductor generated file: /tmp/cnnbench-bert-fold-weaagmnf/repeat_02/base/attempt_01/torchinductor/tmp5mgaaf9a/m5/cm5rpuin46rzdtknz5aqqo3766n3gtq6mrme23rqyrl6joqahvar.debug/fx_graph_runnable.py =====

import os
os.environ['TORCHINDUCTOR_FORCE_DISABLE_CACHES'] = '1'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/cnnbench-bert-fold-weaagmnf/repeat_02/base/attempt_01/torchinductor/tmp5mgaaf9a'
os.environ['TRITON_CACHE_DIR'] = '/tmp/cnnbench-bert-fold-weaagmnf/repeat_02/base/attempt_01/torchinductor/tmp5mgaaf9a/triton'

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
        self.register_buffer('_tensor_constant0', tensor(1.))

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1):
        slice_1 = torch.ops.aten.slice.Tensor(arg1_1, 1, 0, 128);  arg1_1 = None
        expand = torch.ops.aten.expand.default(slice_1, [8, 128]);  slice_1 = None
        slice_2 = torch.ops.aten.slice.Tensor(arg2_1, 1, 0, 128);  arg2_1 = None
        embedding = torch.ops.aten.embedding.default(arg3_1, arg0_1, 0);  arg3_1 = arg0_1 = None
        embedding_1 = torch.ops.aten.embedding.default(arg4_1, expand);  arg4_1 = expand = None
        add = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        embedding_2 = torch.ops.aten.embedding.default(arg5_1, slice_2);  arg5_1 = slice_2 = None
        add_1 = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_2 = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, arg6_1);  mul = arg6_1 = None
        add_3 = torch.ops.aten.add.Tensor(mul_1, arg7_1);  mul_1 = arg7_1 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(arg8_1, 1);  arg8_1 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        expand_1 = torch.ops.aten.expand.default(unsqueeze_1, [8, 1, 128, 128]);  unsqueeze_1 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(expand_1, torch.float32);  expand_1 = None
        _tensor_constant0 = self._tensor_constant0;  _tensor_constant0 = None
        full_default = torch.ops.aten.full.default([], 1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        sub_1 = torch.ops.aten.sub.Tensor(full_default, convert_element_type);  full_default = convert_element_type = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(sub_1, torch.bool)
        full_default_1 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(convert_element_type_1, full_default_1, sub_1);  convert_element_type_1 = full_default_1 = sub_1 = None
        view = torch.ops.aten.view.default(add_3, [1024, 768])
        permute = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm = torch.ops.aten.addmm.default(arg10_1, view, permute);  arg10_1 = view = permute = None
        view_1 = torch.ops.aten.view.default(addmm, [8, 128, 768]);  addmm = None
        view_2 = torch.ops.aten.view.default(view_1, [8, -1, 12, 64]);  view_1 = None
        permute_1 = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        view_3 = torch.ops.aten.view.default(add_3, [1024, 768])
        permute_2 = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg12_1, view_3, permute_2);  arg12_1 = view_3 = permute_2 = None
        view_4 = torch.ops.aten.view.default(addmm_1, [8, 128, 768]);  addmm_1 = None
        view_5 = torch.ops.aten.view.default(view_4, [8, -1, 12, 64]);  view_4 = None
        permute_3 = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        view_6 = torch.ops.aten.view.default(add_3, [1024, 768])
        permute_4 = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_2 = torch.ops.aten.addmm.default(arg14_1, view_6, permute_4);  arg14_1 = view_6 = permute_4 = None
        view_7 = torch.ops.aten.view.default(addmm_2, [8, 128, 768]);  addmm_2 = None
        view_8 = torch.ops.aten.view.default(view_7, [8, -1, 12, 64]);  view_7 = None
        permute_5 = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        expand_2 = torch.ops.aten.expand.default(where, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_1, permute_3, permute_5, expand_2, False);  permute_1 = permute_3 = permute_5 = expand_2 = None
        getitem_2 = _scaled_dot_product_efficient_attention[0];  _scaled_dot_product_efficient_attention = None
        permute_6 = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3]);  getitem_2 = None
        view_9 = torch.ops.aten.view.default(permute_6, [8, 128, 768]);  permute_6 = None
        view_10 = torch.ops.aten.view.default(view_9, [1024, 768]);  view_9 = None
        permute_7 = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_3 = torch.ops.aten.addmm.default(arg16_1, view_10, permute_7);  arg16_1 = view_10 = permute_7 = None
        view_11 = torch.ops.aten.view.default(addmm_3, [8, 128, 768]);  addmm_3 = None
        add_4 = torch.ops.aten.add.Tensor(view_11, add_3);  view_11 = add_3 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_1[0]
        getitem_7 = var_mean_1[1];  var_mean_1 = None
        add_5 = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_2 = torch.ops.aten.sub.Tensor(add_4, getitem_7);  add_4 = getitem_7 = None
        mul_2 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_3 = torch.ops.aten.mul.Tensor(mul_2, arg17_1);  mul_2 = arg17_1 = None
        add_6 = torch.ops.aten.add.Tensor(mul_3, arg18_1);  mul_3 = arg18_1 = None
        view_12 = torch.ops.aten.view.default(add_6, [1024, 768])
        permute_8 = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        addmm_4 = torch.ops.aten.addmm.default(arg20_1, view_12, permute_8);  arg20_1 = view_12 = permute_8 = None
        view_13 = torch.ops.aten.view.default(addmm_4, [8, 128, 3072]);  addmm_4 = None
        mul_4 = torch.ops.aten.mul.Tensor(view_13, 0.5)
        mul_5 = torch.ops.aten.mul.Tensor(view_13, 0.7071067811865476);  view_13 = None
        erf = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_7 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6 = torch.ops.aten.mul.Tensor(mul_4, add_7);  mul_4 = add_7 = None
        view_14 = torch.ops.aten.view.default(mul_6, [1024, 3072]);  mul_6 = None
        permute_9 = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_5 = torch.ops.aten.addmm.default(arg22_1, view_14, permute_9);  arg22_1 = view_14 = permute_9 = None
        view_15 = torch.ops.aten.view.default(addmm_5, [8, 128, 768]);  addmm_5 = None
        add_8 = torch.ops.aten.add.Tensor(view_15, add_6);  view_15 = add_6 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
        getitem_8 = var_mean_2[0]
        getitem_9 = var_mean_2[1];  var_mean_2 = None
        add_9 = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        sub_3 = torch.ops.aten.sub.Tensor(add_8, getitem_9);  add_8 = getitem_9 = None
        mul_7 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_8 = torch.ops.aten.mul.Tensor(mul_7, arg23_1);  mul_7 = arg23_1 = None
        add_10 = torch.ops.aten.add.Tensor(mul_8, arg24_1);  mul_8 = arg24_1 = None
        view_16 = torch.ops.aten.view.default(add_10, [1024, 768])
        permute_10 = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_6 = torch.ops.aten.addmm.default(arg26_1, view_16, permute_10);  arg26_1 = view_16 = permute_10 = None
        view_17 = torch.ops.aten.view.default(addmm_6, [8, 128, 768]);  addmm_6 = None
        view_18 = torch.ops.aten.view.default(view_17, [8, -1, 12, 64]);  view_17 = None
        permute_11 = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        view_19 = torch.ops.aten.view.default(add_10, [1024, 768])
        permute_12 = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_7 = torch.ops.aten.addmm.default(arg28_1, view_19, permute_12);  arg28_1 = view_19 = permute_12 = None
        view_20 = torch.ops.aten.view.default(addmm_7, [8, 128, 768]);  addmm_7 = None
        view_21 = torch.ops.aten.view.default(view_20, [8, -1, 12, 64]);  view_20 = None
        permute_13 = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
        view_22 = torch.ops.aten.view.default(add_10, [1024, 768])
        permute_14 = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        addmm_8 = torch.ops.aten.addmm.default(arg30_1, view_22, permute_14);  arg30_1 = view_22 = permute_14 = None
        view_23 = torch.ops.aten.view.default(addmm_8, [8, 128, 768]);  addmm_8 = None
        view_24 = torch.ops.aten.view.default(view_23, [8, -1, 12, 64]);  view_23 = None
        permute_15 = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        expand_3 = torch.ops.aten.expand.default(where, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_11, permute_13, permute_15, expand_3, False);  permute_11 = permute_13 = permute_15 = expand_3 = None
        getitem_10 = _scaled_dot_product_efficient_attention_1[0];  _scaled_dot_product_efficient_attention_1 = None
        permute_16 = torch.ops.aten.permute.default(getitem_10, [0, 2, 1, 3]);  getitem_10 = None
        view_25 = torch.ops.aten.view.default(permute_16, [8, 128, 768]);  permute_16 = None
        view_26 = torch.ops.aten.view.default(view_25, [1024, 768]);  view_25 = None
        permute_17 = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_9 = torch.ops.aten.addmm.default(arg32_1, view_26, permute_17);  arg32_1 = view_26 = permute_17 = None
        view_27 = torch.ops.aten.view.default(addmm_9, [8, 128, 768]);  addmm_9 = None
        add_11 = torch.ops.aten.add.Tensor(view_27, add_10);  view_27 = add_10 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
        getitem_14 = var_mean_3[0]
        getitem_15 = var_mean_3[1];  var_mean_3 = None
        add_12 = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        sub_4 = torch.ops.aten.sub.Tensor(add_11, getitem_15);  add_11 = getitem_15 = None
        mul_9 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = rsqrt_3 = None
        mul_10 = torch.ops.aten.mul.Tensor(mul_9, arg33_1);  mul_9 = arg33_1 = None
        add_13 = torch.ops.aten.add.Tensor(mul_10, arg34_1);  mul_10 = arg34_1 = None
        view_28 = torch.ops.aten.view.default(add_13, [1024, 768])
        permute_18 = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        addmm_10 = torch.ops.aten.addmm.default(arg36_1, view_28, permute_18);  arg36_1 = view_28 = permute_18 = None
        view_29 = torch.ops.aten.view.default(addmm_10, [8, 128, 3072]);  addmm_10 = None
        mul_11 = torch.ops.aten.mul.Tensor(view_29, 0.5)
        mul_12 = torch.ops.aten.mul.Tensor(view_29, 0.7071067811865476);  view_29 = None
        erf_1 = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_14 = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_11, add_14);  mul_11 = add_14 = None
        view_30 = torch.ops.aten.view.default(mul_13, [1024, 3072]);  mul_13 = None
        permute_19 = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        addmm_11 = torch.ops.aten.addmm.default(arg38_1, view_30, permute_19);  arg38_1 = view_30 = permute_19 = None
        view_31 = torch.ops.aten.view.default(addmm_11, [8, 128, 768]);  addmm_11 = None
        add_15 = torch.ops.aten.add.Tensor(view_31, add_13);  view_31 = add_13 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
        getitem_16 = var_mean_4[0]
        getitem_17 = var_mean_4[1];  var_mean_4 = None
        add_16 = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_5 = torch.ops.aten.sub.Tensor(add_15, getitem_17);  add_15 = getitem_17 = None
        mul_14 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_4);  sub_5 = rsqrt_4 = None
        mul_15 = torch.ops.aten.mul.Tensor(mul_14, arg39_1);  mul_14 = arg39_1 = None
        add_17 = torch.ops.aten.add.Tensor(mul_15, arg40_1);  mul_15 = arg40_1 = None
        view_32 = torch.ops.aten.view.default(add_17, [1024, 768])
        permute_20 = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_12 = torch.ops.aten.addmm.default(arg42_1, view_32, permute_20);  arg42_1 = view_32 = permute_20 = None
        view_33 = torch.ops.aten.view.default(addmm_12, [8, 128, 768]);  addmm_12 = None
        view_34 = torch.ops.aten.view.default(view_33, [8, -1, 12, 64]);  view_33 = None
        permute_21 = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None
        view_35 = torch.ops.aten.view.default(add_17, [1024, 768])
        permute_22 = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_13 = torch.ops.aten.addmm.default(arg44_1, view_35, permute_22);  arg44_1 = view_35 = permute_22 = None
        view_36 = torch.ops.aten.view.default(addmm_13, [8, 128, 768]);  addmm_13 = None
        view_37 = torch.ops.aten.view.default(view_36, [8, -1, 12, 64]);  view_36 = None
        permute_23 = torch.ops.aten.permute.default(view_37, [0, 2, 1, 3]);  view_37 = None
        view_38 = torch.ops.aten.view.default(add_17, [1024, 768])
        permute_24 = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        addmm_14 = torch.ops.aten.addmm.default(arg46_1, view_38, permute_24);  arg46_1 = view_38 = permute_24 = None
        view_39 = torch.ops.aten.view.default(addmm_14, [8, 128, 768]);  addmm_14 = None
        view_40 = torch.ops.aten.view.default(view_39, [8, -1, 12, 64]);  view_39 = None
        permute_25 = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        expand_4 = torch.ops.aten.expand.default(where, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_21, permute_23, permute_25, expand_4, False);  permute_21 = permute_23 = permute_25 = expand_4 = None
        getitem_18 = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None
        permute_26 = torch.ops.aten.permute.default(getitem_18, [0, 2, 1, 3]);  getitem_18 = None
        view_41 = torch.ops.aten.view.default(permute_26, [8, 128, 768]);  permute_26 = None
        view_42 = torch.ops.aten.view.default(view_41, [1024, 768]);  view_41 = None
        permute_27 = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_15 = torch.ops.aten.addmm.default(arg48_1, view_42, permute_27);  arg48_1 = view_42 = permute_27 = None
        view_43 = torch.ops.aten.view.default(addmm_15, [8, 128, 768]);  addmm_15 = None
        add_18 = torch.ops.aten.add.Tensor(view_43, add_17);  view_43 = add_17 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_18, [2], correction = 0, keepdim = True)
        getitem_22 = var_mean_5[0]
        getitem_23 = var_mean_5[1];  var_mean_5 = None
        add_19 = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        sub_6 = torch.ops.aten.sub.Tensor(add_18, getitem_23);  add_18 = getitem_23 = None
        mul_16 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_5);  sub_6 = rsqrt_5 = None
        mul_17 = torch.ops.aten.mul.Tensor(mul_16, arg49_1);  mul_16 = arg49_1 = None
        add_20 = torch.ops.aten.add.Tensor(mul_17, arg50_1);  mul_17 = arg50_1 = None
        view_44 = torch.ops.aten.view.default(add_20, [1024, 768])
        permute_28 = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        addmm_16 = torch.ops.aten.addmm.default(arg52_1, view_44, permute_28);  arg52_1 = view_44 = permute_28 = None
        view_45 = torch.ops.aten.view.default(addmm_16, [8, 128, 3072]);  addmm_16 = None
        mul_18 = torch.ops.aten.mul.Tensor(view_45, 0.5)
        mul_19 = torch.ops.aten.mul.Tensor(view_45, 0.7071067811865476);  view_45 = None
        erf_2 = torch.ops.aten.erf.default(mul_19);  mul_19 = None
        add_21 = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_20 = torch.ops.aten.mul.Tensor(mul_18, add_21);  mul_18 = add_21 = None
        view_46 = torch.ops.aten.view.default(mul_20, [1024, 3072]);  mul_20 = None
        permute_29 = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_17 = torch.ops.aten.addmm.default(arg54_1, view_46, permute_29);  arg54_1 = view_46 = permute_29 = None
        view_47 = torch.ops.aten.view.default(addmm_17, [8, 128, 768]);  addmm_17 = None
        add_22 = torch.ops.aten.add.Tensor(view_47, add_20);  view_47 = add_20 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_22, [2], correction = 0, keepdim = True)
        getitem_24 = var_mean_6[0]
        getitem_25 = var_mean_6[1];  var_mean_6 = None
        add_23 = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        sub_7 = torch.ops.aten.sub.Tensor(add_22, getitem_25);  add_22 = getitem_25 = None
        mul_21 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_6);  sub_7 = rsqrt_6 = None
        mul_22 = torch.ops.aten.mul.Tensor(mul_21, arg55_1);  mul_21 = arg55_1 = None
        add_24 = torch.ops.aten.add.Tensor(mul_22, arg56_1);  mul_22 = arg56_1 = None
        view_48 = torch.ops.aten.view.default(add_24, [1024, 768])
        permute_30 = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        addmm_18 = torch.ops.aten.addmm.default(arg58_1, view_48, permute_30);  arg58_1 = view_48 = permute_30 = None
        view_49 = torch.ops.aten.view.default(addmm_18, [8, 128, 768]);  addmm_18 = None
        view_50 = torch.ops.aten.view.default(view_49, [8, -1, 12, 64]);  view_49 = None
        permute_31 = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        view_51 = torch.ops.aten.view.default(add_24, [1024, 768])
        permute_32 = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        addmm_19 = torch.ops.aten.addmm.default(arg60_1, view_51, permute_32);  arg60_1 = view_51 = permute_32 = None
        view_52 = torch.ops.aten.view.default(addmm_19, [8, 128, 768]);  addmm_19 = None
        view_53 = torch.ops.aten.view.default(view_52, [8, -1, 12, 64]);  view_52 = None
        permute_33 = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
        view_54 = torch.ops.aten.view.default(add_24, [1024, 768])
        permute_34 = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm_20 = torch.ops.aten.addmm.default(arg62_1, view_54, permute_34);  arg62_1 = view_54 = permute_34 = None
        view_55 = torch.ops.aten.view.default(addmm_20, [8, 128, 768]);  addmm_20 = None
        view_56 = torch.ops.aten.view.default(view_55, [8, -1, 12, 64]);  view_55 = None
        permute_35 = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None
        expand_5 = torch.ops.aten.expand.default(where, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_31, permute_33, permute_35, expand_5, False);  permute_31 = permute_33 = permute_35 = expand_5 = None
        getitem_26 = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None
        permute_36 = torch.ops.aten.permute.default(getitem_26, [0, 2, 1, 3]);  getitem_26 = None
        view_57 = torch.ops.aten.view.default(permute_36, [8, 128, 768]);  permute_36 = None
        view_58 = torch.ops.aten.view.default(view_57, [1024, 768]);  view_57 = None
        permute_37 = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        addmm_21 = torch.ops.aten.addmm.default(arg64_1, view_58, permute_37);  arg64_1 = view_58 = permute_37 = None
        view_59 = torch.ops.aten.view.default(addmm_21, [8, 128, 768]);  addmm_21 = None
        add_25 = torch.ops.aten.add.Tensor(view_59, add_24);  view_59 = add_24 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_30 = var_mean_7[0]
        getitem_31 = var_mean_7[1];  var_mean_7 = None
        add_26 = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_8 = torch.ops.aten.sub.Tensor(add_25, getitem_31);  add_25 = getitem_31 = None
        mul_23 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_7);  sub_8 = rsqrt_7 = None
        mul_24 = torch.ops.aten.mul.Tensor(mul_23, arg65_1);  mul_23 = arg65_1 = None
        add_27 = torch.ops.aten.add.Tensor(mul_24, arg66_1);  mul_24 = arg66_1 = None
        view_60 = torch.ops.aten.view.default(add_27, [1024, 768])
        permute_38 = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        addmm_22 = torch.ops.aten.addmm.default(arg68_1, view_60, permute_38);  arg68_1 = view_60 = permute_38 = None
        view_61 = torch.ops.aten.view.default(addmm_22, [8, 128, 3072]);  addmm_22 = None
        mul_25 = torch.ops.aten.mul.Tensor(view_61, 0.5)
        mul_26 = torch.ops.aten.mul.Tensor(view_61, 0.7071067811865476);  view_61 = None
        erf_3 = torch.ops.aten.erf.default(mul_26);  mul_26 = None
        add_28 = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_27 = torch.ops.aten.mul.Tensor(mul_25, add_28);  mul_25 = add_28 = None
        view_62 = torch.ops.aten.view.default(mul_27, [1024, 3072]);  mul_27 = None
        permute_39 = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        addmm_23 = torch.ops.aten.addmm.default(arg70_1, view_62, permute_39);  arg70_1 = view_62 = permute_39 = None
        view_63 = torch.ops.aten.view.default(addmm_23, [8, 128, 768]);  addmm_23 = None
        add_29 = torch.ops.aten.add.Tensor(view_63, add_27);  view_63 = add_27 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
        getitem_32 = var_mean_8[0]
        getitem_33 = var_mean_8[1];  var_mean_8 = None
        add_30 = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        sub_9 = torch.ops.aten.sub.Tensor(add_29, getitem_33);  add_29 = getitem_33 = None
        mul_28 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_8);  sub_9 = rsqrt_8 = None
        mul_29 = torch.ops.aten.mul.Tensor(mul_28, arg71_1);  mul_28 = arg71_1 = None
        add_31 = torch.ops.aten.add.Tensor(mul_29, arg72_1);  mul_29 = arg72_1 = None
        view_64 = torch.ops.aten.view.default(add_31, [1024, 768])
        permute_40 = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_24 = torch.ops.aten.addmm.default(arg74_1, view_64, permute_40);  arg74_1 = view_64 = permute_40 = None
        view_65 = torch.ops.aten.view.default(addmm_24, [8, 128, 768]);  addmm_24 = None
        view_66 = torch.ops.aten.view.default(view_65, [8, -1, 12, 64]);  view_65 = None
        permute_41 = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
        view_67 = torch.ops.aten.view.default(add_31, [1024, 768])
        permute_42 = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        addmm_25 = torch.ops.aten.addmm.default(arg76_1, view_67, permute_42);  arg76_1 = view_67 = permute_42 = None
        view_68 = torch.ops.aten.view.default(addmm_25, [8, 128, 768]);  addmm_25 = None
        view_69 = torch.ops.aten.view.default(view_68, [8, -1, 12, 64]);  view_68 = None
        permute_43 = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None
        view_70 = torch.ops.aten.view.default(add_31, [1024, 768])
        permute_44 = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        addmm_26 = torch.ops.aten.addmm.default(arg78_1, view_70, permute_44);  arg78_1 = view_70 = permute_44 = None
        view_71 = torch.ops.aten.view.default(addmm_26, [8, 128, 768]);  addmm_26 = None
        view_72 = torch.ops.aten.view.default(view_71, [8, -1, 12, 64]);  view_71 = None
        permute_45 = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
        expand_6 = torch.ops.aten.expand.default(where, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_41, permute_43, permute_45, expand_6, False);  permute_41 = permute_43 = permute_45 = expand_6 = None
        getitem_34 = _scaled_dot_product_efficient_attention_4[0];  _scaled_dot_product_efficient_attention_4 = None
        permute_46 = torch.ops.aten.permute.default(getitem_34, [0, 2, 1, 3]);  getitem_34 = None
        view_73 = torch.ops.aten.view.default(permute_46, [8, 128, 768]);  permute_46 = None
        view_74 = torch.ops.aten.view.default(view_73, [1024, 768]);  view_73 = None
        permute_47 = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_27 = torch.ops.aten.addmm.default(arg80_1, view_74, permute_47);  arg80_1 = view_74 = permute_47 = None
        view_75 = torch.ops.aten.view.default(addmm_27, [8, 128, 768]);  addmm_27 = None
        add_32 = torch.ops.aten.add.Tensor(view_75, add_31);  view_75 = add_31 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
        getitem_38 = var_mean_9[0]
        getitem_39 = var_mean_9[1];  var_mean_9 = None
        add_33 = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        sub_10 = torch.ops.aten.sub.Tensor(add_32, getitem_39);  add_32 = getitem_39 = None
        mul_30 = torch.ops.aten.mul.Tensor(sub_10, rsqrt_9);  sub_10 = rsqrt_9 = None
        mul_31 = torch.ops.aten.mul.Tensor(mul_30, arg81_1);  mul_30 = arg81_1 = None
        add_34 = torch.ops.aten.add.Tensor(mul_31, arg82_1);  mul_31 = arg82_1 = None
        view_76 = torch.ops.aten.view.default(add_34, [1024, 768])
        permute_48 = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        addmm_28 = torch.ops.aten.addmm.default(arg84_1, view_76, permute_48);  arg84_1 = view_76 = permute_48 = None
        view_77 = torch.ops.aten.view.default(addmm_28, [8, 128, 3072]);  addmm_28 = None
        mul_32 = torch.ops.aten.mul.Tensor(view_77, 0.5)
        mul_33 = torch.ops.aten.mul.Tensor(view_77, 0.7071067811865476);  view_77 = None
        erf_4 = torch.ops.aten.erf.default(mul_33);  mul_33 = None
        add_35 = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_34 = torch.ops.aten.mul.Tensor(mul_32, add_35);  mul_32 = add_35 = None
        view_78 = torch.ops.aten.view.default(mul_34, [1024, 3072]);  mul_34 = None
        permute_49 = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_29 = torch.ops.aten.addmm.default(arg86_1, view_78, permute_49);  arg86_1 = view_78 = permute_49 = None
        view_79 = torch.ops.aten.view.default(addmm_29, [8, 128, 768]);  addmm_29 = None
        add_36 = torch.ops.aten.add.Tensor(view_79, add_34);  view_79 = add_34 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_36, [2], correction = 0, keepdim = True)
        getitem_40 = var_mean_10[0]
        getitem_41 = var_mean_10[1];  var_mean_10 = None
        add_37 = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        sub_11 = torch.ops.aten.sub.Tensor(add_36, getitem_41);  add_36 = getitem_41 = None
        mul_35 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_10);  sub_11 = rsqrt_10 = None
        mul_36 = torch.ops.aten.mul.Tensor(mul_35, arg87_1);  mul_35 = arg87_1 = None
        add_38 = torch.ops.aten.add.Tensor(mul_36, arg88_1);  mul_36 = arg88_1 = None
        view_80 = torch.ops.aten.view.default(add_38, [1024, 768])
        permute_50 = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        addmm_30 = torch.ops.aten.addmm.default(arg90_1, view_80, permute_50);  arg90_1 = view_80 = permute_50 = None
        view_81 = torch.ops.aten.view.default(addmm_30, [8, 128, 768]);  addmm_30 = None
        view_82 = torch.ops.aten.view.default(view_81, [8, -1, 12, 64]);  view_81 = None
        permute_51 = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
        view_83 = torch.ops.aten.view.default(add_38, [1024, 768])
        permute_52 = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_31 = torch.ops.aten.addmm.default(arg92_1, view_83, permute_52);  arg92_1 = view_83 = permute_52 = None
        view_84 = torch.ops.aten.view.default(addmm_31, [8, 128, 768]);  addmm_31 = None
        view_85 = torch.ops.aten.view.default(view_84, [8, -1, 12, 64]);  view_84 = None
        permute_53 = torch.ops.aten.permute.default(view_85, [0, 2, 1, 3]);  view_85 = None
        view_86 = torch.ops.aten.view.default(add_38, [1024, 768])
        permute_54 = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        addmm_32 = torch.ops.aten.addmm.default(arg94_1, view_86, permute_54);  arg94_1 = view_86 = permute_54 = None
        view_87 = torch.ops.aten.view.default(addmm_32, [8, 128, 768]);  addmm_32 = None
        view_88 = torch.ops.aten.view.default(view_87, [8, -1, 12, 64]);  view_87 = None
        permute_55 = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
        expand_7 = torch.ops.aten.expand.default(where, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_51, permute_53, permute_55, expand_7, False);  permute_51 = permute_53 = permute_55 = expand_7 = None
        getitem_42 = _scaled_dot_product_efficient_attention_5[0];  _scaled_dot_product_efficient_attention_5 = None
        permute_56 = torch.ops.aten.permute.default(getitem_42, [0, 2, 1, 3]);  getitem_42 = None
        view_89 = torch.ops.aten.view.default(permute_56, [8, 128, 768]);  permute_56 = None
        view_90 = torch.ops.aten.view.default(view_89, [1024, 768]);  view_89 = None
        permute_57 = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_33 = torch.ops.aten.addmm.default(arg96_1, view_90, permute_57);  arg96_1 = view_90 = permute_57 = None
        view_91 = torch.ops.aten.view.default(addmm_33, [8, 128, 768]);  addmm_33 = None
        add_39 = torch.ops.aten.add.Tensor(view_91, add_38);  view_91 = add_38 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_39, [2], correction = 0, keepdim = True)
        getitem_46 = var_mean_11[0]
        getitem_47 = var_mean_11[1];  var_mean_11 = None
        add_40 = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        sub_12 = torch.ops.aten.sub.Tensor(add_39, getitem_47);  add_39 = getitem_47 = None
        mul_37 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_11);  sub_12 = rsqrt_11 = None
        mul_38 = torch.ops.aten.mul.Tensor(mul_37, arg97_1);  mul_37 = arg97_1 = None
        add_41 = torch.ops.aten.add.Tensor(mul_38, arg98_1);  mul_38 = arg98_1 = None
        view_92 = torch.ops.aten.view.default(add_41, [1024, 768])
        permute_58 = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        addmm_34 = torch.ops.aten.addmm.default(arg100_1, view_92, permute_58);  arg100_1 = view_92 = permute_58 = None
        view_93 = torch.ops.aten.view.default(addmm_34, [8, 128, 3072]);  addmm_34 = None
        mul_39 = torch.ops.aten.mul.Tensor(view_93, 0.5)
        mul_40 = torch.ops.aten.mul.Tensor(view_93, 0.7071067811865476);  view_93 = None
        erf_5 = torch.ops.aten.erf.default(mul_40);  mul_40 = None
        add_42 = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_39, add_42);  mul_39 = add_42 = None
        view_94 = torch.ops.aten.view.default(mul_41, [1024, 3072]);  mul_41 = None
        permute_59 = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        addmm_35 = torch.ops.aten.addmm.default(arg102_1, view_94, permute_59);  arg102_1 = view_94 = permute_59 = None
        view_95 = torch.ops.aten.view.default(addmm_35, [8, 128, 768]);  addmm_35 = None
        add_43 = torch.ops.aten.add.Tensor(view_95, add_41);  view_95 = add_41 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
        getitem_48 = var_mean_12[0]
        getitem_49 = var_mean_12[1];  var_mean_12 = None
        add_44 = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        sub_13 = torch.ops.aten.sub.Tensor(add_43, getitem_49);  add_43 = getitem_49 = None
        mul_42 = torch.ops.aten.mul.Tensor(sub_13, rsqrt_12);  sub_13 = rsqrt_12 = None
        mul_43 = torch.ops.aten.mul.Tensor(mul_42, arg103_1);  mul_42 = arg103_1 = None
        add_45 = torch.ops.aten.add.Tensor(mul_43, arg104_1);  mul_43 = arg104_1 = None
        view_96 = torch.ops.aten.view.default(add_45, [1024, 768])
        permute_60 = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        addmm_36 = torch.ops.aten.addmm.default(arg106_1, view_96, permute_60);  arg106_1 = view_96 = permute_60 = None
        view_97 = torch.ops.aten.view.default(addmm_36, [8, 128, 768]);  addmm_36 = None
        view_98 = torch.ops.aten.view.default(view_97, [8, -1, 12, 64]);  view_97 = None
        permute_61 = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
        view_99 = torch.ops.aten.view.default(add_45, [1024, 768])
        permute_62 = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_37 = torch.ops.aten.addmm.default(arg108_1, view_99, permute_62);  arg108_1 = view_99 = permute_62 = None
        view_100 = torch.ops.aten.view.default(addmm_37, [8, 128, 768]);  addmm_37 = None
        view_101 = torch.ops.aten.view.default(view_100, [8, -1, 12, 64]);  view_100 = None
        permute_63 = torch.ops.aten.permute.default(view_101, [0, 2, 1, 3]);  view_101 = None
        view_102 = torch.ops.aten.view.default(add_45, [1024, 768])
        permute_64 = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_38 = torch.ops.aten.addmm.default(arg110_1, view_102, permute_64);  arg110_1 = view_102 = permute_64 = None
        view_103 = torch.ops.aten.view.default(addmm_38, [8, 128, 768]);  addmm_38 = None
        view_104 = torch.ops.aten.view.default(view_103, [8, -1, 12, 64]);  view_103 = None
        permute_65 = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
        expand_8 = torch.ops.aten.expand.default(where, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_61, permute_63, permute_65, expand_8, False);  permute_61 = permute_63 = permute_65 = expand_8 = None
        getitem_50 = _scaled_dot_product_efficient_attention_6[0];  _scaled_dot_product_efficient_attention_6 = None
        permute_66 = torch.ops.aten.permute.default(getitem_50, [0, 2, 1, 3]);  getitem_50 = None
        view_105 = torch.ops.aten.view.default(permute_66, [8, 128, 768]);  permute_66 = None
        view_106 = torch.ops.aten.view.default(view_105, [1024, 768]);  view_105 = None
        permute_67 = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        addmm_39 = torch.ops.aten.addmm.default(arg112_1, view_106, permute_67);  arg112_1 = view_106 = permute_67 = None
        view_107 = torch.ops.aten.view.default(addmm_39, [8, 128, 768]);  addmm_39 = None
        add_46 = torch.ops.aten.add.Tensor(view_107, add_45);  view_107 = add_45 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_46, [2], correction = 0, keepdim = True)
        getitem_54 = var_mean_13[0]
        getitem_55 = var_mean_13[1];  var_mean_13 = None
        add_47 = torch.ops.aten.add.Tensor(getitem_54, 1e-12);  getitem_54 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        sub_14 = torch.ops.aten.sub.Tensor(add_46, getitem_55);  add_46 = getitem_55 = None
        mul_44 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_13);  sub_14 = rsqrt_13 = None
        mul_45 = torch.ops.aten.mul.Tensor(mul_44, arg113_1);  mul_44 = arg113_1 = None
        add_48 = torch.ops.aten.add.Tensor(mul_45, arg114_1);  mul_45 = arg114_1 = None
        view_108 = torch.ops.aten.view.default(add_48, [1024, 768])
        permute_68 = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        addmm_40 = torch.ops.aten.addmm.default(arg116_1, view_108, permute_68);  arg116_1 = view_108 = permute_68 = None
        view_109 = torch.ops.aten.view.default(addmm_40, [8, 128, 3072]);  addmm_40 = None
        mul_46 = torch.ops.aten.mul.Tensor(view_109, 0.5)
        mul_47 = torch.ops.aten.mul.Tensor(view_109, 0.7071067811865476);  view_109 = None
        erf_6 = torch.ops.aten.erf.default(mul_47);  mul_47 = None
        add_49 = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_48 = torch.ops.aten.mul.Tensor(mul_46, add_49);  mul_46 = add_49 = None
        view_110 = torch.ops.aten.view.default(mul_48, [1024, 3072]);  mul_48 = None
        permute_69 = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        addmm_41 = torch.ops.aten.addmm.default(arg118_1, view_110, permute_69);  arg118_1 = view_110 = permute_69 = None
        view_111 = torch.ops.aten.view.default(addmm_41, [8, 128, 768]);  addmm_41 = None
        add_50 = torch.ops.aten.add.Tensor(view_111, add_48);  view_111 = add_48 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_50, [2], correction = 0, keepdim = True)
        getitem_56 = var_mean_14[0]
        getitem_57 = var_mean_14[1];  var_mean_14 = None
        add_51 = torch.ops.aten.add.Tensor(getitem_56, 1e-12);  getitem_56 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        sub_15 = torch.ops.aten.sub.Tensor(add_50, getitem_57);  add_50 = getitem_57 = None
        mul_49 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_14);  sub_15 = rsqrt_14 = None
        mul_50 = torch.ops.aten.mul.Tensor(mul_49, arg119_1);  mul_49 = arg119_1 = None
        add_52 = torch.ops.aten.add.Tensor(mul_50, arg120_1);  mul_50 = arg120_1 = None
        view_112 = torch.ops.aten.view.default(add_52, [1024, 768])
        permute_70 = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_42 = torch.ops.aten.addmm.default(arg122_1, view_112, permute_70);  arg122_1 = view_112 = permute_70 = None
        view_113 = torch.ops.aten.view.default(addmm_42, [8, 128, 768]);  addmm_42 = None
        view_114 = torch.ops.aten.view.default(view_113, [8, -1, 12, 64]);  view_113 = None
        permute_71 = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
        view_115 = torch.ops.aten.view.default(add_52, [1024, 768])
        permute_72 = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_43 = torch.ops.aten.addmm.default(arg124_1, view_115, permute_72);  arg124_1 = view_115 = permute_72 = None
        view_116 = torch.ops.aten.view.default(addmm_43, [8, 128, 768]);  addmm_43 = None
        view_117 = torch.ops.aten.view.default(view_116, [8, -1, 12, 64]);  view_116 = None
        permute_73 = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None
        view_118 = torch.ops.aten.view.default(add_52, [1024, 768])
        permute_74 = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        addmm_44 = torch.ops.aten.addmm.default(arg126_1, view_118, permute_74);  arg126_1 = view_118 = permute_74 = None
        view_119 = torch.ops.aten.view.default(addmm_44, [8, 128, 768]);  addmm_44 = None
        view_120 = torch.ops.aten.view.default(view_119, [8, -1, 12, 64]);  view_119 = None
        permute_75 = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
        expand_9 = torch.ops.aten.expand.default(where, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_71, permute_73, permute_75, expand_9, False);  permute_71 = permute_73 = permute_75 = expand_9 = None
        getitem_58 = _scaled_dot_product_efficient_attention_7[0];  _scaled_dot_product_efficient_attention_7 = None
        permute_76 = torch.ops.aten.permute.default(getitem_58, [0, 2, 1, 3]);  getitem_58 = None
        view_121 = torch.ops.aten.view.default(permute_76, [8, 128, 768]);  permute_76 = None
        view_122 = torch.ops.aten.view.default(view_121, [1024, 768]);  view_121 = None
        permute_77 = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        addmm_45 = torch.ops.aten.addmm.default(arg128_1, view_122, permute_77);  arg128_1 = view_122 = permute_77 = None
        view_123 = torch.ops.aten.view.default(addmm_45, [8, 128, 768]);  addmm_45 = None
        add_53 = torch.ops.aten.add.Tensor(view_123, add_52);  view_123 = add_52 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
        getitem_62 = var_mean_15[0]
        getitem_63 = var_mean_15[1];  var_mean_15 = None
        add_54 = torch.ops.aten.add.Tensor(getitem_62, 1e-12);  getitem_62 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        sub_16 = torch.ops.aten.sub.Tensor(add_53, getitem_63);  add_53 = getitem_63 = None
        mul_51 = torch.ops.aten.mul.Tensor(sub_16, rsqrt_15);  sub_16 = rsqrt_15 = None
        mul_52 = torch.ops.aten.mul.Tensor(mul_51, arg129_1);  mul_51 = arg129_1 = None
        add_55 = torch.ops.aten.add.Tensor(mul_52, arg130_1);  mul_52 = arg130_1 = None
        view_124 = torch.ops.aten.view.default(add_55, [1024, 768])
        permute_78 = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        addmm_46 = torch.ops.aten.addmm.default(arg132_1, view_124, permute_78);  arg132_1 = view_124 = permute_78 = None
        view_125 = torch.ops.aten.view.default(addmm_46, [8, 128, 3072]);  addmm_46 = None
        mul_53 = torch.ops.aten.mul.Tensor(view_125, 0.5)
        mul_54 = torch.ops.aten.mul.Tensor(view_125, 0.7071067811865476);  view_125 = None
        erf_7 = torch.ops.aten.erf.default(mul_54);  mul_54 = None
        add_56 = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_55 = torch.ops.aten.mul.Tensor(mul_53, add_56);  mul_53 = add_56 = None
        view_126 = torch.ops.aten.view.default(mul_55, [1024, 3072]);  mul_55 = None
        permute_79 = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_47 = torch.ops.aten.addmm.default(arg134_1, view_126, permute_79);  arg134_1 = view_126 = permute_79 = None
        view_127 = torch.ops.aten.view.default(addmm_47, [8, 128, 768]);  addmm_47 = None
        add_57 = torch.ops.aten.add.Tensor(view_127, add_55);  view_127 = add_55 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_64 = var_mean_16[0]
        getitem_65 = var_mean_16[1];  var_mean_16 = None
        add_58 = torch.ops.aten.add.Tensor(getitem_64, 1e-12);  getitem_64 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_17 = torch.ops.aten.sub.Tensor(add_57, getitem_65);  add_57 = getitem_65 = None
        mul_56 = torch.ops.aten.mul.Tensor(sub_17, rsqrt_16);  sub_17 = rsqrt_16 = None
        mul_57 = torch.ops.aten.mul.Tensor(mul_56, arg135_1);  mul_56 = arg135_1 = None
        add_59 = torch.ops.aten.add.Tensor(mul_57, arg136_1);  mul_57 = arg136_1 = None
        view_128 = torch.ops.aten.view.default(add_59, [1024, 768])
        permute_80 = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        addmm_48 = torch.ops.aten.addmm.default(arg138_1, view_128, permute_80);  arg138_1 = view_128 = permute_80 = None
        view_129 = torch.ops.aten.view.default(addmm_48, [8, 128, 768]);  addmm_48 = None
        view_130 = torch.ops.aten.view.default(view_129, [8, -1, 12, 64]);  view_129 = None
        permute_81 = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
        view_131 = torch.ops.aten.view.default(add_59, [1024, 768])
        permute_82 = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_49 = torch.ops.aten.addmm.default(arg140_1, view_131, permute_82);  arg140_1 = view_131 = permute_82 = None
        view_132 = torch.ops.aten.view.default(addmm_49, [8, 128, 768]);  addmm_49 = None
        view_133 = torch.ops.aten.view.default(view_132, [8, -1, 12, 64]);  view_132 = None
        permute_83 = torch.ops.aten.permute.default(view_133, [0, 2, 1, 3]);  view_133 = None
        view_134 = torch.ops.aten.view.default(add_59, [1024, 768])
        permute_84 = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        addmm_50 = torch.ops.aten.addmm.default(arg142_1, view_134, permute_84);  arg142_1 = view_134 = permute_84 = None
        view_135 = torch.ops.aten.view.default(addmm_50, [8, 128, 768]);  addmm_50 = None
        view_136 = torch.ops.aten.view.default(view_135, [8, -1, 12, 64]);  view_135 = None
        permute_85 = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
        expand_10 = torch.ops.aten.expand.default(where, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_81, permute_83, permute_85, expand_10, False);  permute_81 = permute_83 = permute_85 = expand_10 = None
        getitem_66 = _scaled_dot_product_efficient_attention_8[0];  _scaled_dot_product_efficient_attention_8 = None
        permute_86 = torch.ops.aten.permute.default(getitem_66, [0, 2, 1, 3]);  getitem_66 = None
        view_137 = torch.ops.aten.view.default(permute_86, [8, 128, 768]);  permute_86 = None
        view_138 = torch.ops.aten.view.default(view_137, [1024, 768]);  view_137 = None
        permute_87 = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        addmm_51 = torch.ops.aten.addmm.default(arg144_1, view_138, permute_87);  arg144_1 = view_138 = permute_87 = None
        view_139 = torch.ops.aten.view.default(addmm_51, [8, 128, 768]);  addmm_51 = None
        add_60 = torch.ops.aten.add.Tensor(view_139, add_59);  view_139 = add_59 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_60, [2], correction = 0, keepdim = True)
        getitem_70 = var_mean_17[0]
        getitem_71 = var_mean_17[1];  var_mean_17 = None
        add_61 = torch.ops.aten.add.Tensor(getitem_70, 1e-12);  getitem_70 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        sub_18 = torch.ops.aten.sub.Tensor(add_60, getitem_71);  add_60 = getitem_71 = None
        mul_58 = torch.ops.aten.mul.Tensor(sub_18, rsqrt_17);  sub_18 = rsqrt_17 = None
        mul_59 = torch.ops.aten.mul.Tensor(mul_58, arg145_1);  mul_58 = arg145_1 = None
        add_62 = torch.ops.aten.add.Tensor(mul_59, arg146_1);  mul_59 = arg146_1 = None
        view_140 = torch.ops.aten.view.default(add_62, [1024, 768])
        permute_88 = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        addmm_52 = torch.ops.aten.addmm.default(arg148_1, view_140, permute_88);  arg148_1 = view_140 = permute_88 = None
        view_141 = torch.ops.aten.view.default(addmm_52, [8, 128, 3072]);  addmm_52 = None
        mul_60 = torch.ops.aten.mul.Tensor(view_141, 0.5)
        mul_61 = torch.ops.aten.mul.Tensor(view_141, 0.7071067811865476);  view_141 = None
        erf_8 = torch.ops.aten.erf.default(mul_61);  mul_61 = None
        add_63 = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_62 = torch.ops.aten.mul.Tensor(mul_60, add_63);  mul_60 = add_63 = None
        view_142 = torch.ops.aten.view.default(mul_62, [1024, 3072]);  mul_62 = None
        permute_89 = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        addmm_53 = torch.ops.aten.addmm.default(arg150_1, view_142, permute_89);  arg150_1 = view_142 = permute_89 = None
        view_143 = torch.ops.aten.view.default(addmm_53, [8, 128, 768]);  addmm_53 = None
        add_64 = torch.ops.aten.add.Tensor(view_143, add_62);  view_143 = add_62 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_64, [2], correction = 0, keepdim = True)
        getitem_72 = var_mean_18[0]
        getitem_73 = var_mean_18[1];  var_mean_18 = None
        add_65 = torch.ops.aten.add.Tensor(getitem_72, 1e-12);  getitem_72 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        sub_19 = torch.ops.aten.sub.Tensor(add_64, getitem_73);  add_64 = getitem_73 = None
        mul_63 = torch.ops.aten.mul.Tensor(sub_19, rsqrt_18);  sub_19 = rsqrt_18 = None
        mul_64 = torch.ops.aten.mul.Tensor(mul_63, arg151_1);  mul_63 = arg151_1 = None
        add_66 = torch.ops.aten.add.Tensor(mul_64, arg152_1);  mul_64 = arg152_1 = None
        view_144 = torch.ops.aten.view.default(add_66, [1024, 768])
        permute_90 = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        addmm_54 = torch.ops.aten.addmm.default(arg154_1, view_144, permute_90);  arg154_1 = view_144 = permute_90 = None
        view_145 = torch.ops.aten.view.default(addmm_54, [8, 128, 768]);  addmm_54 = None
        view_146 = torch.ops.aten.view.default(view_145, [8, -1, 12, 64]);  view_145 = None
        permute_91 = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
        view_147 = torch.ops.aten.view.default(add_66, [1024, 768])
        permute_92 = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        addmm_55 = torch.ops.aten.addmm.default(arg156_1, view_147, permute_92);  arg156_1 = view_147 = permute_92 = None
        view_148 = torch.ops.aten.view.default(addmm_55, [8, 128, 768]);  addmm_55 = None
        view_149 = torch.ops.aten.view.default(view_148, [8, -1, 12, 64]);  view_148 = None
        permute_93 = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None
        view_150 = torch.ops.aten.view.default(add_66, [1024, 768])
        permute_94 = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        addmm_56 = torch.ops.aten.addmm.default(arg158_1, view_150, permute_94);  arg158_1 = view_150 = permute_94 = None
        view_151 = torch.ops.aten.view.default(addmm_56, [8, 128, 768]);  addmm_56 = None
        view_152 = torch.ops.aten.view.default(view_151, [8, -1, 12, 64]);  view_151 = None
        permute_95 = torch.ops.aten.permute.default(view_152, [0, 2, 1, 3]);  view_152 = None
        expand_11 = torch.ops.aten.expand.default(where, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_91, permute_93, permute_95, expand_11, False);  permute_91 = permute_93 = permute_95 = expand_11 = None
        getitem_74 = _scaled_dot_product_efficient_attention_9[0];  _scaled_dot_product_efficient_attention_9 = None
        permute_96 = torch.ops.aten.permute.default(getitem_74, [0, 2, 1, 3]);  getitem_74 = None
        view_153 = torch.ops.aten.view.default(permute_96, [8, 128, 768]);  permute_96 = None
        view_154 = torch.ops.aten.view.default(view_153, [1024, 768]);  view_153 = None
        permute_97 = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        addmm_57 = torch.ops.aten.addmm.default(arg160_1, view_154, permute_97);  arg160_1 = view_154 = permute_97 = None
        view_155 = torch.ops.aten.view.default(addmm_57, [8, 128, 768]);  addmm_57 = None
        add_67 = torch.ops.aten.add.Tensor(view_155, add_66);  view_155 = add_66 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
        getitem_78 = var_mean_19[0]
        getitem_79 = var_mean_19[1];  var_mean_19 = None
        add_68 = torch.ops.aten.add.Tensor(getitem_78, 1e-12);  getitem_78 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        sub_20 = torch.ops.aten.sub.Tensor(add_67, getitem_79);  add_67 = getitem_79 = None
        mul_65 = torch.ops.aten.mul.Tensor(sub_20, rsqrt_19);  sub_20 = rsqrt_19 = None
        mul_66 = torch.ops.aten.mul.Tensor(mul_65, arg161_1);  mul_65 = arg161_1 = None
        add_69 = torch.ops.aten.add.Tensor(mul_66, arg162_1);  mul_66 = arg162_1 = None
        view_156 = torch.ops.aten.view.default(add_69, [1024, 768])
        permute_98 = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        addmm_58 = torch.ops.aten.addmm.default(arg164_1, view_156, permute_98);  arg164_1 = view_156 = permute_98 = None
        view_157 = torch.ops.aten.view.default(addmm_58, [8, 128, 3072]);  addmm_58 = None
        mul_67 = torch.ops.aten.mul.Tensor(view_157, 0.5)
        mul_68 = torch.ops.aten.mul.Tensor(view_157, 0.7071067811865476);  view_157 = None
        erf_9 = torch.ops.aten.erf.default(mul_68);  mul_68 = None
        add_70 = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_69 = torch.ops.aten.mul.Tensor(mul_67, add_70);  mul_67 = add_70 = None
        view_158 = torch.ops.aten.view.default(mul_69, [1024, 3072]);  mul_69 = None
        permute_99 = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        addmm_59 = torch.ops.aten.addmm.default(arg166_1, view_158, permute_99);  arg166_1 = view_158 = permute_99 = None
        view_159 = torch.ops.aten.view.default(addmm_59, [8, 128, 768]);  addmm_59 = None
        add_71 = torch.ops.aten.add.Tensor(view_159, add_69);  view_159 = add_69 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
        getitem_80 = var_mean_20[0]
        getitem_81 = var_mean_20[1];  var_mean_20 = None
        add_72 = torch.ops.aten.add.Tensor(getitem_80, 1e-12);  getitem_80 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        sub_21 = torch.ops.aten.sub.Tensor(add_71, getitem_81);  add_71 = getitem_81 = None
        mul_70 = torch.ops.aten.mul.Tensor(sub_21, rsqrt_20);  sub_21 = rsqrt_20 = None
        mul_71 = torch.ops.aten.mul.Tensor(mul_70, arg167_1);  mul_70 = arg167_1 = None
        add_73 = torch.ops.aten.add.Tensor(mul_71, arg168_1);  mul_71 = arg168_1 = None
        view_160 = torch.ops.aten.view.default(add_73, [1024, 768])
        permute_100 = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        addmm_60 = torch.ops.aten.addmm.default(arg170_1, view_160, permute_100);  arg170_1 = view_160 = permute_100 = None
        view_161 = torch.ops.aten.view.default(addmm_60, [8, 128, 768]);  addmm_60 = None
        view_162 = torch.ops.aten.view.default(view_161, [8, -1, 12, 64]);  view_161 = None
        permute_101 = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
        view_163 = torch.ops.aten.view.default(add_73, [1024, 768])
        permute_102 = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        addmm_61 = torch.ops.aten.addmm.default(arg172_1, view_163, permute_102);  arg172_1 = view_163 = permute_102 = None
        view_164 = torch.ops.aten.view.default(addmm_61, [8, 128, 768]);  addmm_61 = None
        view_165 = torch.ops.aten.view.default(view_164, [8, -1, 12, 64]);  view_164 = None
        permute_103 = torch.ops.aten.permute.default(view_165, [0, 2, 1, 3]);  view_165 = None
        view_166 = torch.ops.aten.view.default(add_73, [1024, 768])
        permute_104 = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        addmm_62 = torch.ops.aten.addmm.default(arg174_1, view_166, permute_104);  arg174_1 = view_166 = permute_104 = None
        view_167 = torch.ops.aten.view.default(addmm_62, [8, 128, 768]);  addmm_62 = None
        view_168 = torch.ops.aten.view.default(view_167, [8, -1, 12, 64]);  view_167 = None
        permute_105 = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
        expand_12 = torch.ops.aten.expand.default(where, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_101, permute_103, permute_105, expand_12, False);  permute_101 = permute_103 = permute_105 = expand_12 = None
        getitem_82 = _scaled_dot_product_efficient_attention_10[0];  _scaled_dot_product_efficient_attention_10 = None
        permute_106 = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3]);  getitem_82 = None
        view_169 = torch.ops.aten.view.default(permute_106, [8, 128, 768]);  permute_106 = None
        view_170 = torch.ops.aten.view.default(view_169, [1024, 768]);  view_169 = None
        permute_107 = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        addmm_63 = torch.ops.aten.addmm.default(arg176_1, view_170, permute_107);  arg176_1 = view_170 = permute_107 = None
        view_171 = torch.ops.aten.view.default(addmm_63, [8, 128, 768]);  addmm_63 = None
        add_74 = torch.ops.aten.add.Tensor(view_171, add_73);  view_171 = add_73 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_74, [2], correction = 0, keepdim = True)
        getitem_86 = var_mean_21[0]
        getitem_87 = var_mean_21[1];  var_mean_21 = None
        add_75 = torch.ops.aten.add.Tensor(getitem_86, 1e-12);  getitem_86 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        sub_22 = torch.ops.aten.sub.Tensor(add_74, getitem_87);  add_74 = getitem_87 = None
        mul_72 = torch.ops.aten.mul.Tensor(sub_22, rsqrt_21);  sub_22 = rsqrt_21 = None
        mul_73 = torch.ops.aten.mul.Tensor(mul_72, arg177_1);  mul_72 = arg177_1 = None
        add_76 = torch.ops.aten.add.Tensor(mul_73, arg178_1);  mul_73 = arg178_1 = None
        view_172 = torch.ops.aten.view.default(add_76, [1024, 768])
        permute_108 = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        addmm_64 = torch.ops.aten.addmm.default(arg180_1, view_172, permute_108);  arg180_1 = view_172 = permute_108 = None
        view_173 = torch.ops.aten.view.default(addmm_64, [8, 128, 3072]);  addmm_64 = None
        mul_74 = torch.ops.aten.mul.Tensor(view_173, 0.5)
        mul_75 = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476);  view_173 = None
        erf_10 = torch.ops.aten.erf.default(mul_75);  mul_75 = None
        add_77 = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_76 = torch.ops.aten.mul.Tensor(mul_74, add_77);  mul_74 = add_77 = None
        view_174 = torch.ops.aten.view.default(mul_76, [1024, 3072]);  mul_76 = None
        permute_109 = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        addmm_65 = torch.ops.aten.addmm.default(arg182_1, view_174, permute_109);  arg182_1 = view_174 = permute_109 = None
        view_175 = torch.ops.aten.view.default(addmm_65, [8, 128, 768]);  addmm_65 = None
        add_78 = torch.ops.aten.add.Tensor(view_175, add_76);  view_175 = add_76 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_78, [2], correction = 0, keepdim = True)
        getitem_88 = var_mean_22[0]
        getitem_89 = var_mean_22[1];  var_mean_22 = None
        add_79 = torch.ops.aten.add.Tensor(getitem_88, 1e-12);  getitem_88 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
        sub_23 = torch.ops.aten.sub.Tensor(add_78, getitem_89);  add_78 = getitem_89 = None
        mul_77 = torch.ops.aten.mul.Tensor(sub_23, rsqrt_22);  sub_23 = rsqrt_22 = None
        mul_78 = torch.ops.aten.mul.Tensor(mul_77, arg183_1);  mul_77 = arg183_1 = None
        add_80 = torch.ops.aten.add.Tensor(mul_78, arg184_1);  mul_78 = arg184_1 = None
        view_176 = torch.ops.aten.view.default(add_80, [1024, 768])
        permute_110 = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        addmm_66 = torch.ops.aten.addmm.default(arg186_1, view_176, permute_110);  arg186_1 = view_176 = permute_110 = None
        view_177 = torch.ops.aten.view.default(addmm_66, [8, 128, 768]);  addmm_66 = None
        view_178 = torch.ops.aten.view.default(view_177, [8, -1, 12, 64]);  view_177 = None
        permute_111 = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None
        view_179 = torch.ops.aten.view.default(add_80, [1024, 768])
        permute_112 = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        addmm_67 = torch.ops.aten.addmm.default(arg188_1, view_179, permute_112);  arg188_1 = view_179 = permute_112 = None
        view_180 = torch.ops.aten.view.default(addmm_67, [8, 128, 768]);  addmm_67 = None
        view_181 = torch.ops.aten.view.default(view_180, [8, -1, 12, 64]);  view_180 = None
        permute_113 = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None
        view_182 = torch.ops.aten.view.default(add_80, [1024, 768])
        permute_114 = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        addmm_68 = torch.ops.aten.addmm.default(arg190_1, view_182, permute_114);  arg190_1 = view_182 = permute_114 = None
        view_183 = torch.ops.aten.view.default(addmm_68, [8, 128, 768]);  addmm_68 = None
        view_184 = torch.ops.aten.view.default(view_183, [8, -1, 12, 64]);  view_183 = None
        permute_115 = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
        expand_13 = torch.ops.aten.expand.default(where, [8, 12, 128, 128]);  where = None
        _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_111, permute_113, permute_115, expand_13, False);  permute_111 = permute_113 = permute_115 = expand_13 = None
        getitem_90 = _scaled_dot_product_efficient_attention_11[0];  _scaled_dot_product_efficient_attention_11 = None
        permute_116 = torch.ops.aten.permute.default(getitem_90, [0, 2, 1, 3]);  getitem_90 = None
        view_185 = torch.ops.aten.view.default(permute_116, [8, 128, 768]);  permute_116 = None
        view_186 = torch.ops.aten.view.default(view_185, [1024, 768]);  view_185 = None
        permute_117 = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        addmm_69 = torch.ops.aten.addmm.default(arg192_1, view_186, permute_117);  arg192_1 = view_186 = permute_117 = None
        view_187 = torch.ops.aten.view.default(addmm_69, [8, 128, 768]);  addmm_69 = None
        add_81 = torch.ops.aten.add.Tensor(view_187, add_80);  view_187 = add_80 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
        getitem_94 = var_mean_23[0]
        getitem_95 = var_mean_23[1];  var_mean_23 = None
        add_82 = torch.ops.aten.add.Tensor(getitem_94, 1e-12);  getitem_94 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        sub_24 = torch.ops.aten.sub.Tensor(add_81, getitem_95);  add_81 = getitem_95 = None
        mul_79 = torch.ops.aten.mul.Tensor(sub_24, rsqrt_23);  sub_24 = rsqrt_23 = None
        mul_80 = torch.ops.aten.mul.Tensor(mul_79, arg193_1);  mul_79 = arg193_1 = None
        add_83 = torch.ops.aten.add.Tensor(mul_80, arg194_1);  mul_80 = arg194_1 = None
        view_188 = torch.ops.aten.view.default(add_83, [1024, 768])
        permute_118 = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
        addmm_70 = torch.ops.aten.addmm.default(arg196_1, view_188, permute_118);  arg196_1 = view_188 = permute_118 = None
        view_189 = torch.ops.aten.view.default(addmm_70, [8, 128, 3072]);  addmm_70 = None
        mul_81 = torch.ops.aten.mul.Tensor(view_189, 0.5)
        mul_82 = torch.ops.aten.mul.Tensor(view_189, 0.7071067811865476);  view_189 = None
        erf_11 = torch.ops.aten.erf.default(mul_82);  mul_82 = None
        add_84 = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_83 = torch.ops.aten.mul.Tensor(mul_81, add_84);  mul_81 = add_84 = None
        view_190 = torch.ops.aten.view.default(mul_83, [1024, 3072]);  mul_83 = None
        permute_119 = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
        addmm_71 = torch.ops.aten.addmm.default(arg198_1, view_190, permute_119);  arg198_1 = view_190 = permute_119 = None
        view_191 = torch.ops.aten.view.default(addmm_71, [8, 128, 768]);  addmm_71 = None
        add_85 = torch.ops.aten.add.Tensor(view_191, add_83);  view_191 = add_83 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
        getitem_96 = var_mean_24[0]
        getitem_97 = var_mean_24[1];  var_mean_24 = None
        add_86 = torch.ops.aten.add.Tensor(getitem_96, 1e-12);  getitem_96 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        sub_25 = torch.ops.aten.sub.Tensor(add_85, getitem_97);  add_85 = getitem_97 = None
        mul_84 = torch.ops.aten.mul.Tensor(sub_25, rsqrt_24);  sub_25 = rsqrt_24 = None
        mul_85 = torch.ops.aten.mul.Tensor(mul_84, arg199_1);  mul_84 = arg199_1 = None
        add_87 = torch.ops.aten.add.Tensor(mul_85, arg200_1);  mul_85 = arg200_1 = None
        return (add_87,)
        
def load_args(reader):
    buf0 = reader.storage(None, 8192, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (8, 128), dtype=torch.int64, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf1, (1, 512), dtype=torch.int64, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 4096, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf2, (1, 512), dtype=torch.int64, is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 93763584, device=device(type='cuda', index=0))
    reader.tensor(buf3, (30522, 768), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf4, (2, 768), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf5, (512, 768), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf6, (768,), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf7, (768,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 8192, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf8, (8, 128), dtype=torch.int64, is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf9, (768, 768), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf10, (768,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf11, (768, 768), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf12, (768,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf13, (768, 768), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf14, (768,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf15, (768, 768), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf16, (768,), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf17, (768,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf18, (768,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf19, (3072, 768), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf20, (3072,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf21, (768, 3072), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf22, (768,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf23, (768,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf24, (768,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf25, (768, 768), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf26, (768,), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf27, (768, 768), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf28, (768,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf29, (768, 768), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf30, (768,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf31, (768, 768), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf32, (768,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf33, (768,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf34, (768,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf35, (3072, 768), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf36, (3072,), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf37, (768, 3072), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf38, (768,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf39, (768,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf40, (768,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf41, (768, 768), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf42, (768,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf43, (768, 768), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf44, (768,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf45, (768, 768), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf46, (768,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf47, (768, 768), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf48, (768,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf49, (768,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf50, (768,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf51, (3072, 768), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf52, (3072,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf53, (768, 3072), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf54, (768,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf55, (768,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf56, (768,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf57, (768, 768), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf58, (768,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf59, (768, 768), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf60, (768,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf61, (768, 768), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf62, (768,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf63, (768, 768), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf64, (768,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf65, (768,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf66, (768,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf67, (3072, 768), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf68, (3072,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf69, (768, 3072), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf70, (768,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf71, (768,), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf72, (768,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf73, (768, 768), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf74, (768,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf75, (768, 768), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf76, (768,), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf77, (768, 768), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf78, (768,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf79, (768, 768), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf80, (768,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf81, (768,), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf82, (768,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf83, (3072, 768), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf84, (3072,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf85, (768, 3072), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf86, (768,), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf87, (768,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf88, (768,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf89, (768, 768), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf90, (768,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf91, (768, 768), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf92, (768,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf93, (768, 768), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf94, (768,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf95, (768, 768), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf96, (768,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf97, (768,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf98, (768,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf99, (3072, 768), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf100, (3072,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf101, (768, 3072), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf102, (768,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf103, (768,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf104, (768,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf105, (768, 768), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf106, (768,), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf107, (768, 768), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf108, (768,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf109, (768, 768), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf110, (768,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf111, (768, 768), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf112, (768,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf113, (768,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf114, (768,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf115, (3072, 768), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf116, (3072,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf117, (768, 3072), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf118, (768,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf119, (768,), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf120, (768,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf121, (768, 768), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf122, (768,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf123, (768, 768), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf124, (768,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf125, (768, 768), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf126, (768,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf127, (768, 768), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf128, (768,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf129, (768,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf130, (768,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf131, (3072, 768), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf132, (3072,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf133, (768, 3072), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf134, (768,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf135, (768,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf136, (768,), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf137, (768, 768), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf138, (768,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf139, (768, 768), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf140, (768,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf141, (768, 768), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf142, (768,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf143, (768, 768), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf144, (768,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf145, (768,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf146, (768,), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf147, (3072, 768), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf148, (3072,), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf149, (768, 3072), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf150, (768,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf151, (768,), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf152, (768,), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf153, (768, 768), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf154, (768,), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf155, (768, 768), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf156, (768,), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf157, (768, 768), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf158, (768,), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf159, (768, 768), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf160, (768,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf161, (768,), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf162, (768,), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf163, (3072, 768), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf164, (3072,), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf165, (768, 3072), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf166, (768,), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf167, (768,), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf168, (768,), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf169, (768, 768), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf170, (768,), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf171, (768, 768), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf172, (768,), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf173, (768, 768), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf174, (768,), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf175, (768, 768), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf176, (768,), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf177, (768,), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf178, (768,), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf179, (3072, 768), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf180, (3072,), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf181, (768, 3072), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf182, (768,), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf183, (768,), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf184, (768,), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf185, (768, 768), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf186, (768,), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf187, (768, 768), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf188, (768,), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf189, (768, 768), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf190, (768,), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf191, (768, 768), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf192, (768,), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf193, (768,), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf194, (768,), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf195, (3072, 768), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf196, (3072,), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf197, (768, 3072), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf198, (768,), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf199, (768,), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf200, (768,), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf201, (768, 768), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf202, (768,), is_leaf=True)  # arg202_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)

# ===== inductor generated file: /tmp/cnnbench-bert-fold-weaagmnf/repeat_02/base/attempt_01/torchinductor/tmp5mgaaf9a/m5/cm5rpuin46rzdtknz5aqqo3766n3gtq6mrme23rqyrl6joqahvar.debug/fx_graph_transformed.py =====
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "i64[8, 128]", arg1_1: "i64[1, 512]", arg2_1: "i64[1, 512]", arg3_1: "f32[30522, 768]", arg4_1: "f32[2, 768]", arg5_1: "f32[512, 768]", arg6_1: "f32[768]", arg7_1: "f32[768]", arg8_1: "i64[8, 128]", arg9_1: "f32[768, 768]", arg10_1: "f32[768]", arg11_1: "f32[768, 768]", arg12_1: "f32[768]", arg13_1: "f32[768, 768]", arg14_1: "f32[768]", arg15_1: "f32[768, 768]", arg16_1: "f32[768]", arg17_1: "f32[768]", arg18_1: "f32[768]", arg19_1: "f32[3072, 768]", arg20_1: "f32[3072]", arg21_1: "f32[768, 3072]", arg22_1: "f32[768]", arg23_1: "f32[768]", arg24_1: "f32[768]", arg25_1: "f32[768, 768]", arg26_1: "f32[768]", arg27_1: "f32[768, 768]", arg28_1: "f32[768]", arg29_1: "f32[768, 768]", arg30_1: "f32[768]", arg31_1: "f32[768, 768]", arg32_1: "f32[768]", arg33_1: "f32[768]", arg34_1: "f32[768]", arg35_1: "f32[3072, 768]", arg36_1: "f32[3072]", arg37_1: "f32[768, 3072]", arg38_1: "f32[768]", arg39_1: "f32[768]", arg40_1: "f32[768]", arg41_1: "f32[768, 768]", arg42_1: "f32[768]", arg43_1: "f32[768, 768]", arg44_1: "f32[768]", arg45_1: "f32[768, 768]", arg46_1: "f32[768]", arg47_1: "f32[768, 768]", arg48_1: "f32[768]", arg49_1: "f32[768]", arg50_1: "f32[768]", arg51_1: "f32[3072, 768]", arg52_1: "f32[3072]", arg53_1: "f32[768, 3072]", arg54_1: "f32[768]", arg55_1: "f32[768]", arg56_1: "f32[768]", arg57_1: "f32[768, 768]", arg58_1: "f32[768]", arg59_1: "f32[768, 768]", arg60_1: "f32[768]", arg61_1: "f32[768, 768]", arg62_1: "f32[768]", arg63_1: "f32[768, 768]", arg64_1: "f32[768]", arg65_1: "f32[768]", arg66_1: "f32[768]", arg67_1: "f32[3072, 768]", arg68_1: "f32[3072]", arg69_1: "f32[768, 3072]", arg70_1: "f32[768]", arg71_1: "f32[768]", arg72_1: "f32[768]", arg73_1: "f32[768, 768]", arg74_1: "f32[768]", arg75_1: "f32[768, 768]", arg76_1: "f32[768]", arg77_1: "f32[768, 768]", arg78_1: "f32[768]", arg79_1: "f32[768, 768]", arg80_1: "f32[768]", arg81_1: "f32[768]", arg82_1: "f32[768]", arg83_1: "f32[3072, 768]", arg84_1: "f32[3072]", arg85_1: "f32[768, 3072]", arg86_1: "f32[768]", arg87_1: "f32[768]", arg88_1: "f32[768]", arg89_1: "f32[768, 768]", arg90_1: "f32[768]", arg91_1: "f32[768, 768]", arg92_1: "f32[768]", arg93_1: "f32[768, 768]", arg94_1: "f32[768]", arg95_1: "f32[768, 768]", arg96_1: "f32[768]", arg97_1: "f32[768]", arg98_1: "f32[768]", arg99_1: "f32[3072, 768]", arg100_1: "f32[3072]", arg101_1: "f32[768, 3072]", arg102_1: "f32[768]", arg103_1: "f32[768]", arg104_1: "f32[768]", arg105_1: "f32[768, 768]", arg106_1: "f32[768]", arg107_1: "f32[768, 768]", arg108_1: "f32[768]", arg109_1: "f32[768, 768]", arg110_1: "f32[768]", arg111_1: "f32[768, 768]", arg112_1: "f32[768]", arg113_1: "f32[768]", arg114_1: "f32[768]", arg115_1: "f32[3072, 768]", arg116_1: "f32[3072]", arg117_1: "f32[768, 3072]", arg118_1: "f32[768]", arg119_1: "f32[768]", arg120_1: "f32[768]", arg121_1: "f32[768, 768]", arg122_1: "f32[768]", arg123_1: "f32[768, 768]", arg124_1: "f32[768]", arg125_1: "f32[768, 768]", arg126_1: "f32[768]", arg127_1: "f32[768, 768]", arg128_1: "f32[768]", arg129_1: "f32[768]", arg130_1: "f32[768]", arg131_1: "f32[3072, 768]", arg132_1: "f32[3072]", arg133_1: "f32[768, 3072]", arg134_1: "f32[768]", arg135_1: "f32[768]", arg136_1: "f32[768]", arg137_1: "f32[768, 768]", arg138_1: "f32[768]", arg139_1: "f32[768, 768]", arg140_1: "f32[768]", arg141_1: "f32[768, 768]", arg142_1: "f32[768]", arg143_1: "f32[768, 768]", arg144_1: "f32[768]", arg145_1: "f32[768]", arg146_1: "f32[768]", arg147_1: "f32[3072, 768]", arg148_1: "f32[3072]", arg149_1: "f32[768, 3072]", arg150_1: "f32[768]", arg151_1: "f32[768]", arg152_1: "f32[768]", arg153_1: "f32[768, 768]", arg154_1: "f32[768]", arg155_1: "f32[768, 768]", arg156_1: "f32[768]", arg157_1: "f32[768, 768]", arg158_1: "f32[768]", arg159_1: "f32[768, 768]", arg160_1: "f32[768]", arg161_1: "f32[768]", arg162_1: "f32[768]", arg163_1: "f32[3072, 768]", arg164_1: "f32[3072]", arg165_1: "f32[768, 3072]", arg166_1: "f32[768]", arg167_1: "f32[768]", arg168_1: "f32[768]", arg169_1: "f32[768, 768]", arg170_1: "f32[768]", arg171_1: "f32[768, 768]", arg172_1: "f32[768]", arg173_1: "f32[768, 768]", arg174_1: "f32[768]", arg175_1: "f32[768, 768]", arg176_1: "f32[768]", arg177_1: "f32[768]", arg178_1: "f32[768]", arg179_1: "f32[3072, 768]", arg180_1: "f32[3072]", arg181_1: "f32[768, 3072]", arg182_1: "f32[768]", arg183_1: "f32[768]", arg184_1: "f32[768]", arg185_1: "f32[768, 768]", arg186_1: "f32[768]", arg187_1: "f32[768, 768]", arg188_1: "f32[768]", arg189_1: "f32[768, 768]", arg190_1: "f32[768]", arg191_1: "f32[768, 768]", arg192_1: "f32[768]", arg193_1: "f32[768]", arg194_1: "f32[768]", arg195_1: "f32[3072, 768]", arg196_1: "f32[3072]", arg197_1: "f32[768, 3072]", arg198_1: "f32[768]", arg199_1: "f32[768]", arg200_1: "f32[768]", arg201_1: "f32[768, 768]", arg202_1: "f32[768]"):
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding: "f32[8, 128, 768]" = torch.ops.aten.embedding.default(arg3_1, arg0_1, 0);  arg3_1 = arg0_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:929 in forward, code: buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
        slice_1: "i64[1, 128]" = torch.ops.aten.slice.Tensor(arg1_1, 1, 0, 128);  arg1_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:930 in forward, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
        expand: "i64[8, 128]" = torch.ops.aten.expand.default(slice_1, [8, 128]);  slice_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_1: "f32[8, 128, 768]" = torch.ops.aten.embedding.default(arg4_1, expand);  arg4_1 = expand = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:183 in forward, code: embeddings = inputs_embeds + token_type_embeddings
        add: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:166 in forward, code: position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
        slice_2: "i64[1, 128]" = torch.ops.aten.slice.Tensor(arg2_1, 1, 0, 128);  arg2_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:185 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_2: "f32[1, 128, 768]" = torch.ops.aten.embedding.default(arg5_1, slice_2);  arg5_1 = slice_2 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:186 in forward, code: embeddings += position_embeddings
        add_1: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:187 in forward, code: embeddings = self.LayerNorm(embeddings)
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 128, 1]" = var_mean[0]
        getitem_1: "f32[8, 128, 1]" = var_mean[1];  var_mean = None
        sub: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
        add_2: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul, arg6_1);  mul = arg6_1 = None
        add_3: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_1, arg7_1);  mul_1 = arg7_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:363 in forward, code: self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        view: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_3, [1024, 768])
        permute: "f32[768, 768]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg10_1, view, permute);  arg10_1 = view = permute = None
        view_1: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(addmm, [8, 128, 768]);  addmm = None
        view_2: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(view_1, [8, -1, 12, 64]);  view_1 = None
        permute_1: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:386 in forward, code: self.key(current_states)
        view_3: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_3, [1024, 768])
        permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_1: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg12_1, view_3, permute_2);  arg12_1 = view_3 = permute_2 = None
        view_4: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(addmm_1, [8, 128, 768]);  addmm_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:387 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_5: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(view_4, [8, -1, 12, 64]);  view_4 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:388 in forward, code: .transpose(1, 2)
        permute_3: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:391 in forward, code: self.value(current_states)
        view_6: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_3, [1024, 768])
        permute_4: "f32[768, 768]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_2: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg14_1, view_6, permute_4);  arg14_1 = view_6 = permute_4 = None
        view_7: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(addmm_2, [8, 128, 768]);  addmm_2 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:392 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_8: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(view_7, [8, -1, 12, 64]);  view_7 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:393 in forward, code: .transpose(1, 2)
        permute_5: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/modeling_attn_mask_utils.py:196 in _expand_mask, code: inverted_mask = torch.tensor(1.0, dtype=dtype) - expanded_mask
        full_default: "f32[]" = torch.ops.aten.full.default([], 1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/modeling_attn_mask_utils.py:194 in _expand_mask, code: expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        unsqueeze: "i64[8, 1, 128]" = torch.ops.aten.unsqueeze.default(arg8_1, 1);  arg8_1 = None
        unsqueeze_1: "i64[8, 1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        expand_1: "i64[8, 1, 128, 128]" = torch.ops.aten.expand.default(unsqueeze_1, [8, 1, 128, 128]);  unsqueeze_1 = None
        convert_element_type: "f32[8, 1, 128, 128]" = torch.ops.prims.convert_element_type.default(expand_1, torch.float32);  expand_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/modeling_attn_mask_utils.py:196 in _expand_mask, code: inverted_mask = torch.tensor(1.0, dtype=dtype) - expanded_mask
        sub_1: "f32[8, 1, 128, 128]" = torch.ops.aten.sub.Tensor(full_default, convert_element_type);  full_default = convert_element_type = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/modeling_attn_mask_utils.py:198 in _expand_mask, code: return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
        convert_element_type_1: "b8[8, 1, 128, 128]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bool)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[8, 1, 128, 128]" = torch.ops.aten.where.self(convert_element_type_1, full_default_1, sub_1);  convert_element_type_1 = full_default_1 = sub_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:412 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_2: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(where, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_1, permute_3, permute_5, expand_2, False);  permute_1 = permute_3 = permute_5 = expand_2 = None
        getitem_2: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention[0];  _scaled_dot_product_efficient_attention = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:421 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_6: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3]);  getitem_2 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:422 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_9: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(permute_6, [8, 128, 768]);  permute_6 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:435 in forward, code: hidden_states = self.dense(hidden_states)
        view_10: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_9, [1024, 768]);  view_9 = None
        permute_7: "f32[768, 768]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        mm_default_35: "f32[1024, 768]" = torch.ops.aten.mm.default(view_10, permute_7);  view_10 = permute_7 = None
        add_tensor_35: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_35, arg16_1);  mm_default_35 = arg16_1 = None
        view_11: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_35, [8, 128, 768]);  add_tensor_35 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:437 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_4: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_11, add_3);  view_11 = add_3 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
        getitem_6: "f32[8, 128, 1]" = var_mean_1[0]
        getitem_7: "f32[8, 128, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_2: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_4, getitem_7);  add_4 = getitem_7 = None
        add_5: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_1: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        mul_2: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_3: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_2, arg17_1);  mul_2 = arg17_1 = None
        add_6: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_3, arg18_1);  mul_3 = arg18_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:511 in forward, code: hidden_states = self.dense(hidden_states)
        view_12: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_6, [1024, 768])
        permute_8: "f32[768, 3072]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        mm_default_34: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_12, permute_8);  view_12 = permute_8 = None
        add_tensor_34: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_34, arg20_1);  mm_default_34 = arg20_1 = None
        view_13: "f32[8, 128, 3072]" = torch.ops.aten.reshape.default(add_tensor_34, [8, 128, 3072]);  add_tensor_34 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:70 in forward, code: return self.act(input)
        mul_4: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_13, 0.5)
        mul_5: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_13, 0.7071067811865476);  view_13 = None
        erf: "f32[8, 128, 3072]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_7: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_4, add_7);  mul_4 = add_7 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:524 in forward, code: hidden_states = self.dense(hidden_states)
        view_14: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_6, [1024, 3072]);  mul_6 = None
        permute_9: "f32[3072, 768]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        mm_default_33: "f32[1024, 768]" = torch.ops.aten.mm.default(view_14, permute_9);  view_14 = permute_9 = None
        add_tensor_33: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_33, arg22_1);  mm_default_33 = arg22_1 = None
        view_15: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_33, [8, 128, 768]);  add_tensor_33 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:526 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_8: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_15, add_6);  view_15 = add_6 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
        getitem_8: "f32[8, 128, 1]" = var_mean_2[0]
        getitem_9: "f32[8, 128, 1]" = var_mean_2[1];  var_mean_2 = None
        sub_3: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_8, getitem_9);  add_8 = getitem_9 = None
        add_9: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_2: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        mul_7: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_8: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_7, arg23_1);  mul_7 = arg23_1 = None
        add_10: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_8, arg24_1);  mul_8 = arg24_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:363 in forward, code: self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        view_16: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_10, [1024, 768])
        permute_10: "f32[768, 768]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_6: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg26_1, view_16, permute_10);  arg26_1 = view_16 = permute_10 = None
        view_17: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(addmm_6, [8, 128, 768]);  addmm_6 = None
        view_18: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(view_17, [8, -1, 12, 64]);  view_17 = None
        permute_11: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:386 in forward, code: self.key(current_states)
        view_19: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_10, [1024, 768])
        permute_12: "f32[768, 768]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_7: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg28_1, view_19, permute_12);  arg28_1 = view_19 = permute_12 = None
        view_20: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(addmm_7, [8, 128, 768]);  addmm_7 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:387 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_21: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(view_20, [8, -1, 12, 64]);  view_20 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:388 in forward, code: .transpose(1, 2)
        permute_13: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:391 in forward, code: self.value(current_states)
        view_22: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_10, [1024, 768])
        permute_14: "f32[768, 768]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        addmm_8: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg30_1, view_22, permute_14);  arg30_1 = view_22 = permute_14 = None
        view_23: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(addmm_8, [8, 128, 768]);  addmm_8 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:392 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_24: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(view_23, [8, -1, 12, 64]);  view_23 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:393 in forward, code: .transpose(1, 2)
        permute_15: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:412 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_3: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(where, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_11, permute_13, permute_15, expand_3, False);  permute_11 = permute_13 = permute_15 = expand_3 = None
        getitem_10: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_1[0];  _scaled_dot_product_efficient_attention_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:421 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_16: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_10, [0, 2, 1, 3]);  getitem_10 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:422 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_25: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(permute_16, [8, 128, 768]);  permute_16 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:435 in forward, code: hidden_states = self.dense(hidden_states)
        view_26: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_25, [1024, 768]);  view_25 = None
        permute_17: "f32[768, 768]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        mm_default_32: "f32[1024, 768]" = torch.ops.aten.mm.default(view_26, permute_17);  view_26 = permute_17 = None
        add_tensor_32: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_32, arg32_1);  mm_default_32 = arg32_1 = None
        view_27: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_32, [8, 128, 768]);  add_tensor_32 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:437 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_11: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_27, add_10);  view_27 = add_10 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
        getitem_14: "f32[8, 128, 1]" = var_mean_3[0]
        getitem_15: "f32[8, 128, 1]" = var_mean_3[1];  var_mean_3 = None
        sub_4: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_15);  add_11 = getitem_15 = None
        add_12: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_3: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_9: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = rsqrt_3 = None
        mul_10: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_9, arg33_1);  mul_9 = arg33_1 = None
        add_13: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_10, arg34_1);  mul_10 = arg34_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:511 in forward, code: hidden_states = self.dense(hidden_states)
        view_28: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_13, [1024, 768])
        permute_18: "f32[768, 3072]" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        mm_default_31: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_28, permute_18);  view_28 = permute_18 = None
        add_tensor_31: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_31, arg36_1);  mm_default_31 = arg36_1 = None
        view_29: "f32[8, 128, 3072]" = torch.ops.aten.reshape.default(add_tensor_31, [8, 128, 3072]);  add_tensor_31 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:70 in forward, code: return self.act(input)
        mul_11: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_29, 0.5)
        mul_12: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_29, 0.7071067811865476);  view_29 = None
        erf_1: "f32[8, 128, 3072]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_14: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_13: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_11, add_14);  mul_11 = add_14 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:524 in forward, code: hidden_states = self.dense(hidden_states)
        view_30: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_13, [1024, 3072]);  mul_13 = None
        permute_19: "f32[3072, 768]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        mm_default_30: "f32[1024, 768]" = torch.ops.aten.mm.default(view_30, permute_19);  view_30 = permute_19 = None
        add_tensor_30: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_30, arg38_1);  mm_default_30 = arg38_1 = None
        view_31: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_30, [8, 128, 768]);  add_tensor_30 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:526 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_15: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_31, add_13);  view_31 = add_13 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
        getitem_16: "f32[8, 128, 1]" = var_mean_4[0]
        getitem_17: "f32[8, 128, 1]" = var_mean_4[1];  var_mean_4 = None
        sub_5: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_15, getitem_17);  add_15 = getitem_17 = None
        add_16: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_4: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        mul_14: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_4);  sub_5 = rsqrt_4 = None
        mul_15: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_14, arg39_1);  mul_14 = arg39_1 = None
        add_17: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_15, arg40_1);  mul_15 = arg40_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:363 in forward, code: self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        view_32: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_17, [1024, 768])
        permute_20: "f32[768, 768]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_12: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg42_1, view_32, permute_20);  arg42_1 = view_32 = permute_20 = None
        view_33: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(addmm_12, [8, 128, 768]);  addmm_12 = None
        view_34: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(view_33, [8, -1, 12, 64]);  view_33 = None
        permute_21: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:386 in forward, code: self.key(current_states)
        view_35: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_17, [1024, 768])
        permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_13: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg44_1, view_35, permute_22);  arg44_1 = view_35 = permute_22 = None
        view_36: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(addmm_13, [8, 128, 768]);  addmm_13 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:387 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_37: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(view_36, [8, -1, 12, 64]);  view_36 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:388 in forward, code: .transpose(1, 2)
        permute_23: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_37, [0, 2, 1, 3]);  view_37 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:391 in forward, code: self.value(current_states)
        view_38: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_17, [1024, 768])
        permute_24: "f32[768, 768]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        addmm_14: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg46_1, view_38, permute_24);  arg46_1 = view_38 = permute_24 = None
        view_39: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(addmm_14, [8, 128, 768]);  addmm_14 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:392 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_40: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(view_39, [8, -1, 12, 64]);  view_39 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:393 in forward, code: .transpose(1, 2)
        permute_25: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:412 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_4: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(where, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_21, permute_23, permute_25, expand_4, False);  permute_21 = permute_23 = permute_25 = expand_4 = None
        getitem_18: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:421 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_26: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_18, [0, 2, 1, 3]);  getitem_18 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:422 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_41: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(permute_26, [8, 128, 768]);  permute_26 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:435 in forward, code: hidden_states = self.dense(hidden_states)
        view_42: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_41, [1024, 768]);  view_41 = None
        permute_27: "f32[768, 768]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        mm_default_29: "f32[1024, 768]" = torch.ops.aten.mm.default(view_42, permute_27);  view_42 = permute_27 = None
        add_tensor_29: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_29, arg48_1);  mm_default_29 = arg48_1 = None
        view_43: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_29, [8, 128, 768]);  add_tensor_29 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:437 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_18: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_43, add_17);  view_43 = add_17 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_18, [2], correction = 0, keepdim = True)
        getitem_22: "f32[8, 128, 1]" = var_mean_5[0]
        getitem_23: "f32[8, 128, 1]" = var_mean_5[1];  var_mean_5 = None
        sub_6: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_18, getitem_23);  add_18 = getitem_23 = None
        add_19: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_5: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        mul_16: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_5);  sub_6 = rsqrt_5 = None
        mul_17: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_16, arg49_1);  mul_16 = arg49_1 = None
        add_20: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_17, arg50_1);  mul_17 = arg50_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:511 in forward, code: hidden_states = self.dense(hidden_states)
        view_44: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_20, [1024, 768])
        permute_28: "f32[768, 3072]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        mm_default_28: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_44, permute_28);  view_44 = permute_28 = None
        add_tensor_28: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_28, arg52_1);  mm_default_28 = arg52_1 = None
        view_45: "f32[8, 128, 3072]" = torch.ops.aten.reshape.default(add_tensor_28, [8, 128, 3072]);  add_tensor_28 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:70 in forward, code: return self.act(input)
        mul_18: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_45, 0.5)
        mul_19: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_45, 0.7071067811865476);  view_45 = None
        erf_2: "f32[8, 128, 3072]" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
        add_21: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_20: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_18, add_21);  mul_18 = add_21 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:524 in forward, code: hidden_states = self.dense(hidden_states)
        view_46: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_20, [1024, 3072]);  mul_20 = None
        permute_29: "f32[3072, 768]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        mm_default_27: "f32[1024, 768]" = torch.ops.aten.mm.default(view_46, permute_29);  view_46 = permute_29 = None
        add_tensor_27: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_27, arg54_1);  mm_default_27 = arg54_1 = None
        view_47: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_27, [8, 128, 768]);  add_tensor_27 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:526 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_22: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_47, add_20);  view_47 = add_20 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_22, [2], correction = 0, keepdim = True)
        getitem_24: "f32[8, 128, 1]" = var_mean_6[0]
        getitem_25: "f32[8, 128, 1]" = var_mean_6[1];  var_mean_6 = None
        sub_7: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_22, getitem_25);  add_22 = getitem_25 = None
        add_23: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_6: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        mul_21: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_6);  sub_7 = rsqrt_6 = None
        mul_22: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_21, arg55_1);  mul_21 = arg55_1 = None
        add_24: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_22, arg56_1);  mul_22 = arg56_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:363 in forward, code: self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        view_48: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_24, [1024, 768])
        permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        addmm_18: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg58_1, view_48, permute_30);  arg58_1 = view_48 = permute_30 = None
        view_49: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(addmm_18, [8, 128, 768]);  addmm_18 = None
        view_50: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(view_49, [8, -1, 12, 64]);  view_49 = None
        permute_31: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:386 in forward, code: self.key(current_states)
        view_51: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_24, [1024, 768])
        permute_32: "f32[768, 768]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        addmm_19: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg60_1, view_51, permute_32);  arg60_1 = view_51 = permute_32 = None
        view_52: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(addmm_19, [8, 128, 768]);  addmm_19 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:387 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_53: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(view_52, [8, -1, 12, 64]);  view_52 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:388 in forward, code: .transpose(1, 2)
        permute_33: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:391 in forward, code: self.value(current_states)
        view_54: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_24, [1024, 768])
        permute_34: "f32[768, 768]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm_20: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg62_1, view_54, permute_34);  arg62_1 = view_54 = permute_34 = None
        view_55: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(addmm_20, [8, 128, 768]);  addmm_20 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:392 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_56: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(view_55, [8, -1, 12, 64]);  view_55 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:393 in forward, code: .transpose(1, 2)
        permute_35: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:412 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_5: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(where, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_31, permute_33, permute_35, expand_5, False);  permute_31 = permute_33 = permute_35 = expand_5 = None
        getitem_26: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:421 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_36: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_26, [0, 2, 1, 3]);  getitem_26 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:422 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_57: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(permute_36, [8, 128, 768]);  permute_36 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:435 in forward, code: hidden_states = self.dense(hidden_states)
        view_58: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_57, [1024, 768]);  view_57 = None
        permute_37: "f32[768, 768]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        mm_default_26: "f32[1024, 768]" = torch.ops.aten.mm.default(view_58, permute_37);  view_58 = permute_37 = None
        add_tensor_26: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_26, arg64_1);  mm_default_26 = arg64_1 = None
        view_59: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_26, [8, 128, 768]);  add_tensor_26 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:437 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_25: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_59, add_24);  view_59 = add_24 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_30: "f32[8, 128, 1]" = var_mean_7[0]
        getitem_31: "f32[8, 128, 1]" = var_mean_7[1];  var_mean_7 = None
        sub_8: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_25, getitem_31);  add_25 = getitem_31 = None
        add_26: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_7: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        mul_23: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_7);  sub_8 = rsqrt_7 = None
        mul_24: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_23, arg65_1);  mul_23 = arg65_1 = None
        add_27: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_24, arg66_1);  mul_24 = arg66_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:511 in forward, code: hidden_states = self.dense(hidden_states)
        view_60: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_27, [1024, 768])
        permute_38: "f32[768, 3072]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        mm_default_25: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_60, permute_38);  view_60 = permute_38 = None
        add_tensor_25: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_25, arg68_1);  mm_default_25 = arg68_1 = None
        view_61: "f32[8, 128, 3072]" = torch.ops.aten.reshape.default(add_tensor_25, [8, 128, 3072]);  add_tensor_25 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:70 in forward, code: return self.act(input)
        mul_25: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_61, 0.5)
        mul_26: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_61, 0.7071067811865476);  view_61 = None
        erf_3: "f32[8, 128, 3072]" = torch.ops.aten.erf.default(mul_26);  mul_26 = None
        add_28: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_27: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_25, add_28);  mul_25 = add_28 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:524 in forward, code: hidden_states = self.dense(hidden_states)
        view_62: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_27, [1024, 3072]);  mul_27 = None
        permute_39: "f32[3072, 768]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        mm_default_24: "f32[1024, 768]" = torch.ops.aten.mm.default(view_62, permute_39);  view_62 = permute_39 = None
        add_tensor_24: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_24, arg70_1);  mm_default_24 = arg70_1 = None
        view_63: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_24, [8, 128, 768]);  add_tensor_24 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:526 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_29: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_63, add_27);  view_63 = add_27 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
        getitem_32: "f32[8, 128, 1]" = var_mean_8[0]
        getitem_33: "f32[8, 128, 1]" = var_mean_8[1];  var_mean_8 = None
        sub_9: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_29, getitem_33);  add_29 = getitem_33 = None
        add_30: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_8: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_28: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_8);  sub_9 = rsqrt_8 = None
        mul_29: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_28, arg71_1);  mul_28 = arg71_1 = None
        add_31: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_29, arg72_1);  mul_29 = arg72_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:363 in forward, code: self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        view_64: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_31, [1024, 768])
        permute_40: "f32[768, 768]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_24: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg74_1, view_64, permute_40);  arg74_1 = view_64 = permute_40 = None
        view_65: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(addmm_24, [8, 128, 768]);  addmm_24 = None
        view_66: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(view_65, [8, -1, 12, 64]);  view_65 = None
        permute_41: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:386 in forward, code: self.key(current_states)
        view_67: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_31, [1024, 768])
        permute_42: "f32[768, 768]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        addmm_25: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg76_1, view_67, permute_42);  arg76_1 = view_67 = permute_42 = None
        view_68: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(addmm_25, [8, 128, 768]);  addmm_25 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:387 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_69: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(view_68, [8, -1, 12, 64]);  view_68 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:388 in forward, code: .transpose(1, 2)
        permute_43: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:391 in forward, code: self.value(current_states)
        view_70: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_31, [1024, 768])
        permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        addmm_26: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg78_1, view_70, permute_44);  arg78_1 = view_70 = permute_44 = None
        view_71: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(addmm_26, [8, 128, 768]);  addmm_26 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:392 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_72: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(view_71, [8, -1, 12, 64]);  view_71 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:393 in forward, code: .transpose(1, 2)
        permute_45: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:412 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_6: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(where, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_41, permute_43, permute_45, expand_6, False);  permute_41 = permute_43 = permute_45 = expand_6 = None
        getitem_34: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_4[0];  _scaled_dot_product_efficient_attention_4 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:421 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_46: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_34, [0, 2, 1, 3]);  getitem_34 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:422 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_73: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(permute_46, [8, 128, 768]);  permute_46 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:435 in forward, code: hidden_states = self.dense(hidden_states)
        view_74: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_73, [1024, 768]);  view_73 = None
        permute_47: "f32[768, 768]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        mm_default_23: "f32[1024, 768]" = torch.ops.aten.mm.default(view_74, permute_47);  view_74 = permute_47 = None
        add_tensor_23: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_23, arg80_1);  mm_default_23 = arg80_1 = None
        view_75: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_23, [8, 128, 768]);  add_tensor_23 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:437 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_32: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_75, add_31);  view_75 = add_31 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
        getitem_38: "f32[8, 128, 1]" = var_mean_9[0]
        getitem_39: "f32[8, 128, 1]" = var_mean_9[1];  var_mean_9 = None
        sub_10: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_32, getitem_39);  add_32 = getitem_39 = None
        add_33: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_9: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        mul_30: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_9);  sub_10 = rsqrt_9 = None
        mul_31: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_30, arg81_1);  mul_30 = arg81_1 = None
        add_34: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_31, arg82_1);  mul_31 = arg82_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:511 in forward, code: hidden_states = self.dense(hidden_states)
        view_76: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_34, [1024, 768])
        permute_48: "f32[768, 3072]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        mm_default_22: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_76, permute_48);  view_76 = permute_48 = None
        add_tensor_22: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_22, arg84_1);  mm_default_22 = arg84_1 = None
        view_77: "f32[8, 128, 3072]" = torch.ops.aten.reshape.default(add_tensor_22, [8, 128, 3072]);  add_tensor_22 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:70 in forward, code: return self.act(input)
        mul_32: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_77, 0.5)
        mul_33: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_77, 0.7071067811865476);  view_77 = None
        erf_4: "f32[8, 128, 3072]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
        add_35: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_34: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_32, add_35);  mul_32 = add_35 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:524 in forward, code: hidden_states = self.dense(hidden_states)
        view_78: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_34, [1024, 3072]);  mul_34 = None
        permute_49: "f32[3072, 768]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        mm_default_21: "f32[1024, 768]" = torch.ops.aten.mm.default(view_78, permute_49);  view_78 = permute_49 = None
        add_tensor_21: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_21, arg86_1);  mm_default_21 = arg86_1 = None
        view_79: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_21, [8, 128, 768]);  add_tensor_21 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:526 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_36: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_79, add_34);  view_79 = add_34 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_36, [2], correction = 0, keepdim = True)
        getitem_40: "f32[8, 128, 1]" = var_mean_10[0]
        getitem_41: "f32[8, 128, 1]" = var_mean_10[1];  var_mean_10 = None
        sub_11: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_36, getitem_41);  add_36 = getitem_41 = None
        add_37: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_10: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        mul_35: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_10);  sub_11 = rsqrt_10 = None
        mul_36: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_35, arg87_1);  mul_35 = arg87_1 = None
        add_38: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_36, arg88_1);  mul_36 = arg88_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:363 in forward, code: self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        view_80: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_38, [1024, 768])
        permute_50: "f32[768, 768]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        addmm_30: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg90_1, view_80, permute_50);  arg90_1 = view_80 = permute_50 = None
        view_81: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(addmm_30, [8, 128, 768]);  addmm_30 = None
        view_82: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(view_81, [8, -1, 12, 64]);  view_81 = None
        permute_51: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:386 in forward, code: self.key(current_states)
        view_83: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_38, [1024, 768])
        permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_31: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg92_1, view_83, permute_52);  arg92_1 = view_83 = permute_52 = None
        view_84: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(addmm_31, [8, 128, 768]);  addmm_31 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:387 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_85: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(view_84, [8, -1, 12, 64]);  view_84 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:388 in forward, code: .transpose(1, 2)
        permute_53: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_85, [0, 2, 1, 3]);  view_85 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:391 in forward, code: self.value(current_states)
        view_86: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_38, [1024, 768])
        permute_54: "f32[768, 768]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        addmm_32: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg94_1, view_86, permute_54);  arg94_1 = view_86 = permute_54 = None
        view_87: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(addmm_32, [8, 128, 768]);  addmm_32 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:392 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_88: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(view_87, [8, -1, 12, 64]);  view_87 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:393 in forward, code: .transpose(1, 2)
        permute_55: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:412 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_7: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(where, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_51, permute_53, permute_55, expand_7, False);  permute_51 = permute_53 = permute_55 = expand_7 = None
        getitem_42: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_5[0];  _scaled_dot_product_efficient_attention_5 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:421 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_56: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_42, [0, 2, 1, 3]);  getitem_42 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:422 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_89: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(permute_56, [8, 128, 768]);  permute_56 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:435 in forward, code: hidden_states = self.dense(hidden_states)
        view_90: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_89, [1024, 768]);  view_89 = None
        permute_57: "f32[768, 768]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        mm_default_20: "f32[1024, 768]" = torch.ops.aten.mm.default(view_90, permute_57);  view_90 = permute_57 = None
        add_tensor_20: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_20, arg96_1);  mm_default_20 = arg96_1 = None
        view_91: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_20, [8, 128, 768]);  add_tensor_20 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:437 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_39: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_91, add_38);  view_91 = add_38 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_39, [2], correction = 0, keepdim = True)
        getitem_46: "f32[8, 128, 1]" = var_mean_11[0]
        getitem_47: "f32[8, 128, 1]" = var_mean_11[1];  var_mean_11 = None
        sub_12: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_39, getitem_47);  add_39 = getitem_47 = None
        add_40: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_11: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        mul_37: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_11);  sub_12 = rsqrt_11 = None
        mul_38: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_37, arg97_1);  mul_37 = arg97_1 = None
        add_41: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_38, arg98_1);  mul_38 = arg98_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:511 in forward, code: hidden_states = self.dense(hidden_states)
        view_92: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_41, [1024, 768])
        permute_58: "f32[768, 3072]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        mm_default_19: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_92, permute_58);  view_92 = permute_58 = None
        add_tensor_19: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_19, arg100_1);  mm_default_19 = arg100_1 = None
        view_93: "f32[8, 128, 3072]" = torch.ops.aten.reshape.default(add_tensor_19, [8, 128, 3072]);  add_tensor_19 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:70 in forward, code: return self.act(input)
        mul_39: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_93, 0.5)
        mul_40: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_93, 0.7071067811865476);  view_93 = None
        erf_5: "f32[8, 128, 3072]" = torch.ops.aten.erf.default(mul_40);  mul_40 = None
        add_42: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_41: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_39, add_42);  mul_39 = add_42 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:524 in forward, code: hidden_states = self.dense(hidden_states)
        view_94: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_41, [1024, 3072]);  mul_41 = None
        permute_59: "f32[3072, 768]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        mm_default_18: "f32[1024, 768]" = torch.ops.aten.mm.default(view_94, permute_59);  view_94 = permute_59 = None
        add_tensor_18: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_18, arg102_1);  mm_default_18 = arg102_1 = None
        view_95: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_18, [8, 128, 768]);  add_tensor_18 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:526 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_43: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_95, add_41);  view_95 = add_41 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
        getitem_48: "f32[8, 128, 1]" = var_mean_12[0]
        getitem_49: "f32[8, 128, 1]" = var_mean_12[1];  var_mean_12 = None
        sub_13: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_49);  add_43 = getitem_49 = None
        add_44: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_12: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        mul_42: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_12);  sub_13 = rsqrt_12 = None
        mul_43: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_42, arg103_1);  mul_42 = arg103_1 = None
        add_45: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_43, arg104_1);  mul_43 = arg104_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:363 in forward, code: self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        view_96: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_45, [1024, 768])
        permute_60: "f32[768, 768]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        addmm_36: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg106_1, view_96, permute_60);  arg106_1 = view_96 = permute_60 = None
        view_97: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(addmm_36, [8, 128, 768]);  addmm_36 = None
        view_98: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(view_97, [8, -1, 12, 64]);  view_97 = None
        permute_61: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:386 in forward, code: self.key(current_states)
        view_99: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_45, [1024, 768])
        permute_62: "f32[768, 768]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_37: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg108_1, view_99, permute_62);  arg108_1 = view_99 = permute_62 = None
        view_100: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(addmm_37, [8, 128, 768]);  addmm_37 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:387 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_101: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(view_100, [8, -1, 12, 64]);  view_100 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:388 in forward, code: .transpose(1, 2)
        permute_63: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_101, [0, 2, 1, 3]);  view_101 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:391 in forward, code: self.value(current_states)
        view_102: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_45, [1024, 768])
        permute_64: "f32[768, 768]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_38: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg110_1, view_102, permute_64);  arg110_1 = view_102 = permute_64 = None
        view_103: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(addmm_38, [8, 128, 768]);  addmm_38 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:392 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_104: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(view_103, [8, -1, 12, 64]);  view_103 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:393 in forward, code: .transpose(1, 2)
        permute_65: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:412 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_8: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(where, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_61, permute_63, permute_65, expand_8, False);  permute_61 = permute_63 = permute_65 = expand_8 = None
        getitem_50: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_6[0];  _scaled_dot_product_efficient_attention_6 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:421 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_66: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_50, [0, 2, 1, 3]);  getitem_50 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:422 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_105: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(permute_66, [8, 128, 768]);  permute_66 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:435 in forward, code: hidden_states = self.dense(hidden_states)
        view_106: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_105, [1024, 768]);  view_105 = None
        permute_67: "f32[768, 768]" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        mm_default_17: "f32[1024, 768]" = torch.ops.aten.mm.default(view_106, permute_67);  view_106 = permute_67 = None
        add_tensor_17: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_17, arg112_1);  mm_default_17 = arg112_1 = None
        view_107: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_17, [8, 128, 768]);  add_tensor_17 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:437 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_46: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_107, add_45);  view_107 = add_45 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_46, [2], correction = 0, keepdim = True)
        getitem_54: "f32[8, 128, 1]" = var_mean_13[0]
        getitem_55: "f32[8, 128, 1]" = var_mean_13[1];  var_mean_13 = None
        sub_14: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_46, getitem_55);  add_46 = getitem_55 = None
        add_47: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-12);  getitem_54 = None
        rsqrt_13: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        mul_44: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_13);  sub_14 = rsqrt_13 = None
        mul_45: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_44, arg113_1);  mul_44 = arg113_1 = None
        add_48: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_45, arg114_1);  mul_45 = arg114_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:511 in forward, code: hidden_states = self.dense(hidden_states)
        view_108: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_48, [1024, 768])
        permute_68: "f32[768, 3072]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        mm_default_16: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_108, permute_68);  view_108 = permute_68 = None
        add_tensor_16: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_16, arg116_1);  mm_default_16 = arg116_1 = None
        view_109: "f32[8, 128, 3072]" = torch.ops.aten.reshape.default(add_tensor_16, [8, 128, 3072]);  add_tensor_16 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:70 in forward, code: return self.act(input)
        mul_46: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_109, 0.5)
        mul_47: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_109, 0.7071067811865476);  view_109 = None
        erf_6: "f32[8, 128, 3072]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
        add_49: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_48: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_46, add_49);  mul_46 = add_49 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:524 in forward, code: hidden_states = self.dense(hidden_states)
        view_110: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_48, [1024, 3072]);  mul_48 = None
        permute_69: "f32[3072, 768]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        mm_default_15: "f32[1024, 768]" = torch.ops.aten.mm.default(view_110, permute_69);  view_110 = permute_69 = None
        add_tensor_15: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_15, arg118_1);  mm_default_15 = arg118_1 = None
        view_111: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_15, [8, 128, 768]);  add_tensor_15 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:526 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_50: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_111, add_48);  view_111 = add_48 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_50, [2], correction = 0, keepdim = True)
        getitem_56: "f32[8, 128, 1]" = var_mean_14[0]
        getitem_57: "f32[8, 128, 1]" = var_mean_14[1];  var_mean_14 = None
        sub_15: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_50, getitem_57);  add_50 = getitem_57 = None
        add_51: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-12);  getitem_56 = None
        rsqrt_14: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        mul_49: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_14);  sub_15 = rsqrt_14 = None
        mul_50: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_49, arg119_1);  mul_49 = arg119_1 = None
        add_52: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_50, arg120_1);  mul_50 = arg120_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:363 in forward, code: self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        view_112: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_52, [1024, 768])
        permute_70: "f32[768, 768]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_42: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg122_1, view_112, permute_70);  arg122_1 = view_112 = permute_70 = None
        view_113: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(addmm_42, [8, 128, 768]);  addmm_42 = None
        view_114: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(view_113, [8, -1, 12, 64]);  view_113 = None
        permute_71: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:386 in forward, code: self.key(current_states)
        view_115: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_52, [1024, 768])
        permute_72: "f32[768, 768]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_43: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg124_1, view_115, permute_72);  arg124_1 = view_115 = permute_72 = None
        view_116: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(addmm_43, [8, 128, 768]);  addmm_43 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:387 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_117: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(view_116, [8, -1, 12, 64]);  view_116 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:388 in forward, code: .transpose(1, 2)
        permute_73: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:391 in forward, code: self.value(current_states)
        view_118: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_52, [1024, 768])
        permute_74: "f32[768, 768]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        addmm_44: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg126_1, view_118, permute_74);  arg126_1 = view_118 = permute_74 = None
        view_119: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(addmm_44, [8, 128, 768]);  addmm_44 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:392 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_120: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(view_119, [8, -1, 12, 64]);  view_119 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:393 in forward, code: .transpose(1, 2)
        permute_75: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:412 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_9: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(where, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_71, permute_73, permute_75, expand_9, False);  permute_71 = permute_73 = permute_75 = expand_9 = None
        getitem_58: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_7[0];  _scaled_dot_product_efficient_attention_7 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:421 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_76: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_58, [0, 2, 1, 3]);  getitem_58 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:422 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_121: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(permute_76, [8, 128, 768]);  permute_76 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:435 in forward, code: hidden_states = self.dense(hidden_states)
        view_122: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_121, [1024, 768]);  view_121 = None
        permute_77: "f32[768, 768]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        mm_default_14: "f32[1024, 768]" = torch.ops.aten.mm.default(view_122, permute_77);  view_122 = permute_77 = None
        add_tensor_14: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_14, arg128_1);  mm_default_14 = arg128_1 = None
        view_123: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_14, [8, 128, 768]);  add_tensor_14 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:437 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_53: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_123, add_52);  view_123 = add_52 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
        getitem_62: "f32[8, 128, 1]" = var_mean_15[0]
        getitem_63: "f32[8, 128, 1]" = var_mean_15[1];  var_mean_15 = None
        sub_16: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_53, getitem_63);  add_53 = getitem_63 = None
        add_54: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-12);  getitem_62 = None
        rsqrt_15: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        mul_51: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_15);  sub_16 = rsqrt_15 = None
        mul_52: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_51, arg129_1);  mul_51 = arg129_1 = None
        add_55: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_52, arg130_1);  mul_52 = arg130_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:511 in forward, code: hidden_states = self.dense(hidden_states)
        view_124: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_55, [1024, 768])
        permute_78: "f32[768, 3072]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        mm_default_13: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_124, permute_78);  view_124 = permute_78 = None
        add_tensor_13: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_13, arg132_1);  mm_default_13 = arg132_1 = None
        view_125: "f32[8, 128, 3072]" = torch.ops.aten.reshape.default(add_tensor_13, [8, 128, 3072]);  add_tensor_13 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:70 in forward, code: return self.act(input)
        mul_53: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_125, 0.5)
        mul_54: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_125, 0.7071067811865476);  view_125 = None
        erf_7: "f32[8, 128, 3072]" = torch.ops.aten.erf.default(mul_54);  mul_54 = None
        add_56: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_55: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_53, add_56);  mul_53 = add_56 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:524 in forward, code: hidden_states = self.dense(hidden_states)
        view_126: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_55, [1024, 3072]);  mul_55 = None
        permute_79: "f32[3072, 768]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        mm_default_12: "f32[1024, 768]" = torch.ops.aten.mm.default(view_126, permute_79);  view_126 = permute_79 = None
        add_tensor_12: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_12, arg134_1);  mm_default_12 = arg134_1 = None
        view_127: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_12, [8, 128, 768]);  add_tensor_12 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:526 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_57: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_127, add_55);  view_127 = add_55 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_64: "f32[8, 128, 1]" = var_mean_16[0]
        getitem_65: "f32[8, 128, 1]" = var_mean_16[1];  var_mean_16 = None
        sub_17: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_57, getitem_65);  add_57 = getitem_65 = None
        add_58: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-12);  getitem_64 = None
        rsqrt_16: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_56: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_16);  sub_17 = rsqrt_16 = None
        mul_57: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_56, arg135_1);  mul_56 = arg135_1 = None
        add_59: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_57, arg136_1);  mul_57 = arg136_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:363 in forward, code: self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        view_128: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_59, [1024, 768])
        permute_80: "f32[768, 768]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        addmm_48: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg138_1, view_128, permute_80);  arg138_1 = view_128 = permute_80 = None
        view_129: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(addmm_48, [8, 128, 768]);  addmm_48 = None
        view_130: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(view_129, [8, -1, 12, 64]);  view_129 = None
        permute_81: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:386 in forward, code: self.key(current_states)
        view_131: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_59, [1024, 768])
        permute_82: "f32[768, 768]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_49: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg140_1, view_131, permute_82);  arg140_1 = view_131 = permute_82 = None
        view_132: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(addmm_49, [8, 128, 768]);  addmm_49 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:387 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_133: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(view_132, [8, -1, 12, 64]);  view_132 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:388 in forward, code: .transpose(1, 2)
        permute_83: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_133, [0, 2, 1, 3]);  view_133 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:391 in forward, code: self.value(current_states)
        view_134: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_59, [1024, 768])
        permute_84: "f32[768, 768]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        addmm_50: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg142_1, view_134, permute_84);  arg142_1 = view_134 = permute_84 = None
        view_135: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(addmm_50, [8, 128, 768]);  addmm_50 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:392 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_136: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(view_135, [8, -1, 12, 64]);  view_135 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:393 in forward, code: .transpose(1, 2)
        permute_85: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:412 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_10: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(where, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_81, permute_83, permute_85, expand_10, False);  permute_81 = permute_83 = permute_85 = expand_10 = None
        getitem_66: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_8[0];  _scaled_dot_product_efficient_attention_8 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:421 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_86: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_66, [0, 2, 1, 3]);  getitem_66 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:422 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_137: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(permute_86, [8, 128, 768]);  permute_86 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:435 in forward, code: hidden_states = self.dense(hidden_states)
        view_138: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_137, [1024, 768]);  view_137 = None
        permute_87: "f32[768, 768]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        mm_default_11: "f32[1024, 768]" = torch.ops.aten.mm.default(view_138, permute_87);  view_138 = permute_87 = None
        add_tensor_11: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_11, arg144_1);  mm_default_11 = arg144_1 = None
        view_139: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_11, [8, 128, 768]);  add_tensor_11 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:437 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_60: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_139, add_59);  view_139 = add_59 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_60, [2], correction = 0, keepdim = True)
        getitem_70: "f32[8, 128, 1]" = var_mean_17[0]
        getitem_71: "f32[8, 128, 1]" = var_mean_17[1];  var_mean_17 = None
        sub_18: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_60, getitem_71);  add_60 = getitem_71 = None
        add_61: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-12);  getitem_70 = None
        rsqrt_17: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        mul_58: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_17);  sub_18 = rsqrt_17 = None
        mul_59: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_58, arg145_1);  mul_58 = arg145_1 = None
        add_62: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_59, arg146_1);  mul_59 = arg146_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:511 in forward, code: hidden_states = self.dense(hidden_states)
        view_140: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_62, [1024, 768])
        permute_88: "f32[768, 3072]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        mm_default_10: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_140, permute_88);  view_140 = permute_88 = None
        add_tensor_10: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_10, arg148_1);  mm_default_10 = arg148_1 = None
        view_141: "f32[8, 128, 3072]" = torch.ops.aten.reshape.default(add_tensor_10, [8, 128, 3072]);  add_tensor_10 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:70 in forward, code: return self.act(input)
        mul_60: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_141, 0.5)
        mul_61: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_141, 0.7071067811865476);  view_141 = None
        erf_8: "f32[8, 128, 3072]" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
        add_63: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_62: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_60, add_63);  mul_60 = add_63 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:524 in forward, code: hidden_states = self.dense(hidden_states)
        view_142: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_62, [1024, 3072]);  mul_62 = None
        permute_89: "f32[3072, 768]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        mm_default_9: "f32[1024, 768]" = torch.ops.aten.mm.default(view_142, permute_89);  view_142 = permute_89 = None
        add_tensor_9: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_9, arg150_1);  mm_default_9 = arg150_1 = None
        view_143: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_9, [8, 128, 768]);  add_tensor_9 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:526 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_64: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_143, add_62);  view_143 = add_62 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_64, [2], correction = 0, keepdim = True)
        getitem_72: "f32[8, 128, 1]" = var_mean_18[0]
        getitem_73: "f32[8, 128, 1]" = var_mean_18[1];  var_mean_18 = None
        sub_19: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_64, getitem_73);  add_64 = getitem_73 = None
        add_65: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-12);  getitem_72 = None
        rsqrt_18: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        mul_63: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_18);  sub_19 = rsqrt_18 = None
        mul_64: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_63, arg151_1);  mul_63 = arg151_1 = None
        add_66: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_64, arg152_1);  mul_64 = arg152_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:363 in forward, code: self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        view_144: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_66, [1024, 768])
        permute_90: "f32[768, 768]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        addmm_54: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg154_1, view_144, permute_90);  arg154_1 = view_144 = permute_90 = None
        view_145: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(addmm_54, [8, 128, 768]);  addmm_54 = None
        view_146: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(view_145, [8, -1, 12, 64]);  view_145 = None
        permute_91: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:386 in forward, code: self.key(current_states)
        view_147: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_66, [1024, 768])
        permute_92: "f32[768, 768]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        addmm_55: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg156_1, view_147, permute_92);  arg156_1 = view_147 = permute_92 = None
        view_148: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(addmm_55, [8, 128, 768]);  addmm_55 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:387 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_149: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(view_148, [8, -1, 12, 64]);  view_148 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:388 in forward, code: .transpose(1, 2)
        permute_93: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:391 in forward, code: self.value(current_states)
        view_150: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_66, [1024, 768])
        permute_94: "f32[768, 768]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        addmm_56: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg158_1, view_150, permute_94);  arg158_1 = view_150 = permute_94 = None
        view_151: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(addmm_56, [8, 128, 768]);  addmm_56 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:392 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_152: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(view_151, [8, -1, 12, 64]);  view_151 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:393 in forward, code: .transpose(1, 2)
        permute_95: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_152, [0, 2, 1, 3]);  view_152 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:412 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_11: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(where, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_91, permute_93, permute_95, expand_11, False);  permute_91 = permute_93 = permute_95 = expand_11 = None
        getitem_74: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_9[0];  _scaled_dot_product_efficient_attention_9 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:421 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_96: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_74, [0, 2, 1, 3]);  getitem_74 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:422 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_153: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(permute_96, [8, 128, 768]);  permute_96 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:435 in forward, code: hidden_states = self.dense(hidden_states)
        view_154: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_153, [1024, 768]);  view_153 = None
        permute_97: "f32[768, 768]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        mm_default_8: "f32[1024, 768]" = torch.ops.aten.mm.default(view_154, permute_97);  view_154 = permute_97 = None
        add_tensor_8: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_8, arg160_1);  mm_default_8 = arg160_1 = None
        view_155: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_8, [8, 128, 768]);  add_tensor_8 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:437 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_67: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_155, add_66);  view_155 = add_66 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
        getitem_78: "f32[8, 128, 1]" = var_mean_19[0]
        getitem_79: "f32[8, 128, 1]" = var_mean_19[1];  var_mean_19 = None
        sub_20: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_67, getitem_79);  add_67 = getitem_79 = None
        add_68: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-12);  getitem_78 = None
        rsqrt_19: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        mul_65: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_19);  sub_20 = rsqrt_19 = None
        mul_66: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_65, arg161_1);  mul_65 = arg161_1 = None
        add_69: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_66, arg162_1);  mul_66 = arg162_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:511 in forward, code: hidden_states = self.dense(hidden_states)
        view_156: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_69, [1024, 768])
        permute_98: "f32[768, 3072]" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        mm_default_7: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_156, permute_98);  view_156 = permute_98 = None
        add_tensor_7: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_7, arg164_1);  mm_default_7 = arg164_1 = None
        view_157: "f32[8, 128, 3072]" = torch.ops.aten.reshape.default(add_tensor_7, [8, 128, 3072]);  add_tensor_7 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:70 in forward, code: return self.act(input)
        mul_67: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_157, 0.5)
        mul_68: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_157, 0.7071067811865476);  view_157 = None
        erf_9: "f32[8, 128, 3072]" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
        add_70: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_69: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_67, add_70);  mul_67 = add_70 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:524 in forward, code: hidden_states = self.dense(hidden_states)
        view_158: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_69, [1024, 3072]);  mul_69 = None
        permute_99: "f32[3072, 768]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        mm_default_6: "f32[1024, 768]" = torch.ops.aten.mm.default(view_158, permute_99);  view_158 = permute_99 = None
        add_tensor_6: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_6, arg166_1);  mm_default_6 = arg166_1 = None
        view_159: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_6, [8, 128, 768]);  add_tensor_6 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:526 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_71: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_159, add_69);  view_159 = add_69 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
        getitem_80: "f32[8, 128, 1]" = var_mean_20[0]
        getitem_81: "f32[8, 128, 1]" = var_mean_20[1];  var_mean_20 = None
        sub_21: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_71, getitem_81);  add_71 = getitem_81 = None
        add_72: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-12);  getitem_80 = None
        rsqrt_20: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        mul_70: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_20);  sub_21 = rsqrt_20 = None
        mul_71: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_70, arg167_1);  mul_70 = arg167_1 = None
        add_73: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_71, arg168_1);  mul_71 = arg168_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:363 in forward, code: self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        view_160: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_73, [1024, 768])
        permute_100: "f32[768, 768]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        addmm_60: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg170_1, view_160, permute_100);  arg170_1 = view_160 = permute_100 = None
        view_161: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(addmm_60, [8, 128, 768]);  addmm_60 = None
        view_162: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(view_161, [8, -1, 12, 64]);  view_161 = None
        permute_101: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:386 in forward, code: self.key(current_states)
        view_163: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_73, [1024, 768])
        permute_102: "f32[768, 768]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        addmm_61: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg172_1, view_163, permute_102);  arg172_1 = view_163 = permute_102 = None
        view_164: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(addmm_61, [8, 128, 768]);  addmm_61 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:387 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_165: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(view_164, [8, -1, 12, 64]);  view_164 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:388 in forward, code: .transpose(1, 2)
        permute_103: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_165, [0, 2, 1, 3]);  view_165 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:391 in forward, code: self.value(current_states)
        view_166: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_73, [1024, 768])
        permute_104: "f32[768, 768]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        addmm_62: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg174_1, view_166, permute_104);  arg174_1 = view_166 = permute_104 = None
        view_167: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(addmm_62, [8, 128, 768]);  addmm_62 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:392 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_168: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(view_167, [8, -1, 12, 64]);  view_167 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:393 in forward, code: .transpose(1, 2)
        permute_105: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:412 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_12: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(where, [8, 12, 128, 128])
        _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_101, permute_103, permute_105, expand_12, False);  permute_101 = permute_103 = permute_105 = expand_12 = None
        getitem_82: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_10[0];  _scaled_dot_product_efficient_attention_10 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:421 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_106: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3]);  getitem_82 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:422 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_169: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(permute_106, [8, 128, 768]);  permute_106 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:435 in forward, code: hidden_states = self.dense(hidden_states)
        view_170: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_169, [1024, 768]);  view_169 = None
        permute_107: "f32[768, 768]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        mm_default_5: "f32[1024, 768]" = torch.ops.aten.mm.default(view_170, permute_107);  view_170 = permute_107 = None
        add_tensor_5: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_5, arg176_1);  mm_default_5 = arg176_1 = None
        view_171: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_5, [8, 128, 768]);  add_tensor_5 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:437 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_74: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_171, add_73);  view_171 = add_73 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_74, [2], correction = 0, keepdim = True)
        getitem_86: "f32[8, 128, 1]" = var_mean_21[0]
        getitem_87: "f32[8, 128, 1]" = var_mean_21[1];  var_mean_21 = None
        sub_22: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_74, getitem_87);  add_74 = getitem_87 = None
        add_75: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-12);  getitem_86 = None
        rsqrt_21: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        mul_72: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_21);  sub_22 = rsqrt_21 = None
        mul_73: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_72, arg177_1);  mul_72 = arg177_1 = None
        add_76: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_73, arg178_1);  mul_73 = arg178_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:511 in forward, code: hidden_states = self.dense(hidden_states)
        view_172: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_76, [1024, 768])
        permute_108: "f32[768, 3072]" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        mm_default_4: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_172, permute_108);  view_172 = permute_108 = None
        add_tensor_4: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_4, arg180_1);  mm_default_4 = arg180_1 = None
        view_173: "f32[8, 128, 3072]" = torch.ops.aten.reshape.default(add_tensor_4, [8, 128, 3072]);  add_tensor_4 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:70 in forward, code: return self.act(input)
        mul_74: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.5)
        mul_75: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476);  view_173 = None
        erf_10: "f32[8, 128, 3072]" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
        add_77: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_76: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_74, add_77);  mul_74 = add_77 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:524 in forward, code: hidden_states = self.dense(hidden_states)
        view_174: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_76, [1024, 3072]);  mul_76 = None
        permute_109: "f32[3072, 768]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        mm_default_3: "f32[1024, 768]" = torch.ops.aten.mm.default(view_174, permute_109);  view_174 = permute_109 = None
        add_tensor_3: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_3, arg182_1);  mm_default_3 = arg182_1 = None
        view_175: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_3, [8, 128, 768]);  add_tensor_3 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:526 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_78: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_175, add_76);  view_175 = add_76 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_78, [2], correction = 0, keepdim = True)
        getitem_88: "f32[8, 128, 1]" = var_mean_22[0]
        getitem_89: "f32[8, 128, 1]" = var_mean_22[1];  var_mean_22 = None
        sub_23: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_78, getitem_89);  add_78 = getitem_89 = None
        add_79: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-12);  getitem_88 = None
        rsqrt_22: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
        mul_77: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_22);  sub_23 = rsqrt_22 = None
        mul_78: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_77, arg183_1);  mul_77 = arg183_1 = None
        add_80: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_78, arg184_1);  mul_78 = arg184_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:363 in forward, code: self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        view_176: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_80, [1024, 768])
        permute_110: "f32[768, 768]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        addmm_66: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg186_1, view_176, permute_110);  arg186_1 = view_176 = permute_110 = None
        view_177: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(addmm_66, [8, 128, 768]);  addmm_66 = None
        view_178: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(view_177, [8, -1, 12, 64]);  view_177 = None
        permute_111: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:386 in forward, code: self.key(current_states)
        view_179: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_80, [1024, 768])
        permute_112: "f32[768, 768]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        addmm_67: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg188_1, view_179, permute_112);  arg188_1 = view_179 = permute_112 = None
        view_180: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(addmm_67, [8, 128, 768]);  addmm_67 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:387 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_181: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(view_180, [8, -1, 12, 64]);  view_180 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:388 in forward, code: .transpose(1, 2)
        permute_113: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:391 in forward, code: self.value(current_states)
        view_182: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_80, [1024, 768])
        permute_114: "f32[768, 768]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        addmm_68: "f32[1024, 768]" = torch.ops.aten.addmm.default(arg190_1, view_182, permute_114);  arg190_1 = view_182 = permute_114 = None
        view_183: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(addmm_68, [8, 128, 768]);  addmm_68 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:392 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_184: "f32[8, 128, 12, 64]" = torch.ops.aten.reshape.default(view_183, [8, -1, 12, 64]);  view_183 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:393 in forward, code: .transpose(1, 2)
        permute_115: "f32[8, 12, 128, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:412 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_13: "f32[8, 12, 128, 128]" = torch.ops.aten.expand.default(where, [8, 12, 128, 128]);  where = None
        _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_111, permute_113, permute_115, expand_13, False);  permute_111 = permute_113 = permute_115 = expand_13 = None
        getitem_90: "f32[8, 12, 128, 64]" = _scaled_dot_product_efficient_attention_11[0];  _scaled_dot_product_efficient_attention_11 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:421 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_116: "f32[8, 128, 12, 64]" = torch.ops.aten.permute.default(getitem_90, [0, 2, 1, 3]);  getitem_90 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:422 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_185: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(permute_116, [8, 128, 768]);  permute_116 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:435 in forward, code: hidden_states = self.dense(hidden_states)
        view_186: "f32[1024, 768]" = torch.ops.aten.reshape.default(view_185, [1024, 768]);  view_185 = None
        permute_117: "f32[768, 768]" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        mm_default_2: "f32[1024, 768]" = torch.ops.aten.mm.default(view_186, permute_117);  view_186 = permute_117 = None
        add_tensor_2: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default_2, arg192_1);  mm_default_2 = arg192_1 = None
        view_187: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor_2, [8, 128, 768]);  add_tensor_2 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:437 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_81: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_187, add_80);  view_187 = add_80 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
        getitem_94: "f32[8, 128, 1]" = var_mean_23[0]
        getitem_95: "f32[8, 128, 1]" = var_mean_23[1];  var_mean_23 = None
        sub_24: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_81, getitem_95);  add_81 = getitem_95 = None
        add_82: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-12);  getitem_94 = None
        rsqrt_23: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        mul_79: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_23);  sub_24 = rsqrt_23 = None
        mul_80: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_79, arg193_1);  mul_79 = arg193_1 = None
        add_83: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_80, arg194_1);  mul_80 = arg194_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:511 in forward, code: hidden_states = self.dense(hidden_states)
        view_188: "f32[1024, 768]" = torch.ops.aten.reshape.default(add_83, [1024, 768])
        permute_118: "f32[768, 3072]" = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
        mm_default_1: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_188, permute_118);  view_188 = permute_118 = None
        add_tensor_1: "f32[1024, 3072]" = torch.ops.aten.add.Tensor(mm_default_1, arg196_1);  mm_default_1 = arg196_1 = None
        view_189: "f32[8, 128, 3072]" = torch.ops.aten.reshape.default(add_tensor_1, [8, 128, 3072]);  add_tensor_1 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/activations.py:70 in forward, code: return self.act(input)
        mul_81: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_189, 0.5)
        mul_82: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(view_189, 0.7071067811865476);  view_189 = None
        erf_11: "f32[8, 128, 3072]" = torch.ops.aten.erf.default(mul_82);  mul_82 = None
        add_84: "f32[8, 128, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_83: "f32[8, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_81, add_84);  mul_81 = add_84 = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:524 in forward, code: hidden_states = self.dense(hidden_states)
        view_190: "f32[1024, 3072]" = torch.ops.aten.reshape.default(mul_83, [1024, 3072]);  mul_83 = None
        permute_119: "f32[3072, 768]" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
        mm_default: "f32[1024, 768]" = torch.ops.aten.mm.default(view_190, permute_119);  view_190 = permute_119 = None
        add_tensor: "f32[1024, 768]" = torch.ops.aten.add.Tensor(mm_default, arg198_1);  mm_default = arg198_1 = None
        view_191: "f32[8, 128, 768]" = torch.ops.aten.reshape.default(add_tensor, [8, 128, 768]);  add_tensor = None
        
         # File: /opt/venvs/xla/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:526 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_85: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(view_191, add_83);  view_191 = add_83 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
        getitem_96: "f32[8, 128, 1]" = var_mean_24[0]
        getitem_97: "f32[8, 128, 1]" = var_mean_24[1];  var_mean_24 = None
        sub_25: "f32[8, 128, 768]" = torch.ops.aten.sub.Tensor(add_85, getitem_97);  add_85 = getitem_97 = None
        add_86: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-12);  getitem_96 = None
        rsqrt_24: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        mul_84: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_24);  sub_25 = rsqrt_24 = None
        mul_85: "f32[8, 128, 768]" = torch.ops.aten.mul.Tensor(mul_84, arg199_1);  mul_84 = arg199_1 = None
        add_87: "f32[8, 128, 768]" = torch.ops.aten.add.Tensor(mul_85, arg200_1);  mul_85 = arg200_1 = None
        return (add_87,)
        

# ===== inductor generated file: /tmp/cnnbench-bert-fold-weaagmnf/repeat_02/base/attempt_01/torchinductor/tmp5mgaaf9a/r6/cr6dvhidsakzazbjgcr5uj2e5qory6i6pdmdzxcxjo4omejryiae.py =====

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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_gelu_view_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 37761024}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_gelu_view_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)


# ===== inductor generated file: /tmp/cnnbench-bert-fold-weaagmnf/repeat_02/base/attempt_01/torchinductor/tmp5mgaaf9a/sq/csqnegz2evgpsehsmnndtg3xifvb72hammko7tbs3n2yijtya4yw.py =====

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=20, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_lift_fresh_masked_fill_sub_transpose_unsqueeze_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4B00B69860CF477DDAE6C49CED1F342CC0360AE2DD87517C34B7D29D1AE73394', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2105344}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_expand_lift_fresh_masked_fill_sub_transpose_unsqueeze_view_4(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 128)
    x2 = xindex // 16384
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x2), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp2 - tmp1
    tmp4 = (tmp3 != 0)
    tmp5 = -3.4028234663852886e+38
    tmp6 = tl.where(tmp4, tmp5, tmp3)
    tl.store(out_ptr0 + (x3), tmp6, None)
    tl.store(out_ptr1 + (x3), tmp6, None)
