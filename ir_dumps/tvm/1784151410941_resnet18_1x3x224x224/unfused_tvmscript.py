# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def adaptive_avg_pool2d(lv85: T.Buffer((T.int64(1), T.int64(512), T.int64(7), T.int64(7)), "float32"), adaptive_pool_avg: T.Buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        adaptive_pool_sum = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)))
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                for rv0, rv1 in T.grid(T.int64(7), T.int64(7)):
                    with T.block("adaptive_pool_sum"):
                        v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = T.axis.spatial(T.int64(512), ax0_ax1_ax2_ax3_fused_0 * T.int64(512) + ax0_ax1_ax2_ax3_fused_1)
                        v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_rv0, v_rv1 = T.axis.remap("RR", [rv0, rv1])
                        T.reads(lv85[v_ax0, v_ax1, v_ax2 * T.int64(7) + v_rv0, v_ax3 * T.int64(7) + v_rv1])
                        T.writes(adaptive_pool_sum[v_ax0, v_ax1, v_ax2, v_ax3])
                        with T.init():
                            adaptive_pool_sum[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.0)
                        adaptive_pool_sum[v_ax0, v_ax1, v_ax2, v_ax3] = adaptive_pool_sum[v_ax0, v_ax1, v_ax2, v_ax3] + lv85[v_ax0, v_ax1, v_ax2 * T.int64(7) + v_rv0, v_ax3 * T.int64(7) + v_rv1]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                with T.block("adaptive_pool_avg"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(512), ax0_ax1_ax2_ax3_fused_0 * T.int64(512) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(adaptive_pool_sum[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(adaptive_pool_avg[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.block_attr({"schedule_rule": "meta_schedule.adaptive_pool_avg"})
                    adaptive_pool_avg[v_ax0, v_ax1, v_ax2, v_ax3] = adaptive_pool_sum[v_ax0, v_ax1, v_ax2, v_ax3] * T.float32(0.020408163265306121)

    @T.prim_func(private=True)
    def batch_norm(lv: T.Buffer((T.int64(1), T.int64(64), T.int64(112), T.int64(112)), "float32"), B: T.Buffer((T.int64(64),), "float32"), C: T.Buffer((T.int64(64),), "float32"), D: T.Buffer((T.int64(64),), "float32"), E: T.Buffer((T.int64(64),), "float32"), T_add: T.Buffer((T.int64(1), T.int64(64), T.int64(112), T.int64(112)), "float32"), T_add_1: T.Buffer((T.int64(64),), "float32"), T_add_2: T.Buffer((T.int64(64),), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        T_reshape = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(1), T.int64(1)))
        T_subtract = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(112), T.int64(112)))
        T_reshape_1 = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(1), T.int64(1)))
        T_add_3 = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(1), T.int64(1)))
        compute = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(1), T.int64(1)))
        T_divide = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(112), T.int64(112)))
        T_reshape_2 = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(1), T.int64(1)))
        T_multiply = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(112), T.int64(112)))
        T_reshape_3 = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(1), T.int64(1)))
        T_multiply_1 = T.alloc_buffer((T.int64(64),))
        lv_red = T.alloc_buffer((T.int64(64),))
        T_divide_1 = T.alloc_buffer((T.int64(64),))
        T_multiply_2 = T.alloc_buffer((T.int64(64),))
        T_multiply_3 = T.alloc_buffer((T.int64(64),))
        T_reshape_4 = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(1), T.int64(1)))
        T_subtract_1 = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(112), T.int64(112)))
        T_subtract_2 = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(112), T.int64(112)))
        T_multiply_4 = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(112), T.int64(112)))
        T_multiply_red = T.alloc_buffer((T.int64(64),))
        T_divide_2 = T.alloc_buffer((T.int64(64),))
        T_multiply_5 = T.alloc_buffer((T.int64(64),))
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                with T.block("T_reshape"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(64), ax0_ax1_ax2_ax3_fused_0 * T.int64(64) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(D[(v_ax1 + v_ax2 + v_ax3) % T.int64(64)])
                    T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = D[(v_ax1 + v_ax2 + v_ax3) % T.int64(64)]
        for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(4)):
                    with T.block("T_subtract"):
                        v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) // T.int64(12544))
                        v_ax2 = T.axis.spatial(T.int64(112), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(12544) // T.int64(112))
                        v_ax3 = T.axis.spatial(T.int64(112), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(112))
                        T.where((ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1) * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2 < T.int64(802816))
                        T.reads(lv[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                        T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                        T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = lv[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                with T.block("T_reshape_1"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(64), ax0_ax1_ax2_ax3_fused_0 * T.int64(64) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(E[(v_ax1 + v_ax2 + v_ax3) % T.int64(64)])
                    T.writes(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3] = E[(v_ax1 + v_ax2 + v_ax3) % T.int64(64)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                with T.block("T_add"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(64), ax0_ax1_ax2_ax3_fused_0 * T.int64(64) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_add_3[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_add_3[v_ax0, v_ax1, v_ax2, v_ax3] = T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3] + T.float32(1.0000000000000001e-05)
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                with T.block("compute"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(64), i0_i1_i2_i3_fused_0 * T.int64(64) + i0_i1_i2_i3_fused_1)
                    v_i2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(T_add_3[v_i0, v_i1, v_i2, v_i3])
                    T.writes(compute[v_i0, v_i1, v_i2, v_i3])
                    compute[v_i0, v_i1, v_i2, v_i3] = T.sqrt(T_add_3[v_i0, v_i1, v_i2, v_i3])
        for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(4)):
                    with T.block("T_divide"):
                        v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) // T.int64(12544))
                        v_ax2 = T.axis.spatial(T.int64(112), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(12544) // T.int64(112))
                        v_ax3 = T.axis.spatial(T.int64(112), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(112))
                        T.where((ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1) * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2 < T.int64(802816))
                        T.reads(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3], compute[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                        T.writes(T_divide[v_ax0, v_ax1, v_ax2, v_ax3])
                        T_divide[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] / compute[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                with T.block("T_reshape_2"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(64), ax0_ax1_ax2_ax3_fused_0 * T.int64(64) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(B[(v_ax1 + v_ax2 + v_ax3) % T.int64(64)])
                    T.writes(T_reshape_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_2[v_ax0, v_ax1, v_ax2, v_ax3] = B[(v_ax1 + v_ax2 + v_ax3) % T.int64(64)]
        for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(4)):
                    with T.block("T_multiply"):
                        v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) // T.int64(12544))
                        v_ax2 = T.axis.spatial(T.int64(112), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(12544) // T.int64(112))
                        v_ax3 = T.axis.spatial(T.int64(112), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(112))
                        T.where((ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1) * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2 < T.int64(802816))
                        T.reads(T_divide[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_2[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                        T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                        T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T_divide[v_ax0, v_ax1, v_ax2, v_ax3] * T_reshape_2[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                with T.block("T_reshape_3"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(64), ax0_ax1_ax2_ax3_fused_0 * T.int64(64) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(C[(v_ax1 + v_ax2 + v_ax3) % T.int64(64)])
                    T.writes(T_reshape_3[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_3[v_ax0, v_ax1, v_ax2, v_ax3] = C[(v_ax1 + v_ax2 + v_ax3) % T.int64(64)]
        for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(4)):
                    with T.block("T_add_1"):
                        v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) // T.int64(12544))
                        v_ax2 = T.axis.spatial(T.int64(112), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(12544) // T.int64(112))
                        v_ax3 = T.axis.spatial(T.int64(112), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(112))
                        T.where((ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1) * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2 < T.int64(802816))
                        T.reads(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_3[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                        T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                        T_add[v_ax0, v_ax1, v_ax2, v_ax3] = T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] + T_reshape_3[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                with T.block("T_multiply_1"):
                    v_ax0 = T.axis.spatial(T.int64(64), ax0_fused_0 * T.int64(64) + ax0_fused_1)
                    T.reads(D[v_ax0])
                    T.writes(T_multiply_1[v_ax0])
                    T_multiply_1[v_ax0] = T.float32(0.90000000000000002) * D[v_ax0]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                for k0, k2, k3 in T.grid(T.int64(1), T.int64(112), T.int64(112)):
                    with T.block("lv_red"):
                        v_ax0 = T.axis.spatial(T.int64(64), ax0_fused_0 * T.int64(64) + ax0_fused_1)
                        v_k0 = T.axis.reduce(T.int64(1), T.int64(0))
                        v_k2, v_k3 = T.axis.remap("RR", [k2, k3])
                        T.reads(lv[v_k0, v_ax0, v_k2, v_k3])
                        T.writes(lv_red[v_ax0])
                        with T.init():
                            lv_red[v_ax0] = T.float32(0.0)
                        lv_red[v_ax0] = lv_red[v_ax0] + lv[v_k0, v_ax0, v_k2, v_k3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                with T.block("T_divide_1"):
                    v_ax0 = T.axis.spatial(T.int64(64), ax0_fused_0 * T.int64(64) + ax0_fused_1)
                    T.reads(lv_red[v_ax0])
                    T.writes(T_divide_1[v_ax0])
                    T_divide_1[v_ax0] = lv_red[v_ax0] * T.float32(7.9719387755102034e-05)
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                with T.block("T_multiply_2"):
                    v_ax0 = T.axis.spatial(T.int64(64), ax0_fused_0 * T.int64(64) + ax0_fused_1)
                    T.reads(T_divide_1[v_ax0])
                    T.writes(T_multiply_2[v_ax0])
                    T_multiply_2[v_ax0] = T.float32(0.10000000000000001) * T_divide_1[v_ax0]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                with T.block("T_add_2"):
                    v_ax0 = T.axis.spatial(T.int64(64), ax0_fused_0 * T.int64(64) + ax0_fused_1)
                    T.reads(T_multiply_1[v_ax0], T_multiply_2[v_ax0])
                    T.writes(T_add_1[v_ax0])
                    T_add_1[v_ax0] = T_multiply_1[v_ax0] + T_multiply_2[v_ax0]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                with T.block("T_multiply_3"):
                    v_ax0 = T.axis.spatial(T.int64(64), ax0_fused_0 * T.int64(64) + ax0_fused_1)
                    T.reads(E[v_ax0])
                    T.writes(T_multiply_3[v_ax0])
                    T_multiply_3[v_ax0] = T.float32(0.90000000000000002) * E[v_ax0]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                with T.block("T_reshape_4"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(64), ax0_ax1_ax2_ax3_fused_0 * T.int64(64) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(T_divide_1[(v_ax1 + v_ax2 + v_ax3) % T.int64(64)])
                    T.writes(T_reshape_4[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_4[v_ax0, v_ax1, v_ax2, v_ax3] = T_divide_1[(v_ax1 + v_ax2 + v_ax3) % T.int64(64)]
        for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(4)):
                    with T.block("T_subtract_1"):
                        v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) // T.int64(12544))
                        v_ax2 = T.axis.spatial(T.int64(112), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(12544) // T.int64(112))
                        v_ax3 = T.axis.spatial(T.int64(112), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(112))
                        T.where((ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1) * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2 < T.int64(802816))
                        T.reads(lv[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                        T.writes(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3])
                        T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] = lv[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(4)):
                    with T.block("T_subtract_2"):
                        v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) // T.int64(12544))
                        v_ax2 = T.axis.spatial(T.int64(112), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(12544) // T.int64(112))
                        v_ax3 = T.axis.spatial(T.int64(112), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(112))
                        T.where((ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1) * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2 < T.int64(802816))
                        T.reads(lv[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                        T.writes(T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
                        T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3] = lv[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(4)):
                    with T.block("T_multiply_4"):
                        v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) // T.int64(12544))
                        v_ax2 = T.axis.spatial(T.int64(112), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(12544) // T.int64(112))
                        v_ax3 = T.axis.spatial(T.int64(112), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(112))
                        T.where((ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1) * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2 < T.int64(802816))
                        T.reads(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3], T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
                        T.writes(T_multiply_4[v_ax0, v_ax1, v_ax2, v_ax3])
                        T_multiply_4[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] * T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                for k0, k2, k3 in T.grid(T.int64(1), T.int64(112), T.int64(112)):
                    with T.block("T_multiply_red"):
                        v_ax0 = T.axis.spatial(T.int64(64), ax0_fused_0 * T.int64(64) + ax0_fused_1)
                        v_k0 = T.axis.reduce(T.int64(1), T.int64(0))
                        v_k2, v_k3 = T.axis.remap("RR", [k2, k3])
                        T.reads(T_multiply_4[v_k0, v_ax0, v_k2, v_k3])
                        T.writes(T_multiply_red[v_ax0])
                        with T.init():
                            T_multiply_red[v_ax0] = T.float32(0.0)
                        T_multiply_red[v_ax0] = T_multiply_red[v_ax0] + T_multiply_4[v_k0, v_ax0, v_k2, v_k3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                with T.block("T_divide_2"):
                    v_ax0 = T.axis.spatial(T.int64(64), ax0_fused_0 * T.int64(64) + ax0_fused_1)
                    T.reads(T_multiply_red[v_ax0])
                    T.writes(T_divide_2[v_ax0])
                    T_divide_2[v_ax0] = T_multiply_red[v_ax0] * T.float32(7.9719387755102034e-05)
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                with T.block("T_multiply_5"):
                    v_ax0 = T.axis.spatial(T.int64(64), ax0_fused_0 * T.int64(64) + ax0_fused_1)
                    T.reads(T_divide_2[v_ax0])
                    T.writes(T_multiply_5[v_ax0])
                    T_multiply_5[v_ax0] = T.float32(0.10000000000000001) * T_divide_2[v_ax0]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                with T.block("T_add_3"):
                    v_ax0 = T.axis.spatial(T.int64(64), ax0_fused_0 * T.int64(64) + ax0_fused_1)
                    T.reads(T_multiply_3[v_ax0], T_multiply_5[v_ax0])
                    T.writes(T_add_2[v_ax0])
                    T_add_2[v_ax0] = T_multiply_3[v_ax0] + T_multiply_5[v_ax0]

    @T.prim_func(private=True)
    def batch_norm1(lv5: T.Buffer((T.int64(1), T.int64(64), T.int64(56), T.int64(56)), "float32"), B: T.Buffer((T.int64(64),), "float32"), C: T.Buffer((T.int64(64),), "float32"), D: T.Buffer((T.int64(64),), "float32"), E: T.Buffer((T.int64(64),), "float32"), T_add: T.Buffer((T.int64(1), T.int64(64), T.int64(56), T.int64(56)), "float32"), T_add_1: T.Buffer((T.int64(64),), "float32"), T_add_2: T.Buffer((T.int64(64),), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        T_reshape = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(1), T.int64(1)))
        T_subtract = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(56), T.int64(56)))
        T_reshape_1 = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(1), T.int64(1)))
        T_add_3 = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(1), T.int64(1)))
        compute = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(1), T.int64(1)))
        T_divide = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(56), T.int64(56)))
        T_reshape_2 = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(1), T.int64(1)))
        T_multiply = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(56), T.int64(56)))
        T_reshape_3 = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(1), T.int64(1)))
        T_multiply_1 = T.alloc_buffer((T.int64(64),))
        lv5_red = T.alloc_buffer((T.int64(64),))
        T_divide_1 = T.alloc_buffer((T.int64(64),))
        T_multiply_2 = T.alloc_buffer((T.int64(64),))
        T_multiply_3 = T.alloc_buffer((T.int64(64),))
        T_reshape_4 = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(1), T.int64(1)))
        T_subtract_1 = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(56), T.int64(56)))
        T_subtract_2 = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(56), T.int64(56)))
        T_multiply_4 = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(56), T.int64(56)))
        T_multiply_red = T.alloc_buffer((T.int64(64),))
        T_divide_2 = T.alloc_buffer((T.int64(64),))
        T_multiply_5 = T.alloc_buffer((T.int64(64),))
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                with T.block("T_reshape"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(64), ax0_ax1_ax2_ax3_fused_0 * T.int64(64) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(D[(v_ax1 + v_ax2 + v_ax3) % T.int64(64)])
                    T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = D[(v_ax1 + v_ax2 + v_ax3) % T.int64(64)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_subtract"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(3136))
                    v_ax2 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(3136) // T.int64(56))
                    v_ax3 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(56))
                    T.reads(lv5[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = lv5[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                with T.block("T_reshape_1"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(64), ax0_ax1_ax2_ax3_fused_0 * T.int64(64) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(E[(v_ax1 + v_ax2 + v_ax3) % T.int64(64)])
                    T.writes(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3] = E[(v_ax1 + v_ax2 + v_ax3) % T.int64(64)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                with T.block("T_add"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(64), ax0_ax1_ax2_ax3_fused_0 * T.int64(64) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_add_3[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_add_3[v_ax0, v_ax1, v_ax2, v_ax3] = T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3] + T.float32(1.0000000000000001e-05)
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                with T.block("compute"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(64), i0_i1_i2_i3_fused_0 * T.int64(64) + i0_i1_i2_i3_fused_1)
                    v_i2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(T_add_3[v_i0, v_i1, v_i2, v_i3])
                    T.writes(compute[v_i0, v_i1, v_i2, v_i3])
                    compute[v_i0, v_i1, v_i2, v_i3] = T.sqrt(T_add_3[v_i0, v_i1, v_i2, v_i3])
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_divide"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(3136))
                    v_ax2 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(3136) // T.int64(56))
                    v_ax3 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(56))
                    T.reads(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3], compute[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_divide[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_divide[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] / compute[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                with T.block("T_reshape_2"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(64), ax0_ax1_ax2_ax3_fused_0 * T.int64(64) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(B[(v_ax1 + v_ax2 + v_ax3) % T.int64(64)])
                    T.writes(T_reshape_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_2[v_ax0, v_ax1, v_ax2, v_ax3] = B[(v_ax1 + v_ax2 + v_ax3) % T.int64(64)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(3136))
                    v_ax2 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(3136) // T.int64(56))
                    v_ax3 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(56))
                    T.reads(T_divide[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_2[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T_divide[v_ax0, v_ax1, v_ax2, v_ax3] * T_reshape_2[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                with T.block("T_reshape_3"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(64), ax0_ax1_ax2_ax3_fused_0 * T.int64(64) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(C[(v_ax1 + v_ax2 + v_ax3) % T.int64(64)])
                    T.writes(T_reshape_3[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_3[v_ax0, v_ax1, v_ax2, v_ax3] = C[(v_ax1 + v_ax2 + v_ax3) % T.int64(64)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_1"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(3136))
                    v_ax2 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(3136) // T.int64(56))
                    v_ax3 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(56))
                    T.reads(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_3[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_add[v_ax0, v_ax1, v_ax2, v_ax3] = T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] + T_reshape_3[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                with T.block("T_multiply_1"):
                    v_ax0 = T.axis.spatial(T.int64(64), ax0_fused_0 * T.int64(64) + ax0_fused_1)
                    T.reads(D[v_ax0])
                    T.writes(T_multiply_1[v_ax0])
                    T_multiply_1[v_ax0] = T.float32(0.90000000000000002) * D[v_ax0]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                for k0, k2, k3 in T.grid(T.int64(1), T.int64(56), T.int64(56)):
                    with T.block("lv5_red"):
                        v_ax0 = T.axis.spatial(T.int64(64), ax0_fused_0 * T.int64(64) + ax0_fused_1)
                        v_k0 = T.axis.reduce(T.int64(1), T.int64(0))
                        v_k2, v_k3 = T.axis.remap("RR", [k2, k3])
                        T.reads(lv5[v_k0, v_ax0, v_k2, v_k3])
                        T.writes(lv5_red[v_ax0])
                        with T.init():
                            lv5_red[v_ax0] = T.float32(0.0)
                        lv5_red[v_ax0] = lv5_red[v_ax0] + lv5[v_k0, v_ax0, v_k2, v_k3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                with T.block("T_divide_1"):
                    v_ax0 = T.axis.spatial(T.int64(64), ax0_fused_0 * T.int64(64) + ax0_fused_1)
                    T.reads(lv5_red[v_ax0])
                    T.writes(T_divide_1[v_ax0])
                    T_divide_1[v_ax0] = lv5_red[v_ax0] * T.float32(0.00031887755102040814)
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                with T.block("T_multiply_2"):
                    v_ax0 = T.axis.spatial(T.int64(64), ax0_fused_0 * T.int64(64) + ax0_fused_1)
                    T.reads(T_divide_1[v_ax0])
                    T.writes(T_multiply_2[v_ax0])
                    T_multiply_2[v_ax0] = T.float32(0.10000000000000001) * T_divide_1[v_ax0]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                with T.block("T_add_2"):
                    v_ax0 = T.axis.spatial(T.int64(64), ax0_fused_0 * T.int64(64) + ax0_fused_1)
                    T.reads(T_multiply_1[v_ax0], T_multiply_2[v_ax0])
                    T.writes(T_add_1[v_ax0])
                    T_add_1[v_ax0] = T_multiply_1[v_ax0] + T_multiply_2[v_ax0]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                with T.block("T_multiply_3"):
                    v_ax0 = T.axis.spatial(T.int64(64), ax0_fused_0 * T.int64(64) + ax0_fused_1)
                    T.reads(E[v_ax0])
                    T.writes(T_multiply_3[v_ax0])
                    T_multiply_3[v_ax0] = T.float32(0.90000000000000002) * E[v_ax0]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                with T.block("T_reshape_4"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(64), ax0_ax1_ax2_ax3_fused_0 * T.int64(64) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(T_divide_1[(v_ax1 + v_ax2 + v_ax3) % T.int64(64)])
                    T.writes(T_reshape_4[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_4[v_ax0, v_ax1, v_ax2, v_ax3] = T_divide_1[(v_ax1 + v_ax2 + v_ax3) % T.int64(64)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_subtract_1"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(3136))
                    v_ax2 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(3136) // T.int64(56))
                    v_ax3 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(56))
                    T.reads(lv5[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] = lv5[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_subtract_2"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(3136))
                    v_ax2 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(3136) // T.int64(56))
                    v_ax3 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(56))
                    T.reads(lv5[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3] = lv5[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply_4"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(3136))
                    v_ax2 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(3136) // T.int64(56))
                    v_ax3 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(56))
                    T.reads(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3], T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_multiply_4[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_multiply_4[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] * T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                for k0, k2, k3 in T.grid(T.int64(1), T.int64(56), T.int64(56)):
                    with T.block("T_multiply_red"):
                        v_ax0 = T.axis.spatial(T.int64(64), ax0_fused_0 * T.int64(64) + ax0_fused_1)
                        v_k0 = T.axis.reduce(T.int64(1), T.int64(0))
                        v_k2, v_k3 = T.axis.remap("RR", [k2, k3])
                        T.reads(T_multiply_4[v_k0, v_ax0, v_k2, v_k3])
                        T.writes(T_multiply_red[v_ax0])
                        with T.init():
                            T_multiply_red[v_ax0] = T.float32(0.0)
                        T_multiply_red[v_ax0] = T_multiply_red[v_ax0] + T_multiply_4[v_k0, v_ax0, v_k2, v_k3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                with T.block("T_divide_2"):
                    v_ax0 = T.axis.spatial(T.int64(64), ax0_fused_0 * T.int64(64) + ax0_fused_1)
                    T.reads(T_multiply_red[v_ax0])
                    T.writes(T_divide_2[v_ax0])
                    T_divide_2[v_ax0] = T_multiply_red[v_ax0] * T.float32(0.00031887755102040814)
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                with T.block("T_multiply_5"):
                    v_ax0 = T.axis.spatial(T.int64(64), ax0_fused_0 * T.int64(64) + ax0_fused_1)
                    T.reads(T_divide_2[v_ax0])
                    T.writes(T_multiply_5[v_ax0])
                    T_multiply_5[v_ax0] = T.float32(0.10000000000000001) * T_divide_2[v_ax0]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(64), thread="threadIdx.x"):
                with T.block("T_add_3"):
                    v_ax0 = T.axis.spatial(T.int64(64), ax0_fused_0 * T.int64(64) + ax0_fused_1)
                    T.reads(T_multiply_3[v_ax0], T_multiply_5[v_ax0])
                    T.writes(T_add_2[v_ax0])
                    T_add_2[v_ax0] = T_multiply_3[v_ax0] + T_multiply_5[v_ax0]

    @T.prim_func(private=True)
    def batch_norm2(lv23: T.Buffer((T.int64(1), T.int64(128), T.int64(28), T.int64(28)), "float32"), B: T.Buffer((T.int64(128),), "float32"), C: T.Buffer((T.int64(128),), "float32"), D: T.Buffer((T.int64(128),), "float32"), E: T.Buffer((T.int64(128),), "float32"), T_add: T.Buffer((T.int64(1), T.int64(128), T.int64(28), T.int64(28)), "float32"), T_add_1: T.Buffer((T.int64(128),), "float32"), T_add_2: T.Buffer((T.int64(128),), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        T_reshape = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)))
        T_subtract = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(28), T.int64(28)))
        T_reshape_1 = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)))
        T_add_3 = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)))
        compute = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)))
        T_divide = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(28), T.int64(28)))
        T_reshape_2 = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)))
        T_multiply = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(28), T.int64(28)))
        T_reshape_3 = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)))
        T_multiply_1 = T.alloc_buffer((T.int64(128),))
        lv23_red = T.alloc_buffer((T.int64(128),))
        T_divide_1 = T.alloc_buffer((T.int64(128),))
        T_multiply_2 = T.alloc_buffer((T.int64(128),))
        T_multiply_3 = T.alloc_buffer((T.int64(128),))
        T_reshape_4 = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)))
        T_subtract_1 = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(28), T.int64(28)))
        T_subtract_2 = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(28), T.int64(28)))
        T_multiply_4 = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(28), T.int64(28)))
        T_multiply_red = T.alloc_buffer((T.int64(128),))
        T_divide_2 = T.alloc_buffer((T.int64(128),))
        T_multiply_5 = T.alloc_buffer((T.int64(128),))
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                with T.block("T_reshape"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(128), ax0_ax1_ax2_ax3_fused_0 * T.int64(128) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(D[(v_ax1 + v_ax2 + v_ax3) % T.int64(128)])
                    T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = D[(v_ax1 + v_ax2 + v_ax3) % T.int64(128)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_subtract"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(128), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(784))
                    v_ax2 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(784) // T.int64(28))
                    v_ax3 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(28))
                    T.reads(lv23[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = lv23[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                with T.block("T_reshape_1"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(128), ax0_ax1_ax2_ax3_fused_0 * T.int64(128) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(E[(v_ax1 + v_ax2 + v_ax3) % T.int64(128)])
                    T.writes(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3] = E[(v_ax1 + v_ax2 + v_ax3) % T.int64(128)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                with T.block("T_add"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(128), ax0_ax1_ax2_ax3_fused_0 * T.int64(128) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_add_3[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_add_3[v_ax0, v_ax1, v_ax2, v_ax3] = T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3] + T.float32(1.0000000000000001e-05)
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                with T.block("compute"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(128), i0_i1_i2_i3_fused_0 * T.int64(128) + i0_i1_i2_i3_fused_1)
                    v_i2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(T_add_3[v_i0, v_i1, v_i2, v_i3])
                    T.writes(compute[v_i0, v_i1, v_i2, v_i3])
                    compute[v_i0, v_i1, v_i2, v_i3] = T.sqrt(T_add_3[v_i0, v_i1, v_i2, v_i3])
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_divide"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(128), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(784))
                    v_ax2 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(784) // T.int64(28))
                    v_ax3 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(28))
                    T.reads(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3], compute[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_divide[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_divide[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] / compute[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                with T.block("T_reshape_2"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(128), ax0_ax1_ax2_ax3_fused_0 * T.int64(128) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(B[(v_ax1 + v_ax2 + v_ax3) % T.int64(128)])
                    T.writes(T_reshape_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_2[v_ax0, v_ax1, v_ax2, v_ax3] = B[(v_ax1 + v_ax2 + v_ax3) % T.int64(128)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(128), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(784))
                    v_ax2 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(784) // T.int64(28))
                    v_ax3 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(28))
                    T.reads(T_divide[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_2[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T_divide[v_ax0, v_ax1, v_ax2, v_ax3] * T_reshape_2[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                with T.block("T_reshape_3"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(128), ax0_ax1_ax2_ax3_fused_0 * T.int64(128) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(C[(v_ax1 + v_ax2 + v_ax3) % T.int64(128)])
                    T.writes(T_reshape_3[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_3[v_ax0, v_ax1, v_ax2, v_ax3] = C[(v_ax1 + v_ax2 + v_ax3) % T.int64(128)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_1"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(128), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(784))
                    v_ax2 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(784) // T.int64(28))
                    v_ax3 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(28))
                    T.reads(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_3[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_add[v_ax0, v_ax1, v_ax2, v_ax3] = T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] + T_reshape_3[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                with T.block("T_multiply_1"):
                    v_ax0 = T.axis.spatial(T.int64(128), ax0_fused_0 * T.int64(128) + ax0_fused_1)
                    T.reads(D[v_ax0])
                    T.writes(T_multiply_1[v_ax0])
                    T_multiply_1[v_ax0] = T.float32(0.90000000000000002) * D[v_ax0]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for k0, k2, k3 in T.grid(T.int64(1), T.int64(28), T.int64(28)):
                    with T.block("lv23_red"):
                        v_ax0 = T.axis.spatial(T.int64(128), ax0_fused_0 * T.int64(128) + ax0_fused_1)
                        v_k0 = T.axis.reduce(T.int64(1), T.int64(0))
                        v_k2, v_k3 = T.axis.remap("RR", [k2, k3])
                        T.reads(lv23[v_k0, v_ax0, v_k2, v_k3])
                        T.writes(lv23_red[v_ax0])
                        with T.init():
                            lv23_red[v_ax0] = T.float32(0.0)
                        lv23_red[v_ax0] = lv23_red[v_ax0] + lv23[v_k0, v_ax0, v_k2, v_k3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                with T.block("T_divide_1"):
                    v_ax0 = T.axis.spatial(T.int64(128), ax0_fused_0 * T.int64(128) + ax0_fused_1)
                    T.reads(lv23_red[v_ax0])
                    T.writes(T_divide_1[v_ax0])
                    T_divide_1[v_ax0] = lv23_red[v_ax0] * T.float32(0.0012755102040816326)
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                with T.block("T_multiply_2"):
                    v_ax0 = T.axis.spatial(T.int64(128), ax0_fused_0 * T.int64(128) + ax0_fused_1)
                    T.reads(T_divide_1[v_ax0])
                    T.writes(T_multiply_2[v_ax0])
                    T_multiply_2[v_ax0] = T.float32(0.10000000000000001) * T_divide_1[v_ax0]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                with T.block("T_add_2"):
                    v_ax0 = T.axis.spatial(T.int64(128), ax0_fused_0 * T.int64(128) + ax0_fused_1)
                    T.reads(T_multiply_1[v_ax0], T_multiply_2[v_ax0])
                    T.writes(T_add_1[v_ax0])
                    T_add_1[v_ax0] = T_multiply_1[v_ax0] + T_multiply_2[v_ax0]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                with T.block("T_multiply_3"):
                    v_ax0 = T.axis.spatial(T.int64(128), ax0_fused_0 * T.int64(128) + ax0_fused_1)
                    T.reads(E[v_ax0])
                    T.writes(T_multiply_3[v_ax0])
                    T_multiply_3[v_ax0] = T.float32(0.90000000000000002) * E[v_ax0]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                with T.block("T_reshape_4"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(128), ax0_ax1_ax2_ax3_fused_0 * T.int64(128) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(T_divide_1[(v_ax1 + v_ax2 + v_ax3) % T.int64(128)])
                    T.writes(T_reshape_4[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_4[v_ax0, v_ax1, v_ax2, v_ax3] = T_divide_1[(v_ax1 + v_ax2 + v_ax3) % T.int64(128)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_subtract_1"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(128), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(784))
                    v_ax2 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(784) // T.int64(28))
                    v_ax3 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(28))
                    T.reads(lv23[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] = lv23[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_subtract_2"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(128), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(784))
                    v_ax2 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(784) // T.int64(28))
                    v_ax3 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(28))
                    T.reads(lv23[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3] = lv23[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply_4"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(128), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(784))
                    v_ax2 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(784) // T.int64(28))
                    v_ax3 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(28))
                    T.reads(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3], T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_multiply_4[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_multiply_4[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] * T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for k0, k2, k3 in T.grid(T.int64(1), T.int64(28), T.int64(28)):
                    with T.block("T_multiply_red"):
                        v_ax0 = T.axis.spatial(T.int64(128), ax0_fused_0 * T.int64(128) + ax0_fused_1)
                        v_k0 = T.axis.reduce(T.int64(1), T.int64(0))
                        v_k2, v_k3 = T.axis.remap("RR", [k2, k3])
                        T.reads(T_multiply_4[v_k0, v_ax0, v_k2, v_k3])
                        T.writes(T_multiply_red[v_ax0])
                        with T.init():
                            T_multiply_red[v_ax0] = T.float32(0.0)
                        T_multiply_red[v_ax0] = T_multiply_red[v_ax0] + T_multiply_4[v_k0, v_ax0, v_k2, v_k3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                with T.block("T_divide_2"):
                    v_ax0 = T.axis.spatial(T.int64(128), ax0_fused_0 * T.int64(128) + ax0_fused_1)
                    T.reads(T_multiply_red[v_ax0])
                    T.writes(T_divide_2[v_ax0])
                    T_divide_2[v_ax0] = T_multiply_red[v_ax0] * T.float32(0.0012755102040816326)
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                with T.block("T_multiply_5"):
                    v_ax0 = T.axis.spatial(T.int64(128), ax0_fused_0 * T.int64(128) + ax0_fused_1)
                    T.reads(T_divide_2[v_ax0])
                    T.writes(T_multiply_5[v_ax0])
                    T_multiply_5[v_ax0] = T.float32(0.10000000000000001) * T_divide_2[v_ax0]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                with T.block("T_add_3"):
                    v_ax0 = T.axis.spatial(T.int64(128), ax0_fused_0 * T.int64(128) + ax0_fused_1)
                    T.reads(T_multiply_3[v_ax0], T_multiply_5[v_ax0])
                    T.writes(T_add_2[v_ax0])
                    T_add_2[v_ax0] = T_multiply_3[v_ax0] + T_multiply_5[v_ax0]

    @T.prim_func(private=True)
    def batch_norm3(lv44: T.Buffer((T.int64(1), T.int64(256), T.int64(14), T.int64(14)), "float32"), B: T.Buffer((T.int64(256),), "float32"), C: T.Buffer((T.int64(256),), "float32"), D: T.Buffer((T.int64(256),), "float32"), E: T.Buffer((T.int64(256),), "float32"), T_add: T.Buffer((T.int64(1), T.int64(256), T.int64(14), T.int64(14)), "float32"), T_add_1: T.Buffer((T.int64(256),), "float32"), T_add_2: T.Buffer((T.int64(256),), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        T_reshape = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)))
        T_subtract = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(14), T.int64(14)))
        T_reshape_1 = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)))
        T_add_3 = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)))
        compute = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)))
        T_divide = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(14), T.int64(14)))
        T_reshape_2 = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)))
        T_multiply = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(14), T.int64(14)))
        T_reshape_3 = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)))
        T_multiply_1 = T.alloc_buffer((T.int64(256),))
        lv44_red = T.alloc_buffer((T.int64(256),))
        T_divide_1 = T.alloc_buffer((T.int64(256),))
        T_multiply_2 = T.alloc_buffer((T.int64(256),))
        T_multiply_3 = T.alloc_buffer((T.int64(256),))
        T_reshape_4 = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)))
        T_subtract_1 = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(14), T.int64(14)))
        T_subtract_2 = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(14), T.int64(14)))
        T_multiply_4 = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(14), T.int64(14)))
        T_multiply_red = T.alloc_buffer((T.int64(256),))
        T_divide_2 = T.alloc_buffer((T.int64(256),))
        T_multiply_5 = T.alloc_buffer((T.int64(256),))
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                with T.block("T_reshape"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(256), ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(D[(v_ax1 + v_ax2 + v_ax3) % T.int64(256)])
                    T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = D[(v_ax1 + v_ax2 + v_ax3) % T.int64(256)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(49), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_subtract"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(256), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(196))
                    v_ax2 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(196) // T.int64(14))
                    v_ax3 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(14))
                    T.reads(lv44[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = lv44[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                with T.block("T_reshape_1"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(256), ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(E[(v_ax1 + v_ax2 + v_ax3) % T.int64(256)])
                    T.writes(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3] = E[(v_ax1 + v_ax2 + v_ax3) % T.int64(256)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                with T.block("T_add"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(256), ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_add_3[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_add_3[v_ax0, v_ax1, v_ax2, v_ax3] = T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3] + T.float32(1.0000000000000001e-05)
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                with T.block("compute"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(256), i0_i1_i2_i3_fused_0 * T.int64(256) + i0_i1_i2_i3_fused_1)
                    v_i2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(T_add_3[v_i0, v_i1, v_i2, v_i3])
                    T.writes(compute[v_i0, v_i1, v_i2, v_i3])
                    compute[v_i0, v_i1, v_i2, v_i3] = T.sqrt(T_add_3[v_i0, v_i1, v_i2, v_i3])
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(49), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_divide"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(256), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(196))
                    v_ax2 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(196) // T.int64(14))
                    v_ax3 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(14))
                    T.reads(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3], compute[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_divide[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_divide[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] / compute[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                with T.block("T_reshape_2"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(256), ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(B[(v_ax1 + v_ax2 + v_ax3) % T.int64(256)])
                    T.writes(T_reshape_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_2[v_ax0, v_ax1, v_ax2, v_ax3] = B[(v_ax1 + v_ax2 + v_ax3) % T.int64(256)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(49), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(256), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(196))
                    v_ax2 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(196) // T.int64(14))
                    v_ax3 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(14))
                    T.reads(T_divide[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_2[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T_divide[v_ax0, v_ax1, v_ax2, v_ax3] * T_reshape_2[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                with T.block("T_reshape_3"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(256), ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(C[(v_ax1 + v_ax2 + v_ax3) % T.int64(256)])
                    T.writes(T_reshape_3[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_3[v_ax0, v_ax1, v_ax2, v_ax3] = C[(v_ax1 + v_ax2 + v_ax3) % T.int64(256)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(49), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_1"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(256), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(196))
                    v_ax2 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(196) // T.int64(14))
                    v_ax3 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(14))
                    T.reads(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_3[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_add[v_ax0, v_ax1, v_ax2, v_ax3] = T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] + T_reshape_3[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                with T.block("T_multiply_1"):
                    v_ax0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(256) + ax0_fused_1)
                    T.reads(D[v_ax0])
                    T.writes(T_multiply_1[v_ax0])
                    T_multiply_1[v_ax0] = T.float32(0.90000000000000002) * D[v_ax0]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for k0, k2, k3 in T.grid(T.int64(1), T.int64(14), T.int64(14)):
                    with T.block("lv44_red"):
                        v_ax0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(256) + ax0_fused_1)
                        v_k0 = T.axis.reduce(T.int64(1), T.int64(0))
                        v_k2, v_k3 = T.axis.remap("RR", [k2, k3])
                        T.reads(lv44[v_k0, v_ax0, v_k2, v_k3])
                        T.writes(lv44_red[v_ax0])
                        with T.init():
                            lv44_red[v_ax0] = T.float32(0.0)
                        lv44_red[v_ax0] = lv44_red[v_ax0] + lv44[v_k0, v_ax0, v_k2, v_k3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                with T.block("T_divide_1"):
                    v_ax0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(256) + ax0_fused_1)
                    T.reads(lv44_red[v_ax0])
                    T.writes(T_divide_1[v_ax0])
                    T_divide_1[v_ax0] = lv44_red[v_ax0] * T.float32(0.0051020408163265302)
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                with T.block("T_multiply_2"):
                    v_ax0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(256) + ax0_fused_1)
                    T.reads(T_divide_1[v_ax0])
                    T.writes(T_multiply_2[v_ax0])
                    T_multiply_2[v_ax0] = T.float32(0.10000000000000001) * T_divide_1[v_ax0]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                with T.block("T_add_2"):
                    v_ax0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(256) + ax0_fused_1)
                    T.reads(T_multiply_1[v_ax0], T_multiply_2[v_ax0])
                    T.writes(T_add_1[v_ax0])
                    T_add_1[v_ax0] = T_multiply_1[v_ax0] + T_multiply_2[v_ax0]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                with T.block("T_multiply_3"):
                    v_ax0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(256) + ax0_fused_1)
                    T.reads(E[v_ax0])
                    T.writes(T_multiply_3[v_ax0])
                    T_multiply_3[v_ax0] = T.float32(0.90000000000000002) * E[v_ax0]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                with T.block("T_reshape_4"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(256), ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(T_divide_1[(v_ax1 + v_ax2 + v_ax3) % T.int64(256)])
                    T.writes(T_reshape_4[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_4[v_ax0, v_ax1, v_ax2, v_ax3] = T_divide_1[(v_ax1 + v_ax2 + v_ax3) % T.int64(256)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(49), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_subtract_1"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(256), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(196))
                    v_ax2 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(196) // T.int64(14))
                    v_ax3 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(14))
                    T.reads(lv44[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] = lv44[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(49), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_subtract_2"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(256), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(196))
                    v_ax2 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(196) // T.int64(14))
                    v_ax3 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(14))
                    T.reads(lv44[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3] = lv44[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(49), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply_4"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(256), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(196))
                    v_ax2 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(196) // T.int64(14))
                    v_ax3 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(14))
                    T.reads(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3], T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_multiply_4[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_multiply_4[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] * T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for k0, k2, k3 in T.grid(T.int64(1), T.int64(14), T.int64(14)):
                    with T.block("T_multiply_red"):
                        v_ax0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(256) + ax0_fused_1)
                        v_k0 = T.axis.reduce(T.int64(1), T.int64(0))
                        v_k2, v_k3 = T.axis.remap("RR", [k2, k3])
                        T.reads(T_multiply_4[v_k0, v_ax0, v_k2, v_k3])
                        T.writes(T_multiply_red[v_ax0])
                        with T.init():
                            T_multiply_red[v_ax0] = T.float32(0.0)
                        T_multiply_red[v_ax0] = T_multiply_red[v_ax0] + T_multiply_4[v_k0, v_ax0, v_k2, v_k3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                with T.block("T_divide_2"):
                    v_ax0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(256) + ax0_fused_1)
                    T.reads(T_multiply_red[v_ax0])
                    T.writes(T_divide_2[v_ax0])
                    T_divide_2[v_ax0] = T_multiply_red[v_ax0] * T.float32(0.0051020408163265302)
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                with T.block("T_multiply_5"):
                    v_ax0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(256) + ax0_fused_1)
                    T.reads(T_divide_2[v_ax0])
                    T.writes(T_multiply_5[v_ax0])
                    T_multiply_5[v_ax0] = T.float32(0.10000000000000001) * T_divide_2[v_ax0]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                with T.block("T_add_3"):
                    v_ax0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(256) + ax0_fused_1)
                    T.reads(T_multiply_3[v_ax0], T_multiply_5[v_ax0])
                    T.writes(T_add_2[v_ax0])
                    T_add_2[v_ax0] = T_multiply_3[v_ax0] + T_multiply_5[v_ax0]

    @T.prim_func(private=True)
    def batch_norm4(lv65: T.Buffer((T.int64(1), T.int64(512), T.int64(7), T.int64(7)), "float32"), B: T.Buffer((T.int64(512),), "float32"), C: T.Buffer((T.int64(512),), "float32"), D: T.Buffer((T.int64(512),), "float32"), E: T.Buffer((T.int64(512),), "float32"), T_add: T.Buffer((T.int64(1), T.int64(512), T.int64(7), T.int64(7)), "float32"), T_add_1: T.Buffer((T.int64(512),), "float32"), T_add_2: T.Buffer((T.int64(512),), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        T_reshape = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)))
        T_subtract = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(7), T.int64(7)))
        T_reshape_1 = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)))
        T_add_3 = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)))
        compute = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)))
        T_divide = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(7), T.int64(7)))
        T_reshape_2 = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)))
        T_multiply = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(7), T.int64(7)))
        T_reshape_3 = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)))
        T_multiply_1 = T.alloc_buffer((T.int64(512),))
        lv65_red = T.alloc_buffer((T.int64(512),))
        T_divide_1 = T.alloc_buffer((T.int64(512),))
        T_multiply_2 = T.alloc_buffer((T.int64(512),))
        T_multiply_3 = T.alloc_buffer((T.int64(512),))
        T_reshape_4 = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)))
        T_subtract_1 = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(7), T.int64(7)))
        T_subtract_2 = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(7), T.int64(7)))
        T_multiply_4 = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(7), T.int64(7)))
        T_multiply_red = T.alloc_buffer((T.int64(512),))
        T_divide_2 = T.alloc_buffer((T.int64(512),))
        T_multiply_5 = T.alloc_buffer((T.int64(512),))
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                with T.block("T_reshape"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(512), ax0_ax1_ax2_ax3_fused_0 * T.int64(512) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(D[(v_ax1 + v_ax2 + v_ax3) % T.int64(512)])
                    T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = D[(v_ax1 + v_ax2 + v_ax3) % T.int64(512)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(25), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_subtract"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(512), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(49))
                    v_ax2 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(49) // T.int64(7))
                    v_ax3 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(7))
                    T.where(ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1 < T.int64(25088))
                    T.reads(lv65[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = lv65[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                with T.block("T_reshape_1"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(512), ax0_ax1_ax2_ax3_fused_0 * T.int64(512) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(E[(v_ax1 + v_ax2 + v_ax3) % T.int64(512)])
                    T.writes(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3] = E[(v_ax1 + v_ax2 + v_ax3) % T.int64(512)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                with T.block("T_add"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(512), ax0_ax1_ax2_ax3_fused_0 * T.int64(512) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_add_3[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_add_3[v_ax0, v_ax1, v_ax2, v_ax3] = T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3] + T.float32(1.0000000000000001e-05)
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                with T.block("compute"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(512), i0_i1_i2_i3_fused_0 * T.int64(512) + i0_i1_i2_i3_fused_1)
                    v_i2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(T_add_3[v_i0, v_i1, v_i2, v_i3])
                    T.writes(compute[v_i0, v_i1, v_i2, v_i3])
                    compute[v_i0, v_i1, v_i2, v_i3] = T.sqrt(T_add_3[v_i0, v_i1, v_i2, v_i3])
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(25), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_divide"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(512), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(49))
                    v_ax2 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(49) // T.int64(7))
                    v_ax3 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(7))
                    T.where(ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1 < T.int64(25088))
                    T.reads(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3], compute[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_divide[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_divide[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] / compute[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                with T.block("T_reshape_2"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(512), ax0_ax1_ax2_ax3_fused_0 * T.int64(512) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(B[(v_ax1 + v_ax2 + v_ax3) % T.int64(512)])
                    T.writes(T_reshape_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_2[v_ax0, v_ax1, v_ax2, v_ax3] = B[(v_ax1 + v_ax2 + v_ax3) % T.int64(512)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(25), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(512), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(49))
                    v_ax2 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(49) // T.int64(7))
                    v_ax3 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(7))
                    T.where(ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1 < T.int64(25088))
                    T.reads(T_divide[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_2[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T_divide[v_ax0, v_ax1, v_ax2, v_ax3] * T_reshape_2[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                with T.block("T_reshape_3"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(512), ax0_ax1_ax2_ax3_fused_0 * T.int64(512) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(C[(v_ax1 + v_ax2 + v_ax3) % T.int64(512)])
                    T.writes(T_reshape_3[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_3[v_ax0, v_ax1, v_ax2, v_ax3] = C[(v_ax1 + v_ax2 + v_ax3) % T.int64(512)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(25), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_1"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(512), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(49))
                    v_ax2 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(49) // T.int64(7))
                    v_ax3 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(7))
                    T.where(ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1 < T.int64(25088))
                    T.reads(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_3[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_add[v_ax0, v_ax1, v_ax2, v_ax3] = T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] + T_reshape_3[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                with T.block("T_multiply_1"):
                    v_ax0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(512) + ax0_fused_1)
                    T.reads(D[v_ax0])
                    T.writes(T_multiply_1[v_ax0])
                    T_multiply_1[v_ax0] = T.float32(0.90000000000000002) * D[v_ax0]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                for k0, k2, k3 in T.grid(T.int64(1), T.int64(7), T.int64(7)):
                    with T.block("lv65_red"):
                        v_ax0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(512) + ax0_fused_1)
                        v_k0 = T.axis.reduce(T.int64(1), T.int64(0))
                        v_k2, v_k3 = T.axis.remap("RR", [k2, k3])
                        T.reads(lv65[v_k0, v_ax0, v_k2, v_k3])
                        T.writes(lv65_red[v_ax0])
                        with T.init():
                            lv65_red[v_ax0] = T.float32(0.0)
                        lv65_red[v_ax0] = lv65_red[v_ax0] + lv65[v_k0, v_ax0, v_k2, v_k3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                with T.block("T_divide_1"):
                    v_ax0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(512) + ax0_fused_1)
                    T.reads(lv65_red[v_ax0])
                    T.writes(T_divide_1[v_ax0])
                    T_divide_1[v_ax0] = lv65_red[v_ax0] * T.float32(0.020408163265306121)
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                with T.block("T_multiply_2"):
                    v_ax0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(512) + ax0_fused_1)
                    T.reads(T_divide_1[v_ax0])
                    T.writes(T_multiply_2[v_ax0])
                    T_multiply_2[v_ax0] = T.float32(0.10000000000000001) * T_divide_1[v_ax0]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                with T.block("T_add_2"):
                    v_ax0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(512) + ax0_fused_1)
                    T.reads(T_multiply_1[v_ax0], T_multiply_2[v_ax0])
                    T.writes(T_add_1[v_ax0])
                    T_add_1[v_ax0] = T_multiply_1[v_ax0] + T_multiply_2[v_ax0]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                with T.block("T_multiply_3"):
                    v_ax0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(512) + ax0_fused_1)
                    T.reads(E[v_ax0])
                    T.writes(T_multiply_3[v_ax0])
                    T_multiply_3[v_ax0] = T.float32(0.90000000000000002) * E[v_ax0]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                with T.block("T_reshape_4"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(512), ax0_ax1_ax2_ax3_fused_0 * T.int64(512) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(T_divide_1[(v_ax1 + v_ax2 + v_ax3) % T.int64(512)])
                    T.writes(T_reshape_4[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_4[v_ax0, v_ax1, v_ax2, v_ax3] = T_divide_1[(v_ax1 + v_ax2 + v_ax3) % T.int64(512)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(25), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_subtract_1"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(512), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(49))
                    v_ax2 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(49) // T.int64(7))
                    v_ax3 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(7))
                    T.where(ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1 < T.int64(25088))
                    T.reads(lv65[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] = lv65[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(25), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_subtract_2"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(512), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(49))
                    v_ax2 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(49) // T.int64(7))
                    v_ax3 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(7))
                    T.where(ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1 < T.int64(25088))
                    T.reads(lv65[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3] = lv65[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(25), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply_4"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(512), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(49))
                    v_ax2 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(49) // T.int64(7))
                    v_ax3 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(7))
                    T.where(ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1 < T.int64(25088))
                    T.reads(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3], T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_multiply_4[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_multiply_4[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] * T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                for k0, k2, k3 in T.grid(T.int64(1), T.int64(7), T.int64(7)):
                    with T.block("T_multiply_red"):
                        v_ax0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(512) + ax0_fused_1)
                        v_k0 = T.axis.reduce(T.int64(1), T.int64(0))
                        v_k2, v_k3 = T.axis.remap("RR", [k2, k3])
                        T.reads(T_multiply_4[v_k0, v_ax0, v_k2, v_k3])
                        T.writes(T_multiply_red[v_ax0])
                        with T.init():
                            T_multiply_red[v_ax0] = T.float32(0.0)
                        T_multiply_red[v_ax0] = T_multiply_red[v_ax0] + T_multiply_4[v_k0, v_ax0, v_k2, v_k3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                with T.block("T_divide_2"):
                    v_ax0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(512) + ax0_fused_1)
                    T.reads(T_multiply_red[v_ax0])
                    T.writes(T_divide_2[v_ax0])
                    T_divide_2[v_ax0] = T_multiply_red[v_ax0] * T.float32(0.020408163265306121)
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                with T.block("T_multiply_5"):
                    v_ax0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(512) + ax0_fused_1)
                    T.reads(T_divide_2[v_ax0])
                    T.writes(T_multiply_5[v_ax0])
                    T_multiply_5[v_ax0] = T.float32(0.10000000000000001) * T_divide_2[v_ax0]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                with T.block("T_add_3"):
                    v_ax0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(512) + ax0_fused_1)
                    T.reads(T_multiply_3[v_ax0], T_multiply_5[v_ax0])
                    T.writes(T_add_2[v_ax0])
                    T_add_2[v_ax0] = T_multiply_3[v_ax0] + T_multiply_5[v_ax0]

    @T.prim_func(private=True)
    def conv2d(inp_0: T.Buffer((T.int64(1), T.int64(3), T.int64(224), T.int64(224)), "float32"), B: T.Buffer((T.int64(64), T.int64(3), T.int64(7), T.int64(7)), "float32"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(64), T.int64(112), T.int64(112)), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(1), T.int64(3), T.int64(230), T.int64(230)))
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(155), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("pad_temp"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(3), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) // T.int64(52900))
                    v_i2 = T.axis.spatial(T.int64(230), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(52900) // T.int64(230))
                    v_i3 = T.axis.spatial(T.int64(230), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(230))
                    T.where(i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1 < T.int64(158700))
                    T.reads(inp_0[v_i0, v_i1, v_i2 - T.int64(3), v_i3 - T.int64(3)])
                    T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                    pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(3) <= v_i2 and v_i2 < T.int64(227) and T.int64(3) <= v_i3 and v_i3 < T.int64(227), inp_0[v_i0, v_i1, v_i2 - T.int64(3), v_i3 - T.int64(3)], T.float32(0.0))
        for nn_ff_yy_xx_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for nn_ff_yy_xx_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for nn_ff_yy_xx_fused_0, rc, ry, rx in T.grid(T.int64(4), T.int64(3), T.int64(7), T.int64(7)):
                    with T.block("conv2d_nchw"):
                        v_nn = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ff = T.axis.spatial(T.int64(64), (nn_ff_yy_xx_fused_0 * T.int64(262144) + nn_ff_yy_xx_fused_1 * T.int64(1024) + nn_ff_yy_xx_fused_2) // T.int64(12544))
                        v_yy = T.axis.spatial(T.int64(112), (nn_ff_yy_xx_fused_0 * T.int64(262144) + nn_ff_yy_xx_fused_1 * T.int64(1024) + nn_ff_yy_xx_fused_2) % T.int64(12544) // T.int64(112))
                        v_xx = T.axis.spatial(T.int64(112), (nn_ff_yy_xx_fused_0 * T.int64(262144) + nn_ff_yy_xx_fused_1 * T.int64(1024) + nn_ff_yy_xx_fused_2) % T.int64(112))
                        v_rc, v_ry, v_rx = T.axis.remap("RRR", [rc, ry, rx])
                        T.where((nn_ff_yy_xx_fused_0 * T.int64(256) + nn_ff_yy_xx_fused_1) * T.int64(1024) + nn_ff_yy_xx_fused_2 < T.int64(802816))
                        T.reads(pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx], B[v_ff, v_rc, v_ry, v_rx])
                        T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                        with T.init():
                            conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0.0)
                        conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx] * B[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func(private=True)
    def conv2d1(lv4: T.Buffer((T.int64(1), T.int64(64), T.int64(56), T.int64(56)), "float32"), B: T.Buffer((T.int64(64), T.int64(64), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(64), T.int64(56), T.int64(56)), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(58), T.int64(58)))
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(211), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("pad_temp"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(64), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) // T.int64(3364))
                    v_i2 = T.axis.spatial(T.int64(58), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(3364) // T.int64(58))
                    v_i3 = T.axis.spatial(T.int64(58), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(58))
                    T.where(i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1 < T.int64(215296))
                    T.reads(lv4[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                    T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                    pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(57) and T.int64(1) <= v_i3 and v_i3 < T.int64(57), lv4[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0.0))
        for nn_ff_yy_xx_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
            for nn_ff_yy_xx_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for rc, ry, rx in T.grid(T.int64(64), T.int64(3), T.int64(3)):
                    with T.block("conv2d_nchw"):
                        v_nn = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ff = T.axis.spatial(T.int64(64), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) // T.int64(3136))
                        v_yy = T.axis.spatial(T.int64(56), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(3136) // T.int64(56))
                        v_xx = T.axis.spatial(T.int64(56), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(56))
                        v_rc, v_ry, v_rx = T.axis.remap("RRR", [rc, ry, rx])
                        T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], B[v_ff, v_rc, v_ry, v_rx])
                        T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                        with T.init():
                            conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0.0)
                        conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * B[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func(private=True)
    def conv2d10(lv64: T.Buffer((T.int64(1), T.int64(256), T.int64(14), T.int64(14)), "float32"), B: T.Buffer((T.int64(512), T.int64(256), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(512), T.int64(7), T.int64(7)), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(14), T.int64(14)))
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(49), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("pad_temp"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(256), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) // T.int64(196))
                    v_i2 = T.axis.spatial(T.int64(14), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(196) // T.int64(14))
                    v_i3 = T.axis.spatial(T.int64(14), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(14))
                    T.reads(lv64[v_i0, v_i1, v_i2, v_i3])
                    T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                    pad_temp[v_i0, v_i1, v_i2, v_i3] = lv64[v_i0, v_i1, v_i2, v_i3]
        for nn_ff_yy_xx_fused_0 in T.thread_binding(T.int64(25), thread="blockIdx.x"):
            for nn_ff_yy_xx_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for rc, ry, rx in T.grid(T.int64(256), T.int64(1), T.int64(1)):
                    with T.block("conv2d_nchw"):
                        v_nn = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ff = T.axis.spatial(T.int64(512), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) // T.int64(49))
                        v_yy = T.axis.spatial(T.int64(7), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(49) // T.int64(7))
                        v_xx = T.axis.spatial(T.int64(7), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(7))
                        v_rc = T.axis.reduce(T.int64(256), rc)
                        v_ry = T.axis.reduce(T.int64(1), T.int64(0))
                        v_rx = T.axis.reduce(T.int64(1), T.int64(0))
                        T.where(nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1 < T.int64(25088))
                        T.reads(pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx], B[v_ff, v_rc, v_ry, v_rx])
                        T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                        with T.init():
                            conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0.0)
                        conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx] * B[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func(private=True)
    def conv2d2(lv22: T.Buffer((T.int64(1), T.int64(64), T.int64(56), T.int64(56)), "float32"), B: T.Buffer((T.int64(128), T.int64(64), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(128), T.int64(28), T.int64(28)), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(58), T.int64(58)))
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(211), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("pad_temp"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(64), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) // T.int64(3364))
                    v_i2 = T.axis.spatial(T.int64(58), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(3364) // T.int64(58))
                    v_i3 = T.axis.spatial(T.int64(58), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(58))
                    T.where(i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1 < T.int64(215296))
                    T.reads(lv22[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                    T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                    pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(57) and T.int64(1) <= v_i3 and v_i3 < T.int64(57), lv22[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0.0))
        for nn_ff_yy_xx_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for nn_ff_yy_xx_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for rc, ry, rx in T.grid(T.int64(64), T.int64(3), T.int64(3)):
                    with T.block("conv2d_nchw"):
                        v_nn = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ff = T.axis.spatial(T.int64(128), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) // T.int64(784))
                        v_yy = T.axis.spatial(T.int64(28), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(784) // T.int64(28))
                        v_xx = T.axis.spatial(T.int64(28), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(28))
                        v_rc, v_ry, v_rx = T.axis.remap("RRR", [rc, ry, rx])
                        T.reads(pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx], B[v_ff, v_rc, v_ry, v_rx])
                        T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                        with T.init():
                            conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0.0)
                        conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx] * B[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func(private=True)
    def conv2d3(lv26: T.Buffer((T.int64(1), T.int64(128), T.int64(28), T.int64(28)), "float32"), B: T.Buffer((T.int64(128), T.int64(128), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(128), T.int64(28), T.int64(28)), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(30), T.int64(30)))
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(113), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("pad_temp"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(128), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) // T.int64(900))
                    v_i2 = T.axis.spatial(T.int64(30), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(900) // T.int64(30))
                    v_i3 = T.axis.spatial(T.int64(30), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(30))
                    T.where(i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1 < T.int64(115200))
                    T.reads(lv26[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                    T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                    pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(29) and T.int64(1) <= v_i3 and v_i3 < T.int64(29), lv26[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0.0))
        for nn_ff_yy_xx_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for nn_ff_yy_xx_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for rc, ry, rx in T.grid(T.int64(128), T.int64(3), T.int64(3)):
                    with T.block("conv2d_nchw"):
                        v_nn = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ff = T.axis.spatial(T.int64(128), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) // T.int64(784))
                        v_yy = T.axis.spatial(T.int64(28), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(784) // T.int64(28))
                        v_xx = T.axis.spatial(T.int64(28), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(28))
                        v_rc, v_ry, v_rx = T.axis.remap("RRR", [rc, ry, rx])
                        T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], B[v_ff, v_rc, v_ry, v_rx])
                        T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                        with T.init():
                            conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0.0)
                        conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * B[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func(private=True)
    def conv2d4(lv22: T.Buffer((T.int64(1), T.int64(64), T.int64(56), T.int64(56)), "float32"), B: T.Buffer((T.int64(128), T.int64(64), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(128), T.int64(28), T.int64(28)), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(56), T.int64(56)))
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("pad_temp"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(64), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) // T.int64(3136))
                    v_i2 = T.axis.spatial(T.int64(56), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(3136) // T.int64(56))
                    v_i3 = T.axis.spatial(T.int64(56), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(56))
                    T.reads(lv22[v_i0, v_i1, v_i2, v_i3])
                    T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                    pad_temp[v_i0, v_i1, v_i2, v_i3] = lv22[v_i0, v_i1, v_i2, v_i3]
        for nn_ff_yy_xx_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for nn_ff_yy_xx_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for rc, ry, rx in T.grid(T.int64(64), T.int64(1), T.int64(1)):
                    with T.block("conv2d_nchw"):
                        v_nn = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ff = T.axis.spatial(T.int64(128), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) // T.int64(784))
                        v_yy = T.axis.spatial(T.int64(28), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(784) // T.int64(28))
                        v_xx = T.axis.spatial(T.int64(28), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(28))
                        v_rc = T.axis.reduce(T.int64(64), rc)
                        v_ry = T.axis.reduce(T.int64(1), T.int64(0))
                        v_rx = T.axis.reduce(T.int64(1), T.int64(0))
                        T.reads(pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx], B[v_ff, v_rc, v_ry, v_rx])
                        T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                        with T.init():
                            conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0.0)
                        conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx] * B[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func(private=True)
    def conv2d5(lv43: T.Buffer((T.int64(1), T.int64(128), T.int64(28), T.int64(28)), "float32"), B: T.Buffer((T.int64(256), T.int64(128), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(256), T.int64(14), T.int64(14)), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(30), T.int64(30)))
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(113), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("pad_temp"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(128), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) // T.int64(900))
                    v_i2 = T.axis.spatial(T.int64(30), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(900) // T.int64(30))
                    v_i3 = T.axis.spatial(T.int64(30), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(30))
                    T.where(i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1 < T.int64(115200))
                    T.reads(lv43[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                    T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                    pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(29) and T.int64(1) <= v_i3 and v_i3 < T.int64(29), lv43[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0.0))
        for nn_ff_yy_xx_fused_0 in T.thread_binding(T.int64(49), thread="blockIdx.x"):
            for nn_ff_yy_xx_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for rc, ry, rx in T.grid(T.int64(128), T.int64(3), T.int64(3)):
                    with T.block("conv2d_nchw"):
                        v_nn = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ff = T.axis.spatial(T.int64(256), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) // T.int64(196))
                        v_yy = T.axis.spatial(T.int64(14), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(196) // T.int64(14))
                        v_xx = T.axis.spatial(T.int64(14), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(14))
                        v_rc, v_ry, v_rx = T.axis.remap("RRR", [rc, ry, rx])
                        T.reads(pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx], B[v_ff, v_rc, v_ry, v_rx])
                        T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                        with T.init():
                            conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0.0)
                        conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx] * B[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func(private=True)
    def conv2d6(lv47: T.Buffer((T.int64(1), T.int64(256), T.int64(14), T.int64(14)), "float32"), B: T.Buffer((T.int64(256), T.int64(256), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(256), T.int64(14), T.int64(14)), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(16), T.int64(16)))
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(64), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("pad_temp"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(256), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) // T.int64(256))
                    v_i2 = T.axis.spatial(T.int64(16), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(256) // T.int64(16))
                    v_i3 = T.axis.spatial(T.int64(16), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(16))
                    T.reads(lv47[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                    T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                    pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(15) and T.int64(1) <= v_i3 and v_i3 < T.int64(15), lv47[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0.0))
        for nn_ff_yy_xx_fused_0 in T.thread_binding(T.int64(49), thread="blockIdx.x"):
            for nn_ff_yy_xx_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for rc, ry, rx in T.grid(T.int64(256), T.int64(3), T.int64(3)):
                    with T.block("conv2d_nchw"):
                        v_nn = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ff = T.axis.spatial(T.int64(256), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) // T.int64(196))
                        v_yy = T.axis.spatial(T.int64(14), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(196) // T.int64(14))
                        v_xx = T.axis.spatial(T.int64(14), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(14))
                        v_rc, v_ry, v_rx = T.axis.remap("RRR", [rc, ry, rx])
                        T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], B[v_ff, v_rc, v_ry, v_rx])
                        T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                        with T.init():
                            conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0.0)
                        conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * B[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func(private=True)
    def conv2d7(lv43: T.Buffer((T.int64(1), T.int64(128), T.int64(28), T.int64(28)), "float32"), B: T.Buffer((T.int64(256), T.int64(128), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(256), T.int64(14), T.int64(14)), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(28), T.int64(28)))
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("pad_temp"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(128), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) // T.int64(784))
                    v_i2 = T.axis.spatial(T.int64(28), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(784) // T.int64(28))
                    v_i3 = T.axis.spatial(T.int64(28), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(28))
                    T.reads(lv43[v_i0, v_i1, v_i2, v_i3])
                    T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                    pad_temp[v_i0, v_i1, v_i2, v_i3] = lv43[v_i0, v_i1, v_i2, v_i3]
        for nn_ff_yy_xx_fused_0 in T.thread_binding(T.int64(49), thread="blockIdx.x"):
            for nn_ff_yy_xx_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for rc, ry, rx in T.grid(T.int64(128), T.int64(1), T.int64(1)):
                    with T.block("conv2d_nchw"):
                        v_nn = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ff = T.axis.spatial(T.int64(256), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) // T.int64(196))
                        v_yy = T.axis.spatial(T.int64(14), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(196) // T.int64(14))
                        v_xx = T.axis.spatial(T.int64(14), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(14))
                        v_rc = T.axis.reduce(T.int64(128), rc)
                        v_ry = T.axis.reduce(T.int64(1), T.int64(0))
                        v_rx = T.axis.reduce(T.int64(1), T.int64(0))
                        T.reads(pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx], B[v_ff, v_rc, v_ry, v_rx])
                        T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                        with T.init():
                            conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0.0)
                        conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx] * B[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func(private=True)
    def conv2d8(lv64: T.Buffer((T.int64(1), T.int64(256), T.int64(14), T.int64(14)), "float32"), B: T.Buffer((T.int64(512), T.int64(256), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(512), T.int64(7), T.int64(7)), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(16), T.int64(16)))
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(64), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("pad_temp"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(256), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) // T.int64(256))
                    v_i2 = T.axis.spatial(T.int64(16), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(256) // T.int64(16))
                    v_i3 = T.axis.spatial(T.int64(16), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(16))
                    T.reads(lv64[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                    T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                    pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(15) and T.int64(1) <= v_i3 and v_i3 < T.int64(15), lv64[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0.0))
        for nn_ff_yy_xx_fused_0 in T.thread_binding(T.int64(25), thread="blockIdx.x"):
            for nn_ff_yy_xx_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for rc, ry, rx in T.grid(T.int64(256), T.int64(3), T.int64(3)):
                    with T.block("conv2d_nchw"):
                        v_nn = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ff = T.axis.spatial(T.int64(512), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) // T.int64(49))
                        v_yy = T.axis.spatial(T.int64(7), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(49) // T.int64(7))
                        v_xx = T.axis.spatial(T.int64(7), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(7))
                        v_rc, v_ry, v_rx = T.axis.remap("RRR", [rc, ry, rx])
                        T.where(nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1 < T.int64(25088))
                        T.reads(pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx], B[v_ff, v_rc, v_ry, v_rx])
                        T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                        with T.init():
                            conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0.0)
                        conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx] * B[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func(private=True)
    def conv2d9(lv68: T.Buffer((T.int64(1), T.int64(512), T.int64(7), T.int64(7)), "float32"), B: T.Buffer((T.int64(512), T.int64(512), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(512), T.int64(7), T.int64(7)), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(9), T.int64(9)))
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(41), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("pad_temp"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(512), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) // T.int64(81))
                    v_i2 = T.axis.spatial(T.int64(9), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(81) // T.int64(9))
                    v_i3 = T.axis.spatial(T.int64(9), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(9))
                    T.where(i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1 < T.int64(41472))
                    T.reads(lv68[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                    T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                    pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(8) and T.int64(1) <= v_i3 and v_i3 < T.int64(8), lv68[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0.0))
        for nn_ff_yy_xx_fused_0 in T.thread_binding(T.int64(25), thread="blockIdx.x"):
            for nn_ff_yy_xx_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for rc, ry, rx in T.grid(T.int64(512), T.int64(3), T.int64(3)):
                    with T.block("conv2d_nchw"):
                        v_nn = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ff = T.axis.spatial(T.int64(512), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) // T.int64(49))
                        v_yy = T.axis.spatial(T.int64(7), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(49) // T.int64(7))
                        v_xx = T.axis.spatial(T.int64(7), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(7))
                        v_rc, v_ry, v_rx = T.axis.remap("RRR", [rc, ry, rx])
                        T.where(nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1 < T.int64(25088))
                        T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], B[v_ff, v_rc, v_ry, v_rx])
                        T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                        with T.init():
                            conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0.0)
                        conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * B[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func(private=True)
    def fused_add1_relu2(lv28_0: T.Buffer((T.int64(1), T.int64(128), T.int64(28), T.int64(28)), "float32"), lv31_0: T.Buffer((T.int64(1), T.int64(128), T.int64(28), T.int64(28)), "float32"), compute_intermediate: T.Buffer((T.int64(1), T.int64(128), T.int64(28), T.int64(28)), "float32")):
        T.func_attr({"target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(28), T.int64(28)))
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(128), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(784))
                    v_ax2 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(784) // T.int64(28))
                    v_ax3 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(28))
                    T.reads(lv28_0[v_ax0, v_ax1, v_ax2, v_ax3], lv31_0[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv28_0[v_ax0, v_ax1, v_ax2, v_ax3] + lv31_0[v_ax0, v_ax1, v_ax2, v_ax3]
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(128), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) // T.int64(784))
                    v_i2 = T.axis.spatial(T.int64(28), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(784) // T.int64(28))
                    v_i3 = T.axis.spatial(T.int64(28), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(28))
                    T.reads(T_add_intermediate[v_i0, v_i1, v_i2, v_i3])
                    T.writes(compute_intermediate[v_i0, v_i1, v_i2, v_i3])
                    compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.max(T_add_intermediate[v_i0, v_i1, v_i2, v_i3], T.float32(0.0))

    @T.prim_func(private=True)
    def fused_add2_relu3(lv49_0: T.Buffer((T.int64(1), T.int64(256), T.int64(14), T.int64(14)), "float32"), lv52_0: T.Buffer((T.int64(1), T.int64(256), T.int64(14), T.int64(14)), "float32"), compute_intermediate: T.Buffer((T.int64(1), T.int64(256), T.int64(14), T.int64(14)), "float32")):
        T.func_attr({"target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(14), T.int64(14)))
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(49), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(256), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(196))
                    v_ax2 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(196) // T.int64(14))
                    v_ax3 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(14))
                    T.reads(lv49_0[v_ax0, v_ax1, v_ax2, v_ax3], lv52_0[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv49_0[v_ax0, v_ax1, v_ax2, v_ax3] + lv52_0[v_ax0, v_ax1, v_ax2, v_ax3]
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(49), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(256), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) // T.int64(196))
                    v_i2 = T.axis.spatial(T.int64(14), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(196) // T.int64(14))
                    v_i3 = T.axis.spatial(T.int64(14), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(14))
                    T.reads(T_add_intermediate[v_i0, v_i1, v_i2, v_i3])
                    T.writes(compute_intermediate[v_i0, v_i1, v_i2, v_i3])
                    compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.max(T_add_intermediate[v_i0, v_i1, v_i2, v_i3], T.float32(0.0))

    @T.prim_func(private=True)
    def fused_add3_relu4(lv70_0: T.Buffer((T.int64(1), T.int64(512), T.int64(7), T.int64(7)), "float32"), lv73_0: T.Buffer((T.int64(1), T.int64(512), T.int64(7), T.int64(7)), "float32"), compute_intermediate: T.Buffer((T.int64(1), T.int64(512), T.int64(7), T.int64(7)), "float32")):
        T.func_attr({"target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(7), T.int64(7)))
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(25), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(512), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(49))
                    v_ax2 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(49) // T.int64(7))
                    v_ax3 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(7))
                    T.where(ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1 < T.int64(25088))
                    T.reads(lv70_0[v_ax0, v_ax1, v_ax2, v_ax3], lv73_0[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv70_0[v_ax0, v_ax1, v_ax2, v_ax3] + lv73_0[v_ax0, v_ax1, v_ax2, v_ax3]
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(25), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(512), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) // T.int64(49))
                    v_i2 = T.axis.spatial(T.int64(7), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(49) // T.int64(7))
                    v_i3 = T.axis.spatial(T.int64(7), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(7))
                    T.where(i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1 < T.int64(25088))
                    T.reads(T_add_intermediate[v_i0, v_i1, v_i2, v_i3])
                    T.writes(compute_intermediate[v_i0, v_i1, v_i2, v_i3])
                    compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.max(T_add_intermediate[v_i0, v_i1, v_i2, v_i3], T.float32(0.0))

    @T.prim_func(private=True)
    def fused_add_relu1(lv10_0: T.Buffer((T.int64(1), T.int64(64), T.int64(56), T.int64(56)), "float32"), lv4: T.Buffer((T.int64(1), T.int64(64), T.int64(56), T.int64(56)), "float32"), compute_intermediate: T.Buffer((T.int64(1), T.int64(64), T.int64(56), T.int64(56)), "float32")):
        T.func_attr({"target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(56), T.int64(56)))
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(3136))
                    v_ax2 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(3136) // T.int64(56))
                    v_ax3 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(56))
                    T.reads(lv10_0[v_ax0, v_ax1, v_ax2, v_ax3], lv4[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv10_0[v_ax0, v_ax1, v_ax2, v_ax3] + lv4[v_ax0, v_ax1, v_ax2, v_ax3]
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(64), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) // T.int64(3136))
                    v_i2 = T.axis.spatial(T.int64(56), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(3136) // T.int64(56))
                    v_i3 = T.axis.spatial(T.int64(56), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(56))
                    T.reads(T_add_intermediate[v_i0, v_i1, v_i2, v_i3])
                    T.writes(compute_intermediate[v_i0, v_i1, v_i2, v_i3])
                    compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.max(T_add_intermediate[v_i0, v_i1, v_i2, v_i3], T.float32(0.0))

    @T.prim_func(private=True)
    def fused_matmul_add4(lv87: T.Buffer((T.int64(1), T.int64(512)), "float32"), lv88: T.Buffer((T.int64(512), T.int64(1000)), "float32"), param_0: T.Buffer((T.int64(1000),), "float32"), T_add_intermediate: T.Buffer((T.int64(1), T.int64(1000)), "float32")):
        T.func_attr({"target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1000)))
        for i0_i1_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for i0_i1_fused_1 in T.thread_binding(T.int64(1000), thread="threadIdx.x"):
                for k in range(T.int64(512)):
                    with T.block("matmul"):
                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i1 = T.axis.spatial(T.int64(1000), i0_i1_fused_0 * T.int64(1000) + i0_i1_fused_1)
                        v_k = T.axis.reduce(T.int64(512), k)
                        T.reads(lv87[v_i0, v_k], lv88[v_k, v_i1])
                        T.writes(matmul_intermediate[v_i0, v_i1])
                        with T.init():
                            matmul_intermediate[v_i0, v_i1] = T.float32(0.0)
                        matmul_intermediate[v_i0, v_i1] = matmul_intermediate[v_i0, v_i1] + lv87[v_i0, v_k] * lv88[v_k, v_i1]
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1000), thread="threadIdx.x"):
                with T.block("T_add"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(1000), ax0_ax1_fused_0 * T.int64(1000) + ax0_ax1_fused_1)
                    T.reads(matmul_intermediate[v_ax0, v_ax1], param_0[v_ax1])
                    T.writes(T_add_intermediate[v_ax0, v_ax1])
                    T_add_intermediate[v_ax0, v_ax1] = matmul_intermediate[v_ax0, v_ax1] + param_0[v_ax1]

    @T.prim_func(private=True)
    def fused_relu(lv1_0: T.Buffer((T.int64(1), T.int64(64), T.int64(112), T.int64(112)), "float32"), compute_intermediate: T.Buffer((T.int64(1), T.int64(64), T.int64(112), T.int64(112)), "float32")):
        T.func_attr({"target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for i0_i1_i2_i3_fused_0 in range(T.int64(4)):
                    with T.block("compute"):
                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i1 = T.axis.spatial(T.int64(64), (i0_i1_i2_i3_fused_0 * T.int64(262144) + i0_i1_i2_i3_fused_1 * T.int64(1024) + i0_i1_i2_i3_fused_2) // T.int64(12544))
                        v_i2 = T.axis.spatial(T.int64(112), (i0_i1_i2_i3_fused_0 * T.int64(262144) + i0_i1_i2_i3_fused_1 * T.int64(1024) + i0_i1_i2_i3_fused_2) % T.int64(12544) // T.int64(112))
                        v_i3 = T.axis.spatial(T.int64(112), (i0_i1_i2_i3_fused_0 * T.int64(262144) + i0_i1_i2_i3_fused_1 * T.int64(1024) + i0_i1_i2_i3_fused_2) % T.int64(112))
                        T.where((i0_i1_i2_i3_fused_0 * T.int64(256) + i0_i1_i2_i3_fused_1) * T.int64(1024) + i0_i1_i2_i3_fused_2 < T.int64(802816))
                        T.reads(lv1_0[v_i0, v_i1, v_i2, v_i3])
                        T.writes(compute_intermediate[v_i0, v_i1, v_i2, v_i3])
                        compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.max(lv1_0[v_i0, v_i1, v_i2, v_i3], T.float32(0.0))

    @T.prim_func(private=True)
    def fused_relu1(lv6_0: T.Buffer((T.int64(1), T.int64(64), T.int64(56), T.int64(56)), "float32"), compute_intermediate: T.Buffer((T.int64(1), T.int64(64), T.int64(56), T.int64(56)), "float32")):
        T.func_attr({"target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(64), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) // T.int64(3136))
                    v_i2 = T.axis.spatial(T.int64(56), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(3136) // T.int64(56))
                    v_i3 = T.axis.spatial(T.int64(56), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(56))
                    T.reads(lv6_0[v_i0, v_i1, v_i2, v_i3])
                    T.writes(compute_intermediate[v_i0, v_i1, v_i2, v_i3])
                    compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.max(lv6_0[v_i0, v_i1, v_i2, v_i3], T.float32(0.0))

    @T.prim_func(private=True)
    def fused_relu2(lv24_0: T.Buffer((T.int64(1), T.int64(128), T.int64(28), T.int64(28)), "float32"), compute_intermediate: T.Buffer((T.int64(1), T.int64(128), T.int64(28), T.int64(28)), "float32")):
        T.func_attr({"target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(128), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) // T.int64(784))
                    v_i2 = T.axis.spatial(T.int64(28), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(784) // T.int64(28))
                    v_i3 = T.axis.spatial(T.int64(28), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(28))
                    T.reads(lv24_0[v_i0, v_i1, v_i2, v_i3])
                    T.writes(compute_intermediate[v_i0, v_i1, v_i2, v_i3])
                    compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.max(lv24_0[v_i0, v_i1, v_i2, v_i3], T.float32(0.0))

    @T.prim_func(private=True)
    def fused_relu3(lv45_0: T.Buffer((T.int64(1), T.int64(256), T.int64(14), T.int64(14)), "float32"), compute_intermediate: T.Buffer((T.int64(1), T.int64(256), T.int64(14), T.int64(14)), "float32")):
        T.func_attr({"target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(49), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(256), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) // T.int64(196))
                    v_i2 = T.axis.spatial(T.int64(14), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(196) // T.int64(14))
                    v_i3 = T.axis.spatial(T.int64(14), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(14))
                    T.reads(lv45_0[v_i0, v_i1, v_i2, v_i3])
                    T.writes(compute_intermediate[v_i0, v_i1, v_i2, v_i3])
                    compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.max(lv45_0[v_i0, v_i1, v_i2, v_i3], T.float32(0.0))

    @T.prim_func(private=True)
    def fused_relu4(lv66_0: T.Buffer((T.int64(1), T.int64(512), T.int64(7), T.int64(7)), "float32"), compute_intermediate: T.Buffer((T.int64(1), T.int64(512), T.int64(7), T.int64(7)), "float32")):
        T.func_attr({"target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(25), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(512), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) // T.int64(49))
                    v_i2 = T.axis.spatial(T.int64(7), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(49) // T.int64(7))
                    v_i3 = T.axis.spatial(T.int64(7), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(7))
                    T.where(i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1 < T.int64(25088))
                    T.reads(lv66_0[v_i0, v_i1, v_i2, v_i3])
                    T.writes(compute_intermediate[v_i0, v_i1, v_i2, v_i3])
                    compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.max(lv66_0[v_i0, v_i1, v_i2, v_i3], T.float32(0.0))

    @T.prim_func(private=True)
    def max_pool2d(lv3: T.Buffer((T.int64(1), T.int64(64), T.int64(112), T.int64(112)), "float32"), pool_max: T.Buffer((T.int64(1), T.int64(64), T.int64(56), T.int64(56)), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(114), T.int64(114)))
        for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(4)):
                    with T.block("pad_temp"):
                        v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) // T.int64(12996))
                        v_ax2 = T.axis.spatial(T.int64(114), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(12996) // T.int64(114))
                        v_ax3 = T.axis.spatial(T.int64(114), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(114))
                        T.where((ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1) * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2 < T.int64(831744))
                        T.reads(lv3[v_ax0, v_ax1, v_ax2 - T.int64(1), v_ax3 - T.int64(1)])
                        T.writes(pad_temp[v_ax0, v_ax1, v_ax2, v_ax3])
                        pad_temp[v_ax0, v_ax1, v_ax2, v_ax3] = T.if_then_else(T.int64(1) <= v_ax2 and v_ax2 < T.int64(113) and T.int64(1) <= v_ax3 and v_ax3 < T.int64(113), lv3[v_ax0, v_ax1, v_ax2 - T.int64(1), v_ax3 - T.int64(1)], T.float32(-340282346638528859811704183484516925440.0))
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for rv0, rv1 in T.grid(T.int64(3), T.int64(3)):
                    with T.block("pool_max"):
                        v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(3136))
                        v_ax2 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(3136) // T.int64(56))
                        v_ax3 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(56))
                        v_rv0, v_rv1 = T.axis.remap("RR", [rv0, rv1])
                        T.reads(pad_temp[v_ax0, v_ax1, v_ax2 * T.int64(2) + v_rv0, v_ax3 * T.int64(2) + v_rv1])
                        T.writes(pool_max[v_ax0, v_ax1, v_ax2, v_ax3])
                        T.block_attr({"schedule_rule": "meta_schedule.pool_max"})
                        with T.init():
                            pool_max[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(-340282346638528859811704183484516925440.0)
                        pool_max[v_ax0, v_ax1, v_ax2, v_ax3] = T.max(pool_max[v_ax0, v_ax1, v_ax2, v_ax3], pad_temp[v_ax0, v_ax1, v_ax2 * T.int64(2) + v_rv0, v_ax3 * T.int64(2) + v_rv1])

    @T.prim_func(private=True)
    def reshape(lv86: T.Buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)), "float32"), T_reshape: T.Buffer((T.int64(1), T.int64(512)), "float32")):
        T.func_attr({"op_pattern": 2, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                with T.block("T_reshape"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(512), ax0_ax1_fused_0 * T.int64(512) + ax0_ax1_fused_1)
                    T.reads(lv86[T.int64(0), v_ax1 % T.int64(512), T.int64(0), T.int64(0)])
                    T.writes(T_reshape[v_ax0, v_ax1])
                    T_reshape[v_ax0, v_ax1] = lv86[T.int64(0), v_ax1 % T.int64(512), T.int64(0), T.int64(0)]

    @T.prim_func(private=True)
    def transpose(A: T.Buffer((T.int64(1000), T.int64(512)), "float32"), T_transpose: T.Buffer((T.int64(512), T.int64(1000)), "float32")):
        T.func_attr({"op_pattern": 2, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for ax0_ax1_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for ax0_ax1_fused_0 in range(T.int64(2)):
                    with T.block("T_transpose"):
                        v_ax0 = T.axis.spatial(T.int64(512), (ax0_ax1_fused_0 * T.int64(262144) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2) // T.int64(1000))
                        v_ax1 = T.axis.spatial(T.int64(1000), (ax0_ax1_fused_0 * T.int64(262144) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2) % T.int64(1000))
                        T.where((ax0_ax1_fused_0 * T.int64(256) + ax0_ax1_fused_1) * T.int64(1024) + ax0_ax1_fused_2 < T.int64(512000))
                        T.reads(A[v_ax1, v_ax0])
                        T.writes(T_transpose[v_ax0, v_ax1])
                        T_transpose[v_ax0, v_ax1] = A[v_ax1, v_ax0]

    @R.function
    def main(inp_0: R.Tensor((1, 3, 224, 224), dtype="float32")) -> R.Tensor((1, 1000), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.conv2d, (inp_0, metadata["relax.expr.Constant"][0]), out_sinfo=R.Tensor((1, 64, 112, 112), dtype="float32"))
            lv1 = R.call_tir(cls.batch_norm, (lv, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2], metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=[R.Tensor((1, 64, 112, 112), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")])
            lv_1: R.Tensor((1, 64, 112, 112), dtype="float32") = lv1[0]
            lv1_1 = R.call_tir(cls.fused_relu, (lv_1,), out_sinfo=R.Tensor((1, 64, 112, 112), dtype="float32"))
            lv4 = R.call_tir(cls.max_pool2d, (lv1_1,), out_sinfo=R.Tensor((1, 64, 56, 56), dtype="float32"))
            lv5 = R.call_tir(cls.conv2d1, (lv4, metadata["relax.expr.Constant"][5]), out_sinfo=R.Tensor((1, 64, 56, 56), dtype="float32"))
            lv6 = R.call_tir(cls.batch_norm1, (lv5, metadata["relax.expr.Constant"][6], metadata["relax.expr.Constant"][7], metadata["relax.expr.Constant"][8], metadata["relax.expr.Constant"][9]), out_sinfo=[R.Tensor((1, 64, 56, 56), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")])
            lv2: R.Tensor((1, 64, 56, 56), dtype="float32") = lv6[0]
            lv3 = R.call_tir(cls.fused_relu1, (lv2,), out_sinfo=R.Tensor((1, 64, 56, 56), dtype="float32"))
            lv9 = R.call_tir(cls.conv2d1, (lv3, metadata["relax.expr.Constant"][10]), out_sinfo=R.Tensor((1, 64, 56, 56), dtype="float32"))
            lv10 = R.call_tir(cls.batch_norm1, (lv9, metadata["relax.expr.Constant"][11], metadata["relax.expr.Constant"][12], metadata["relax.expr.Constant"][13], metadata["relax.expr.Constant"][14]), out_sinfo=[R.Tensor((1, 64, 56, 56), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")])
            lv4_1: R.Tensor((1, 64, 56, 56), dtype="float32") = lv10[0]
            lv5_1 = R.call_tir(cls.fused_add_relu1, (lv4_1, lv4), out_sinfo=R.Tensor((1, 64, 56, 56), dtype="float32"))
            lv14 = R.call_tir(cls.conv2d1, (lv5_1, metadata["relax.expr.Constant"][15]), out_sinfo=R.Tensor((1, 64, 56, 56), dtype="float32"))
            lv15 = R.call_tir(cls.batch_norm1, (lv14, metadata["relax.expr.Constant"][16], metadata["relax.expr.Constant"][17], metadata["relax.expr.Constant"][18], metadata["relax.expr.Constant"][19]), out_sinfo=[R.Tensor((1, 64, 56, 56), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")])
            lv6_1: R.Tensor((1, 64, 56, 56), dtype="float32") = lv15[0]
            lv7 = R.call_tir(cls.fused_relu1, (lv6_1,), out_sinfo=R.Tensor((1, 64, 56, 56), dtype="float32"))
            lv18 = R.call_tir(cls.conv2d1, (lv7, metadata["relax.expr.Constant"][20]), out_sinfo=R.Tensor((1, 64, 56, 56), dtype="float32"))
            lv19 = R.call_tir(cls.batch_norm1, (lv18, metadata["relax.expr.Constant"][21], metadata["relax.expr.Constant"][22], metadata["relax.expr.Constant"][23], metadata["relax.expr.Constant"][24]), out_sinfo=[R.Tensor((1, 64, 56, 56), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")])
            lv8: R.Tensor((1, 64, 56, 56), dtype="float32") = lv19[0]
            lv9_1 = R.call_tir(cls.fused_add_relu1, (lv8, lv5_1), out_sinfo=R.Tensor((1, 64, 56, 56), dtype="float32"))
            lv23 = R.call_tir(cls.conv2d2, (lv9_1, metadata["relax.expr.Constant"][25]), out_sinfo=R.Tensor((1, 128, 28, 28), dtype="float32"))
            lv24 = R.call_tir(cls.batch_norm2, (lv23, metadata["relax.expr.Constant"][26], metadata["relax.expr.Constant"][27], metadata["relax.expr.Constant"][28], metadata["relax.expr.Constant"][29]), out_sinfo=[R.Tensor((1, 128, 28, 28), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")])
            lv10_1: R.Tensor((1, 128, 28, 28), dtype="float32") = lv24[0]
            lv11 = R.call_tir(cls.fused_relu2, (lv10_1,), out_sinfo=R.Tensor((1, 128, 28, 28), dtype="float32"))
            lv27 = R.call_tir(cls.conv2d3, (lv11, metadata["relax.expr.Constant"][30]), out_sinfo=R.Tensor((1, 128, 28, 28), dtype="float32"))
            lv28 = R.call_tir(cls.batch_norm2, (lv27, metadata["relax.expr.Constant"][31], metadata["relax.expr.Constant"][32], metadata["relax.expr.Constant"][33], metadata["relax.expr.Constant"][34]), out_sinfo=[R.Tensor((1, 128, 28, 28), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")])
            lv30 = R.call_tir(cls.conv2d4, (lv9_1, metadata["relax.expr.Constant"][35]), out_sinfo=R.Tensor((1, 128, 28, 28), dtype="float32"))
            lv31 = R.call_tir(cls.batch_norm2, (lv30, metadata["relax.expr.Constant"][36], metadata["relax.expr.Constant"][37], metadata["relax.expr.Constant"][38], metadata["relax.expr.Constant"][39]), out_sinfo=[R.Tensor((1, 128, 28, 28), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")])
            lv12: R.Tensor((1, 128, 28, 28), dtype="float32") = lv28[0]
            lv13: R.Tensor((1, 128, 28, 28), dtype="float32") = lv31[0]
            lv14_1 = R.call_tir(cls.fused_add1_relu2, (lv12, lv13), out_sinfo=R.Tensor((1, 128, 28, 28), dtype="float32"))
            lv35 = R.call_tir(cls.conv2d3, (lv14_1, metadata["relax.expr.Constant"][40]), out_sinfo=R.Tensor((1, 128, 28, 28), dtype="float32"))
            lv36 = R.call_tir(cls.batch_norm2, (lv35, metadata["relax.expr.Constant"][41], metadata["relax.expr.Constant"][42], metadata["relax.expr.Constant"][43], metadata["relax.expr.Constant"][44]), out_sinfo=[R.Tensor((1, 128, 28, 28), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")])
            lv15_1: R.Tensor((1, 128, 28, 28), dtype="float32") = lv36[0]
            lv16 = R.call_tir(cls.fused_relu2, (lv15_1,), out_sinfo=R.Tensor((1, 128, 28, 28), dtype="float32"))
            lv39 = R.call_tir(cls.conv2d3, (lv16, metadata["relax.expr.Constant"][45]), out_sinfo=R.Tensor((1, 128, 28, 28), dtype="float32"))
            lv40 = R.call_tir(cls.batch_norm2, (lv39, metadata["relax.expr.Constant"][46], metadata["relax.expr.Constant"][47], metadata["relax.expr.Constant"][48], metadata["relax.expr.Constant"][49]), out_sinfo=[R.Tensor((1, 128, 28, 28), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")])
            lv17: R.Tensor((1, 128, 28, 28), dtype="float32") = lv40[0]
            lv18_1 = R.call_tir(cls.fused_add1_relu2, (lv17, lv14_1), out_sinfo=R.Tensor((1, 128, 28, 28), dtype="float32"))
            lv44 = R.call_tir(cls.conv2d5, (lv18_1, metadata["relax.expr.Constant"][50]), out_sinfo=R.Tensor((1, 256, 14, 14), dtype="float32"))
            lv45 = R.call_tir(cls.batch_norm3, (lv44, metadata["relax.expr.Constant"][51], metadata["relax.expr.Constant"][52], metadata["relax.expr.Constant"][53], metadata["relax.expr.Constant"][54]), out_sinfo=[R.Tensor((1, 256, 14, 14), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")])
            lv19_1: R.Tensor((1, 256, 14, 14), dtype="float32") = lv45[0]
            lv20 = R.call_tir(cls.fused_relu3, (lv19_1,), out_sinfo=R.Tensor((1, 256, 14, 14), dtype="float32"))
            lv48 = R.call_tir(cls.conv2d6, (lv20, metadata["relax.expr.Constant"][55]), out_sinfo=R.Tensor((1, 256, 14, 14), dtype="float32"))
            lv49 = R.call_tir(cls.batch_norm3, (lv48, metadata["relax.expr.Constant"][56], metadata["relax.expr.Constant"][57], metadata["relax.expr.Constant"][58], metadata["relax.expr.Constant"][59]), out_sinfo=[R.Tensor((1, 256, 14, 14), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")])
            lv51 = R.call_tir(cls.conv2d7, (lv18_1, metadata["relax.expr.Constant"][60]), out_sinfo=R.Tensor((1, 256, 14, 14), dtype="float32"))
            lv52 = R.call_tir(cls.batch_norm3, (lv51, metadata["relax.expr.Constant"][61], metadata["relax.expr.Constant"][62], metadata["relax.expr.Constant"][63], metadata["relax.expr.Constant"][64]), out_sinfo=[R.Tensor((1, 256, 14, 14), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")])
            lv21: R.Tensor((1, 256, 14, 14), dtype="float32") = lv49[0]
            lv22: R.Tensor((1, 256, 14, 14), dtype="float32") = lv52[0]
            lv23_1 = R.call_tir(cls.fused_add2_relu3, (lv21, lv22), out_sinfo=R.Tensor((1, 256, 14, 14), dtype="float32"))
            lv56 = R.call_tir(cls.conv2d6, (lv23_1, metadata["relax.expr.Constant"][65]), out_sinfo=R.Tensor((1, 256, 14, 14), dtype="float32"))
            lv57 = R.call_tir(cls.batch_norm3, (lv56, metadata["relax.expr.Constant"][66], metadata["relax.expr.Constant"][67], metadata["relax.expr.Constant"][68], metadata["relax.expr.Constant"][69]), out_sinfo=[R.Tensor((1, 256, 14, 14), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")])
            lv24_1: R.Tensor((1, 256, 14, 14), dtype="float32") = lv57[0]
            lv25 = R.call_tir(cls.fused_relu3, (lv24_1,), out_sinfo=R.Tensor((1, 256, 14, 14), dtype="float32"))
            lv60 = R.call_tir(cls.conv2d6, (lv25, metadata["relax.expr.Constant"][70]), out_sinfo=R.Tensor((1, 256, 14, 14), dtype="float32"))
            lv61 = R.call_tir(cls.batch_norm3, (lv60, metadata["relax.expr.Constant"][71], metadata["relax.expr.Constant"][72], metadata["relax.expr.Constant"][73], metadata["relax.expr.Constant"][74]), out_sinfo=[R.Tensor((1, 256, 14, 14), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")])
            lv26: R.Tensor((1, 256, 14, 14), dtype="float32") = lv61[0]
            lv27_1 = R.call_tir(cls.fused_add2_relu3, (lv26, lv23_1), out_sinfo=R.Tensor((1, 256, 14, 14), dtype="float32"))
            lv65 = R.call_tir(cls.conv2d8, (lv27_1, metadata["relax.expr.Constant"][75]), out_sinfo=R.Tensor((1, 512, 7, 7), dtype="float32"))
            lv66 = R.call_tir(cls.batch_norm4, (lv65, metadata["relax.expr.Constant"][76], metadata["relax.expr.Constant"][77], metadata["relax.expr.Constant"][78], metadata["relax.expr.Constant"][79]), out_sinfo=[R.Tensor((1, 512, 7, 7), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")])
            lv28_1: R.Tensor((1, 512, 7, 7), dtype="float32") = lv66[0]
            lv29 = R.call_tir(cls.fused_relu4, (lv28_1,), out_sinfo=R.Tensor((1, 512, 7, 7), dtype="float32"))
            lv69 = R.call_tir(cls.conv2d9, (lv29, metadata["relax.expr.Constant"][80]), out_sinfo=R.Tensor((1, 512, 7, 7), dtype="float32"))
            lv70 = R.call_tir(cls.batch_norm4, (lv69, metadata["relax.expr.Constant"][81], metadata["relax.expr.Constant"][82], metadata["relax.expr.Constant"][83], metadata["relax.expr.Constant"][84]), out_sinfo=[R.Tensor((1, 512, 7, 7), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")])
            lv72 = R.call_tir(cls.conv2d10, (lv27_1, metadata["relax.expr.Constant"][85]), out_sinfo=R.Tensor((1, 512, 7, 7), dtype="float32"))
            lv73 = R.call_tir(cls.batch_norm4, (lv72, metadata["relax.expr.Constant"][86], metadata["relax.expr.Constant"][87], metadata["relax.expr.Constant"][88], metadata["relax.expr.Constant"][89]), out_sinfo=[R.Tensor((1, 512, 7, 7), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")])
            lv30_1: R.Tensor((1, 512, 7, 7), dtype="float32") = lv70[0]
            lv31_1: R.Tensor((1, 512, 7, 7), dtype="float32") = lv73[0]
            lv32 = R.call_tir(cls.fused_add3_relu4, (lv30_1, lv31_1), out_sinfo=R.Tensor((1, 512, 7, 7), dtype="float32"))
            lv77 = R.call_tir(cls.conv2d9, (lv32, metadata["relax.expr.Constant"][90]), out_sinfo=R.Tensor((1, 512, 7, 7), dtype="float32"))
            lv78 = R.call_tir(cls.batch_norm4, (lv77, metadata["relax.expr.Constant"][91], metadata["relax.expr.Constant"][92], metadata["relax.expr.Constant"][93], metadata["relax.expr.Constant"][94]), out_sinfo=[R.Tensor((1, 512, 7, 7), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")])
            lv33: R.Tensor((1, 512, 7, 7), dtype="float32") = lv78[0]
            lv34 = R.call_tir(cls.fused_relu4, (lv33,), out_sinfo=R.Tensor((1, 512, 7, 7), dtype="float32"))
            lv81 = R.call_tir(cls.conv2d9, (lv34, metadata["relax.expr.Constant"][95]), out_sinfo=R.Tensor((1, 512, 7, 7), dtype="float32"))
            lv82 = R.call_tir(cls.batch_norm4, (lv81, metadata["relax.expr.Constant"][96], metadata["relax.expr.Constant"][97], metadata["relax.expr.Constant"][98], metadata["relax.expr.Constant"][99]), out_sinfo=[R.Tensor((1, 512, 7, 7), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")])
            lv35_1: R.Tensor((1, 512, 7, 7), dtype="float32") = lv82[0]
            lv36_1 = R.call_tir(cls.fused_add3_relu4, (lv35_1, lv32), out_sinfo=R.Tensor((1, 512, 7, 7), dtype="float32"))
            lv86 = R.call_tir(cls.adaptive_avg_pool2d, (lv36_1,), out_sinfo=R.Tensor((1, 512, 1, 1), dtype="float32"))
            lv87 = R.call_tir(cls.reshape, (lv86,), out_sinfo=R.Tensor((1, 512), dtype="float32"))
            lv88 = R.call_tir(cls.transpose, (metadata["relax.expr.Constant"][100],), out_sinfo=R.Tensor((512, 1000), dtype="float32"))
            gv = R.call_tir(cls.fused_matmul_add4, (lv87, lv88, metadata["relax.expr.Constant"][101]), out_sinfo=R.Tensor((1, 1000), dtype="float32"))
            R.output(gv)
        return gv

# Metadata omitted. Use show_meta=True in script() method to show it.