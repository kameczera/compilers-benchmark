# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def adaptive_avg_pool2d(lv224: T.Buffer((T.int64(1), T.int64(2048), T.int64(7), T.int64(7)), "float32"), adaptive_pool_avg: T.Buffer((T.int64(1), T.int64(2048), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        adaptive_pool_sum = T.alloc_buffer((T.int64(1), T.int64(2048), T.int64(1), T.int64(1)))
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(2), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for rv0, rv1 in T.grid(T.int64(7), T.int64(7)):
                    with T.block("adaptive_pool_sum"):
                        v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = T.axis.spatial(T.int64(2048), ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1)
                        v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_rv0, v_rv1 = T.axis.remap("RR", [rv0, rv1])
                        T.reads(lv224[v_ax0, v_ax1, v_ax2 * T.int64(7) + v_rv0, v_ax3 * T.int64(7) + v_rv1])
                        T.writes(adaptive_pool_sum[v_ax0, v_ax1, v_ax2, v_ax3])
                        with T.init():
                            adaptive_pool_sum[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.0)
                        adaptive_pool_sum[v_ax0, v_ax1, v_ax2, v_ax3] = adaptive_pool_sum[v_ax0, v_ax1, v_ax2, v_ax3] + lv224[v_ax0, v_ax1, v_ax2 * T.int64(7) + v_rv0, v_ax3 * T.int64(7) + v_rv1]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(2), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("adaptive_pool_avg"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(2048), ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1)
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
    def batch_norm10(lv187: T.Buffer((T.int64(1), T.int64(512), T.int64(7), T.int64(7)), "float32"), B: T.Buffer((T.int64(512),), "float32"), C: T.Buffer((T.int64(512),), "float32"), D: T.Buffer((T.int64(512),), "float32"), E: T.Buffer((T.int64(512),), "float32"), T_add: T.Buffer((T.int64(1), T.int64(512), T.int64(7), T.int64(7)), "float32"), T_add_1: T.Buffer((T.int64(512),), "float32"), T_add_2: T.Buffer((T.int64(512),), "float32")):
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
        lv187_red = T.alloc_buffer((T.int64(512),))
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
                    T.reads(lv187[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = lv187[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape[v_ax0, v_ax1, T.int64(0), T.int64(0)]
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
                    with T.block("lv187_red"):
                        v_ax0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(512) + ax0_fused_1)
                        v_k0 = T.axis.reduce(T.int64(1), T.int64(0))
                        v_k2, v_k3 = T.axis.remap("RR", [k2, k3])
                        T.reads(lv187[v_k0, v_ax0, v_k2, v_k3])
                        T.writes(lv187_red[v_ax0])
                        with T.init():
                            lv187_red[v_ax0] = T.float32(0.0)
                        lv187_red[v_ax0] = lv187_red[v_ax0] + lv187[v_k0, v_ax0, v_k2, v_k3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                with T.block("T_divide_1"):
                    v_ax0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(512) + ax0_fused_1)
                    T.reads(lv187_red[v_ax0])
                    T.writes(T_divide_1[v_ax0])
                    T_divide_1[v_ax0] = lv187_red[v_ax0] * T.float32(0.020408163265306121)
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
                    T.reads(lv187[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] = lv187[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(25), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_subtract_2"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(512), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(49))
                    v_ax2 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(49) // T.int64(7))
                    v_ax3 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(7))
                    T.where(ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1 < T.int64(25088))
                    T.reads(lv187[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3] = lv187[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)]
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
    def batch_norm11(lv191: T.Buffer((T.int64(1), T.int64(2048), T.int64(7), T.int64(7)), "float32"), B: T.Buffer((T.int64(2048),), "float32"), C: T.Buffer((T.int64(2048),), "float32"), D: T.Buffer((T.int64(2048),), "float32"), E: T.Buffer((T.int64(2048),), "float32"), T_add: T.Buffer((T.int64(1), T.int64(2048), T.int64(7), T.int64(7)), "float32"), T_add_1: T.Buffer((T.int64(2048),), "float32"), T_add_2: T.Buffer((T.int64(2048),), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        T_reshape = T.alloc_buffer((T.int64(1), T.int64(2048), T.int64(1), T.int64(1)))
        T_subtract = T.alloc_buffer((T.int64(1), T.int64(2048), T.int64(7), T.int64(7)))
        T_reshape_1 = T.alloc_buffer((T.int64(1), T.int64(2048), T.int64(1), T.int64(1)))
        T_add_3 = T.alloc_buffer((T.int64(1), T.int64(2048), T.int64(1), T.int64(1)))
        compute = T.alloc_buffer((T.int64(1), T.int64(2048), T.int64(1), T.int64(1)))
        T_divide = T.alloc_buffer((T.int64(1), T.int64(2048), T.int64(7), T.int64(7)))
        T_reshape_2 = T.alloc_buffer((T.int64(1), T.int64(2048), T.int64(1), T.int64(1)))
        T_multiply = T.alloc_buffer((T.int64(1), T.int64(2048), T.int64(7), T.int64(7)))
        T_reshape_3 = T.alloc_buffer((T.int64(1), T.int64(2048), T.int64(1), T.int64(1)))
        T_multiply_1 = T.alloc_buffer((T.int64(2048),))
        lv191_red = T.alloc_buffer((T.int64(2048),))
        T_divide_1 = T.alloc_buffer((T.int64(2048),))
        T_multiply_2 = T.alloc_buffer((T.int64(2048),))
        T_multiply_3 = T.alloc_buffer((T.int64(2048),))
        T_reshape_4 = T.alloc_buffer((T.int64(1), T.int64(2048), T.int64(1), T.int64(1)))
        T_subtract_1 = T.alloc_buffer((T.int64(1), T.int64(2048), T.int64(7), T.int64(7)))
        T_subtract_2 = T.alloc_buffer((T.int64(1), T.int64(2048), T.int64(7), T.int64(7)))
        T_multiply_4 = T.alloc_buffer((T.int64(1), T.int64(2048), T.int64(7), T.int64(7)))
        T_multiply_red = T.alloc_buffer((T.int64(2048),))
        T_divide_2 = T.alloc_buffer((T.int64(2048),))
        T_multiply_5 = T.alloc_buffer((T.int64(2048),))
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(2), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_reshape"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(2048), ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(D[(v_ax1 + v_ax2 + v_ax3) % T.int64(2048)])
                    T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = D[(v_ax1 + v_ax2 + v_ax3) % T.int64(2048)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_subtract"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(2048), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(49))
                    v_ax2 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(49) // T.int64(7))
                    v_ax3 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(7))
                    T.reads(lv191[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = lv191[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(2), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_reshape_1"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(2048), ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(E[(v_ax1 + v_ax2 + v_ax3) % T.int64(2048)])
                    T.writes(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3] = E[(v_ax1 + v_ax2 + v_ax3) % T.int64(2048)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(2), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(2048), ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_add_3[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_add_3[v_ax0, v_ax1, v_ax2, v_ax3] = T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3] + T.float32(1.0000000000000001e-05)
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(2), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(2048), i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1)
                    v_i2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(T_add_3[v_i0, v_i1, v_i2, v_i3])
                    T.writes(compute[v_i0, v_i1, v_i2, v_i3])
                    compute[v_i0, v_i1, v_i2, v_i3] = T.sqrt(T_add_3[v_i0, v_i1, v_i2, v_i3])
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_divide"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(2048), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(49))
                    v_ax2 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(49) // T.int64(7))
                    v_ax3 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(7))
                    T.reads(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3], compute[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_divide[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_divide[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] / compute[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(2), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_reshape_2"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(2048), ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(B[(v_ax1 + v_ax2 + v_ax3) % T.int64(2048)])
                    T.writes(T_reshape_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_2[v_ax0, v_ax1, v_ax2, v_ax3] = B[(v_ax1 + v_ax2 + v_ax3) % T.int64(2048)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(2048), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(49))
                    v_ax2 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(49) // T.int64(7))
                    v_ax3 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(7))
                    T.reads(T_divide[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_2[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T_divide[v_ax0, v_ax1, v_ax2, v_ax3] * T_reshape_2[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(2), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_reshape_3"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(2048), ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(C[(v_ax1 + v_ax2 + v_ax3) % T.int64(2048)])
                    T.writes(T_reshape_3[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_3[v_ax0, v_ax1, v_ax2, v_ax3] = C[(v_ax1 + v_ax2 + v_ax3) % T.int64(2048)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_1"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(2048), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(49))
                    v_ax2 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(49) // T.int64(7))
                    v_ax3 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(7))
                    T.reads(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_3[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_add[v_ax0, v_ax1, v_ax2, v_ax3] = T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] + T_reshape_3[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_fused_0 in T.thread_binding(T.int64(2), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply_1"):
                    v_ax0 = T.axis.spatial(T.int64(2048), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.reads(D[v_ax0])
                    T.writes(T_multiply_1[v_ax0])
                    T_multiply_1[v_ax0] = T.float32(0.90000000000000002) * D[v_ax0]
        for ax0_fused_0 in T.thread_binding(T.int64(2), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for k0, k2, k3 in T.grid(T.int64(1), T.int64(7), T.int64(7)):
                    with T.block("lv191_red"):
                        v_ax0 = T.axis.spatial(T.int64(2048), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                        v_k0 = T.axis.reduce(T.int64(1), T.int64(0))
                        v_k2, v_k3 = T.axis.remap("RR", [k2, k3])
                        T.reads(lv191[v_k0, v_ax0, v_k2, v_k3])
                        T.writes(lv191_red[v_ax0])
                        with T.init():
                            lv191_red[v_ax0] = T.float32(0.0)
                        lv191_red[v_ax0] = lv191_red[v_ax0] + lv191[v_k0, v_ax0, v_k2, v_k3]
        for ax0_fused_0 in T.thread_binding(T.int64(2), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_divide_1"):
                    v_ax0 = T.axis.spatial(T.int64(2048), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.reads(lv191_red[v_ax0])
                    T.writes(T_divide_1[v_ax0])
                    T_divide_1[v_ax0] = lv191_red[v_ax0] * T.float32(0.020408163265306121)
        for ax0_fused_0 in T.thread_binding(T.int64(2), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply_2"):
                    v_ax0 = T.axis.spatial(T.int64(2048), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.reads(T_divide_1[v_ax0])
                    T.writes(T_multiply_2[v_ax0])
                    T_multiply_2[v_ax0] = T.float32(0.10000000000000001) * T_divide_1[v_ax0]
        for ax0_fused_0 in T.thread_binding(T.int64(2), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_2"):
                    v_ax0 = T.axis.spatial(T.int64(2048), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.reads(T_multiply_1[v_ax0], T_multiply_2[v_ax0])
                    T.writes(T_add_1[v_ax0])
                    T_add_1[v_ax0] = T_multiply_1[v_ax0] + T_multiply_2[v_ax0]
        for ax0_fused_0 in T.thread_binding(T.int64(2), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply_3"):
                    v_ax0 = T.axis.spatial(T.int64(2048), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.reads(E[v_ax0])
                    T.writes(T_multiply_3[v_ax0])
                    T_multiply_3[v_ax0] = T.float32(0.90000000000000002) * E[v_ax0]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(2), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_reshape_4"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(2048), ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(T_divide_1[(v_ax1 + v_ax2 + v_ax3) % T.int64(2048)])
                    T.writes(T_reshape_4[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_4[v_ax0, v_ax1, v_ax2, v_ax3] = T_divide_1[(v_ax1 + v_ax2 + v_ax3) % T.int64(2048)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_subtract_1"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(2048), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(49))
                    v_ax2 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(49) // T.int64(7))
                    v_ax3 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(7))
                    T.reads(lv191[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] = lv191[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_subtract_2"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(2048), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(49))
                    v_ax2 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(49) // T.int64(7))
                    v_ax3 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(7))
                    T.reads(lv191[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3] = lv191[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply_4"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(2048), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(49))
                    v_ax2 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(49) // T.int64(7))
                    v_ax3 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(7))
                    T.reads(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3], T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_multiply_4[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_multiply_4[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] * T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3]
        for ax0_fused_0 in T.thread_binding(T.int64(2), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for k0, k2, k3 in T.grid(T.int64(1), T.int64(7), T.int64(7)):
                    with T.block("T_multiply_red"):
                        v_ax0 = T.axis.spatial(T.int64(2048), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                        v_k0 = T.axis.reduce(T.int64(1), T.int64(0))
                        v_k2, v_k3 = T.axis.remap("RR", [k2, k3])
                        T.reads(T_multiply_4[v_k0, v_ax0, v_k2, v_k3])
                        T.writes(T_multiply_red[v_ax0])
                        with T.init():
                            T_multiply_red[v_ax0] = T.float32(0.0)
                        T_multiply_red[v_ax0] = T_multiply_red[v_ax0] + T_multiply_4[v_k0, v_ax0, v_k2, v_k3]
        for ax0_fused_0 in T.thread_binding(T.int64(2), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_divide_2"):
                    v_ax0 = T.axis.spatial(T.int64(2048), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.reads(T_multiply_red[v_ax0])
                    T.writes(T_divide_2[v_ax0])
                    T_divide_2[v_ax0] = T_multiply_red[v_ax0] * T.float32(0.020408163265306121)
        for ax0_fused_0 in T.thread_binding(T.int64(2), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply_5"):
                    v_ax0 = T.axis.spatial(T.int64(2048), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.reads(T_divide_2[v_ax0])
                    T.writes(T_multiply_5[v_ax0])
                    T_multiply_5[v_ax0] = T.float32(0.10000000000000001) * T_divide_2[v_ax0]
        for ax0_fused_0 in T.thread_binding(T.int64(2), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_3"):
                    v_ax0 = T.axis.spatial(T.int64(2048), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.reads(T_multiply_3[v_ax0], T_multiply_5[v_ax0])
                    T.writes(T_add_2[v_ax0])
                    T_add_2[v_ax0] = T_multiply_3[v_ax0] + T_multiply_5[v_ax0]

    @T.prim_func(private=True)
    def batch_norm2(lv13: T.Buffer((T.int64(1), T.int64(256), T.int64(56), T.int64(56)), "float32"), B: T.Buffer((T.int64(256),), "float32"), C: T.Buffer((T.int64(256),), "float32"), D: T.Buffer((T.int64(256),), "float32"), E: T.Buffer((T.int64(256),), "float32"), T_add: T.Buffer((T.int64(1), T.int64(256), T.int64(56), T.int64(56)), "float32"), T_add_1: T.Buffer((T.int64(256),), "float32"), T_add_2: T.Buffer((T.int64(256),), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        T_reshape = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)))
        T_subtract = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(56), T.int64(56)))
        T_reshape_1 = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)))
        T_add_3 = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)))
        compute = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)))
        T_divide = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(56), T.int64(56)))
        T_reshape_2 = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)))
        T_multiply = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(56), T.int64(56)))
        T_reshape_3 = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)))
        T_multiply_1 = T.alloc_buffer((T.int64(256),))
        lv13_red = T.alloc_buffer((T.int64(256),))
        T_divide_1 = T.alloc_buffer((T.int64(256),))
        T_multiply_2 = T.alloc_buffer((T.int64(256),))
        T_multiply_3 = T.alloc_buffer((T.int64(256),))
        T_reshape_4 = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)))
        T_subtract_1 = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(56), T.int64(56)))
        T_subtract_2 = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(56), T.int64(56)))
        T_multiply_4 = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(56), T.int64(56)))
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
        for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(4)):
                    with T.block("T_subtract"):
                        v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = T.axis.spatial(T.int64(256), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) // T.int64(3136))
                        v_ax2 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(3136) // T.int64(56))
                        v_ax3 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(56))
                        T.where((ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1) * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2 < T.int64(802816))
                        T.reads(lv13[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                        T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                        T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = lv13[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape[v_ax0, v_ax1, T.int64(0), T.int64(0)]
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
        for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(4)):
                    with T.block("T_divide"):
                        v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = T.axis.spatial(T.int64(256), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) // T.int64(3136))
                        v_ax2 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(3136) // T.int64(56))
                        v_ax3 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(56))
                        T.where((ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1) * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2 < T.int64(802816))
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
        for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(4)):
                    with T.block("T_multiply"):
                        v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = T.axis.spatial(T.int64(256), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) // T.int64(3136))
                        v_ax2 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(3136) // T.int64(56))
                        v_ax3 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(56))
                        T.where((ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1) * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2 < T.int64(802816))
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
        for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(4)):
                    with T.block("T_add_1"):
                        v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = T.axis.spatial(T.int64(256), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) // T.int64(3136))
                        v_ax2 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(3136) // T.int64(56))
                        v_ax3 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(56))
                        T.where((ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1) * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2 < T.int64(802816))
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
                for k0, k2, k3 in T.grid(T.int64(1), T.int64(56), T.int64(56)):
                    with T.block("lv13_red"):
                        v_ax0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(256) + ax0_fused_1)
                        v_k0 = T.axis.reduce(T.int64(1), T.int64(0))
                        v_k2, v_k3 = T.axis.remap("RR", [k2, k3])
                        T.reads(lv13[v_k0, v_ax0, v_k2, v_k3])
                        T.writes(lv13_red[v_ax0])
                        with T.init():
                            lv13_red[v_ax0] = T.float32(0.0)
                        lv13_red[v_ax0] = lv13_red[v_ax0] + lv13[v_k0, v_ax0, v_k2, v_k3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                with T.block("T_divide_1"):
                    v_ax0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(256) + ax0_fused_1)
                    T.reads(lv13_red[v_ax0])
                    T.writes(T_divide_1[v_ax0])
                    T_divide_1[v_ax0] = lv13_red[v_ax0] * T.float32(0.00031887755102040814)
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
        for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(4)):
                    with T.block("T_subtract_1"):
                        v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = T.axis.spatial(T.int64(256), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) // T.int64(3136))
                        v_ax2 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(3136) // T.int64(56))
                        v_ax3 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(56))
                        T.where((ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1) * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2 < T.int64(802816))
                        T.reads(lv13[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                        T.writes(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3])
                        T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] = lv13[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(4)):
                    with T.block("T_subtract_2"):
                        v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = T.axis.spatial(T.int64(256), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) // T.int64(3136))
                        v_ax2 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(3136) // T.int64(56))
                        v_ax3 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(56))
                        T.where((ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1) * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2 < T.int64(802816))
                        T.reads(lv13[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                        T.writes(T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
                        T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3] = lv13[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(4)):
                    with T.block("T_multiply_4"):
                        v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = T.axis.spatial(T.int64(256), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) // T.int64(3136))
                        v_ax2 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(3136) // T.int64(56))
                        v_ax3 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(56))
                        T.where((ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1) * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2 < T.int64(802816))
                        T.reads(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3], T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
                        T.writes(T_multiply_4[v_ax0, v_ax1, v_ax2, v_ax3])
                        T_multiply_4[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] * T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for k0, k2, k3 in T.grid(T.int64(1), T.int64(56), T.int64(56)):
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
                    T_divide_2[v_ax0] = T_multiply_red[v_ax0] * T.float32(0.00031887755102040814)
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
    def batch_norm3(lv47: T.Buffer((T.int64(1), T.int64(128), T.int64(56), T.int64(56)), "float32"), B: T.Buffer((T.int64(128),), "float32"), C: T.Buffer((T.int64(128),), "float32"), D: T.Buffer((T.int64(128),), "float32"), E: T.Buffer((T.int64(128),), "float32"), T_add: T.Buffer((T.int64(1), T.int64(128), T.int64(56), T.int64(56)), "float32"), T_add_1: T.Buffer((T.int64(128),), "float32"), T_add_2: T.Buffer((T.int64(128),), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        T_reshape = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)))
        T_subtract = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(56), T.int64(56)))
        T_reshape_1 = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)))
        T_add_3 = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)))
        compute = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)))
        T_divide = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(56), T.int64(56)))
        T_reshape_2 = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)))
        T_multiply = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(56), T.int64(56)))
        T_reshape_3 = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)))
        T_multiply_1 = T.alloc_buffer((T.int64(128),))
        lv47_red = T.alloc_buffer((T.int64(128),))
        T_divide_1 = T.alloc_buffer((T.int64(128),))
        T_multiply_2 = T.alloc_buffer((T.int64(128),))
        T_multiply_3 = T.alloc_buffer((T.int64(128),))
        T_reshape_4 = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)))
        T_subtract_1 = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(56), T.int64(56)))
        T_subtract_2 = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(56), T.int64(56)))
        T_multiply_4 = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(56), T.int64(56)))
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
        for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(2)):
                    with T.block("T_subtract"):
                        v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = T.axis.spatial(T.int64(128), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) // T.int64(3136))
                        v_ax2 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(3136) // T.int64(56))
                        v_ax3 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(56))
                        T.where((ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1) * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2 < T.int64(401408))
                        T.reads(lv47[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                        T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                        T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = lv47[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape[v_ax0, v_ax1, T.int64(0), T.int64(0)]
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
        for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(2)):
                    with T.block("T_divide"):
                        v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = T.axis.spatial(T.int64(128), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) // T.int64(3136))
                        v_ax2 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(3136) // T.int64(56))
                        v_ax3 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(56))
                        T.where((ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1) * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2 < T.int64(401408))
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
        for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(2)):
                    with T.block("T_multiply"):
                        v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = T.axis.spatial(T.int64(128), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) // T.int64(3136))
                        v_ax2 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(3136) // T.int64(56))
                        v_ax3 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(56))
                        T.where((ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1) * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2 < T.int64(401408))
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
        for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(2)):
                    with T.block("T_add_1"):
                        v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = T.axis.spatial(T.int64(128), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) // T.int64(3136))
                        v_ax2 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(3136) // T.int64(56))
                        v_ax3 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(56))
                        T.where((ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1) * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2 < T.int64(401408))
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
                for k0, k2, k3 in T.grid(T.int64(1), T.int64(56), T.int64(56)):
                    with T.block("lv47_red"):
                        v_ax0 = T.axis.spatial(T.int64(128), ax0_fused_0 * T.int64(128) + ax0_fused_1)
                        v_k0 = T.axis.reduce(T.int64(1), T.int64(0))
                        v_k2, v_k3 = T.axis.remap("RR", [k2, k3])
                        T.reads(lv47[v_k0, v_ax0, v_k2, v_k3])
                        T.writes(lv47_red[v_ax0])
                        with T.init():
                            lv47_red[v_ax0] = T.float32(0.0)
                        lv47_red[v_ax0] = lv47_red[v_ax0] + lv47[v_k0, v_ax0, v_k2, v_k3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                with T.block("T_divide_1"):
                    v_ax0 = T.axis.spatial(T.int64(128), ax0_fused_0 * T.int64(128) + ax0_fused_1)
                    T.reads(lv47_red[v_ax0])
                    T.writes(T_divide_1[v_ax0])
                    T_divide_1[v_ax0] = lv47_red[v_ax0] * T.float32(0.00031887755102040814)
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
        for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(2)):
                    with T.block("T_subtract_1"):
                        v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = T.axis.spatial(T.int64(128), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) // T.int64(3136))
                        v_ax2 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(3136) // T.int64(56))
                        v_ax3 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(56))
                        T.where((ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1) * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2 < T.int64(401408))
                        T.reads(lv47[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                        T.writes(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3])
                        T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] = lv47[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(2)):
                    with T.block("T_subtract_2"):
                        v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = T.axis.spatial(T.int64(128), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) // T.int64(3136))
                        v_ax2 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(3136) // T.int64(56))
                        v_ax3 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(56))
                        T.where((ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1) * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2 < T.int64(401408))
                        T.reads(lv47[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                        T.writes(T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
                        T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3] = lv47[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(2)):
                    with T.block("T_multiply_4"):
                        v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = T.axis.spatial(T.int64(128), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) // T.int64(3136))
                        v_ax2 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(3136) // T.int64(56))
                        v_ax3 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(56))
                        T.where((ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1) * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2 < T.int64(401408))
                        T.reads(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3], T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
                        T.writes(T_multiply_4[v_ax0, v_ax1, v_ax2, v_ax3])
                        T_multiply_4[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] * T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                for k0, k2, k3 in T.grid(T.int64(1), T.int64(56), T.int64(56)):
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
                    T_divide_2[v_ax0] = T_multiply_red[v_ax0] * T.float32(0.00031887755102040814)
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
    def batch_norm4(lv51: T.Buffer((T.int64(1), T.int64(128), T.int64(28), T.int64(28)), "float32"), B: T.Buffer((T.int64(128),), "float32"), C: T.Buffer((T.int64(128),), "float32"), D: T.Buffer((T.int64(128),), "float32"), E: T.Buffer((T.int64(128),), "float32"), T_add: T.Buffer((T.int64(1), T.int64(128), T.int64(28), T.int64(28)), "float32"), T_add_1: T.Buffer((T.int64(128),), "float32"), T_add_2: T.Buffer((T.int64(128),), "float32")):
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
        lv51_red = T.alloc_buffer((T.int64(128),))
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
                    T.reads(lv51[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = lv51[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape[v_ax0, v_ax1, T.int64(0), T.int64(0)]
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
                    with T.block("lv51_red"):
                        v_ax0 = T.axis.spatial(T.int64(128), ax0_fused_0 * T.int64(128) + ax0_fused_1)
                        v_k0 = T.axis.reduce(T.int64(1), T.int64(0))
                        v_k2, v_k3 = T.axis.remap("RR", [k2, k3])
                        T.reads(lv51[v_k0, v_ax0, v_k2, v_k3])
                        T.writes(lv51_red[v_ax0])
                        with T.init():
                            lv51_red[v_ax0] = T.float32(0.0)
                        lv51_red[v_ax0] = lv51_red[v_ax0] + lv51[v_k0, v_ax0, v_k2, v_k3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(128), thread="threadIdx.x"):
                with T.block("T_divide_1"):
                    v_ax0 = T.axis.spatial(T.int64(128), ax0_fused_0 * T.int64(128) + ax0_fused_1)
                    T.reads(lv51_red[v_ax0])
                    T.writes(T_divide_1[v_ax0])
                    T_divide_1[v_ax0] = lv51_red[v_ax0] * T.float32(0.0012755102040816326)
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
                    T.reads(lv51[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] = lv51[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_subtract_2"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(128), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(784))
                    v_ax2 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(784) // T.int64(28))
                    v_ax3 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(28))
                    T.reads(lv51[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3] = lv51[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)]
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
    def batch_norm5(lv55: T.Buffer((T.int64(1), T.int64(512), T.int64(28), T.int64(28)), "float32"), B: T.Buffer((T.int64(512),), "float32"), C: T.Buffer((T.int64(512),), "float32"), D: T.Buffer((T.int64(512),), "float32"), E: T.Buffer((T.int64(512),), "float32"), T_add: T.Buffer((T.int64(1), T.int64(512), T.int64(28), T.int64(28)), "float32"), T_add_1: T.Buffer((T.int64(512),), "float32"), T_add_2: T.Buffer((T.int64(512),), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        T_reshape = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)))
        T_subtract = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(28), T.int64(28)))
        T_reshape_1 = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)))
        T_add_3 = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)))
        compute = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)))
        T_divide = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(28), T.int64(28)))
        T_reshape_2 = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)))
        T_multiply = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(28), T.int64(28)))
        T_reshape_3 = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)))
        T_multiply_1 = T.alloc_buffer((T.int64(512),))
        lv55_red = T.alloc_buffer((T.int64(512),))
        T_divide_1 = T.alloc_buffer((T.int64(512),))
        T_multiply_2 = T.alloc_buffer((T.int64(512),))
        T_multiply_3 = T.alloc_buffer((T.int64(512),))
        T_reshape_4 = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)))
        T_subtract_1 = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(28), T.int64(28)))
        T_subtract_2 = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(28), T.int64(28)))
        T_multiply_4 = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(28), T.int64(28)))
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
        for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(2)):
                    with T.block("T_subtract"):
                        v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = T.axis.spatial(T.int64(512), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) // T.int64(784))
                        v_ax2 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(784) // T.int64(28))
                        v_ax3 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(28))
                        T.where((ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1) * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2 < T.int64(401408))
                        T.reads(lv55[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                        T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                        T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = lv55[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape[v_ax0, v_ax1, T.int64(0), T.int64(0)]
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
        for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(2)):
                    with T.block("T_divide"):
                        v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = T.axis.spatial(T.int64(512), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) // T.int64(784))
                        v_ax2 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(784) // T.int64(28))
                        v_ax3 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(28))
                        T.where((ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1) * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2 < T.int64(401408))
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
        for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(2)):
                    with T.block("T_multiply"):
                        v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = T.axis.spatial(T.int64(512), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) // T.int64(784))
                        v_ax2 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(784) // T.int64(28))
                        v_ax3 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(28))
                        T.where((ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1) * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2 < T.int64(401408))
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
        for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(2)):
                    with T.block("T_add_1"):
                        v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = T.axis.spatial(T.int64(512), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) // T.int64(784))
                        v_ax2 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(784) // T.int64(28))
                        v_ax3 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(28))
                        T.where((ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1) * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2 < T.int64(401408))
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
                for k0, k2, k3 in T.grid(T.int64(1), T.int64(28), T.int64(28)):
                    with T.block("lv55_red"):
                        v_ax0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(512) + ax0_fused_1)
                        v_k0 = T.axis.reduce(T.int64(1), T.int64(0))
                        v_k2, v_k3 = T.axis.remap("RR", [k2, k3])
                        T.reads(lv55[v_k0, v_ax0, v_k2, v_k3])
                        T.writes(lv55_red[v_ax0])
                        with T.init():
                            lv55_red[v_ax0] = T.float32(0.0)
                        lv55_red[v_ax0] = lv55_red[v_ax0] + lv55[v_k0, v_ax0, v_k2, v_k3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                with T.block("T_divide_1"):
                    v_ax0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(512) + ax0_fused_1)
                    T.reads(lv55_red[v_ax0])
                    T.writes(T_divide_1[v_ax0])
                    T_divide_1[v_ax0] = lv55_red[v_ax0] * T.float32(0.0012755102040816326)
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
        for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(2)):
                    with T.block("T_subtract_1"):
                        v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = T.axis.spatial(T.int64(512), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) // T.int64(784))
                        v_ax2 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(784) // T.int64(28))
                        v_ax3 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(28))
                        T.where((ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1) * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2 < T.int64(401408))
                        T.reads(lv55[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                        T.writes(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3])
                        T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] = lv55[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(2)):
                    with T.block("T_subtract_2"):
                        v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = T.axis.spatial(T.int64(512), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) // T.int64(784))
                        v_ax2 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(784) // T.int64(28))
                        v_ax3 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(28))
                        T.where((ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1) * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2 < T.int64(401408))
                        T.reads(lv55[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                        T.writes(T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
                        T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3] = lv55[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(2)):
                    with T.block("T_multiply_4"):
                        v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = T.axis.spatial(T.int64(512), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) // T.int64(784))
                        v_ax2 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(784) // T.int64(28))
                        v_ax3 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(28))
                        T.where((ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1) * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2 < T.int64(401408))
                        T.reads(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3], T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
                        T.writes(T_multiply_4[v_ax0, v_ax1, v_ax2, v_ax3])
                        T_multiply_4[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] * T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                for k0, k2, k3 in T.grid(T.int64(1), T.int64(28), T.int64(28)):
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
                    T_divide_2[v_ax0] = T_multiply_red[v_ax0] * T.float32(0.0012755102040816326)
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
    def batch_norm6(lv102: T.Buffer((T.int64(1), T.int64(256), T.int64(28), T.int64(28)), "float32"), B: T.Buffer((T.int64(256),), "float32"), C: T.Buffer((T.int64(256),), "float32"), D: T.Buffer((T.int64(256),), "float32"), E: T.Buffer((T.int64(256),), "float32"), T_add: T.Buffer((T.int64(1), T.int64(256), T.int64(28), T.int64(28)), "float32"), T_add_1: T.Buffer((T.int64(256),), "float32"), T_add_2: T.Buffer((T.int64(256),), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        T_reshape = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)))
        T_subtract = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(28), T.int64(28)))
        T_reshape_1 = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)))
        T_add_3 = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)))
        compute = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)))
        T_divide = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(28), T.int64(28)))
        T_reshape_2 = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)))
        T_multiply = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(28), T.int64(28)))
        T_reshape_3 = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)))
        T_multiply_1 = T.alloc_buffer((T.int64(256),))
        lv102_red = T.alloc_buffer((T.int64(256),))
        T_divide_1 = T.alloc_buffer((T.int64(256),))
        T_multiply_2 = T.alloc_buffer((T.int64(256),))
        T_multiply_3 = T.alloc_buffer((T.int64(256),))
        T_reshape_4 = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)))
        T_subtract_1 = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(28), T.int64(28)))
        T_subtract_2 = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(28), T.int64(28)))
        T_multiply_4 = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(28), T.int64(28)))
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
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_subtract"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(256), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(784))
                    v_ax2 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(784) // T.int64(28))
                    v_ax3 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(28))
                    T.reads(lv102[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = lv102[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape[v_ax0, v_ax1, T.int64(0), T.int64(0)]
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
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_divide"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(256), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(784))
                    v_ax2 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(784) // T.int64(28))
                    v_ax3 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(28))
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
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(256), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(784))
                    v_ax2 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(784) // T.int64(28))
                    v_ax3 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(28))
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
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_1"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(256), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(784))
                    v_ax2 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(784) // T.int64(28))
                    v_ax3 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(28))
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
                for k0, k2, k3 in T.grid(T.int64(1), T.int64(28), T.int64(28)):
                    with T.block("lv102_red"):
                        v_ax0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(256) + ax0_fused_1)
                        v_k0 = T.axis.reduce(T.int64(1), T.int64(0))
                        v_k2, v_k3 = T.axis.remap("RR", [k2, k3])
                        T.reads(lv102[v_k0, v_ax0, v_k2, v_k3])
                        T.writes(lv102_red[v_ax0])
                        with T.init():
                            lv102_red[v_ax0] = T.float32(0.0)
                        lv102_red[v_ax0] = lv102_red[v_ax0] + lv102[v_k0, v_ax0, v_k2, v_k3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                with T.block("T_divide_1"):
                    v_ax0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(256) + ax0_fused_1)
                    T.reads(lv102_red[v_ax0])
                    T.writes(T_divide_1[v_ax0])
                    T_divide_1[v_ax0] = lv102_red[v_ax0] * T.float32(0.0012755102040816326)
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
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_subtract_1"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(256), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(784))
                    v_ax2 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(784) // T.int64(28))
                    v_ax3 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(28))
                    T.reads(lv102[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] = lv102[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_subtract_2"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(256), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(784))
                    v_ax2 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(784) // T.int64(28))
                    v_ax3 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(28))
                    T.reads(lv102[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3] = lv102[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply_4"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(256), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(784))
                    v_ax2 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(784) // T.int64(28))
                    v_ax3 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(28))
                    T.reads(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3], T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_multiply_4[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_multiply_4[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] * T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for k0, k2, k3 in T.grid(T.int64(1), T.int64(28), T.int64(28)):
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
                    T_divide_2[v_ax0] = T_multiply_red[v_ax0] * T.float32(0.0012755102040816326)
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
    def batch_norm7(lv106: T.Buffer((T.int64(1), T.int64(256), T.int64(14), T.int64(14)), "float32"), B: T.Buffer((T.int64(256),), "float32"), C: T.Buffer((T.int64(256),), "float32"), D: T.Buffer((T.int64(256),), "float32"), E: T.Buffer((T.int64(256),), "float32"), T_add: T.Buffer((T.int64(1), T.int64(256), T.int64(14), T.int64(14)), "float32"), T_add_1: T.Buffer((T.int64(256),), "float32"), T_add_2: T.Buffer((T.int64(256),), "float32")):
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
        lv106_red = T.alloc_buffer((T.int64(256),))
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
                    T.reads(lv106[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = lv106[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape[v_ax0, v_ax1, T.int64(0), T.int64(0)]
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
                    with T.block("lv106_red"):
                        v_ax0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(256) + ax0_fused_1)
                        v_k0 = T.axis.reduce(T.int64(1), T.int64(0))
                        v_k2, v_k3 = T.axis.remap("RR", [k2, k3])
                        T.reads(lv106[v_k0, v_ax0, v_k2, v_k3])
                        T.writes(lv106_red[v_ax0])
                        with T.init():
                            lv106_red[v_ax0] = T.float32(0.0)
                        lv106_red[v_ax0] = lv106_red[v_ax0] + lv106[v_k0, v_ax0, v_k2, v_k3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                with T.block("T_divide_1"):
                    v_ax0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(256) + ax0_fused_1)
                    T.reads(lv106_red[v_ax0])
                    T.writes(T_divide_1[v_ax0])
                    T_divide_1[v_ax0] = lv106_red[v_ax0] * T.float32(0.0051020408163265302)
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
                    T.reads(lv106[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] = lv106[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(49), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_subtract_2"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(256), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(196))
                    v_ax2 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(196) // T.int64(14))
                    v_ax3 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(14))
                    T.reads(lv106[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3] = lv106[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)]
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
    def batch_norm8(lv110: T.Buffer((T.int64(1), T.int64(1024), T.int64(14), T.int64(14)), "float32"), B: T.Buffer((T.int64(1024),), "float32"), C: T.Buffer((T.int64(1024),), "float32"), D: T.Buffer((T.int64(1024),), "float32"), E: T.Buffer((T.int64(1024),), "float32"), T_add: T.Buffer((T.int64(1), T.int64(1024), T.int64(14), T.int64(14)), "float32"), T_add_1: T.Buffer((T.int64(1024),), "float32"), T_add_2: T.Buffer((T.int64(1024),), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        T_reshape = T.alloc_buffer((T.int64(1), T.int64(1024), T.int64(1), T.int64(1)))
        T_subtract = T.alloc_buffer((T.int64(1), T.int64(1024), T.int64(14), T.int64(14)))
        T_reshape_1 = T.alloc_buffer((T.int64(1), T.int64(1024), T.int64(1), T.int64(1)))
        T_add_3 = T.alloc_buffer((T.int64(1), T.int64(1024), T.int64(1), T.int64(1)))
        compute = T.alloc_buffer((T.int64(1), T.int64(1024), T.int64(1), T.int64(1)))
        T_divide = T.alloc_buffer((T.int64(1), T.int64(1024), T.int64(14), T.int64(14)))
        T_reshape_2 = T.alloc_buffer((T.int64(1), T.int64(1024), T.int64(1), T.int64(1)))
        T_multiply = T.alloc_buffer((T.int64(1), T.int64(1024), T.int64(14), T.int64(14)))
        T_reshape_3 = T.alloc_buffer((T.int64(1), T.int64(1024), T.int64(1), T.int64(1)))
        T_multiply_1 = T.alloc_buffer((T.int64(1024),))
        lv110_red = T.alloc_buffer((T.int64(1024),))
        T_divide_1 = T.alloc_buffer((T.int64(1024),))
        T_multiply_2 = T.alloc_buffer((T.int64(1024),))
        T_multiply_3 = T.alloc_buffer((T.int64(1024),))
        T_reshape_4 = T.alloc_buffer((T.int64(1), T.int64(1024), T.int64(1), T.int64(1)))
        T_subtract_1 = T.alloc_buffer((T.int64(1), T.int64(1024), T.int64(14), T.int64(14)))
        T_subtract_2 = T.alloc_buffer((T.int64(1), T.int64(1024), T.int64(14), T.int64(14)))
        T_multiply_4 = T.alloc_buffer((T.int64(1), T.int64(1024), T.int64(14), T.int64(14)))
        T_multiply_red = T.alloc_buffer((T.int64(1024),))
        T_divide_2 = T.alloc_buffer((T.int64(1024),))
        T_multiply_5 = T.alloc_buffer((T.int64(1024),))
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_reshape"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(1024), ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(D[(v_ax1 + v_ax2 + v_ax3) % T.int64(1024)])
                    T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = D[(v_ax1 + v_ax2 + v_ax3) % T.int64(1024)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_subtract"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(1024), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(196))
                    v_ax2 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(196) // T.int64(14))
                    v_ax3 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(14))
                    T.reads(lv110[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = lv110[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_reshape_1"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(1024), ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(E[(v_ax1 + v_ax2 + v_ax3) % T.int64(1024)])
                    T.writes(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3] = E[(v_ax1 + v_ax2 + v_ax3) % T.int64(1024)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(1024), ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_add_3[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_add_3[v_ax0, v_ax1, v_ax2, v_ax3] = T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3] + T.float32(1.0000000000000001e-05)
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(1024), i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1)
                    v_i2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(T_add_3[v_i0, v_i1, v_i2, v_i3])
                    T.writes(compute[v_i0, v_i1, v_i2, v_i3])
                    compute[v_i0, v_i1, v_i2, v_i3] = T.sqrt(T_add_3[v_i0, v_i1, v_i2, v_i3])
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_divide"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(1024), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(196))
                    v_ax2 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(196) // T.int64(14))
                    v_ax3 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(14))
                    T.reads(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3], compute[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_divide[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_divide[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] / compute[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_reshape_2"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(1024), ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(B[(v_ax1 + v_ax2 + v_ax3) % T.int64(1024)])
                    T.writes(T_reshape_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_2[v_ax0, v_ax1, v_ax2, v_ax3] = B[(v_ax1 + v_ax2 + v_ax3) % T.int64(1024)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(1024), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(196))
                    v_ax2 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(196) // T.int64(14))
                    v_ax3 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(14))
                    T.reads(T_divide[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_2[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T_divide[v_ax0, v_ax1, v_ax2, v_ax3] * T_reshape_2[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_reshape_3"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(1024), ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(C[(v_ax1 + v_ax2 + v_ax3) % T.int64(1024)])
                    T.writes(T_reshape_3[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_3[v_ax0, v_ax1, v_ax2, v_ax3] = C[(v_ax1 + v_ax2 + v_ax3) % T.int64(1024)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_1"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(1024), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(196))
                    v_ax2 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(196) // T.int64(14))
                    v_ax3 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(14))
                    T.reads(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_3[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_add[v_ax0, v_ax1, v_ax2, v_ax3] = T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] + T_reshape_3[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply_1"):
                    v_ax0 = T.axis.spatial(T.int64(1024), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.reads(D[v_ax0])
                    T.writes(T_multiply_1[v_ax0])
                    T_multiply_1[v_ax0] = T.float32(0.90000000000000002) * D[v_ax0]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for k0, k2, k3 in T.grid(T.int64(1), T.int64(14), T.int64(14)):
                    with T.block("lv110_red"):
                        v_ax0 = T.axis.spatial(T.int64(1024), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                        v_k0 = T.axis.reduce(T.int64(1), T.int64(0))
                        v_k2, v_k3 = T.axis.remap("RR", [k2, k3])
                        T.reads(lv110[v_k0, v_ax0, v_k2, v_k3])
                        T.writes(lv110_red[v_ax0])
                        with T.init():
                            lv110_red[v_ax0] = T.float32(0.0)
                        lv110_red[v_ax0] = lv110_red[v_ax0] + lv110[v_k0, v_ax0, v_k2, v_k3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_divide_1"):
                    v_ax0 = T.axis.spatial(T.int64(1024), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.reads(lv110_red[v_ax0])
                    T.writes(T_divide_1[v_ax0])
                    T_divide_1[v_ax0] = lv110_red[v_ax0] * T.float32(0.0051020408163265302)
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply_2"):
                    v_ax0 = T.axis.spatial(T.int64(1024), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.reads(T_divide_1[v_ax0])
                    T.writes(T_multiply_2[v_ax0])
                    T_multiply_2[v_ax0] = T.float32(0.10000000000000001) * T_divide_1[v_ax0]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_2"):
                    v_ax0 = T.axis.spatial(T.int64(1024), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.reads(T_multiply_1[v_ax0], T_multiply_2[v_ax0])
                    T.writes(T_add_1[v_ax0])
                    T_add_1[v_ax0] = T_multiply_1[v_ax0] + T_multiply_2[v_ax0]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply_3"):
                    v_ax0 = T.axis.spatial(T.int64(1024), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.reads(E[v_ax0])
                    T.writes(T_multiply_3[v_ax0])
                    T_multiply_3[v_ax0] = T.float32(0.90000000000000002) * E[v_ax0]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_reshape_4"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(1024), ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1)
                    v_ax2 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax3 = T.axis.spatial(T.int64(1), T.int64(0))
                    T.reads(T_divide_1[(v_ax1 + v_ax2 + v_ax3) % T.int64(1024)])
                    T.writes(T_reshape_4[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_4[v_ax0, v_ax1, v_ax2, v_ax3] = T_divide_1[(v_ax1 + v_ax2 + v_ax3) % T.int64(1024)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_subtract_1"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(1024), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(196))
                    v_ax2 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(196) // T.int64(14))
                    v_ax3 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(14))
                    T.reads(lv110[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] = lv110[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_subtract_2"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(1024), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(196))
                    v_ax2 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(196) // T.int64(14))
                    v_ax3 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(14))
                    T.reads(lv110[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3] = lv110[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply_4"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(1024), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(196))
                    v_ax2 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(196) // T.int64(14))
                    v_ax3 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(14))
                    T.reads(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3], T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_multiply_4[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_multiply_4[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] * T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for k0, k2, k3 in T.grid(T.int64(1), T.int64(14), T.int64(14)):
                    with T.block("T_multiply_red"):
                        v_ax0 = T.axis.spatial(T.int64(1024), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                        v_k0 = T.axis.reduce(T.int64(1), T.int64(0))
                        v_k2, v_k3 = T.axis.remap("RR", [k2, k3])
                        T.reads(T_multiply_4[v_k0, v_ax0, v_k2, v_k3])
                        T.writes(T_multiply_red[v_ax0])
                        with T.init():
                            T_multiply_red[v_ax0] = T.float32(0.0)
                        T_multiply_red[v_ax0] = T_multiply_red[v_ax0] + T_multiply_4[v_k0, v_ax0, v_k2, v_k3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_divide_2"):
                    v_ax0 = T.axis.spatial(T.int64(1024), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.reads(T_multiply_red[v_ax0])
                    T.writes(T_divide_2[v_ax0])
                    T_divide_2[v_ax0] = T_multiply_red[v_ax0] * T.float32(0.0051020408163265302)
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply_5"):
                    v_ax0 = T.axis.spatial(T.int64(1024), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.reads(T_divide_2[v_ax0])
                    T.writes(T_multiply_5[v_ax0])
                    T_multiply_5[v_ax0] = T.float32(0.10000000000000001) * T_divide_2[v_ax0]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_3"):
                    v_ax0 = T.axis.spatial(T.int64(1024), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.reads(T_multiply_3[v_ax0], T_multiply_5[v_ax0])
                    T.writes(T_add_2[v_ax0])
                    T_add_2[v_ax0] = T_multiply_3[v_ax0] + T_multiply_5[v_ax0]

    @T.prim_func(private=True)
    def batch_norm9(lv183: T.Buffer((T.int64(1), T.int64(512), T.int64(14), T.int64(14)), "float32"), B: T.Buffer((T.int64(512),), "float32"), C: T.Buffer((T.int64(512),), "float32"), D: T.Buffer((T.int64(512),), "float32"), E: T.Buffer((T.int64(512),), "float32"), T_add: T.Buffer((T.int64(1), T.int64(512), T.int64(14), T.int64(14)), "float32"), T_add_1: T.Buffer((T.int64(512),), "float32"), T_add_2: T.Buffer((T.int64(512),), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        T_reshape = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)))
        T_subtract = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(14), T.int64(14)))
        T_reshape_1 = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)))
        T_add_3 = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)))
        compute = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)))
        T_divide = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(14), T.int64(14)))
        T_reshape_2 = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)))
        T_multiply = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(14), T.int64(14)))
        T_reshape_3 = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)))
        T_multiply_1 = T.alloc_buffer((T.int64(512),))
        lv183_red = T.alloc_buffer((T.int64(512),))
        T_divide_1 = T.alloc_buffer((T.int64(512),))
        T_multiply_2 = T.alloc_buffer((T.int64(512),))
        T_multiply_3 = T.alloc_buffer((T.int64(512),))
        T_reshape_4 = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)))
        T_subtract_1 = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(14), T.int64(14)))
        T_subtract_2 = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(14), T.int64(14)))
        T_multiply_4 = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(14), T.int64(14)))
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
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_subtract"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(512), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(196))
                    v_ax2 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(196) // T.int64(14))
                    v_ax3 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(14))
                    T.reads(lv183[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = lv183[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape[v_ax0, v_ax1, T.int64(0), T.int64(0)]
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
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_divide"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(512), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(196))
                    v_ax2 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(196) // T.int64(14))
                    v_ax3 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(14))
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
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(512), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(196))
                    v_ax2 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(196) // T.int64(14))
                    v_ax3 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(14))
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
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_1"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(512), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(196))
                    v_ax2 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(196) // T.int64(14))
                    v_ax3 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(14))
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
                for k0, k2, k3 in T.grid(T.int64(1), T.int64(14), T.int64(14)):
                    with T.block("lv183_red"):
                        v_ax0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(512) + ax0_fused_1)
                        v_k0 = T.axis.reduce(T.int64(1), T.int64(0))
                        v_k2, v_k3 = T.axis.remap("RR", [k2, k3])
                        T.reads(lv183[v_k0, v_ax0, v_k2, v_k3])
                        T.writes(lv183_red[v_ax0])
                        with T.init():
                            lv183_red[v_ax0] = T.float32(0.0)
                        lv183_red[v_ax0] = lv183_red[v_ax0] + lv183[v_k0, v_ax0, v_k2, v_k3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                with T.block("T_divide_1"):
                    v_ax0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(512) + ax0_fused_1)
                    T.reads(lv183_red[v_ax0])
                    T.writes(T_divide_1[v_ax0])
                    T_divide_1[v_ax0] = lv183_red[v_ax0] * T.float32(0.0051020408163265302)
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
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_subtract_1"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(512), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(196))
                    v_ax2 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(196) // T.int64(14))
                    v_ax3 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(14))
                    T.reads(lv183[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] = lv183[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_subtract_2"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(512), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(196))
                    v_ax2 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(196) // T.int64(14))
                    v_ax3 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(14))
                    T.reads(lv183[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3] = lv183[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape_4[v_ax0, v_ax1, T.int64(0), T.int64(0)]
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply_4"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(512), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(196))
                    v_ax2 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(196) // T.int64(14))
                    v_ax3 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(14))
                    T.reads(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3], T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_multiply_4[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_multiply_4[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] * T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(512), thread="threadIdx.x"):
                for k0, k2, k3 in T.grid(T.int64(1), T.int64(14), T.int64(14)):
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
                    T_divide_2[v_ax0] = T_multiply_red[v_ax0] * T.float32(0.0051020408163265302)
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
    def conv2d1(lv4: T.Buffer((T.int64(1), T.int64(64), T.int64(56), T.int64(56)), "float32"), B: T.Buffer((T.int64(64), T.int64(64), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(64), T.int64(56), T.int64(56)), "float32")):
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
                    T.reads(lv4[v_i0, v_i1, v_i2, v_i3])
                    T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                    pad_temp[v_i0, v_i1, v_i2, v_i3] = lv4[v_i0, v_i1, v_i2, v_i3]
        for nn_ff_yy_xx_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
            for nn_ff_yy_xx_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for rc, ry, rx in T.grid(T.int64(64), T.int64(1), T.int64(1)):
                    with T.block("conv2d_nchw"):
                        v_nn = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ff = T.axis.spatial(T.int64(64), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) // T.int64(3136))
                        v_yy = T.axis.spatial(T.int64(56), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(3136) // T.int64(56))
                        v_xx = T.axis.spatial(T.int64(56), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(56))
                        v_rc = T.axis.reduce(T.int64(64), rc)
                        v_ry = T.axis.reduce(T.int64(1), T.int64(0))
                        v_rx = T.axis.reduce(T.int64(1), T.int64(0))
                        T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], B[v_ff, v_rc, v_ry, v_rx])
                        T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                        with T.init():
                            conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0.0)
                        conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * B[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func(private=True)
    def conv2d10(lv66: T.Buffer((T.int64(1), T.int64(128), T.int64(28), T.int64(28)), "float32"), B: T.Buffer((T.int64(128), T.int64(128), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(128), T.int64(28), T.int64(28)), "float32")):
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
                    T.reads(lv66[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                    T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                    pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(29) and T.int64(1) <= v_i3 and v_i3 < T.int64(29), lv66[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0.0))
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
    def conv2d11(lv101: T.Buffer((T.int64(1), T.int64(512), T.int64(28), T.int64(28)), "float32"), B: T.Buffer((T.int64(256), T.int64(512), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(256), T.int64(28), T.int64(28)), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(28), T.int64(28)))
        for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for i0_i1_i2_i3_fused_0 in range(T.int64(2)):
                    with T.block("pad_temp"):
                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i1 = T.axis.spatial(T.int64(512), (i0_i1_i2_i3_fused_0 * T.int64(262144) + i0_i1_i2_i3_fused_1 * T.int64(1024) + i0_i1_i2_i3_fused_2) // T.int64(784))
                        v_i2 = T.axis.spatial(T.int64(28), (i0_i1_i2_i3_fused_0 * T.int64(262144) + i0_i1_i2_i3_fused_1 * T.int64(1024) + i0_i1_i2_i3_fused_2) % T.int64(784) // T.int64(28))
                        v_i3 = T.axis.spatial(T.int64(28), (i0_i1_i2_i3_fused_0 * T.int64(262144) + i0_i1_i2_i3_fused_1 * T.int64(1024) + i0_i1_i2_i3_fused_2) % T.int64(28))
                        T.where((i0_i1_i2_i3_fused_0 * T.int64(256) + i0_i1_i2_i3_fused_1) * T.int64(1024) + i0_i1_i2_i3_fused_2 < T.int64(401408))
                        T.reads(lv101[v_i0, v_i1, v_i2, v_i3])
                        T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                        pad_temp[v_i0, v_i1, v_i2, v_i3] = lv101[v_i0, v_i1, v_i2, v_i3]
        for nn_ff_yy_xx_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
            for nn_ff_yy_xx_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for rc, ry, rx in T.grid(T.int64(512), T.int64(1), T.int64(1)):
                    with T.block("conv2d_nchw"):
                        v_nn = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ff = T.axis.spatial(T.int64(256), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) // T.int64(784))
                        v_yy = T.axis.spatial(T.int64(28), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(784) // T.int64(28))
                        v_xx = T.axis.spatial(T.int64(28), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(28))
                        v_rc = T.axis.reduce(T.int64(512), rc)
                        v_ry = T.axis.reduce(T.int64(1), T.int64(0))
                        v_rx = T.axis.reduce(T.int64(1), T.int64(0))
                        T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], B[v_ff, v_rc, v_ry, v_rx])
                        T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                        with T.init():
                            conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0.0)
                        conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * B[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func(private=True)
    def conv2d12(lv105: T.Buffer((T.int64(1), T.int64(256), T.int64(28), T.int64(28)), "float32"), B: T.Buffer((T.int64(256), T.int64(256), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(256), T.int64(14), T.int64(14)), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(30), T.int64(30)))
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(225), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("pad_temp"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(256), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) // T.int64(900))
                    v_i2 = T.axis.spatial(T.int64(30), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(900) // T.int64(30))
                    v_i3 = T.axis.spatial(T.int64(30), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(30))
                    T.reads(lv105[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                    T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                    pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(29) and T.int64(1) <= v_i3 and v_i3 < T.int64(29), lv105[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0.0))
        for nn_ff_yy_xx_fused_0 in T.thread_binding(T.int64(49), thread="blockIdx.x"):
            for nn_ff_yy_xx_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for rc, ry, rx in T.grid(T.int64(256), T.int64(3), T.int64(3)):
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
    def conv2d13(lv109: T.Buffer((T.int64(1), T.int64(256), T.int64(14), T.int64(14)), "float32"), B: T.Buffer((T.int64(1024), T.int64(256), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(1024), T.int64(14), T.int64(14)), "float32")):
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
                    T.reads(lv109[v_i0, v_i1, v_i2, v_i3])
                    T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                    pad_temp[v_i0, v_i1, v_i2, v_i3] = lv109[v_i0, v_i1, v_i2, v_i3]
        for nn_ff_yy_xx_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
            for nn_ff_yy_xx_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for rc, ry, rx in T.grid(T.int64(256), T.int64(1), T.int64(1)):
                    with T.block("conv2d_nchw"):
                        v_nn = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ff = T.axis.spatial(T.int64(1024), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) // T.int64(196))
                        v_yy = T.axis.spatial(T.int64(14), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(196) // T.int64(14))
                        v_xx = T.axis.spatial(T.int64(14), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(14))
                        v_rc = T.axis.reduce(T.int64(256), rc)
                        v_ry = T.axis.reduce(T.int64(1), T.int64(0))
                        v_rx = T.axis.reduce(T.int64(1), T.int64(0))
                        T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], B[v_ff, v_rc, v_ry, v_rx])
                        T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                        with T.init():
                            conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0.0)
                        conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * B[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func(private=True)
    def conv2d14(lv101: T.Buffer((T.int64(1), T.int64(512), T.int64(28), T.int64(28)), "float32"), B: T.Buffer((T.int64(1024), T.int64(512), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(1024), T.int64(14), T.int64(14)), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(28), T.int64(28)))
        for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for i0_i1_i2_i3_fused_0 in range(T.int64(2)):
                    with T.block("pad_temp"):
                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i1 = T.axis.spatial(T.int64(512), (i0_i1_i2_i3_fused_0 * T.int64(262144) + i0_i1_i2_i3_fused_1 * T.int64(1024) + i0_i1_i2_i3_fused_2) // T.int64(784))
                        v_i2 = T.axis.spatial(T.int64(28), (i0_i1_i2_i3_fused_0 * T.int64(262144) + i0_i1_i2_i3_fused_1 * T.int64(1024) + i0_i1_i2_i3_fused_2) % T.int64(784) // T.int64(28))
                        v_i3 = T.axis.spatial(T.int64(28), (i0_i1_i2_i3_fused_0 * T.int64(262144) + i0_i1_i2_i3_fused_1 * T.int64(1024) + i0_i1_i2_i3_fused_2) % T.int64(28))
                        T.where((i0_i1_i2_i3_fused_0 * T.int64(256) + i0_i1_i2_i3_fused_1) * T.int64(1024) + i0_i1_i2_i3_fused_2 < T.int64(401408))
                        T.reads(lv101[v_i0, v_i1, v_i2, v_i3])
                        T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                        pad_temp[v_i0, v_i1, v_i2, v_i3] = lv101[v_i0, v_i1, v_i2, v_i3]
        for nn_ff_yy_xx_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
            for nn_ff_yy_xx_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for rc, ry, rx in T.grid(T.int64(512), T.int64(1), T.int64(1)):
                    with T.block("conv2d_nchw"):
                        v_nn = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ff = T.axis.spatial(T.int64(1024), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) // T.int64(196))
                        v_yy = T.axis.spatial(T.int64(14), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(196) // T.int64(14))
                        v_xx = T.axis.spatial(T.int64(14), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(14))
                        v_rc = T.axis.reduce(T.int64(512), rc)
                        v_ry = T.axis.reduce(T.int64(1), T.int64(0))
                        v_rx = T.axis.reduce(T.int64(1), T.int64(0))
                        T.reads(pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx], B[v_ff, v_rc, v_ry, v_rx])
                        T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                        with T.init():
                            conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0.0)
                        conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx] * B[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func(private=True)
    def conv2d15(lv117: T.Buffer((T.int64(1), T.int64(1024), T.int64(14), T.int64(14)), "float32"), B: T.Buffer((T.int64(256), T.int64(1024), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(256), T.int64(14), T.int64(14)), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(1), T.int64(1024), T.int64(14), T.int64(14)))
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("pad_temp"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(1024), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) // T.int64(196))
                    v_i2 = T.axis.spatial(T.int64(14), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(196) // T.int64(14))
                    v_i3 = T.axis.spatial(T.int64(14), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(14))
                    T.reads(lv117[v_i0, v_i1, v_i2, v_i3])
                    T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                    pad_temp[v_i0, v_i1, v_i2, v_i3] = lv117[v_i0, v_i1, v_i2, v_i3]
        for nn_ff_yy_xx_fused_0 in T.thread_binding(T.int64(49), thread="blockIdx.x"):
            for nn_ff_yy_xx_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for rc, ry, rx in T.grid(T.int64(1024), T.int64(1), T.int64(1)):
                    with T.block("conv2d_nchw"):
                        v_nn = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ff = T.axis.spatial(T.int64(256), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) // T.int64(196))
                        v_yy = T.axis.spatial(T.int64(14), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(196) // T.int64(14))
                        v_xx = T.axis.spatial(T.int64(14), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(14))
                        v_rc = T.axis.reduce(T.int64(1024), rc)
                        v_ry = T.axis.reduce(T.int64(1), T.int64(0))
                        v_rx = T.axis.reduce(T.int64(1), T.int64(0))
                        T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], B[v_ff, v_rc, v_ry, v_rx])
                        T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                        with T.init():
                            conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0.0)
                        conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * B[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func(private=True)
    def conv2d16(lv121: T.Buffer((T.int64(1), T.int64(256), T.int64(14), T.int64(14)), "float32"), B: T.Buffer((T.int64(256), T.int64(256), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(256), T.int64(14), T.int64(14)), "float32")):
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
                    T.reads(lv121[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                    T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                    pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(15) and T.int64(1) <= v_i3 and v_i3 < T.int64(15), lv121[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0.0))
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
    def conv2d17(lv182: T.Buffer((T.int64(1), T.int64(1024), T.int64(14), T.int64(14)), "float32"), B: T.Buffer((T.int64(512), T.int64(1024), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(512), T.int64(14), T.int64(14)), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(1), T.int64(1024), T.int64(14), T.int64(14)))
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("pad_temp"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(1024), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) // T.int64(196))
                    v_i2 = T.axis.spatial(T.int64(14), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(196) // T.int64(14))
                    v_i3 = T.axis.spatial(T.int64(14), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(14))
                    T.reads(lv182[v_i0, v_i1, v_i2, v_i3])
                    T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                    pad_temp[v_i0, v_i1, v_i2, v_i3] = lv182[v_i0, v_i1, v_i2, v_i3]
        for nn_ff_yy_xx_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for nn_ff_yy_xx_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for rc, ry, rx in T.grid(T.int64(1024), T.int64(1), T.int64(1)):
                    with T.block("conv2d_nchw"):
                        v_nn = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ff = T.axis.spatial(T.int64(512), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) // T.int64(196))
                        v_yy = T.axis.spatial(T.int64(14), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(196) // T.int64(14))
                        v_xx = T.axis.spatial(T.int64(14), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(14))
                        v_rc = T.axis.reduce(T.int64(1024), rc)
                        v_ry = T.axis.reduce(T.int64(1), T.int64(0))
                        v_rx = T.axis.reduce(T.int64(1), T.int64(0))
                        T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], B[v_ff, v_rc, v_ry, v_rx])
                        T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                        with T.init():
                            conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0.0)
                        conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * B[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func(private=True)
    def conv2d18(lv186: T.Buffer((T.int64(1), T.int64(512), T.int64(14), T.int64(14)), "float32"), B: T.Buffer((T.int64(512), T.int64(512), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(512), T.int64(7), T.int64(7)), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(16), T.int64(16)))
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(128), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("pad_temp"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(512), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) // T.int64(256))
                    v_i2 = T.axis.spatial(T.int64(16), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(256) // T.int64(16))
                    v_i3 = T.axis.spatial(T.int64(16), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(16))
                    T.reads(lv186[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                    T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                    pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(15) and T.int64(1) <= v_i3 and v_i3 < T.int64(15), lv186[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0.0))
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
                        T.reads(pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx], B[v_ff, v_rc, v_ry, v_rx])
                        T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                        with T.init():
                            conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0.0)
                        conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx] * B[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func(private=True)
    def conv2d19(lv190: T.Buffer((T.int64(1), T.int64(512), T.int64(7), T.int64(7)), "float32"), B: T.Buffer((T.int64(2048), T.int64(512), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(2048), T.int64(7), T.int64(7)), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(7), T.int64(7)))
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(25), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("pad_temp"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(512), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) // T.int64(49))
                    v_i2 = T.axis.spatial(T.int64(7), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(49) // T.int64(7))
                    v_i3 = T.axis.spatial(T.int64(7), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(7))
                    T.where(i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1 < T.int64(25088))
                    T.reads(lv190[v_i0, v_i1, v_i2, v_i3])
                    T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                    pad_temp[v_i0, v_i1, v_i2, v_i3] = lv190[v_i0, v_i1, v_i2, v_i3]
        for nn_ff_yy_xx_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for nn_ff_yy_xx_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for rc, ry, rx in T.grid(T.int64(512), T.int64(1), T.int64(1)):
                    with T.block("conv2d_nchw"):
                        v_nn = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ff = T.axis.spatial(T.int64(2048), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) // T.int64(49))
                        v_yy = T.axis.spatial(T.int64(7), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(49) // T.int64(7))
                        v_xx = T.axis.spatial(T.int64(7), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(7))
                        v_rc = T.axis.reduce(T.int64(512), rc)
                        v_ry = T.axis.reduce(T.int64(1), T.int64(0))
                        v_rx = T.axis.reduce(T.int64(1), T.int64(0))
                        T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], B[v_ff, v_rc, v_ry, v_rx])
                        T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                        with T.init():
                            conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0.0)
                        conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * B[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func(private=True)
    def conv2d2(lv8: T.Buffer((T.int64(1), T.int64(64), T.int64(56), T.int64(56)), "float32"), B: T.Buffer((T.int64(64), T.int64(64), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(64), T.int64(56), T.int64(56)), "float32")):
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
                    T.reads(lv8[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                    T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                    pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(57) and T.int64(1) <= v_i3 and v_i3 < T.int64(57), lv8[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0.0))
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
    def conv2d20(lv182: T.Buffer((T.int64(1), T.int64(1024), T.int64(14), T.int64(14)), "float32"), B: T.Buffer((T.int64(2048), T.int64(1024), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(2048), T.int64(7), T.int64(7)), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(1), T.int64(1024), T.int64(14), T.int64(14)))
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("pad_temp"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(1024), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) // T.int64(196))
                    v_i2 = T.axis.spatial(T.int64(14), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(196) // T.int64(14))
                    v_i3 = T.axis.spatial(T.int64(14), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(14))
                    T.reads(lv182[v_i0, v_i1, v_i2, v_i3])
                    T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                    pad_temp[v_i0, v_i1, v_i2, v_i3] = lv182[v_i0, v_i1, v_i2, v_i3]
        for nn_ff_yy_xx_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for nn_ff_yy_xx_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for rc, ry, rx in T.grid(T.int64(1024), T.int64(1), T.int64(1)):
                    with T.block("conv2d_nchw"):
                        v_nn = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ff = T.axis.spatial(T.int64(2048), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) // T.int64(49))
                        v_yy = T.axis.spatial(T.int64(7), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(49) // T.int64(7))
                        v_xx = T.axis.spatial(T.int64(7), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(7))
                        v_rc = T.axis.reduce(T.int64(1024), rc)
                        v_ry = T.axis.reduce(T.int64(1), T.int64(0))
                        v_rx = T.axis.reduce(T.int64(1), T.int64(0))
                        T.reads(pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx], B[v_ff, v_rc, v_ry, v_rx])
                        T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                        with T.init():
                            conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0.0)
                        conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx] * B[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func(private=True)
    def conv2d21(lv198: T.Buffer((T.int64(1), T.int64(2048), T.int64(7), T.int64(7)), "float32"), B: T.Buffer((T.int64(512), T.int64(2048), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(512), T.int64(7), T.int64(7)), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(1), T.int64(2048), T.int64(7), T.int64(7)))
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("pad_temp"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(2048), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) // T.int64(49))
                    v_i2 = T.axis.spatial(T.int64(7), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(49) // T.int64(7))
                    v_i3 = T.axis.spatial(T.int64(7), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(7))
                    T.reads(lv198[v_i0, v_i1, v_i2, v_i3])
                    T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                    pad_temp[v_i0, v_i1, v_i2, v_i3] = lv198[v_i0, v_i1, v_i2, v_i3]
        for nn_ff_yy_xx_fused_0 in T.thread_binding(T.int64(25), thread="blockIdx.x"):
            for nn_ff_yy_xx_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for rc, ry, rx in T.grid(T.int64(2048), T.int64(1), T.int64(1)):
                    with T.block("conv2d_nchw"):
                        v_nn = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ff = T.axis.spatial(T.int64(512), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) // T.int64(49))
                        v_yy = T.axis.spatial(T.int64(7), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(49) // T.int64(7))
                        v_xx = T.axis.spatial(T.int64(7), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(7))
                        v_rc = T.axis.reduce(T.int64(2048), rc)
                        v_ry = T.axis.reduce(T.int64(1), T.int64(0))
                        v_rx = T.axis.reduce(T.int64(1), T.int64(0))
                        T.where(nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1 < T.int64(25088))
                        T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], B[v_ff, v_rc, v_ry, v_rx])
                        T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                        with T.init():
                            conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0.0)
                        conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * B[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func(private=True)
    def conv2d22(lv202: T.Buffer((T.int64(1), T.int64(512), T.int64(7), T.int64(7)), "float32"), B: T.Buffer((T.int64(512), T.int64(512), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(512), T.int64(7), T.int64(7)), "float32")):
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
                    T.reads(lv202[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                    T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                    pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(8) and T.int64(1) <= v_i3 and v_i3 < T.int64(8), lv202[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0.0))
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
    def conv2d3(lv12: T.Buffer((T.int64(1), T.int64(64), T.int64(56), T.int64(56)), "float32"), B: T.Buffer((T.int64(256), T.int64(64), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(256), T.int64(56), T.int64(56)), "float32")):
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
                    T.reads(lv12[v_i0, v_i1, v_i2, v_i3])
                    T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                    pad_temp[v_i0, v_i1, v_i2, v_i3] = lv12[v_i0, v_i1, v_i2, v_i3]
        for nn_ff_yy_xx_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for nn_ff_yy_xx_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for nn_ff_yy_xx_fused_0, rc, ry, rx in T.grid(T.int64(4), T.int64(64), T.int64(1), T.int64(1)):
                    with T.block("conv2d_nchw"):
                        v_nn = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ff = T.axis.spatial(T.int64(256), (nn_ff_yy_xx_fused_0 * T.int64(262144) + nn_ff_yy_xx_fused_1 * T.int64(1024) + nn_ff_yy_xx_fused_2) // T.int64(3136))
                        v_yy = T.axis.spatial(T.int64(56), (nn_ff_yy_xx_fused_0 * T.int64(262144) + nn_ff_yy_xx_fused_1 * T.int64(1024) + nn_ff_yy_xx_fused_2) % T.int64(3136) // T.int64(56))
                        v_xx = T.axis.spatial(T.int64(56), (nn_ff_yy_xx_fused_0 * T.int64(262144) + nn_ff_yy_xx_fused_1 * T.int64(1024) + nn_ff_yy_xx_fused_2) % T.int64(56))
                        v_rc = T.axis.reduce(T.int64(64), rc)
                        v_ry = T.axis.reduce(T.int64(1), T.int64(0))
                        v_rx = T.axis.reduce(T.int64(1), T.int64(0))
                        T.where((nn_ff_yy_xx_fused_0 * T.int64(256) + nn_ff_yy_xx_fused_1) * T.int64(1024) + nn_ff_yy_xx_fused_2 < T.int64(802816))
                        T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], B[v_ff, v_rc, v_ry, v_rx])
                        T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                        with T.init():
                            conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0.0)
                        conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * B[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func(private=True)
    def conv2d4(lv20: T.Buffer((T.int64(1), T.int64(256), T.int64(56), T.int64(56)), "float32"), B: T.Buffer((T.int64(64), T.int64(256), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(64), T.int64(56), T.int64(56)), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(56), T.int64(56)))
        for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for i0_i1_i2_i3_fused_0 in range(T.int64(4)):
                    with T.block("pad_temp"):
                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i1 = T.axis.spatial(T.int64(256), (i0_i1_i2_i3_fused_0 * T.int64(262144) + i0_i1_i2_i3_fused_1 * T.int64(1024) + i0_i1_i2_i3_fused_2) // T.int64(3136))
                        v_i2 = T.axis.spatial(T.int64(56), (i0_i1_i2_i3_fused_0 * T.int64(262144) + i0_i1_i2_i3_fused_1 * T.int64(1024) + i0_i1_i2_i3_fused_2) % T.int64(3136) // T.int64(56))
                        v_i3 = T.axis.spatial(T.int64(56), (i0_i1_i2_i3_fused_0 * T.int64(262144) + i0_i1_i2_i3_fused_1 * T.int64(1024) + i0_i1_i2_i3_fused_2) % T.int64(56))
                        T.where((i0_i1_i2_i3_fused_0 * T.int64(256) + i0_i1_i2_i3_fused_1) * T.int64(1024) + i0_i1_i2_i3_fused_2 < T.int64(802816))
                        T.reads(lv20[v_i0, v_i1, v_i2, v_i3])
                        T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                        pad_temp[v_i0, v_i1, v_i2, v_i3] = lv20[v_i0, v_i1, v_i2, v_i3]
        for nn_ff_yy_xx_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
            for nn_ff_yy_xx_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for rc, ry, rx in T.grid(T.int64(256), T.int64(1), T.int64(1)):
                    with T.block("conv2d_nchw"):
                        v_nn = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ff = T.axis.spatial(T.int64(64), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) // T.int64(3136))
                        v_yy = T.axis.spatial(T.int64(56), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(3136) // T.int64(56))
                        v_xx = T.axis.spatial(T.int64(56), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(56))
                        v_rc = T.axis.reduce(T.int64(256), rc)
                        v_ry = T.axis.reduce(T.int64(1), T.int64(0))
                        v_rx = T.axis.reduce(T.int64(1), T.int64(0))
                        T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], B[v_ff, v_rc, v_ry, v_rx])
                        T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                        with T.init():
                            conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0.0)
                        conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * B[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func(private=True)
    def conv2d5(lv46: T.Buffer((T.int64(1), T.int64(256), T.int64(56), T.int64(56)), "float32"), B: T.Buffer((T.int64(128), T.int64(256), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(128), T.int64(56), T.int64(56)), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(56), T.int64(56)))
        for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for i0_i1_i2_i3_fused_0 in range(T.int64(4)):
                    with T.block("pad_temp"):
                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i1 = T.axis.spatial(T.int64(256), (i0_i1_i2_i3_fused_0 * T.int64(262144) + i0_i1_i2_i3_fused_1 * T.int64(1024) + i0_i1_i2_i3_fused_2) // T.int64(3136))
                        v_i2 = T.axis.spatial(T.int64(56), (i0_i1_i2_i3_fused_0 * T.int64(262144) + i0_i1_i2_i3_fused_1 * T.int64(1024) + i0_i1_i2_i3_fused_2) % T.int64(3136) // T.int64(56))
                        v_i3 = T.axis.spatial(T.int64(56), (i0_i1_i2_i3_fused_0 * T.int64(262144) + i0_i1_i2_i3_fused_1 * T.int64(1024) + i0_i1_i2_i3_fused_2) % T.int64(56))
                        T.where((i0_i1_i2_i3_fused_0 * T.int64(256) + i0_i1_i2_i3_fused_1) * T.int64(1024) + i0_i1_i2_i3_fused_2 < T.int64(802816))
                        T.reads(lv46[v_i0, v_i1, v_i2, v_i3])
                        T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                        pad_temp[v_i0, v_i1, v_i2, v_i3] = lv46[v_i0, v_i1, v_i2, v_i3]
        for nn_ff_yy_xx_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for nn_ff_yy_xx_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for nn_ff_yy_xx_fused_0, rc, ry, rx in T.grid(T.int64(2), T.int64(256), T.int64(1), T.int64(1)):
                    with T.block("conv2d_nchw"):
                        v_nn = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ff = T.axis.spatial(T.int64(128), (nn_ff_yy_xx_fused_0 * T.int64(262144) + nn_ff_yy_xx_fused_1 * T.int64(1024) + nn_ff_yy_xx_fused_2) // T.int64(3136))
                        v_yy = T.axis.spatial(T.int64(56), (nn_ff_yy_xx_fused_0 * T.int64(262144) + nn_ff_yy_xx_fused_1 * T.int64(1024) + nn_ff_yy_xx_fused_2) % T.int64(3136) // T.int64(56))
                        v_xx = T.axis.spatial(T.int64(56), (nn_ff_yy_xx_fused_0 * T.int64(262144) + nn_ff_yy_xx_fused_1 * T.int64(1024) + nn_ff_yy_xx_fused_2) % T.int64(56))
                        v_rc = T.axis.reduce(T.int64(256), rc)
                        v_ry = T.axis.reduce(T.int64(1), T.int64(0))
                        v_rx = T.axis.reduce(T.int64(1), T.int64(0))
                        T.where((nn_ff_yy_xx_fused_0 * T.int64(256) + nn_ff_yy_xx_fused_1) * T.int64(1024) + nn_ff_yy_xx_fused_2 < T.int64(401408))
                        T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], B[v_ff, v_rc, v_ry, v_rx])
                        T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                        with T.init():
                            conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0.0)
                        conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * B[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func(private=True)
    def conv2d6(lv50: T.Buffer((T.int64(1), T.int64(128), T.int64(56), T.int64(56)), "float32"), B: T.Buffer((T.int64(128), T.int64(128), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(128), T.int64(28), T.int64(28)), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(58), T.int64(58)))
        for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for i0_i1_i2_i3_fused_0 in range(T.int64(2)):
                    with T.block("pad_temp"):
                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i1 = T.axis.spatial(T.int64(128), (i0_i1_i2_i3_fused_0 * T.int64(262144) + i0_i1_i2_i3_fused_1 * T.int64(1024) + i0_i1_i2_i3_fused_2) // T.int64(3364))
                        v_i2 = T.axis.spatial(T.int64(58), (i0_i1_i2_i3_fused_0 * T.int64(262144) + i0_i1_i2_i3_fused_1 * T.int64(1024) + i0_i1_i2_i3_fused_2) % T.int64(3364) // T.int64(58))
                        v_i3 = T.axis.spatial(T.int64(58), (i0_i1_i2_i3_fused_0 * T.int64(262144) + i0_i1_i2_i3_fused_1 * T.int64(1024) + i0_i1_i2_i3_fused_2) % T.int64(58))
                        T.where((i0_i1_i2_i3_fused_0 * T.int64(256) + i0_i1_i2_i3_fused_1) * T.int64(1024) + i0_i1_i2_i3_fused_2 < T.int64(430592))
                        T.reads(lv50[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                        T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                        pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(57) and T.int64(1) <= v_i3 and v_i3 < T.int64(57), lv50[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0.0))
        for nn_ff_yy_xx_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for nn_ff_yy_xx_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for rc, ry, rx in T.grid(T.int64(128), T.int64(3), T.int64(3)):
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
    def conv2d7(lv54: T.Buffer((T.int64(1), T.int64(128), T.int64(28), T.int64(28)), "float32"), B: T.Buffer((T.int64(512), T.int64(128), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(512), T.int64(28), T.int64(28)), "float32")):
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
                    T.reads(lv54[v_i0, v_i1, v_i2, v_i3])
                    T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                    pad_temp[v_i0, v_i1, v_i2, v_i3] = lv54[v_i0, v_i1, v_i2, v_i3]
        for nn_ff_yy_xx_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for nn_ff_yy_xx_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for nn_ff_yy_xx_fused_0, rc, ry, rx in T.grid(T.int64(2), T.int64(128), T.int64(1), T.int64(1)):
                    with T.block("conv2d_nchw"):
                        v_nn = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ff = T.axis.spatial(T.int64(512), (nn_ff_yy_xx_fused_0 * T.int64(262144) + nn_ff_yy_xx_fused_1 * T.int64(1024) + nn_ff_yy_xx_fused_2) // T.int64(784))
                        v_yy = T.axis.spatial(T.int64(28), (nn_ff_yy_xx_fused_0 * T.int64(262144) + nn_ff_yy_xx_fused_1 * T.int64(1024) + nn_ff_yy_xx_fused_2) % T.int64(784) // T.int64(28))
                        v_xx = T.axis.spatial(T.int64(28), (nn_ff_yy_xx_fused_0 * T.int64(262144) + nn_ff_yy_xx_fused_1 * T.int64(1024) + nn_ff_yy_xx_fused_2) % T.int64(28))
                        v_rc = T.axis.reduce(T.int64(128), rc)
                        v_ry = T.axis.reduce(T.int64(1), T.int64(0))
                        v_rx = T.axis.reduce(T.int64(1), T.int64(0))
                        T.where((nn_ff_yy_xx_fused_0 * T.int64(256) + nn_ff_yy_xx_fused_1) * T.int64(1024) + nn_ff_yy_xx_fused_2 < T.int64(401408))
                        T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], B[v_ff, v_rc, v_ry, v_rx])
                        T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                        with T.init():
                            conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0.0)
                        conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * B[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func(private=True)
    def conv2d8(lv46: T.Buffer((T.int64(1), T.int64(256), T.int64(56), T.int64(56)), "float32"), B: T.Buffer((T.int64(512), T.int64(256), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(512), T.int64(28), T.int64(28)), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(56), T.int64(56)))
        for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for i0_i1_i2_i3_fused_0 in range(T.int64(4)):
                    with T.block("pad_temp"):
                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i1 = T.axis.spatial(T.int64(256), (i0_i1_i2_i3_fused_0 * T.int64(262144) + i0_i1_i2_i3_fused_1 * T.int64(1024) + i0_i1_i2_i3_fused_2) // T.int64(3136))
                        v_i2 = T.axis.spatial(T.int64(56), (i0_i1_i2_i3_fused_0 * T.int64(262144) + i0_i1_i2_i3_fused_1 * T.int64(1024) + i0_i1_i2_i3_fused_2) % T.int64(3136) // T.int64(56))
                        v_i3 = T.axis.spatial(T.int64(56), (i0_i1_i2_i3_fused_0 * T.int64(262144) + i0_i1_i2_i3_fused_1 * T.int64(1024) + i0_i1_i2_i3_fused_2) % T.int64(56))
                        T.where((i0_i1_i2_i3_fused_0 * T.int64(256) + i0_i1_i2_i3_fused_1) * T.int64(1024) + i0_i1_i2_i3_fused_2 < T.int64(802816))
                        T.reads(lv46[v_i0, v_i1, v_i2, v_i3])
                        T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                        pad_temp[v_i0, v_i1, v_i2, v_i3] = lv46[v_i0, v_i1, v_i2, v_i3]
        for nn_ff_yy_xx_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for nn_ff_yy_xx_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for nn_ff_yy_xx_fused_0, rc, ry, rx in T.grid(T.int64(2), T.int64(256), T.int64(1), T.int64(1)):
                    with T.block("conv2d_nchw"):
                        v_nn = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ff = T.axis.spatial(T.int64(512), (nn_ff_yy_xx_fused_0 * T.int64(262144) + nn_ff_yy_xx_fused_1 * T.int64(1024) + nn_ff_yy_xx_fused_2) // T.int64(784))
                        v_yy = T.axis.spatial(T.int64(28), (nn_ff_yy_xx_fused_0 * T.int64(262144) + nn_ff_yy_xx_fused_1 * T.int64(1024) + nn_ff_yy_xx_fused_2) % T.int64(784) // T.int64(28))
                        v_xx = T.axis.spatial(T.int64(28), (nn_ff_yy_xx_fused_0 * T.int64(262144) + nn_ff_yy_xx_fused_1 * T.int64(1024) + nn_ff_yy_xx_fused_2) % T.int64(28))
                        v_rc = T.axis.reduce(T.int64(256), rc)
                        v_ry = T.axis.reduce(T.int64(1), T.int64(0))
                        v_rx = T.axis.reduce(T.int64(1), T.int64(0))
                        T.where((nn_ff_yy_xx_fused_0 * T.int64(256) + nn_ff_yy_xx_fused_1) * T.int64(1024) + nn_ff_yy_xx_fused_2 < T.int64(401408))
                        T.reads(pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx], B[v_ff, v_rc, v_ry, v_rx])
                        T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                        with T.init():
                            conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0.0)
                        conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx] * B[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func(private=True)
    def conv2d9(lv62: T.Buffer((T.int64(1), T.int64(512), T.int64(28), T.int64(28)), "float32"), B: T.Buffer((T.int64(128), T.int64(512), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(1), T.int64(128), T.int64(28), T.int64(28)), "float32")):
        T.func_attr({"op_pattern": 4, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        pad_temp = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(28), T.int64(28)))
        for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for i0_i1_i2_i3_fused_0 in range(T.int64(2)):
                    with T.block("pad_temp"):
                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i1 = T.axis.spatial(T.int64(512), (i0_i1_i2_i3_fused_0 * T.int64(262144) + i0_i1_i2_i3_fused_1 * T.int64(1024) + i0_i1_i2_i3_fused_2) // T.int64(784))
                        v_i2 = T.axis.spatial(T.int64(28), (i0_i1_i2_i3_fused_0 * T.int64(262144) + i0_i1_i2_i3_fused_1 * T.int64(1024) + i0_i1_i2_i3_fused_2) % T.int64(784) // T.int64(28))
                        v_i3 = T.axis.spatial(T.int64(28), (i0_i1_i2_i3_fused_0 * T.int64(262144) + i0_i1_i2_i3_fused_1 * T.int64(1024) + i0_i1_i2_i3_fused_2) % T.int64(28))
                        T.where((i0_i1_i2_i3_fused_0 * T.int64(256) + i0_i1_i2_i3_fused_1) * T.int64(1024) + i0_i1_i2_i3_fused_2 < T.int64(401408))
                        T.reads(lv62[v_i0, v_i1, v_i2, v_i3])
                        T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                        pad_temp[v_i0, v_i1, v_i2, v_i3] = lv62[v_i0, v_i1, v_i2, v_i3]
        for nn_ff_yy_xx_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for nn_ff_yy_xx_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for rc, ry, rx in T.grid(T.int64(512), T.int64(1), T.int64(1)):
                    with T.block("conv2d_nchw"):
                        v_nn = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ff = T.axis.spatial(T.int64(128), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) // T.int64(784))
                        v_yy = T.axis.spatial(T.int64(28), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(784) // T.int64(28))
                        v_xx = T.axis.spatial(T.int64(28), (nn_ff_yy_xx_fused_0 * T.int64(1024) + nn_ff_yy_xx_fused_1) % T.int64(28))
                        v_rc = T.axis.reduce(T.int64(512), rc)
                        v_ry = T.axis.reduce(T.int64(1), T.int64(0))
                        v_rx = T.axis.reduce(T.int64(1), T.int64(0))
                        T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], B[v_ff, v_rc, v_ry, v_rx])
                        T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
                        with T.init():
                            conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0.0)
                        conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = conv2d_nchw[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * B[v_ff, v_rc, v_ry, v_rx]

    @T.prim_func(private=True)
    def fused_add1_relu5(lv56_0: T.Buffer((T.int64(1), T.int64(512), T.int64(28), T.int64(28)), "float32"), lv59_0: T.Buffer((T.int64(1), T.int64(512), T.int64(28), T.int64(28)), "float32"), compute_intermediate: T.Buffer((T.int64(1), T.int64(512), T.int64(28), T.int64(28)), "float32")):
        T.func_attr({"target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(28), T.int64(28)))
        for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(2)):
                    with T.block("T_add"):
                        v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = T.axis.spatial(T.int64(512), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) // T.int64(784))
                        v_ax2 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(784) // T.int64(28))
                        v_ax3 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(28))
                        T.where((ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1) * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2 < T.int64(401408))
                        T.reads(lv56_0[v_ax0, v_ax1, v_ax2, v_ax3], lv59_0[v_ax0, v_ax1, v_ax2, v_ax3])
                        T.writes(T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                        T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv56_0[v_ax0, v_ax1, v_ax2, v_ax3] + lv59_0[v_ax0, v_ax1, v_ax2, v_ax3]
        for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for i0_i1_i2_i3_fused_0 in range(T.int64(2)):
                    with T.block("compute"):
                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i1 = T.axis.spatial(T.int64(512), (i0_i1_i2_i3_fused_0 * T.int64(262144) + i0_i1_i2_i3_fused_1 * T.int64(1024) + i0_i1_i2_i3_fused_2) // T.int64(784))
                        v_i2 = T.axis.spatial(T.int64(28), (i0_i1_i2_i3_fused_0 * T.int64(262144) + i0_i1_i2_i3_fused_1 * T.int64(1024) + i0_i1_i2_i3_fused_2) % T.int64(784) // T.int64(28))
                        v_i3 = T.axis.spatial(T.int64(28), (i0_i1_i2_i3_fused_0 * T.int64(262144) + i0_i1_i2_i3_fused_1 * T.int64(1024) + i0_i1_i2_i3_fused_2) % T.int64(28))
                        T.where((i0_i1_i2_i3_fused_0 * T.int64(256) + i0_i1_i2_i3_fused_1) * T.int64(1024) + i0_i1_i2_i3_fused_2 < T.int64(401408))
                        T.reads(T_add_intermediate[v_i0, v_i1, v_i2, v_i3])
                        T.writes(compute_intermediate[v_i0, v_i1, v_i2, v_i3])
                        compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.max(T_add_intermediate[v_i0, v_i1, v_i2, v_i3], T.float32(0.0))

    @T.prim_func(private=True)
    def fused_add2_relu8(lv111_0: T.Buffer((T.int64(1), T.int64(1024), T.int64(14), T.int64(14)), "float32"), lv114_0: T.Buffer((T.int64(1), T.int64(1024), T.int64(14), T.int64(14)), "float32"), compute_intermediate: T.Buffer((T.int64(1), T.int64(1024), T.int64(14), T.int64(14)), "float32")):
        T.func_attr({"target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(1024), T.int64(14), T.int64(14)))
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(1024), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(196))
                    v_ax2 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(196) // T.int64(14))
                    v_ax3 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(14))
                    T.reads(lv111_0[v_ax0, v_ax1, v_ax2, v_ax3], lv114_0[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv111_0[v_ax0, v_ax1, v_ax2, v_ax3] + lv114_0[v_ax0, v_ax1, v_ax2, v_ax3]
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(1024), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) // T.int64(196))
                    v_i2 = T.axis.spatial(T.int64(14), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(196) // T.int64(14))
                    v_i3 = T.axis.spatial(T.int64(14), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(14))
                    T.reads(T_add_intermediate[v_i0, v_i1, v_i2, v_i3])
                    T.writes(compute_intermediate[v_i0, v_i1, v_i2, v_i3])
                    compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.max(T_add_intermediate[v_i0, v_i1, v_i2, v_i3], T.float32(0.0))

    @T.prim_func(private=True)
    def fused_add3_relu11(lv192_0: T.Buffer((T.int64(1), T.int64(2048), T.int64(7), T.int64(7)), "float32"), lv195_0: T.Buffer((T.int64(1), T.int64(2048), T.int64(7), T.int64(7)), "float32"), compute_intermediate: T.Buffer((T.int64(1), T.int64(2048), T.int64(7), T.int64(7)), "float32")):
        T.func_attr({"target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(2048), T.int64(7), T.int64(7)))
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(2048), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(49))
                    v_ax2 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(49) // T.int64(7))
                    v_ax3 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(7))
                    T.reads(lv192_0[v_ax0, v_ax1, v_ax2, v_ax3], lv195_0[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv192_0[v_ax0, v_ax1, v_ax2, v_ax3] + lv195_0[v_ax0, v_ax1, v_ax2, v_ax3]
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(2048), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) // T.int64(49))
                    v_i2 = T.axis.spatial(T.int64(7), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(49) // T.int64(7))
                    v_i3 = T.axis.spatial(T.int64(7), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(7))
                    T.reads(T_add_intermediate[v_i0, v_i1, v_i2, v_i3])
                    T.writes(compute_intermediate[v_i0, v_i1, v_i2, v_i3])
                    compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.max(T_add_intermediate[v_i0, v_i1, v_i2, v_i3], T.float32(0.0))

    @T.prim_func(private=True)
    def fused_add_relu2(lv14_0: T.Buffer((T.int64(1), T.int64(256), T.int64(56), T.int64(56)), "float32"), lv17_0: T.Buffer((T.int64(1), T.int64(256), T.int64(56), T.int64(56)), "float32"), compute_intermediate: T.Buffer((T.int64(1), T.int64(256), T.int64(56), T.int64(56)), "float32")):
        T.func_attr({"target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(56), T.int64(56)))
        for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(4)):
                    with T.block("T_add"):
                        v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_ax1 = T.axis.spatial(T.int64(256), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) // T.int64(3136))
                        v_ax2 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(3136) // T.int64(56))
                        v_ax3 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(56))
                        T.where((ax0_ax1_ax2_ax3_fused_0 * T.int64(256) + ax0_ax1_ax2_ax3_fused_1) * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2 < T.int64(802816))
                        T.reads(lv14_0[v_ax0, v_ax1, v_ax2, v_ax3], lv17_0[v_ax0, v_ax1, v_ax2, v_ax3])
                        T.writes(T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                        T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv14_0[v_ax0, v_ax1, v_ax2, v_ax3] + lv17_0[v_ax0, v_ax1, v_ax2, v_ax3]
        for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for i0_i1_i2_i3_fused_0 in range(T.int64(4)):
                    with T.block("compute"):
                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i1 = T.axis.spatial(T.int64(256), (i0_i1_i2_i3_fused_0 * T.int64(262144) + i0_i1_i2_i3_fused_1 * T.int64(1024) + i0_i1_i2_i3_fused_2) // T.int64(3136))
                        v_i2 = T.axis.spatial(T.int64(56), (i0_i1_i2_i3_fused_0 * T.int64(262144) + i0_i1_i2_i3_fused_1 * T.int64(1024) + i0_i1_i2_i3_fused_2) % T.int64(3136) // T.int64(56))
                        v_i3 = T.axis.spatial(T.int64(56), (i0_i1_i2_i3_fused_0 * T.int64(262144) + i0_i1_i2_i3_fused_1 * T.int64(1024) + i0_i1_i2_i3_fused_2) % T.int64(56))
                        T.where((i0_i1_i2_i3_fused_0 * T.int64(256) + i0_i1_i2_i3_fused_1) * T.int64(1024) + i0_i1_i2_i3_fused_2 < T.int64(802816))
                        T.reads(T_add_intermediate[v_i0, v_i1, v_i2, v_i3])
                        T.writes(compute_intermediate[v_i0, v_i1, v_i2, v_i3])
                        compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.max(T_add_intermediate[v_i0, v_i1, v_i2, v_i3], T.float32(0.0))

    @T.prim_func(private=True)
    def fused_matmul_add4(lv226: T.Buffer((T.int64(1), T.int64(2048)), "float32"), lv227: T.Buffer((T.int64(2048), T.int64(1000)), "float32"), param_0: T.Buffer((T.int64(1000),), "float32"), T_add_intermediate: T.Buffer((T.int64(1), T.int64(1000)), "float32")):
        T.func_attr({"target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1000)))
        for i0_i1_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for i0_i1_fused_1 in T.thread_binding(T.int64(1000), thread="threadIdx.x"):
                for k in range(T.int64(2048)):
                    with T.block("matmul"):
                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i1 = T.axis.spatial(T.int64(1000), i0_i1_fused_0 * T.int64(1000) + i0_i1_fused_1)
                        v_k = T.axis.reduce(T.int64(2048), k)
                        T.reads(lv226[v_i0, v_k], lv227[v_k, v_i1])
                        T.writes(matmul_intermediate[v_i0, v_i1])
                        with T.init():
                            matmul_intermediate[v_i0, v_i1] = T.float32(0.0)
                        matmul_intermediate[v_i0, v_i1] = matmul_intermediate[v_i0, v_i1] + lv226[v_i0, v_k] * lv227[v_k, v_i1]
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
    def fused_relu10(lv188_0: T.Buffer((T.int64(1), T.int64(512), T.int64(7), T.int64(7)), "float32"), compute_intermediate: T.Buffer((T.int64(1), T.int64(512), T.int64(7), T.int64(7)), "float32")):
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
                    T.reads(lv188_0[v_i0, v_i1, v_i2, v_i3])
                    T.writes(compute_intermediate[v_i0, v_i1, v_i2, v_i3])
                    compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.max(lv188_0[v_i0, v_i1, v_i2, v_i3], T.float32(0.0))

    @T.prim_func(private=True)
    def fused_relu3(lv48_0: T.Buffer((T.int64(1), T.int64(128), T.int64(56), T.int64(56)), "float32"), compute_intermediate: T.Buffer((T.int64(1), T.int64(128), T.int64(56), T.int64(56)), "float32")):
        T.func_attr({"target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for i0_i1_i2_i3_fused_0 in range(T.int64(2)):
                    with T.block("compute"):
                        v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                        v_i1 = T.axis.spatial(T.int64(128), (i0_i1_i2_i3_fused_0 * T.int64(262144) + i0_i1_i2_i3_fused_1 * T.int64(1024) + i0_i1_i2_i3_fused_2) // T.int64(3136))
                        v_i2 = T.axis.spatial(T.int64(56), (i0_i1_i2_i3_fused_0 * T.int64(262144) + i0_i1_i2_i3_fused_1 * T.int64(1024) + i0_i1_i2_i3_fused_2) % T.int64(3136) // T.int64(56))
                        v_i3 = T.axis.spatial(T.int64(56), (i0_i1_i2_i3_fused_0 * T.int64(262144) + i0_i1_i2_i3_fused_1 * T.int64(1024) + i0_i1_i2_i3_fused_2) % T.int64(56))
                        T.where((i0_i1_i2_i3_fused_0 * T.int64(256) + i0_i1_i2_i3_fused_1) * T.int64(1024) + i0_i1_i2_i3_fused_2 < T.int64(401408))
                        T.reads(lv48_0[v_i0, v_i1, v_i2, v_i3])
                        T.writes(compute_intermediate[v_i0, v_i1, v_i2, v_i3])
                        compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.max(lv48_0[v_i0, v_i1, v_i2, v_i3], T.float32(0.0))

    @T.prim_func(private=True)
    def fused_relu4(lv52_0: T.Buffer((T.int64(1), T.int64(128), T.int64(28), T.int64(28)), "float32"), compute_intermediate: T.Buffer((T.int64(1), T.int64(128), T.int64(28), T.int64(28)), "float32")):
        T.func_attr({"target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(128), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) // T.int64(784))
                    v_i2 = T.axis.spatial(T.int64(28), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(784) // T.int64(28))
                    v_i3 = T.axis.spatial(T.int64(28), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(28))
                    T.reads(lv52_0[v_i0, v_i1, v_i2, v_i3])
                    T.writes(compute_intermediate[v_i0, v_i1, v_i2, v_i3])
                    compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.max(lv52_0[v_i0, v_i1, v_i2, v_i3], T.float32(0.0))

    @T.prim_func(private=True)
    def fused_relu6(lv103_0: T.Buffer((T.int64(1), T.int64(256), T.int64(28), T.int64(28)), "float32"), compute_intermediate: T.Buffer((T.int64(1), T.int64(256), T.int64(28), T.int64(28)), "float32")):
        T.func_attr({"target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(196), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(256), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) // T.int64(784))
                    v_i2 = T.axis.spatial(T.int64(28), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(784) // T.int64(28))
                    v_i3 = T.axis.spatial(T.int64(28), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(28))
                    T.reads(lv103_0[v_i0, v_i1, v_i2, v_i3])
                    T.writes(compute_intermediate[v_i0, v_i1, v_i2, v_i3])
                    compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.max(lv103_0[v_i0, v_i1, v_i2, v_i3], T.float32(0.0))

    @T.prim_func(private=True)
    def fused_relu7(lv107_0: T.Buffer((T.int64(1), T.int64(256), T.int64(14), T.int64(14)), "float32"), compute_intermediate: T.Buffer((T.int64(1), T.int64(256), T.int64(14), T.int64(14)), "float32")):
        T.func_attr({"target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(49), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(256), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) // T.int64(196))
                    v_i2 = T.axis.spatial(T.int64(14), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(196) // T.int64(14))
                    v_i3 = T.axis.spatial(T.int64(14), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(14))
                    T.reads(lv107_0[v_i0, v_i1, v_i2, v_i3])
                    T.writes(compute_intermediate[v_i0, v_i1, v_i2, v_i3])
                    compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.max(lv107_0[v_i0, v_i1, v_i2, v_i3], T.float32(0.0))

    @T.prim_func(private=True)
    def fused_relu9(lv184_0: T.Buffer((T.int64(1), T.int64(512), T.int64(14), T.int64(14)), "float32"), compute_intermediate: T.Buffer((T.int64(1), T.int64(512), T.int64(14), T.int64(14)), "float32")):
        T.func_attr({"target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for i0_i1_i2_i3_fused_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
            for i0_i1_i2_i3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v_i0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_i1 = T.axis.spatial(T.int64(512), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) // T.int64(196))
                    v_i2 = T.axis.spatial(T.int64(14), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(196) // T.int64(14))
                    v_i3 = T.axis.spatial(T.int64(14), (i0_i1_i2_i3_fused_0 * T.int64(1024) + i0_i1_i2_i3_fused_1) % T.int64(14))
                    T.reads(lv184_0[v_i0, v_i1, v_i2, v_i3])
                    T.writes(compute_intermediate[v_i0, v_i1, v_i2, v_i3])
                    compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.max(lv184_0[v_i0, v_i1, v_i2, v_i3], T.float32(0.0))

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
    def reshape(lv225: T.Buffer((T.int64(1), T.int64(2048), T.int64(1), T.int64(1)), "float32"), T_reshape: T.Buffer((T.int64(1), T.int64(2048)), "float32")):
        T.func_attr({"op_pattern": 2, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(2), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_reshape"):
                    v_ax0 = T.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1 = T.axis.spatial(T.int64(2048), ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1)
                    T.reads(lv225[T.int64(0), v_ax1 % T.int64(2048), T.int64(0), T.int64(0)])
                    T.writes(T_reshape[v_ax0, v_ax1])
                    T_reshape[v_ax0, v_ax1] = lv225[T.int64(0), v_ax1 % T.int64(2048), T.int64(0), T.int64(0)]

    @T.prim_func(private=True)
    def transpose(A: T.Buffer((T.int64(1000), T.int64(2048)), "float32"), T_transpose: T.Buffer((T.int64(2048), T.int64(1000)), "float32")):
        T.func_attr({"op_pattern": 2, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "max_shared_memory_per_block": 49152, "max_threads_per_block": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
            for ax0_ax1_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                for ax0_ax1_fused_0 in range(T.int64(8)):
                    with T.block("T_transpose"):
                        v_ax0 = T.axis.spatial(T.int64(2048), (ax0_ax1_fused_0 * T.int64(262144) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2) // T.int64(1000))
                        v_ax1 = T.axis.spatial(T.int64(1000), (ax0_ax1_fused_0 * T.int64(262144) + ax0_ax1_fused_1 * T.int64(1024) + ax0_ax1_fused_2) % T.int64(1000))
                        T.where((ax0_ax1_fused_0 * T.int64(256) + ax0_ax1_fused_1) * T.int64(1024) + ax0_ax1_fused_2 < T.int64(2048000))
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
            lv9 = R.call_tir(cls.conv2d2, (lv3, metadata["relax.expr.Constant"][10]), out_sinfo=R.Tensor((1, 64, 56, 56), dtype="float32"))
            lv10 = R.call_tir(cls.batch_norm1, (lv9, metadata["relax.expr.Constant"][11], metadata["relax.expr.Constant"][12], metadata["relax.expr.Constant"][13], metadata["relax.expr.Constant"][14]), out_sinfo=[R.Tensor((1, 64, 56, 56), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")])
            lv4_1: R.Tensor((1, 64, 56, 56), dtype="float32") = lv10[0]
            lv5_1 = R.call_tir(cls.fused_relu1, (lv4_1,), out_sinfo=R.Tensor((1, 64, 56, 56), dtype="float32"))
            lv13 = R.call_tir(cls.conv2d3, (lv5_1, metadata["relax.expr.Constant"][15]), out_sinfo=R.Tensor((1, 256, 56, 56), dtype="float32"))
            lv14 = R.call_tir(cls.batch_norm2, (lv13, metadata["relax.expr.Constant"][16], metadata["relax.expr.Constant"][17], metadata["relax.expr.Constant"][18], metadata["relax.expr.Constant"][19]), out_sinfo=[R.Tensor((1, 256, 56, 56), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")])
            lv16 = R.call_tir(cls.conv2d3, (lv4, metadata["relax.expr.Constant"][20]), out_sinfo=R.Tensor((1, 256, 56, 56), dtype="float32"))
            lv17 = R.call_tir(cls.batch_norm2, (lv16, metadata["relax.expr.Constant"][21], metadata["relax.expr.Constant"][22], metadata["relax.expr.Constant"][23], metadata["relax.expr.Constant"][24]), out_sinfo=[R.Tensor((1, 256, 56, 56), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")])
            lv6_1: R.Tensor((1, 256, 56, 56), dtype="float32") = lv14[0]
            lv7: R.Tensor((1, 256, 56, 56), dtype="float32") = lv17[0]
            lv8 = R.call_tir(cls.fused_add_relu2, (lv6_1, lv7), out_sinfo=R.Tensor((1, 256, 56, 56), dtype="float32"))
            lv21 = R.call_tir(cls.conv2d4, (lv8, metadata["relax.expr.Constant"][25]), out_sinfo=R.Tensor((1, 64, 56, 56), dtype="float32"))
            lv22 = R.call_tir(cls.batch_norm1, (lv21, metadata["relax.expr.Constant"][26], metadata["relax.expr.Constant"][27], metadata["relax.expr.Constant"][28], metadata["relax.expr.Constant"][29]), out_sinfo=[R.Tensor((1, 64, 56, 56), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")])
            lv9_1: R.Tensor((1, 64, 56, 56), dtype="float32") = lv22[0]
            lv10_1 = R.call_tir(cls.fused_relu1, (lv9_1,), out_sinfo=R.Tensor((1, 64, 56, 56), dtype="float32"))
            lv25 = R.call_tir(cls.conv2d2, (lv10_1, metadata["relax.expr.Constant"][30]), out_sinfo=R.Tensor((1, 64, 56, 56), dtype="float32"))
            lv26 = R.call_tir(cls.batch_norm1, (lv25, metadata["relax.expr.Constant"][31], metadata["relax.expr.Constant"][32], metadata["relax.expr.Constant"][33], metadata["relax.expr.Constant"][34]), out_sinfo=[R.Tensor((1, 64, 56, 56), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")])
            lv11: R.Tensor((1, 64, 56, 56), dtype="float32") = lv26[0]
            lv12 = R.call_tir(cls.fused_relu1, (lv11,), out_sinfo=R.Tensor((1, 64, 56, 56), dtype="float32"))
            lv29 = R.call_tir(cls.conv2d3, (lv12, metadata["relax.expr.Constant"][35]), out_sinfo=R.Tensor((1, 256, 56, 56), dtype="float32"))
            lv30 = R.call_tir(cls.batch_norm2, (lv29, metadata["relax.expr.Constant"][36], metadata["relax.expr.Constant"][37], metadata["relax.expr.Constant"][38], metadata["relax.expr.Constant"][39]), out_sinfo=[R.Tensor((1, 256, 56, 56), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")])
            lv13_1: R.Tensor((1, 256, 56, 56), dtype="float32") = lv30[0]
            lv14_1 = R.call_tir(cls.fused_add_relu2, (lv13_1, lv8), out_sinfo=R.Tensor((1, 256, 56, 56), dtype="float32"))
            lv34 = R.call_tir(cls.conv2d4, (lv14_1, metadata["relax.expr.Constant"][40]), out_sinfo=R.Tensor((1, 64, 56, 56), dtype="float32"))
            lv35 = R.call_tir(cls.batch_norm1, (lv34, metadata["relax.expr.Constant"][41], metadata["relax.expr.Constant"][42], metadata["relax.expr.Constant"][43], metadata["relax.expr.Constant"][44]), out_sinfo=[R.Tensor((1, 64, 56, 56), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")])
            lv15: R.Tensor((1, 64, 56, 56), dtype="float32") = lv35[0]
            lv16_1 = R.call_tir(cls.fused_relu1, (lv15,), out_sinfo=R.Tensor((1, 64, 56, 56), dtype="float32"))
            lv38 = R.call_tir(cls.conv2d2, (lv16_1, metadata["relax.expr.Constant"][45]), out_sinfo=R.Tensor((1, 64, 56, 56), dtype="float32"))
            lv39 = R.call_tir(cls.batch_norm1, (lv38, metadata["relax.expr.Constant"][46], metadata["relax.expr.Constant"][47], metadata["relax.expr.Constant"][48], metadata["relax.expr.Constant"][49]), out_sinfo=[R.Tensor((1, 64, 56, 56), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")])
            lv17_1: R.Tensor((1, 64, 56, 56), dtype="float32") = lv39[0]
            lv18 = R.call_tir(cls.fused_relu1, (lv17_1,), out_sinfo=R.Tensor((1, 64, 56, 56), dtype="float32"))
            lv42 = R.call_tir(cls.conv2d3, (lv18, metadata["relax.expr.Constant"][50]), out_sinfo=R.Tensor((1, 256, 56, 56), dtype="float32"))
            lv43 = R.call_tir(cls.batch_norm2, (lv42, metadata["relax.expr.Constant"][51], metadata["relax.expr.Constant"][52], metadata["relax.expr.Constant"][53], metadata["relax.expr.Constant"][54]), out_sinfo=[R.Tensor((1, 256, 56, 56), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")])
            lv19: R.Tensor((1, 256, 56, 56), dtype="float32") = lv43[0]
            lv20 = R.call_tir(cls.fused_add_relu2, (lv19, lv14_1), out_sinfo=R.Tensor((1, 256, 56, 56), dtype="float32"))
            lv47 = R.call_tir(cls.conv2d5, (lv20, metadata["relax.expr.Constant"][55]), out_sinfo=R.Tensor((1, 128, 56, 56), dtype="float32"))
            lv48 = R.call_tir(cls.batch_norm3, (lv47, metadata["relax.expr.Constant"][56], metadata["relax.expr.Constant"][57], metadata["relax.expr.Constant"][58], metadata["relax.expr.Constant"][59]), out_sinfo=[R.Tensor((1, 128, 56, 56), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")])
            lv21_1: R.Tensor((1, 128, 56, 56), dtype="float32") = lv48[0]
            lv22_1 = R.call_tir(cls.fused_relu3, (lv21_1,), out_sinfo=R.Tensor((1, 128, 56, 56), dtype="float32"))
            lv51 = R.call_tir(cls.conv2d6, (lv22_1, metadata["relax.expr.Constant"][60]), out_sinfo=R.Tensor((1, 128, 28, 28), dtype="float32"))
            lv52 = R.call_tir(cls.batch_norm4, (lv51, metadata["relax.expr.Constant"][61], metadata["relax.expr.Constant"][62], metadata["relax.expr.Constant"][63], metadata["relax.expr.Constant"][64]), out_sinfo=[R.Tensor((1, 128, 28, 28), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")])
            lv23: R.Tensor((1, 128, 28, 28), dtype="float32") = lv52[0]
            lv24 = R.call_tir(cls.fused_relu4, (lv23,), out_sinfo=R.Tensor((1, 128, 28, 28), dtype="float32"))
            lv55 = R.call_tir(cls.conv2d7, (lv24, metadata["relax.expr.Constant"][65]), out_sinfo=R.Tensor((1, 512, 28, 28), dtype="float32"))
            lv56 = R.call_tir(cls.batch_norm5, (lv55, metadata["relax.expr.Constant"][66], metadata["relax.expr.Constant"][67], metadata["relax.expr.Constant"][68], metadata["relax.expr.Constant"][69]), out_sinfo=[R.Tensor((1, 512, 28, 28), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")])
            lv58 = R.call_tir(cls.conv2d8, (lv20, metadata["relax.expr.Constant"][70]), out_sinfo=R.Tensor((1, 512, 28, 28), dtype="float32"))
            lv59 = R.call_tir(cls.batch_norm5, (lv58, metadata["relax.expr.Constant"][71], metadata["relax.expr.Constant"][72], metadata["relax.expr.Constant"][73], metadata["relax.expr.Constant"][74]), out_sinfo=[R.Tensor((1, 512, 28, 28), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")])
            lv25_1: R.Tensor((1, 512, 28, 28), dtype="float32") = lv56[0]
            lv26_1: R.Tensor((1, 512, 28, 28), dtype="float32") = lv59[0]
            lv27 = R.call_tir(cls.fused_add1_relu5, (lv25_1, lv26_1), out_sinfo=R.Tensor((1, 512, 28, 28), dtype="float32"))
            lv63 = R.call_tir(cls.conv2d9, (lv27, metadata["relax.expr.Constant"][75]), out_sinfo=R.Tensor((1, 128, 28, 28), dtype="float32"))
            lv64 = R.call_tir(cls.batch_norm4, (lv63, metadata["relax.expr.Constant"][76], metadata["relax.expr.Constant"][77], metadata["relax.expr.Constant"][78], metadata["relax.expr.Constant"][79]), out_sinfo=[R.Tensor((1, 128, 28, 28), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")])
            lv28: R.Tensor((1, 128, 28, 28), dtype="float32") = lv64[0]
            lv29_1 = R.call_tir(cls.fused_relu4, (lv28,), out_sinfo=R.Tensor((1, 128, 28, 28), dtype="float32"))
            lv67 = R.call_tir(cls.conv2d10, (lv29_1, metadata["relax.expr.Constant"][80]), out_sinfo=R.Tensor((1, 128, 28, 28), dtype="float32"))
            lv68 = R.call_tir(cls.batch_norm4, (lv67, metadata["relax.expr.Constant"][81], metadata["relax.expr.Constant"][82], metadata["relax.expr.Constant"][83], metadata["relax.expr.Constant"][84]), out_sinfo=[R.Tensor((1, 128, 28, 28), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")])
            lv30_1: R.Tensor((1, 128, 28, 28), dtype="float32") = lv68[0]
            lv31 = R.call_tir(cls.fused_relu4, (lv30_1,), out_sinfo=R.Tensor((1, 128, 28, 28), dtype="float32"))
            lv71 = R.call_tir(cls.conv2d7, (lv31, metadata["relax.expr.Constant"][85]), out_sinfo=R.Tensor((1, 512, 28, 28), dtype="float32"))
            lv72 = R.call_tir(cls.batch_norm5, (lv71, metadata["relax.expr.Constant"][86], metadata["relax.expr.Constant"][87], metadata["relax.expr.Constant"][88], metadata["relax.expr.Constant"][89]), out_sinfo=[R.Tensor((1, 512, 28, 28), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")])
            lv32: R.Tensor((1, 512, 28, 28), dtype="float32") = lv72[0]
            lv33 = R.call_tir(cls.fused_add1_relu5, (lv32, lv27), out_sinfo=R.Tensor((1, 512, 28, 28), dtype="float32"))
            lv76 = R.call_tir(cls.conv2d9, (lv33, metadata["relax.expr.Constant"][90]), out_sinfo=R.Tensor((1, 128, 28, 28), dtype="float32"))
            lv77 = R.call_tir(cls.batch_norm4, (lv76, metadata["relax.expr.Constant"][91], metadata["relax.expr.Constant"][92], metadata["relax.expr.Constant"][93], metadata["relax.expr.Constant"][94]), out_sinfo=[R.Tensor((1, 128, 28, 28), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")])
            lv34_1: R.Tensor((1, 128, 28, 28), dtype="float32") = lv77[0]
            lv35_1 = R.call_tir(cls.fused_relu4, (lv34_1,), out_sinfo=R.Tensor((1, 128, 28, 28), dtype="float32"))
            lv80 = R.call_tir(cls.conv2d10, (lv35_1, metadata["relax.expr.Constant"][95]), out_sinfo=R.Tensor((1, 128, 28, 28), dtype="float32"))
            lv81 = R.call_tir(cls.batch_norm4, (lv80, metadata["relax.expr.Constant"][96], metadata["relax.expr.Constant"][97], metadata["relax.expr.Constant"][98], metadata["relax.expr.Constant"][99]), out_sinfo=[R.Tensor((1, 128, 28, 28), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")])
            lv36: R.Tensor((1, 128, 28, 28), dtype="float32") = lv81[0]
            lv37 = R.call_tir(cls.fused_relu4, (lv36,), out_sinfo=R.Tensor((1, 128, 28, 28), dtype="float32"))
            lv84 = R.call_tir(cls.conv2d7, (lv37, metadata["relax.expr.Constant"][100]), out_sinfo=R.Tensor((1, 512, 28, 28), dtype="float32"))
            lv85 = R.call_tir(cls.batch_norm5, (lv84, metadata["relax.expr.Constant"][101], metadata["relax.expr.Constant"][102], metadata["relax.expr.Constant"][103], metadata["relax.expr.Constant"][104]), out_sinfo=[R.Tensor((1, 512, 28, 28), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")])
            lv38_1: R.Tensor((1, 512, 28, 28), dtype="float32") = lv85[0]
            lv39_1 = R.call_tir(cls.fused_add1_relu5, (lv38_1, lv33), out_sinfo=R.Tensor((1, 512, 28, 28), dtype="float32"))
            lv89 = R.call_tir(cls.conv2d9, (lv39_1, metadata["relax.expr.Constant"][105]), out_sinfo=R.Tensor((1, 128, 28, 28), dtype="float32"))
            lv90 = R.call_tir(cls.batch_norm4, (lv89, metadata["relax.expr.Constant"][106], metadata["relax.expr.Constant"][107], metadata["relax.expr.Constant"][108], metadata["relax.expr.Constant"][109]), out_sinfo=[R.Tensor((1, 128, 28, 28), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")])
            lv40: R.Tensor((1, 128, 28, 28), dtype="float32") = lv90[0]
            lv41 = R.call_tir(cls.fused_relu4, (lv40,), out_sinfo=R.Tensor((1, 128, 28, 28), dtype="float32"))
            lv93 = R.call_tir(cls.conv2d10, (lv41, metadata["relax.expr.Constant"][110]), out_sinfo=R.Tensor((1, 128, 28, 28), dtype="float32"))
            lv94 = R.call_tir(cls.batch_norm4, (lv93, metadata["relax.expr.Constant"][111], metadata["relax.expr.Constant"][112], metadata["relax.expr.Constant"][113], metadata["relax.expr.Constant"][114]), out_sinfo=[R.Tensor((1, 128, 28, 28), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")])
            lv42_1: R.Tensor((1, 128, 28, 28), dtype="float32") = lv94[0]
            lv43_1 = R.call_tir(cls.fused_relu4, (lv42_1,), out_sinfo=R.Tensor((1, 128, 28, 28), dtype="float32"))
            lv97 = R.call_tir(cls.conv2d7, (lv43_1, metadata["relax.expr.Constant"][115]), out_sinfo=R.Tensor((1, 512, 28, 28), dtype="float32"))
            lv98 = R.call_tir(cls.batch_norm5, (lv97, metadata["relax.expr.Constant"][116], metadata["relax.expr.Constant"][117], metadata["relax.expr.Constant"][118], metadata["relax.expr.Constant"][119]), out_sinfo=[R.Tensor((1, 512, 28, 28), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")])
            lv44: R.Tensor((1, 512, 28, 28), dtype="float32") = lv98[0]
            lv45 = R.call_tir(cls.fused_add1_relu5, (lv44, lv39_1), out_sinfo=R.Tensor((1, 512, 28, 28), dtype="float32"))
            lv102 = R.call_tir(cls.conv2d11, (lv45, metadata["relax.expr.Constant"][120]), out_sinfo=R.Tensor((1, 256, 28, 28), dtype="float32"))
            lv103 = R.call_tir(cls.batch_norm6, (lv102, metadata["relax.expr.Constant"][121], metadata["relax.expr.Constant"][122], metadata["relax.expr.Constant"][123], metadata["relax.expr.Constant"][124]), out_sinfo=[R.Tensor((1, 256, 28, 28), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")])
            lv46: R.Tensor((1, 256, 28, 28), dtype="float32") = lv103[0]
            lv47_1 = R.call_tir(cls.fused_relu6, (lv46,), out_sinfo=R.Tensor((1, 256, 28, 28), dtype="float32"))
            lv106 = R.call_tir(cls.conv2d12, (lv47_1, metadata["relax.expr.Constant"][125]), out_sinfo=R.Tensor((1, 256, 14, 14), dtype="float32"))
            lv107 = R.call_tir(cls.batch_norm7, (lv106, metadata["relax.expr.Constant"][126], metadata["relax.expr.Constant"][127], metadata["relax.expr.Constant"][128], metadata["relax.expr.Constant"][129]), out_sinfo=[R.Tensor((1, 256, 14, 14), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")])
            lv48_1: R.Tensor((1, 256, 14, 14), dtype="float32") = lv107[0]
            lv49 = R.call_tir(cls.fused_relu7, (lv48_1,), out_sinfo=R.Tensor((1, 256, 14, 14), dtype="float32"))
            lv110 = R.call_tir(cls.conv2d13, (lv49, metadata["relax.expr.Constant"][130]), out_sinfo=R.Tensor((1, 1024, 14, 14), dtype="float32"))
            lv111 = R.call_tir(cls.batch_norm8, (lv110, metadata["relax.expr.Constant"][131], metadata["relax.expr.Constant"][132], metadata["relax.expr.Constant"][133], metadata["relax.expr.Constant"][134]), out_sinfo=[R.Tensor((1, 1024, 14, 14), dtype="float32"), R.Tensor((1024,), dtype="float32"), R.Tensor((1024,), dtype="float32")])
            lv113 = R.call_tir(cls.conv2d14, (lv45, metadata["relax.expr.Constant"][135]), out_sinfo=R.Tensor((1, 1024, 14, 14), dtype="float32"))
            lv114 = R.call_tir(cls.batch_norm8, (lv113, metadata["relax.expr.Constant"][136], metadata["relax.expr.Constant"][137], metadata["relax.expr.Constant"][138], metadata["relax.expr.Constant"][139]), out_sinfo=[R.Tensor((1, 1024, 14, 14), dtype="float32"), R.Tensor((1024,), dtype="float32"), R.Tensor((1024,), dtype="float32")])
            lv50: R.Tensor((1, 1024, 14, 14), dtype="float32") = lv111[0]
            lv51_1: R.Tensor((1, 1024, 14, 14), dtype="float32") = lv114[0]
            lv52_1 = R.call_tir(cls.fused_add2_relu8, (lv50, lv51_1), out_sinfo=R.Tensor((1, 1024, 14, 14), dtype="float32"))
            lv118 = R.call_tir(cls.conv2d15, (lv52_1, metadata["relax.expr.Constant"][140]), out_sinfo=R.Tensor((1, 256, 14, 14), dtype="float32"))
            lv119 = R.call_tir(cls.batch_norm7, (lv118, metadata["relax.expr.Constant"][141], metadata["relax.expr.Constant"][142], metadata["relax.expr.Constant"][143], metadata["relax.expr.Constant"][144]), out_sinfo=[R.Tensor((1, 256, 14, 14), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")])
            lv53: R.Tensor((1, 256, 14, 14), dtype="float32") = lv119[0]
            lv54 = R.call_tir(cls.fused_relu7, (lv53,), out_sinfo=R.Tensor((1, 256, 14, 14), dtype="float32"))
            lv122 = R.call_tir(cls.conv2d16, (lv54, metadata["relax.expr.Constant"][145]), out_sinfo=R.Tensor((1, 256, 14, 14), dtype="float32"))
            lv123 = R.call_tir(cls.batch_norm7, (lv122, metadata["relax.expr.Constant"][146], metadata["relax.expr.Constant"][147], metadata["relax.expr.Constant"][148], metadata["relax.expr.Constant"][149]), out_sinfo=[R.Tensor((1, 256, 14, 14), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")])
            lv55_1: R.Tensor((1, 256, 14, 14), dtype="float32") = lv123[0]
            lv56_1 = R.call_tir(cls.fused_relu7, (lv55_1,), out_sinfo=R.Tensor((1, 256, 14, 14), dtype="float32"))
            lv126 = R.call_tir(cls.conv2d13, (lv56_1, metadata["relax.expr.Constant"][150]), out_sinfo=R.Tensor((1, 1024, 14, 14), dtype="float32"))
            lv127 = R.call_tir(cls.batch_norm8, (lv126, metadata["relax.expr.Constant"][151], metadata["relax.expr.Constant"][152], metadata["relax.expr.Constant"][153], metadata["relax.expr.Constant"][154]), out_sinfo=[R.Tensor((1, 1024, 14, 14), dtype="float32"), R.Tensor((1024,), dtype="float32"), R.Tensor((1024,), dtype="float32")])
            lv57: R.Tensor((1, 1024, 14, 14), dtype="float32") = lv127[0]
            lv58_1 = R.call_tir(cls.fused_add2_relu8, (lv57, lv52_1), out_sinfo=R.Tensor((1, 1024, 14, 14), dtype="float32"))
            lv131 = R.call_tir(cls.conv2d15, (lv58_1, metadata["relax.expr.Constant"][155]), out_sinfo=R.Tensor((1, 256, 14, 14), dtype="float32"))
            lv132 = R.call_tir(cls.batch_norm7, (lv131, metadata["relax.expr.Constant"][156], metadata["relax.expr.Constant"][157], metadata["relax.expr.Constant"][158], metadata["relax.expr.Constant"][159]), out_sinfo=[R.Tensor((1, 256, 14, 14), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")])
            lv59_1: R.Tensor((1, 256, 14, 14), dtype="float32") = lv132[0]
            lv60 = R.call_tir(cls.fused_relu7, (lv59_1,), out_sinfo=R.Tensor((1, 256, 14, 14), dtype="float32"))
            lv135 = R.call_tir(cls.conv2d16, (lv60, metadata["relax.expr.Constant"][160]), out_sinfo=R.Tensor((1, 256, 14, 14), dtype="float32"))
            lv136 = R.call_tir(cls.batch_norm7, (lv135, metadata["relax.expr.Constant"][161], metadata["relax.expr.Constant"][162], metadata["relax.expr.Constant"][163], metadata["relax.expr.Constant"][164]), out_sinfo=[R.Tensor((1, 256, 14, 14), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")])
            lv61: R.Tensor((1, 256, 14, 14), dtype="float32") = lv136[0]
            lv62 = R.call_tir(cls.fused_relu7, (lv61,), out_sinfo=R.Tensor((1, 256, 14, 14), dtype="float32"))
            lv139 = R.call_tir(cls.conv2d13, (lv62, metadata["relax.expr.Constant"][165]), out_sinfo=R.Tensor((1, 1024, 14, 14), dtype="float32"))
            lv140 = R.call_tir(cls.batch_norm8, (lv139, metadata["relax.expr.Constant"][166], metadata["relax.expr.Constant"][167], metadata["relax.expr.Constant"][168], metadata["relax.expr.Constant"][169]), out_sinfo=[R.Tensor((1, 1024, 14, 14), dtype="float32"), R.Tensor((1024,), dtype="float32"), R.Tensor((1024,), dtype="float32")])
            lv63_1: R.Tensor((1, 1024, 14, 14), dtype="float32") = lv140[0]
            lv64_1 = R.call_tir(cls.fused_add2_relu8, (lv63_1, lv58_1), out_sinfo=R.Tensor((1, 1024, 14, 14), dtype="float32"))
            lv144 = R.call_tir(cls.conv2d15, (lv64_1, metadata["relax.expr.Constant"][170]), out_sinfo=R.Tensor((1, 256, 14, 14), dtype="float32"))
            lv145 = R.call_tir(cls.batch_norm7, (lv144, metadata["relax.expr.Constant"][171], metadata["relax.expr.Constant"][172], metadata["relax.expr.Constant"][173], metadata["relax.expr.Constant"][174]), out_sinfo=[R.Tensor((1, 256, 14, 14), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")])
            lv65: R.Tensor((1, 256, 14, 14), dtype="float32") = lv145[0]
            lv66 = R.call_tir(cls.fused_relu7, (lv65,), out_sinfo=R.Tensor((1, 256, 14, 14), dtype="float32"))
            lv148 = R.call_tir(cls.conv2d16, (lv66, metadata["relax.expr.Constant"][175]), out_sinfo=R.Tensor((1, 256, 14, 14), dtype="float32"))
            lv149 = R.call_tir(cls.batch_norm7, (lv148, metadata["relax.expr.Constant"][176], metadata["relax.expr.Constant"][177], metadata["relax.expr.Constant"][178], metadata["relax.expr.Constant"][179]), out_sinfo=[R.Tensor((1, 256, 14, 14), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")])
            lv67_1: R.Tensor((1, 256, 14, 14), dtype="float32") = lv149[0]
            lv68_1 = R.call_tir(cls.fused_relu7, (lv67_1,), out_sinfo=R.Tensor((1, 256, 14, 14), dtype="float32"))
            lv152 = R.call_tir(cls.conv2d13, (lv68_1, metadata["relax.expr.Constant"][180]), out_sinfo=R.Tensor((1, 1024, 14, 14), dtype="float32"))
            lv153 = R.call_tir(cls.batch_norm8, (lv152, metadata["relax.expr.Constant"][181], metadata["relax.expr.Constant"][182], metadata["relax.expr.Constant"][183], metadata["relax.expr.Constant"][184]), out_sinfo=[R.Tensor((1, 1024, 14, 14), dtype="float32"), R.Tensor((1024,), dtype="float32"), R.Tensor((1024,), dtype="float32")])
            lv69: R.Tensor((1, 1024, 14, 14), dtype="float32") = lv153[0]
            lv70 = R.call_tir(cls.fused_add2_relu8, (lv69, lv64_1), out_sinfo=R.Tensor((1, 1024, 14, 14), dtype="float32"))
            lv157 = R.call_tir(cls.conv2d15, (lv70, metadata["relax.expr.Constant"][185]), out_sinfo=R.Tensor((1, 256, 14, 14), dtype="float32"))
            lv158 = R.call_tir(cls.batch_norm7, (lv157, metadata["relax.expr.Constant"][186], metadata["relax.expr.Constant"][187], metadata["relax.expr.Constant"][188], metadata["relax.expr.Constant"][189]), out_sinfo=[R.Tensor((1, 256, 14, 14), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")])
            lv71_1: R.Tensor((1, 256, 14, 14), dtype="float32") = lv158[0]
            lv72_1 = R.call_tir(cls.fused_relu7, (lv71_1,), out_sinfo=R.Tensor((1, 256, 14, 14), dtype="float32"))
            lv161 = R.call_tir(cls.conv2d16, (lv72_1, metadata["relax.expr.Constant"][190]), out_sinfo=R.Tensor((1, 256, 14, 14), dtype="float32"))
            lv162 = R.call_tir(cls.batch_norm7, (lv161, metadata["relax.expr.Constant"][191], metadata["relax.expr.Constant"][192], metadata["relax.expr.Constant"][193], metadata["relax.expr.Constant"][194]), out_sinfo=[R.Tensor((1, 256, 14, 14), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")])
            lv73: R.Tensor((1, 256, 14, 14), dtype="float32") = lv162[0]
            lv74 = R.call_tir(cls.fused_relu7, (lv73,), out_sinfo=R.Tensor((1, 256, 14, 14), dtype="float32"))
            lv165 = R.call_tir(cls.conv2d13, (lv74, metadata["relax.expr.Constant"][195]), out_sinfo=R.Tensor((1, 1024, 14, 14), dtype="float32"))
            lv166 = R.call_tir(cls.batch_norm8, (lv165, metadata["relax.expr.Constant"][196], metadata["relax.expr.Constant"][197], metadata["relax.expr.Constant"][198], metadata["relax.expr.Constant"][199]), out_sinfo=[R.Tensor((1, 1024, 14, 14), dtype="float32"), R.Tensor((1024,), dtype="float32"), R.Tensor((1024,), dtype="float32")])
            lv75: R.Tensor((1, 1024, 14, 14), dtype="float32") = lv166[0]
            lv76_1 = R.call_tir(cls.fused_add2_relu8, (lv75, lv70), out_sinfo=R.Tensor((1, 1024, 14, 14), dtype="float32"))
            lv170 = R.call_tir(cls.conv2d15, (lv76_1, metadata["relax.expr.Constant"][200]), out_sinfo=R.Tensor((1, 256, 14, 14), dtype="float32"))
            lv171 = R.call_tir(cls.batch_norm7, (lv170, metadata["relax.expr.Constant"][201], metadata["relax.expr.Constant"][202], metadata["relax.expr.Constant"][203], metadata["relax.expr.Constant"][204]), out_sinfo=[R.Tensor((1, 256, 14, 14), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")])
            lv77_1: R.Tensor((1, 256, 14, 14), dtype="float32") = lv171[0]
            lv78 = R.call_tir(cls.fused_relu7, (lv77_1,), out_sinfo=R.Tensor((1, 256, 14, 14), dtype="float32"))
            lv174 = R.call_tir(cls.conv2d16, (lv78, metadata["relax.expr.Constant"][205]), out_sinfo=R.Tensor((1, 256, 14, 14), dtype="float32"))
            lv175 = R.call_tir(cls.batch_norm7, (lv174, metadata["relax.expr.Constant"][206], metadata["relax.expr.Constant"][207], metadata["relax.expr.Constant"][208], metadata["relax.expr.Constant"][209]), out_sinfo=[R.Tensor((1, 256, 14, 14), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")])
            lv79: R.Tensor((1, 256, 14, 14), dtype="float32") = lv175[0]
            lv80_1 = R.call_tir(cls.fused_relu7, (lv79,), out_sinfo=R.Tensor((1, 256, 14, 14), dtype="float32"))
            lv178 = R.call_tir(cls.conv2d13, (lv80_1, metadata["relax.expr.Constant"][210]), out_sinfo=R.Tensor((1, 1024, 14, 14), dtype="float32"))
            lv179 = R.call_tir(cls.batch_norm8, (lv178, metadata["relax.expr.Constant"][211], metadata["relax.expr.Constant"][212], metadata["relax.expr.Constant"][213], metadata["relax.expr.Constant"][214]), out_sinfo=[R.Tensor((1, 1024, 14, 14), dtype="float32"), R.Tensor((1024,), dtype="float32"), R.Tensor((1024,), dtype="float32")])
            lv81_1: R.Tensor((1, 1024, 14, 14), dtype="float32") = lv179[0]
            lv82 = R.call_tir(cls.fused_add2_relu8, (lv81_1, lv76_1), out_sinfo=R.Tensor((1, 1024, 14, 14), dtype="float32"))
            lv183 = R.call_tir(cls.conv2d17, (lv82, metadata["relax.expr.Constant"][215]), out_sinfo=R.Tensor((1, 512, 14, 14), dtype="float32"))
            lv184 = R.call_tir(cls.batch_norm9, (lv183, metadata["relax.expr.Constant"][216], metadata["relax.expr.Constant"][217], metadata["relax.expr.Constant"][218], metadata["relax.expr.Constant"][219]), out_sinfo=[R.Tensor((1, 512, 14, 14), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")])
            lv83: R.Tensor((1, 512, 14, 14), dtype="float32") = lv184[0]
            lv84_1 = R.call_tir(cls.fused_relu9, (lv83,), out_sinfo=R.Tensor((1, 512, 14, 14), dtype="float32"))
            lv187 = R.call_tir(cls.conv2d18, (lv84_1, metadata["relax.expr.Constant"][220]), out_sinfo=R.Tensor((1, 512, 7, 7), dtype="float32"))
            lv188 = R.call_tir(cls.batch_norm10, (lv187, metadata["relax.expr.Constant"][221], metadata["relax.expr.Constant"][222], metadata["relax.expr.Constant"][223], metadata["relax.expr.Constant"][224]), out_sinfo=[R.Tensor((1, 512, 7, 7), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")])
            lv85_1: R.Tensor((1, 512, 7, 7), dtype="float32") = lv188[0]
            lv86 = R.call_tir(cls.fused_relu10, (lv85_1,), out_sinfo=R.Tensor((1, 512, 7, 7), dtype="float32"))
            lv191 = R.call_tir(cls.conv2d19, (lv86, metadata["relax.expr.Constant"][225]), out_sinfo=R.Tensor((1, 2048, 7, 7), dtype="float32"))
            lv192 = R.call_tir(cls.batch_norm11, (lv191, metadata["relax.expr.Constant"][226], metadata["relax.expr.Constant"][227], metadata["relax.expr.Constant"][228], metadata["relax.expr.Constant"][229]), out_sinfo=[R.Tensor((1, 2048, 7, 7), dtype="float32"), R.Tensor((2048,), dtype="float32"), R.Tensor((2048,), dtype="float32")])
            lv194 = R.call_tir(cls.conv2d20, (lv82, metadata["relax.expr.Constant"][230]), out_sinfo=R.Tensor((1, 2048, 7, 7), dtype="float32"))
            lv195 = R.call_tir(cls.batch_norm11, (lv194, metadata["relax.expr.Constant"][231], metadata["relax.expr.Constant"][232], metadata["relax.expr.Constant"][233], metadata["relax.expr.Constant"][234]), out_sinfo=[R.Tensor((1, 2048, 7, 7), dtype="float32"), R.Tensor((2048,), dtype="float32"), R.Tensor((2048,), dtype="float32")])
            lv87: R.Tensor((1, 2048, 7, 7), dtype="float32") = lv192[0]
            lv88: R.Tensor((1, 2048, 7, 7), dtype="float32") = lv195[0]
            lv89_1 = R.call_tir(cls.fused_add3_relu11, (lv87, lv88), out_sinfo=R.Tensor((1, 2048, 7, 7), dtype="float32"))
            lv199 = R.call_tir(cls.conv2d21, (lv89_1, metadata["relax.expr.Constant"][235]), out_sinfo=R.Tensor((1, 512, 7, 7), dtype="float32"))
            lv200 = R.call_tir(cls.batch_norm10, (lv199, metadata["relax.expr.Constant"][236], metadata["relax.expr.Constant"][237], metadata["relax.expr.Constant"][238], metadata["relax.expr.Constant"][239]), out_sinfo=[R.Tensor((1, 512, 7, 7), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")])
            lv90_1: R.Tensor((1, 512, 7, 7), dtype="float32") = lv200[0]
            lv91 = R.call_tir(cls.fused_relu10, (lv90_1,), out_sinfo=R.Tensor((1, 512, 7, 7), dtype="float32"))
            lv203 = R.call_tir(cls.conv2d22, (lv91, metadata["relax.expr.Constant"][240]), out_sinfo=R.Tensor((1, 512, 7, 7), dtype="float32"))
            lv204 = R.call_tir(cls.batch_norm10, (lv203, metadata["relax.expr.Constant"][241], metadata["relax.expr.Constant"][242], metadata["relax.expr.Constant"][243], metadata["relax.expr.Constant"][244]), out_sinfo=[R.Tensor((1, 512, 7, 7), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")])
            lv92: R.Tensor((1, 512, 7, 7), dtype="float32") = lv204[0]
            lv93_1 = R.call_tir(cls.fused_relu10, (lv92,), out_sinfo=R.Tensor((1, 512, 7, 7), dtype="float32"))
            lv207 = R.call_tir(cls.conv2d19, (lv93_1, metadata["relax.expr.Constant"][245]), out_sinfo=R.Tensor((1, 2048, 7, 7), dtype="float32"))
            lv208 = R.call_tir(cls.batch_norm11, (lv207, metadata["relax.expr.Constant"][246], metadata["relax.expr.Constant"][247], metadata["relax.expr.Constant"][248], metadata["relax.expr.Constant"][249]), out_sinfo=[R.Tensor((1, 2048, 7, 7), dtype="float32"), R.Tensor((2048,), dtype="float32"), R.Tensor((2048,), dtype="float32")])
            lv94_1: R.Tensor((1, 2048, 7, 7), dtype="float32") = lv208[0]
            lv95 = R.call_tir(cls.fused_add3_relu11, (lv94_1, lv89_1), out_sinfo=R.Tensor((1, 2048, 7, 7), dtype="float32"))
            lv212 = R.call_tir(cls.conv2d21, (lv95, metadata["relax.expr.Constant"][250]), out_sinfo=R.Tensor((1, 512, 7, 7), dtype="float32"))
            lv213 = R.call_tir(cls.batch_norm10, (lv212, metadata["relax.expr.Constant"][251], metadata["relax.expr.Constant"][252], metadata["relax.expr.Constant"][253], metadata["relax.expr.Constant"][254]), out_sinfo=[R.Tensor((1, 512, 7, 7), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")])
            lv96: R.Tensor((1, 512, 7, 7), dtype="float32") = lv213[0]
            lv97_1 = R.call_tir(cls.fused_relu10, (lv96,), out_sinfo=R.Tensor((1, 512, 7, 7), dtype="float32"))
            lv216 = R.call_tir(cls.conv2d22, (lv97_1, metadata["relax.expr.Constant"][255]), out_sinfo=R.Tensor((1, 512, 7, 7), dtype="float32"))
            lv217 = R.call_tir(cls.batch_norm10, (lv216, metadata["relax.expr.Constant"][256], metadata["relax.expr.Constant"][257], metadata["relax.expr.Constant"][258], metadata["relax.expr.Constant"][259]), out_sinfo=[R.Tensor((1, 512, 7, 7), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")])
            lv98_1: R.Tensor((1, 512, 7, 7), dtype="float32") = lv217[0]
            lv99 = R.call_tir(cls.fused_relu10, (lv98_1,), out_sinfo=R.Tensor((1, 512, 7, 7), dtype="float32"))
            lv220 = R.call_tir(cls.conv2d19, (lv99, metadata["relax.expr.Constant"][260]), out_sinfo=R.Tensor((1, 2048, 7, 7), dtype="float32"))
            lv221 = R.call_tir(cls.batch_norm11, (lv220, metadata["relax.expr.Constant"][261], metadata["relax.expr.Constant"][262], metadata["relax.expr.Constant"][263], metadata["relax.expr.Constant"][264]), out_sinfo=[R.Tensor((1, 2048, 7, 7), dtype="float32"), R.Tensor((2048,), dtype="float32"), R.Tensor((2048,), dtype="float32")])
            lv100: R.Tensor((1, 2048, 7, 7), dtype="float32") = lv221[0]
            lv101 = R.call_tir(cls.fused_add3_relu11, (lv100, lv95), out_sinfo=R.Tensor((1, 2048, 7, 7), dtype="float32"))
            lv225 = R.call_tir(cls.adaptive_avg_pool2d, (lv101,), out_sinfo=R.Tensor((1, 2048, 1, 1), dtype="float32"))
            lv226 = R.call_tir(cls.reshape, (lv225,), out_sinfo=R.Tensor((1, 2048), dtype="float32"))
            lv227 = R.call_tir(cls.transpose, (metadata["relax.expr.Constant"][265],), out_sinfo=R.Tensor((2048, 1000), dtype="float32"))
            gv = R.call_tir(cls.fused_matmul_add4, (lv226, lv227, metadata["relax.expr.Constant"][266]), out_sinfo=R.Tensor((1, 1000), dtype="float32"))
            R.output(gv)
        return gv

# Metadata omitted. Use show_meta=True in script() method to show it.