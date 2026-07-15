# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def adaptive_avg_pool2d(lv224: T.Buffer((T.int64(16), T.int64(2048), T.int64(7), T.int64(7)), "float32"), adaptive_pool_avg: T.Buffer((T.int64(16), T.int64(2048), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        adaptive_pool_sum_local = T.alloc_buffer((T.int64(16), T.int64(2048), T.int64(1), T.int64(1)), scope="local")
        adaptive_pool_sum_rf_local = T.alloc_buffer((T.int64(32), T.int64(16), T.int64(2048), T.int64(1), T.int64(1)), scope="local")
        for ax0_ax1_fused in T.thread_binding(T.int64(32768), thread="blockIdx.x"):
            for ax2_ax3_fused_1 in T.thread_binding(T.int64(32), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                with T.block("adaptive_pool_sum_rf_init"):
                    vax2_ax3_fused_1 = T.axis.spatial(T.int64(32), ax2_ax3_fused_1)
                    v0 = T.axis.spatial(T.int64(16), ax0_ax1_fused // T.int64(2048))
                    v1 = T.axis.spatial(T.int64(2048), ax0_ax1_fused % T.int64(2048))
                    T.reads()
                    T.writes(adaptive_pool_sum_rf_local[vax2_ax3_fused_1, v0, v1, T.int64(0), T.int64(0)])
                    adaptive_pool_sum_rf_local[vax2_ax3_fused_1, v0, v1, T.int64(0), T.int64(0)] = T.float32(0.0)
                for ax2_ax3_fused_0, u in T.grid(T.int64(2), 1):
                    with T.block("adaptive_pool_sum_rf_update"):
                        vax2_ax3_fused_1 = T.axis.spatial(T.int64(32), ax2_ax3_fused_1)
                        v0 = T.axis.spatial(T.int64(16), ax0_ax1_fused // T.int64(2048))
                        v1 = T.axis.spatial(T.int64(2048), ax0_ax1_fused % T.int64(2048))
                        vax2_ax3_fused_0 = T.axis.reduce(T.int64(2), ax2_ax3_fused_0)
                        T.where(ax2_ax3_fused_0 * T.int64(32) + ax2_ax3_fused_1 < T.int64(49))
                        T.reads(adaptive_pool_sum_rf_local[vax2_ax3_fused_1, v0, v1, T.int64(0), T.int64(0)], lv224[v0, v1, (vax2_ax3_fused_0 * T.int64(32) + vax2_ax3_fused_1) // T.int64(7), (vax2_ax3_fused_0 * T.int64(32) + vax2_ax3_fused_1) % T.int64(7)])
                        T.writes(adaptive_pool_sum_rf_local[vax2_ax3_fused_1, v0, v1, T.int64(0), T.int64(0)])
                        adaptive_pool_sum_rf_local[vax2_ax3_fused_1, v0, v1, T.int64(0), T.int64(0)] = adaptive_pool_sum_rf_local[vax2_ax3_fused_1, v0, v1, T.int64(0), T.int64(0)] + lv224[v0, v1, (vax2_ax3_fused_0 * T.int64(32) + vax2_ax3_fused_1) // T.int64(7), (vax2_ax3_fused_0 * T.int64(32) + vax2_ax3_fused_1) % T.int64(7)]
            for ax1_ax2_fused in range(T.int64(1)):
                for ax0 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                    with T.block("adaptive_pool_sum"):
                        vax2_ax3_fused_1 = T.axis.reduce(T.int64(32), ax0)
                        v0 = T.axis.spatial(T.int64(16), ax0_ax1_fused // T.int64(2048))
                        v1 = T.axis.spatial(T.int64(2048), ax0_ax1_fused % T.int64(2048))
                        T.reads(adaptive_pool_sum_rf_local[vax2_ax3_fused_1, v0, v1, T.int64(0), T.int64(0)])
                        T.writes(adaptive_pool_sum_local[v0, v1, T.int64(0), T.int64(0)])
                        with T.init():
                            adaptive_pool_sum_local[v0, v1, T.int64(0), T.int64(0)] = T.float32(0.0)
                        adaptive_pool_sum_local[v0, v1, T.int64(0), T.int64(0)] = adaptive_pool_sum_local[v0, v1, T.int64(0), T.int64(0)] + adaptive_pool_sum_rf_local[vax2_ax3_fused_1, v0, v1, T.int64(0), T.int64(0)]
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                with T.block("adaptive_pool_avg"):
                    v0 = T.axis.spatial(T.int64(16), ax0_ax1_fused // T.int64(2048) + ax0)
                    v1 = T.axis.spatial(T.int64(2048), ax0_ax1_fused % T.int64(2048) + ax1)
                    T.reads(adaptive_pool_sum_local[v0, v1, T.int64(0), T.int64(0)])
                    T.writes(adaptive_pool_avg[v0, v1, T.int64(0), T.int64(0)])
                    T.block_attr({"schedule_rule": "meta_schedule.adaptive_pool_avg"})
                    adaptive_pool_avg[v0, v1, T.int64(0), T.int64(0)] = adaptive_pool_sum_local[v0, v1, T.int64(0), T.int64(0)] * T.float32(0.020408163265306121)

    @T.prim_func
    def batch_norm(lv: T.Buffer((T.int64(16), T.int64(64), T.int64(112), T.int64(112)), "float32"), B: T.Buffer((T.int64(64),), "float32"), C: T.Buffer((T.int64(64),), "float32"), D: T.Buffer((T.int64(64),), "float32"), E: T.Buffer((T.int64(64),), "float32"), T_add: T.Buffer((T.int64(16), T.int64(64), T.int64(112), T.int64(112)), "float32"), T_add_1: T.Buffer((T.int64(64),), "float32"), T_add_2: T.Buffer((T.int64(64),), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        lv_red = T.alloc_buffer((T.int64(64),))
        T_multiply_red = T.alloc_buffer((T.int64(64),))
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(12544), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_1"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(802816))
                    v1 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(802816) // T.int64(12544))
                    v2 = T.axis.spatial(T.int64(112), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(12544) // T.int64(112))
                    v3 = T.axis.spatial(T.int64(112), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(112))
                    T.reads(lv[v0, v1, v2, v3], D[v1], E[v1], B[v1], C[v1])
                    T.writes(T_add[v0, v1, v2, v3])
                    T_add[v0, v1, v2, v3] = (lv[v0, v1, v2, v3] - D[v1]) / T.sqrt(E[v1] + T.float32(1.0000000000000001e-05)) * B[v1] + C[v1]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("lv_red_init"):
                    v0 = T.axis.spatial(T.int64(64), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(64))
                    T.reads()
                    T.writes(lv_red[v0])
                    lv_red[v0] = T.float32(0.0)
                for ax1, ax2, ax3 in T.grid(T.int64(16), T.int64(112), T.int64(112)):
                    with T.block("lv_red_update"):
                        v0 = T.axis.spatial(T.int64(64), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                        v1, v2, v3 = T.axis.remap("RRR", [ax1, ax2, ax3])
                        T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(64))
                        T.reads(lv_red[v0], lv[v1, v0, v2, v3])
                        T.writes(lv_red[v0])
                        lv_red[v0] = lv_red[v0] + lv[v1, v0, v2, v3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_2"):
                    v0 = T.axis.spatial(T.int64(64), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(64))
                    T.reads(D[v0], lv_red[v0])
                    T.writes(T_add_1[v0])
                    T_add_1[v0] = T.float32(0.90000000000000002) * D[v0] + T.float32(0.10000000000000001) * (lv_red[v0] * T.float32(4.9824617346938772e-06))
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply_red_init"):
                    v0 = T.axis.spatial(T.int64(64), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(64))
                    T.reads()
                    T.writes(T_multiply_red[v0])
                    T_multiply_red[v0] = T.float32(0.0)
                for ax1, ax2, ax3 in T.grid(T.int64(16), T.int64(112), T.int64(112)):
                    with T.block("T_multiply_red_update"):
                        v0 = T.axis.spatial(T.int64(64), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                        v1, v2, v3 = T.axis.remap("RRR", [ax1, ax2, ax3])
                        T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(64))
                        T.reads(T_multiply_red[v0], lv[v1, v0, v2, v3], lv_red[v0])
                        T.writes(T_multiply_red[v0])
                        T_multiply_red[v0] = T_multiply_red[v0] + (lv[v1, v0, v2, v3] - lv_red[v0] * T.float32(4.9824617346938772e-06)) * (lv[v1, v0, v2, v3] - lv_red[v0] * T.float32(4.9824617346938772e-06))
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_3"):
                    v0 = T.axis.spatial(T.int64(64), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(64))
                    T.reads(E[v0], T_multiply_red[v0])
                    T.writes(T_add_2[v0])
                    T_add_2[v0] = T.float32(0.90000000000000002) * E[v0] + T.float32(0.10000000000000001) * (T_multiply_red[v0] * T.float32(4.9824617346938772e-06))

    @T.prim_func
    def batch_norm1(lv5: T.Buffer((T.int64(16), T.int64(64), T.int64(56), T.int64(56)), "float32"), B: T.Buffer((T.int64(64),), "float32"), C: T.Buffer((T.int64(64),), "float32"), D: T.Buffer((T.int64(64),), "float32"), E: T.Buffer((T.int64(64),), "float32"), T_add: T.Buffer((T.int64(16), T.int64(64), T.int64(56), T.int64(56)), "float32"), T_add_1: T.Buffer((T.int64(64),), "float32"), T_add_2: T.Buffer((T.int64(64),), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        lv5_red = T.alloc_buffer((T.int64(64),))
        T_multiply_red = T.alloc_buffer((T.int64(64),))
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(3136), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_1"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(200704))
                    v1 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(200704) // T.int64(3136))
                    v2 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(3136) // T.int64(56))
                    v3 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(56))
                    T.reads(lv5[v0, v1, v2, v3], D[v1], E[v1], B[v1], C[v1])
                    T.writes(T_add[v0, v1, v2, v3])
                    T_add[v0, v1, v2, v3] = (lv5[v0, v1, v2, v3] - D[v1]) / T.sqrt(E[v1] + T.float32(1.0000000000000001e-05)) * B[v1] + C[v1]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("lv5_red_init"):
                    v0 = T.axis.spatial(T.int64(64), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(64))
                    T.reads()
                    T.writes(lv5_red[v0])
                    lv5_red[v0] = T.float32(0.0)
                for ax1, ax2, ax3 in T.grid(T.int64(16), T.int64(56), T.int64(56)):
                    with T.block("lv5_red_update"):
                        v0 = T.axis.spatial(T.int64(64), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                        v1, v2, v3 = T.axis.remap("RRR", [ax1, ax2, ax3])
                        T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(64))
                        T.reads(lv5_red[v0], lv5[v1, v0, v2, v3])
                        T.writes(lv5_red[v0])
                        lv5_red[v0] = lv5_red[v0] + lv5[v1, v0, v2, v3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_2"):
                    v0 = T.axis.spatial(T.int64(64), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(64))
                    T.reads(D[v0], lv5_red[v0])
                    T.writes(T_add_1[v0])
                    T_add_1[v0] = T.float32(0.90000000000000002) * D[v0] + T.float32(0.10000000000000001) * (lv5_red[v0] * T.float32(1.9929846938775509e-05))
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply_red_init"):
                    v0 = T.axis.spatial(T.int64(64), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(64))
                    T.reads()
                    T.writes(T_multiply_red[v0])
                    T_multiply_red[v0] = T.float32(0.0)
                for ax1, ax2, ax3 in T.grid(T.int64(16), T.int64(56), T.int64(56)):
                    with T.block("T_multiply_red_update"):
                        v0 = T.axis.spatial(T.int64(64), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                        v1, v2, v3 = T.axis.remap("RRR", [ax1, ax2, ax3])
                        T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(64))
                        T.reads(T_multiply_red[v0], lv5[v1, v0, v2, v3], lv5_red[v0])
                        T.writes(T_multiply_red[v0])
                        T_multiply_red[v0] = T_multiply_red[v0] + (lv5[v1, v0, v2, v3] - lv5_red[v0] * T.float32(1.9929846938775509e-05)) * (lv5[v1, v0, v2, v3] - lv5_red[v0] * T.float32(1.9929846938775509e-05))
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_3"):
                    v0 = T.axis.spatial(T.int64(64), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(64))
                    T.reads(E[v0], T_multiply_red[v0])
                    T.writes(T_add_2[v0])
                    T_add_2[v0] = T.float32(0.90000000000000002) * E[v0] + T.float32(0.10000000000000001) * (T_multiply_red[v0] * T.float32(1.9929846938775509e-05))

    @T.prim_func
    def batch_norm10(lv187: T.Buffer((T.int64(16), T.int64(512), T.int64(7), T.int64(7)), "float32"), B: T.Buffer((T.int64(512),), "float32"), C: T.Buffer((T.int64(512),), "float32"), D: T.Buffer((T.int64(512),), "float32"), E: T.Buffer((T.int64(512),), "float32"), T_add: T.Buffer((T.int64(16), T.int64(512), T.int64(7), T.int64(7)), "float32"), T_add_1: T.Buffer((T.int64(512),), "float32"), T_add_2: T.Buffer((T.int64(512),), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        lv187_red = T.alloc_buffer((T.int64(512),))
        T_multiply_red = T.alloc_buffer((T.int64(512),))
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(392), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_1"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(25088))
                    v1 = T.axis.spatial(T.int64(512), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(25088) // T.int64(49))
                    v2 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(49) // T.int64(7))
                    v3 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(7))
                    T.reads(lv187[v0, v1, v2, v3], D[v1], E[v1], B[v1], C[v1])
                    T.writes(T_add[v0, v1, v2, v3])
                    T_add[v0, v1, v2, v3] = (lv187[v0, v1, v2, v3] - D[v1]) / T.sqrt(E[v1] + T.float32(1.0000000000000001e-05)) * B[v1] + C[v1]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("lv187_red_init"):
                    v0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(512))
                    T.reads()
                    T.writes(lv187_red[v0])
                    lv187_red[v0] = T.float32(0.0)
                for ax1, ax2, ax3 in T.grid(T.int64(16), T.int64(7), T.int64(7)):
                    with T.block("lv187_red_update"):
                        v0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                        v1, v2, v3 = T.axis.remap("RRR", [ax1, ax2, ax3])
                        T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(512))
                        T.reads(lv187_red[v0], lv187[v1, v0, v2, v3])
                        T.writes(lv187_red[v0])
                        lv187_red[v0] = lv187_red[v0] + lv187[v1, v0, v2, v3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_2"):
                    v0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(512))
                    T.reads(D[v0], lv187_red[v0])
                    T.writes(T_add_1[v0])
                    T_add_1[v0] = T.float32(0.90000000000000002) * D[v0] + T.float32(0.10000000000000001) * (lv187_red[v0] * T.float32(0.0012755102040816326))
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply_red_init"):
                    v0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(512))
                    T.reads()
                    T.writes(T_multiply_red[v0])
                    T_multiply_red[v0] = T.float32(0.0)
                for ax1, ax2, ax3 in T.grid(T.int64(16), T.int64(7), T.int64(7)):
                    with T.block("T_multiply_red_update"):
                        v0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                        v1, v2, v3 = T.axis.remap("RRR", [ax1, ax2, ax3])
                        T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(512))
                        T.reads(T_multiply_red[v0], lv187[v1, v0, v2, v3], lv187_red[v0])
                        T.writes(T_multiply_red[v0])
                        T_multiply_red[v0] = T_multiply_red[v0] + (lv187[v1, v0, v2, v3] - lv187_red[v0] * T.float32(0.0012755102040816326)) * (lv187[v1, v0, v2, v3] - lv187_red[v0] * T.float32(0.0012755102040816326))
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_3"):
                    v0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(512))
                    T.reads(E[v0], T_multiply_red[v0])
                    T.writes(T_add_2[v0])
                    T_add_2[v0] = T.float32(0.90000000000000002) * E[v0] + T.float32(0.10000000000000001) * (T_multiply_red[v0] * T.float32(0.0012755102040816326))

    @T.prim_func
    def batch_norm11(lv191: T.Buffer((T.int64(16), T.int64(2048), T.int64(7), T.int64(7)), "float32"), B: T.Buffer((T.int64(2048),), "float32"), C: T.Buffer((T.int64(2048),), "float32"), D: T.Buffer((T.int64(2048),), "float32"), E: T.Buffer((T.int64(2048),), "float32"), T_add: T.Buffer((T.int64(16), T.int64(2048), T.int64(7), T.int64(7)), "float32"), T_add_1: T.Buffer((T.int64(2048),), "float32"), T_add_2: T.Buffer((T.int64(2048),), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        lv191_red = T.alloc_buffer((T.int64(2048),))
        T_multiply_red = T.alloc_buffer((T.int64(2048),))
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1568), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_1"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(100352))
                    v1 = T.axis.spatial(T.int64(2048), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(100352) // T.int64(49))
                    v2 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(49) // T.int64(7))
                    v3 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(7))
                    T.reads(lv191[v0, v1, v2, v3], D[v1], E[v1], B[v1], C[v1])
                    T.writes(T_add[v0, v1, v2, v3])
                    T_add[v0, v1, v2, v3] = (lv191[v0, v1, v2, v3] - D[v1]) / T.sqrt(E[v1] + T.float32(1.0000000000000001e-05)) * B[v1] + C[v1]
        for ax0_fused_0 in T.thread_binding(T.int64(2), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("lv191_red_init"):
                    v0 = T.axis.spatial(T.int64(2048), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.reads()
                    T.writes(lv191_red[v0])
                    lv191_red[v0] = T.float32(0.0)
                for ax1, ax2, ax3 in T.grid(T.int64(16), T.int64(7), T.int64(7)):
                    with T.block("lv191_red_update"):
                        v0 = T.axis.spatial(T.int64(2048), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                        v1, v2, v3 = T.axis.remap("RRR", [ax1, ax2, ax3])
                        T.reads(lv191_red[v0], lv191[v1, v0, v2, v3])
                        T.writes(lv191_red[v0])
                        lv191_red[v0] = lv191_red[v0] + lv191[v1, v0, v2, v3]
        for ax0_fused_0 in T.thread_binding(T.int64(2), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_2"):
                    v0 = T.axis.spatial(T.int64(2048), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.reads(D[v0], lv191_red[v0])
                    T.writes(T_add_1[v0])
                    T_add_1[v0] = T.float32(0.90000000000000002) * D[v0] + T.float32(0.10000000000000001) * (lv191_red[v0] * T.float32(0.0012755102040816326))
        for ax0_fused_0 in T.thread_binding(T.int64(2), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply_red_init"):
                    v0 = T.axis.spatial(T.int64(2048), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.reads()
                    T.writes(T_multiply_red[v0])
                    T_multiply_red[v0] = T.float32(0.0)
                for ax1, ax2, ax3 in T.grid(T.int64(16), T.int64(7), T.int64(7)):
                    with T.block("T_multiply_red_update"):
                        v0 = T.axis.spatial(T.int64(2048), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                        v1, v2, v3 = T.axis.remap("RRR", [ax1, ax2, ax3])
                        T.reads(T_multiply_red[v0], lv191[v1, v0, v2, v3], lv191_red[v0])
                        T.writes(T_multiply_red[v0])
                        T_multiply_red[v0] = T_multiply_red[v0] + (lv191[v1, v0, v2, v3] - lv191_red[v0] * T.float32(0.0012755102040816326)) * (lv191[v1, v0, v2, v3] - lv191_red[v0] * T.float32(0.0012755102040816326))
        for ax0_fused_0 in T.thread_binding(T.int64(2), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_3"):
                    v0 = T.axis.spatial(T.int64(2048), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.reads(E[v0], T_multiply_red[v0])
                    T.writes(T_add_2[v0])
                    T_add_2[v0] = T.float32(0.90000000000000002) * E[v0] + T.float32(0.10000000000000001) * (T_multiply_red[v0] * T.float32(0.0012755102040816326))

    @T.prim_func
    def batch_norm2(lv13: T.Buffer((T.int64(16), T.int64(256), T.int64(56), T.int64(56)), "float32"), B: T.Buffer((T.int64(256),), "float32"), C: T.Buffer((T.int64(256),), "float32"), D: T.Buffer((T.int64(256),), "float32"), E: T.Buffer((T.int64(256),), "float32"), T_add: T.Buffer((T.int64(16), T.int64(256), T.int64(56), T.int64(56)), "float32"), T_add_1: T.Buffer((T.int64(256),), "float32"), T_add_2: T.Buffer((T.int64(256),), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        lv13_red = T.alloc_buffer((T.int64(256),))
        T_multiply_red = T.alloc_buffer((T.int64(256),))
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(12544), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_1"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(802816))
                    v1 = T.axis.spatial(T.int64(256), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(802816) // T.int64(3136))
                    v2 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(3136) // T.int64(56))
                    v3 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(56))
                    T.reads(lv13[v0, v1, v2, v3], D[v1], E[v1], B[v1], C[v1])
                    T.writes(T_add[v0, v1, v2, v3])
                    T_add[v0, v1, v2, v3] = (lv13[v0, v1, v2, v3] - D[v1]) / T.sqrt(E[v1] + T.float32(1.0000000000000001e-05)) * B[v1] + C[v1]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("lv13_red_init"):
                    v0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(256))
                    T.reads()
                    T.writes(lv13_red[v0])
                    lv13_red[v0] = T.float32(0.0)
                for ax1, ax2, ax3 in T.grid(T.int64(16), T.int64(56), T.int64(56)):
                    with T.block("lv13_red_update"):
                        v0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                        v1, v2, v3 = T.axis.remap("RRR", [ax1, ax2, ax3])
                        T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(256))
                        T.reads(lv13_red[v0], lv13[v1, v0, v2, v3])
                        T.writes(lv13_red[v0])
                        lv13_red[v0] = lv13_red[v0] + lv13[v1, v0, v2, v3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_2"):
                    v0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(256))
                    T.reads(D[v0], lv13_red[v0])
                    T.writes(T_add_1[v0])
                    T_add_1[v0] = T.float32(0.90000000000000002) * D[v0] + T.float32(0.10000000000000001) * (lv13_red[v0] * T.float32(1.9929846938775509e-05))
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply_red_init"):
                    v0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(256))
                    T.reads()
                    T.writes(T_multiply_red[v0])
                    T_multiply_red[v0] = T.float32(0.0)
                for ax1, ax2, ax3 in T.grid(T.int64(16), T.int64(56), T.int64(56)):
                    with T.block("T_multiply_red_update"):
                        v0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                        v1, v2, v3 = T.axis.remap("RRR", [ax1, ax2, ax3])
                        T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(256))
                        T.reads(T_multiply_red[v0], lv13[v1, v0, v2, v3], lv13_red[v0])
                        T.writes(T_multiply_red[v0])
                        T_multiply_red[v0] = T_multiply_red[v0] + (lv13[v1, v0, v2, v3] - lv13_red[v0] * T.float32(1.9929846938775509e-05)) * (lv13[v1, v0, v2, v3] - lv13_red[v0] * T.float32(1.9929846938775509e-05))
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_3"):
                    v0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(256))
                    T.reads(E[v0], T_multiply_red[v0])
                    T.writes(T_add_2[v0])
                    T_add_2[v0] = T.float32(0.90000000000000002) * E[v0] + T.float32(0.10000000000000001) * (T_multiply_red[v0] * T.float32(1.9929846938775509e-05))

    @T.prim_func
    def batch_norm3(lv47: T.Buffer((T.int64(16), T.int64(128), T.int64(56), T.int64(56)), "float32"), B: T.Buffer((T.int64(128),), "float32"), C: T.Buffer((T.int64(128),), "float32"), D: T.Buffer((T.int64(128),), "float32"), E: T.Buffer((T.int64(128),), "float32"), T_add: T.Buffer((T.int64(16), T.int64(128), T.int64(56), T.int64(56)), "float32"), T_add_1: T.Buffer((T.int64(128),), "float32"), T_add_2: T.Buffer((T.int64(128),), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        lv47_red = T.alloc_buffer((T.int64(128),))
        T_multiply_red = T.alloc_buffer((T.int64(128),))
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(6272), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_1"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(401408))
                    v1 = T.axis.spatial(T.int64(128), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(401408) // T.int64(3136))
                    v2 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(3136) // T.int64(56))
                    v3 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(56))
                    T.reads(lv47[v0, v1, v2, v3], D[v1], E[v1], B[v1], C[v1])
                    T.writes(T_add[v0, v1, v2, v3])
                    T_add[v0, v1, v2, v3] = (lv47[v0, v1, v2, v3] - D[v1]) / T.sqrt(E[v1] + T.float32(1.0000000000000001e-05)) * B[v1] + C[v1]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("lv47_red_init"):
                    v0 = T.axis.spatial(T.int64(128), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(128))
                    T.reads()
                    T.writes(lv47_red[v0])
                    lv47_red[v0] = T.float32(0.0)
                for ax1, ax2, ax3 in T.grid(T.int64(16), T.int64(56), T.int64(56)):
                    with T.block("lv47_red_update"):
                        v0 = T.axis.spatial(T.int64(128), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                        v1, v2, v3 = T.axis.remap("RRR", [ax1, ax2, ax3])
                        T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(128))
                        T.reads(lv47_red[v0], lv47[v1, v0, v2, v3])
                        T.writes(lv47_red[v0])
                        lv47_red[v0] = lv47_red[v0] + lv47[v1, v0, v2, v3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_2"):
                    v0 = T.axis.spatial(T.int64(128), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(128))
                    T.reads(D[v0], lv47_red[v0])
                    T.writes(T_add_1[v0])
                    T_add_1[v0] = T.float32(0.90000000000000002) * D[v0] + T.float32(0.10000000000000001) * (lv47_red[v0] * T.float32(1.9929846938775509e-05))
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply_red_init"):
                    v0 = T.axis.spatial(T.int64(128), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(128))
                    T.reads()
                    T.writes(T_multiply_red[v0])
                    T_multiply_red[v0] = T.float32(0.0)
                for ax1, ax2, ax3 in T.grid(T.int64(16), T.int64(56), T.int64(56)):
                    with T.block("T_multiply_red_update"):
                        v0 = T.axis.spatial(T.int64(128), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                        v1, v2, v3 = T.axis.remap("RRR", [ax1, ax2, ax3])
                        T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(128))
                        T.reads(T_multiply_red[v0], lv47[v1, v0, v2, v3], lv47_red[v0])
                        T.writes(T_multiply_red[v0])
                        T_multiply_red[v0] = T_multiply_red[v0] + (lv47[v1, v0, v2, v3] - lv47_red[v0] * T.float32(1.9929846938775509e-05)) * (lv47[v1, v0, v2, v3] - lv47_red[v0] * T.float32(1.9929846938775509e-05))
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_3"):
                    v0 = T.axis.spatial(T.int64(128), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(128))
                    T.reads(E[v0], T_multiply_red[v0])
                    T.writes(T_add_2[v0])
                    T_add_2[v0] = T.float32(0.90000000000000002) * E[v0] + T.float32(0.10000000000000001) * (T_multiply_red[v0] * T.float32(1.9929846938775509e-05))

    @T.prim_func
    def batch_norm4(lv51: T.Buffer((T.int64(16), T.int64(128), T.int64(28), T.int64(28)), "float32"), B: T.Buffer((T.int64(128),), "float32"), C: T.Buffer((T.int64(128),), "float32"), D: T.Buffer((T.int64(128),), "float32"), E: T.Buffer((T.int64(128),), "float32"), T_add: T.Buffer((T.int64(16), T.int64(128), T.int64(28), T.int64(28)), "float32"), T_add_1: T.Buffer((T.int64(128),), "float32"), T_add_2: T.Buffer((T.int64(128),), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        lv51_red = T.alloc_buffer((T.int64(128),))
        T_multiply_red = T.alloc_buffer((T.int64(128),))
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1568), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_1"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(100352))
                    v1 = T.axis.spatial(T.int64(128), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(100352) // T.int64(784))
                    v2 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(784) // T.int64(28))
                    v3 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(28))
                    T.reads(lv51[v0, v1, v2, v3], D[v1], E[v1], B[v1], C[v1])
                    T.writes(T_add[v0, v1, v2, v3])
                    T_add[v0, v1, v2, v3] = (lv51[v0, v1, v2, v3] - D[v1]) / T.sqrt(E[v1] + T.float32(1.0000000000000001e-05)) * B[v1] + C[v1]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("lv51_red_init"):
                    v0 = T.axis.spatial(T.int64(128), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(128))
                    T.reads()
                    T.writes(lv51_red[v0])
                    lv51_red[v0] = T.float32(0.0)
                for ax1, ax2, ax3 in T.grid(T.int64(16), T.int64(28), T.int64(28)):
                    with T.block("lv51_red_update"):
                        v0 = T.axis.spatial(T.int64(128), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                        v1, v2, v3 = T.axis.remap("RRR", [ax1, ax2, ax3])
                        T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(128))
                        T.reads(lv51_red[v0], lv51[v1, v0, v2, v3])
                        T.writes(lv51_red[v0])
                        lv51_red[v0] = lv51_red[v0] + lv51[v1, v0, v2, v3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_2"):
                    v0 = T.axis.spatial(T.int64(128), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(128))
                    T.reads(D[v0], lv51_red[v0])
                    T.writes(T_add_1[v0])
                    T_add_1[v0] = T.float32(0.90000000000000002) * D[v0] + T.float32(0.10000000000000001) * (lv51_red[v0] * T.float32(7.9719387755102034e-05))
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply_red_init"):
                    v0 = T.axis.spatial(T.int64(128), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(128))
                    T.reads()
                    T.writes(T_multiply_red[v0])
                    T_multiply_red[v0] = T.float32(0.0)
                for ax1, ax2, ax3 in T.grid(T.int64(16), T.int64(28), T.int64(28)):
                    with T.block("T_multiply_red_update"):
                        v0 = T.axis.spatial(T.int64(128), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                        v1, v2, v3 = T.axis.remap("RRR", [ax1, ax2, ax3])
                        T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(128))
                        T.reads(T_multiply_red[v0], lv51[v1, v0, v2, v3], lv51_red[v0])
                        T.writes(T_multiply_red[v0])
                        T_multiply_red[v0] = T_multiply_red[v0] + (lv51[v1, v0, v2, v3] - lv51_red[v0] * T.float32(7.9719387755102034e-05)) * (lv51[v1, v0, v2, v3] - lv51_red[v0] * T.float32(7.9719387755102034e-05))
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_3"):
                    v0 = T.axis.spatial(T.int64(128), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(128))
                    T.reads(E[v0], T_multiply_red[v0])
                    T.writes(T_add_2[v0])
                    T_add_2[v0] = T.float32(0.90000000000000002) * E[v0] + T.float32(0.10000000000000001) * (T_multiply_red[v0] * T.float32(7.9719387755102034e-05))

    @T.prim_func
    def batch_norm5(lv55: T.Buffer((T.int64(16), T.int64(512), T.int64(28), T.int64(28)), "float32"), B: T.Buffer((T.int64(512),), "float32"), C: T.Buffer((T.int64(512),), "float32"), D: T.Buffer((T.int64(512),), "float32"), E: T.Buffer((T.int64(512),), "float32"), T_add: T.Buffer((T.int64(16), T.int64(512), T.int64(28), T.int64(28)), "float32"), T_add_1: T.Buffer((T.int64(512),), "float32"), T_add_2: T.Buffer((T.int64(512),), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        lv55_red = T.alloc_buffer((T.int64(512),))
        T_multiply_red = T.alloc_buffer((T.int64(512),))
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(6272), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_1"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(401408))
                    v1 = T.axis.spatial(T.int64(512), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(401408) // T.int64(784))
                    v2 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(784) // T.int64(28))
                    v3 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(28))
                    T.reads(lv55[v0, v1, v2, v3], D[v1], E[v1], B[v1], C[v1])
                    T.writes(T_add[v0, v1, v2, v3])
                    T_add[v0, v1, v2, v3] = (lv55[v0, v1, v2, v3] - D[v1]) / T.sqrt(E[v1] + T.float32(1.0000000000000001e-05)) * B[v1] + C[v1]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("lv55_red_init"):
                    v0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(512))
                    T.reads()
                    T.writes(lv55_red[v0])
                    lv55_red[v0] = T.float32(0.0)
                for ax1, ax2, ax3 in T.grid(T.int64(16), T.int64(28), T.int64(28)):
                    with T.block("lv55_red_update"):
                        v0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                        v1, v2, v3 = T.axis.remap("RRR", [ax1, ax2, ax3])
                        T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(512))
                        T.reads(lv55_red[v0], lv55[v1, v0, v2, v3])
                        T.writes(lv55_red[v0])
                        lv55_red[v0] = lv55_red[v0] + lv55[v1, v0, v2, v3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_2"):
                    v0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(512))
                    T.reads(D[v0], lv55_red[v0])
                    T.writes(T_add_1[v0])
                    T_add_1[v0] = T.float32(0.90000000000000002) * D[v0] + T.float32(0.10000000000000001) * (lv55_red[v0] * T.float32(7.9719387755102034e-05))
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply_red_init"):
                    v0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(512))
                    T.reads()
                    T.writes(T_multiply_red[v0])
                    T_multiply_red[v0] = T.float32(0.0)
                for ax1, ax2, ax3 in T.grid(T.int64(16), T.int64(28), T.int64(28)):
                    with T.block("T_multiply_red_update"):
                        v0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                        v1, v2, v3 = T.axis.remap("RRR", [ax1, ax2, ax3])
                        T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(512))
                        T.reads(T_multiply_red[v0], lv55[v1, v0, v2, v3], lv55_red[v0])
                        T.writes(T_multiply_red[v0])
                        T_multiply_red[v0] = T_multiply_red[v0] + (lv55[v1, v0, v2, v3] - lv55_red[v0] * T.float32(7.9719387755102034e-05)) * (lv55[v1, v0, v2, v3] - lv55_red[v0] * T.float32(7.9719387755102034e-05))
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_3"):
                    v0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(512))
                    T.reads(E[v0], T_multiply_red[v0])
                    T.writes(T_add_2[v0])
                    T_add_2[v0] = T.float32(0.90000000000000002) * E[v0] + T.float32(0.10000000000000001) * (T_multiply_red[v0] * T.float32(7.9719387755102034e-05))

    @T.prim_func
    def batch_norm6(lv102: T.Buffer((T.int64(16), T.int64(256), T.int64(28), T.int64(28)), "float32"), B: T.Buffer((T.int64(256),), "float32"), C: T.Buffer((T.int64(256),), "float32"), D: T.Buffer((T.int64(256),), "float32"), E: T.Buffer((T.int64(256),), "float32"), T_add: T.Buffer((T.int64(16), T.int64(256), T.int64(28), T.int64(28)), "float32"), T_add_1: T.Buffer((T.int64(256),), "float32"), T_add_2: T.Buffer((T.int64(256),), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        lv102_red = T.alloc_buffer((T.int64(256),))
        T_multiply_red = T.alloc_buffer((T.int64(256),))
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(3136), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_1"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(200704))
                    v1 = T.axis.spatial(T.int64(256), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(200704) // T.int64(784))
                    v2 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(784) // T.int64(28))
                    v3 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(28))
                    T.reads(lv102[v0, v1, v2, v3], D[v1], E[v1], B[v1], C[v1])
                    T.writes(T_add[v0, v1, v2, v3])
                    T_add[v0, v1, v2, v3] = (lv102[v0, v1, v2, v3] - D[v1]) / T.sqrt(E[v1] + T.float32(1.0000000000000001e-05)) * B[v1] + C[v1]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("lv102_red_init"):
                    v0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(256))
                    T.reads()
                    T.writes(lv102_red[v0])
                    lv102_red[v0] = T.float32(0.0)
                for ax1, ax2, ax3 in T.grid(T.int64(16), T.int64(28), T.int64(28)):
                    with T.block("lv102_red_update"):
                        v0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                        v1, v2, v3 = T.axis.remap("RRR", [ax1, ax2, ax3])
                        T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(256))
                        T.reads(lv102_red[v0], lv102[v1, v0, v2, v3])
                        T.writes(lv102_red[v0])
                        lv102_red[v0] = lv102_red[v0] + lv102[v1, v0, v2, v3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_2"):
                    v0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(256))
                    T.reads(D[v0], lv102_red[v0])
                    T.writes(T_add_1[v0])
                    T_add_1[v0] = T.float32(0.90000000000000002) * D[v0] + T.float32(0.10000000000000001) * (lv102_red[v0] * T.float32(7.9719387755102034e-05))
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply_red_init"):
                    v0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(256))
                    T.reads()
                    T.writes(T_multiply_red[v0])
                    T_multiply_red[v0] = T.float32(0.0)
                for ax1, ax2, ax3 in T.grid(T.int64(16), T.int64(28), T.int64(28)):
                    with T.block("T_multiply_red_update"):
                        v0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                        v1, v2, v3 = T.axis.remap("RRR", [ax1, ax2, ax3])
                        T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(256))
                        T.reads(T_multiply_red[v0], lv102[v1, v0, v2, v3], lv102_red[v0])
                        T.writes(T_multiply_red[v0])
                        T_multiply_red[v0] = T_multiply_red[v0] + (lv102[v1, v0, v2, v3] - lv102_red[v0] * T.float32(7.9719387755102034e-05)) * (lv102[v1, v0, v2, v3] - lv102_red[v0] * T.float32(7.9719387755102034e-05))
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_3"):
                    v0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(256))
                    T.reads(E[v0], T_multiply_red[v0])
                    T.writes(T_add_2[v0])
                    T_add_2[v0] = T.float32(0.90000000000000002) * E[v0] + T.float32(0.10000000000000001) * (T_multiply_red[v0] * T.float32(7.9719387755102034e-05))

    @T.prim_func
    def batch_norm7(lv106: T.Buffer((T.int64(16), T.int64(256), T.int64(14), T.int64(14)), "float32"), B: T.Buffer((T.int64(256),), "float32"), C: T.Buffer((T.int64(256),), "float32"), D: T.Buffer((T.int64(256),), "float32"), E: T.Buffer((T.int64(256),), "float32"), T_add: T.Buffer((T.int64(16), T.int64(256), T.int64(14), T.int64(14)), "float32"), T_add_1: T.Buffer((T.int64(256),), "float32"), T_add_2: T.Buffer((T.int64(256),), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        lv106_red = T.alloc_buffer((T.int64(256),))
        T_multiply_red = T.alloc_buffer((T.int64(256),))
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(784), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_1"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(50176))
                    v1 = T.axis.spatial(T.int64(256), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(50176) // T.int64(196))
                    v2 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(196) // T.int64(14))
                    v3 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(14))
                    T.reads(lv106[v0, v1, v2, v3], D[v1], E[v1], B[v1], C[v1])
                    T.writes(T_add[v0, v1, v2, v3])
                    T_add[v0, v1, v2, v3] = (lv106[v0, v1, v2, v3] - D[v1]) / T.sqrt(E[v1] + T.float32(1.0000000000000001e-05)) * B[v1] + C[v1]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("lv106_red_init"):
                    v0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(256))
                    T.reads()
                    T.writes(lv106_red[v0])
                    lv106_red[v0] = T.float32(0.0)
                for ax1, ax2, ax3 in T.grid(T.int64(16), T.int64(14), T.int64(14)):
                    with T.block("lv106_red_update"):
                        v0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                        v1, v2, v3 = T.axis.remap("RRR", [ax1, ax2, ax3])
                        T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(256))
                        T.reads(lv106_red[v0], lv106[v1, v0, v2, v3])
                        T.writes(lv106_red[v0])
                        lv106_red[v0] = lv106_red[v0] + lv106[v1, v0, v2, v3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_2"):
                    v0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(256))
                    T.reads(D[v0], lv106_red[v0])
                    T.writes(T_add_1[v0])
                    T_add_1[v0] = T.float32(0.90000000000000002) * D[v0] + T.float32(0.10000000000000001) * (lv106_red[v0] * T.float32(0.00031887755102040814))
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply_red_init"):
                    v0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(256))
                    T.reads()
                    T.writes(T_multiply_red[v0])
                    T_multiply_red[v0] = T.float32(0.0)
                for ax1, ax2, ax3 in T.grid(T.int64(16), T.int64(14), T.int64(14)):
                    with T.block("T_multiply_red_update"):
                        v0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                        v1, v2, v3 = T.axis.remap("RRR", [ax1, ax2, ax3])
                        T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(256))
                        T.reads(T_multiply_red[v0], lv106[v1, v0, v2, v3], lv106_red[v0])
                        T.writes(T_multiply_red[v0])
                        T_multiply_red[v0] = T_multiply_red[v0] + (lv106[v1, v0, v2, v3] - lv106_red[v0] * T.float32(0.00031887755102040814)) * (lv106[v1, v0, v2, v3] - lv106_red[v0] * T.float32(0.00031887755102040814))
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_3"):
                    v0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(256))
                    T.reads(E[v0], T_multiply_red[v0])
                    T.writes(T_add_2[v0])
                    T_add_2[v0] = T.float32(0.90000000000000002) * E[v0] + T.float32(0.10000000000000001) * (T_multiply_red[v0] * T.float32(0.00031887755102040814))

    @T.prim_func
    def batch_norm8(lv110: T.Buffer((T.int64(16), T.int64(1024), T.int64(14), T.int64(14)), "float32"), B: T.Buffer((T.int64(1024),), "float32"), C: T.Buffer((T.int64(1024),), "float32"), D: T.Buffer((T.int64(1024),), "float32"), E: T.Buffer((T.int64(1024),), "float32"), T_add: T.Buffer((T.int64(16), T.int64(1024), T.int64(14), T.int64(14)), "float32"), T_add_1: T.Buffer((T.int64(1024),), "float32"), T_add_2: T.Buffer((T.int64(1024),), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        lv110_red = T.alloc_buffer((T.int64(1024),))
        T_multiply_red = T.alloc_buffer((T.int64(1024),))
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(3136), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_1"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(200704))
                    v1 = T.axis.spatial(T.int64(1024), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(200704) // T.int64(196))
                    v2 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(196) // T.int64(14))
                    v3 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(14))
                    T.reads(lv110[v0, v1, v2, v3], D[v1], E[v1], B[v1], C[v1])
                    T.writes(T_add[v0, v1, v2, v3])
                    T_add[v0, v1, v2, v3] = (lv110[v0, v1, v2, v3] - D[v1]) / T.sqrt(E[v1] + T.float32(1.0000000000000001e-05)) * B[v1] + C[v1]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("lv110_red_init"):
                    v0 = T.axis.spatial(T.int64(1024), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.reads()
                    T.writes(lv110_red[v0])
                    lv110_red[v0] = T.float32(0.0)
                for ax1, ax2, ax3 in T.grid(T.int64(16), T.int64(14), T.int64(14)):
                    with T.block("lv110_red_update"):
                        v0 = T.axis.spatial(T.int64(1024), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                        v1, v2, v3 = T.axis.remap("RRR", [ax1, ax2, ax3])
                        T.reads(lv110_red[v0], lv110[v1, v0, v2, v3])
                        T.writes(lv110_red[v0])
                        lv110_red[v0] = lv110_red[v0] + lv110[v1, v0, v2, v3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_2"):
                    v0 = T.axis.spatial(T.int64(1024), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.reads(D[v0], lv110_red[v0])
                    T.writes(T_add_1[v0])
                    T_add_1[v0] = T.float32(0.90000000000000002) * D[v0] + T.float32(0.10000000000000001) * (lv110_red[v0] * T.float32(0.00031887755102040814))
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply_red_init"):
                    v0 = T.axis.spatial(T.int64(1024), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.reads()
                    T.writes(T_multiply_red[v0])
                    T_multiply_red[v0] = T.float32(0.0)
                for ax1, ax2, ax3 in T.grid(T.int64(16), T.int64(14), T.int64(14)):
                    with T.block("T_multiply_red_update"):
                        v0 = T.axis.spatial(T.int64(1024), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                        v1, v2, v3 = T.axis.remap("RRR", [ax1, ax2, ax3])
                        T.reads(T_multiply_red[v0], lv110[v1, v0, v2, v3], lv110_red[v0])
                        T.writes(T_multiply_red[v0])
                        T_multiply_red[v0] = T_multiply_red[v0] + (lv110[v1, v0, v2, v3] - lv110_red[v0] * T.float32(0.00031887755102040814)) * (lv110[v1, v0, v2, v3] - lv110_red[v0] * T.float32(0.00031887755102040814))
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_3"):
                    v0 = T.axis.spatial(T.int64(1024), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.reads(E[v0], T_multiply_red[v0])
                    T.writes(T_add_2[v0])
                    T_add_2[v0] = T.float32(0.90000000000000002) * E[v0] + T.float32(0.10000000000000001) * (T_multiply_red[v0] * T.float32(0.00031887755102040814))

    @T.prim_func
    def batch_norm9(lv183: T.Buffer((T.int64(16), T.int64(512), T.int64(14), T.int64(14)), "float32"), B: T.Buffer((T.int64(512),), "float32"), C: T.Buffer((T.int64(512),), "float32"), D: T.Buffer((T.int64(512),), "float32"), E: T.Buffer((T.int64(512),), "float32"), T_add: T.Buffer((T.int64(16), T.int64(512), T.int64(14), T.int64(14)), "float32"), T_add_1: T.Buffer((T.int64(512),), "float32"), T_add_2: T.Buffer((T.int64(512),), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        lv183_red = T.alloc_buffer((T.int64(512),))
        T_multiply_red = T.alloc_buffer((T.int64(512),))
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1568), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_1"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(100352))
                    v1 = T.axis.spatial(T.int64(512), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(100352) // T.int64(196))
                    v2 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(196) // T.int64(14))
                    v3 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(14))
                    T.reads(lv183[v0, v1, v2, v3], D[v1], E[v1], B[v1], C[v1])
                    T.writes(T_add[v0, v1, v2, v3])
                    T_add[v0, v1, v2, v3] = (lv183[v0, v1, v2, v3] - D[v1]) / T.sqrt(E[v1] + T.float32(1.0000000000000001e-05)) * B[v1] + C[v1]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("lv183_red_init"):
                    v0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(512))
                    T.reads()
                    T.writes(lv183_red[v0])
                    lv183_red[v0] = T.float32(0.0)
                for ax1, ax2, ax3 in T.grid(T.int64(16), T.int64(14), T.int64(14)):
                    with T.block("lv183_red_update"):
                        v0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                        v1, v2, v3 = T.axis.remap("RRR", [ax1, ax2, ax3])
                        T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(512))
                        T.reads(lv183_red[v0], lv183[v1, v0, v2, v3])
                        T.writes(lv183_red[v0])
                        lv183_red[v0] = lv183_red[v0] + lv183[v1, v0, v2, v3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_2"):
                    v0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(512))
                    T.reads(D[v0], lv183_red[v0])
                    T.writes(T_add_1[v0])
                    T_add_1[v0] = T.float32(0.90000000000000002) * D[v0] + T.float32(0.10000000000000001) * (lv183_red[v0] * T.float32(0.00031887755102040814))
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply_red_init"):
                    v0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(512))
                    T.reads()
                    T.writes(T_multiply_red[v0])
                    T_multiply_red[v0] = T.float32(0.0)
                for ax1, ax2, ax3 in T.grid(T.int64(16), T.int64(14), T.int64(14)):
                    with T.block("T_multiply_red_update"):
                        v0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                        v1, v2, v3 = T.axis.remap("RRR", [ax1, ax2, ax3])
                        T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(512))
                        T.reads(T_multiply_red[v0], lv183[v1, v0, v2, v3], lv183_red[v0])
                        T.writes(T_multiply_red[v0])
                        T_multiply_red[v0] = T_multiply_red[v0] + (lv183[v1, v0, v2, v3] - lv183_red[v0] * T.float32(0.00031887755102040814)) * (lv183[v1, v0, v2, v3] - lv183_red[v0] * T.float32(0.00031887755102040814))
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_3"):
                    v0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(512))
                    T.reads(E[v0], T_multiply_red[v0])
                    T.writes(T_add_2[v0])
                    T_add_2[v0] = T.float32(0.90000000000000002) * E[v0] + T.float32(0.10000000000000001) * (T_multiply_red[v0] * T.float32(0.00031887755102040814))

    @T.prim_func
    def conv2d(inp_0: T.Buffer((T.int64(16), T.int64(3), T.int64(224), T.int64(224)), "float32"), B: T.Buffer((T.int64(64), T.int64(3), T.int64(7), T.int64(7)), "float32"), conv2d_nchw: T.Buffer((T.int64(16), T.int64(64), T.int64(112), T.int64(112)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        conv2d_nchw_reindex_local = T.alloc_buffer((T.int64(1), T.int64(200704), T.int64(64)), scope="local")
        pad_temp_reindex_pad_shared = T.alloc_buffer((T.int64(1), T.int64(200704), T.int64(160)), scope="shared")
        B_reindex_pad_shared = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(160)), scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(1), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding(T.int64(6272), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_init"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(200704), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = T.axis.spatial(T.int64(64), ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            T.reads()
                                            T.writes(conv2d_nchw_reindex_local[T.int64(0), v1, v2])
                                            conv2d_nchw_reindex_local[T.int64(0), v1, v2] = T.float32(0.0)
                                for ax3_0 in range(T.int64(10)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("pad_temp_reindex_pad_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(200704), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(160), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(inp_0[v1 // T.int64(12544), v2 // T.int64(49), v1 // T.int64(112) % T.int64(112) * T.int64(2) + v2 // T.int64(7) % T.int64(7) - T.int64(3), v1 % T.int64(112) * T.int64(2) + v2 % T.int64(7) - T.int64(3)])
                                                        T.writes(pad_temp_reindex_pad_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        pad_temp_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v2 < T.int64(147), T.if_then_else(T.int64(3) <= v1 // T.int64(112) % T.int64(112) * T.int64(2) + v2 // T.int64(7) % T.int64(7) and v1 // T.int64(112) % T.int64(112) * T.int64(2) + v2 // T.int64(7) % T.int64(7) < T.int64(227) and T.int64(3) <= v1 % T.int64(112) * T.int64(2) + v2 % T.int64(7) and v1 % T.int64(112) * T.int64(2) + v2 % T.int64(7) < T.int64(227), inp_0[v1 // T.int64(12544), v2 // T.int64(49), v1 // T.int64(112) % T.int64(112) * T.int64(2) + v2 // T.int64(7) % T.int64(7) - T.int64(3), v1 % T.int64(112) * T.int64(2) + v2 % T.int64(7) - T.int64(3)], T.float32(0.0)), T.float32(0.0))
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("B_reindex_pad_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(160), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(B[v1, v2 // T.int64(49), v2 // T.int64(7) % T.int64(7), v2 % T.int64(7)])
                                                        T.writes(B_reindex_pad_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        B_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v2 < T.int64(147), B[v1, v2 // T.int64(49), v2 // T.int64(7) % T.int64(7), v2 % T.int64(7)], T.float32(0.0))
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with T.block("conv2d_nchw_update"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial(T.int64(200704), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(64), ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(160), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(conv2d_nchw_reindex_local[T.int64(0), v1, v2], pad_temp_reindex_pad_shared[T.int64(0), v1, v3], B_reindex_pad_shared[T.int64(0), v2, v3])
                                                T.writes(conv2d_nchw_reindex_local[T.int64(0), v1, v2])
                                                conv2d_nchw_reindex_local[T.int64(0), v1, v2] = conv2d_nchw_reindex_local[T.int64(0), v1, v2] + pad_temp_reindex_pad_shared[T.int64(0), v1, v3] * B_reindex_pad_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_reindex_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(T.int64(200704), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(64), ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            T.reads(conv2d_nchw_reindex_local[v0, v1, v2])
                                            T.writes(conv2d_nchw[v1 // T.int64(12544), v2, v1 // T.int64(112) % T.int64(112), v1 % T.int64(112)])
                                            conv2d_nchw[v1 // T.int64(12544), v2, v1 // T.int64(112) % T.int64(112), v1 % T.int64(112)] = conv2d_nchw_reindex_local[v0, v1, v2]

    @T.prim_func
    def conv2d1(lv4: T.Buffer((T.int64(16), T.int64(64), T.int64(56), T.int64(56)), "float32"), B: T.Buffer((T.int64(64), T.int64(64), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(16), T.int64(64), T.int64(56), T.int64(56)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        conv2d_nchw_reindex_local = T.alloc_buffer((T.int64(1), T.int64(50176), T.int64(64)), scope="local")
        pad_temp_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(50176), T.int64(64)), scope="shared")
        B_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(64)), scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(1), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding(T.int64(1568), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_init"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(50176), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = T.axis.spatial(T.int64(64), ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            T.reads()
                                            T.writes(conv2d_nchw_reindex_local[T.int64(0), v1, v2])
                                            conv2d_nchw_reindex_local[T.int64(0), v1, v2] = T.float32(0.0)
                                for ax3_0 in range(T.int64(4)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("pad_temp_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(50176), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(64), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv4[v1 // T.int64(3136), v2, v1 // T.int64(56) % T.int64(56), v1 % T.int64(56)])
                                                        T.writes(pad_temp_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        pad_temp_reindex_shared[v0, v1, v2] = lv4[v1 // T.int64(3136), v2, v1 // T.int64(56) % T.int64(56), v1 % T.int64(56)]
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("B_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(64), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(B[v1, v2, T.int64(0), T.int64(0)])
                                                        T.writes(B_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        B_reindex_shared[v0, v1, v2] = B[v1, v2, T.int64(0), T.int64(0)]
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with T.block("conv2d_nchw_update"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial(T.int64(50176), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(64), ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(64), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(conv2d_nchw_reindex_local[T.int64(0), v1, v2], pad_temp_reindex_shared[T.int64(0), v1, v3], B_reindex_shared[T.int64(0), v2, v3])
                                                T.writes(conv2d_nchw_reindex_local[T.int64(0), v1, v2])
                                                conv2d_nchw_reindex_local[T.int64(0), v1, v2] = conv2d_nchw_reindex_local[T.int64(0), v1, v2] + pad_temp_reindex_shared[T.int64(0), v1, v3] * B_reindex_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_reindex_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(T.int64(50176), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(64), ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            T.reads(conv2d_nchw_reindex_local[v0, v1, v2])
                                            T.writes(conv2d_nchw[v1 // T.int64(3136), v2, v1 // T.int64(56) % T.int64(56), v1 % T.int64(56)])
                                            conv2d_nchw[v1 // T.int64(3136), v2, v1 // T.int64(56) % T.int64(56), v1 % T.int64(56)] = conv2d_nchw_reindex_local[v0, v1, v2]

    @T.prim_func
    def conv2d10(lv66: T.Buffer((T.int64(16), T.int64(128), T.int64(28), T.int64(28)), "float32"), B: T.Buffer((T.int64(128), T.int64(128), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(16), T.int64(128), T.int64(28), T.int64(28)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        conv2d_nchw_reindex_local = T.alloc_buffer((T.int64(1), T.int64(12544), T.int64(128)), scope="local")
        pad_temp_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(12544), T.int64(1152)), scope="shared")
        B_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(1152)), scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(2), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding(T.int64(392), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_init"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(12544), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = T.axis.spatial(T.int64(128), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            T.reads()
                                            T.writes(conv2d_nchw_reindex_local[T.int64(0), v1, v2])
                                            conv2d_nchw_reindex_local[T.int64(0), v1, v2] = T.float32(0.0)
                                for ax3_0 in range(T.int64(72)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("pad_temp_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(12544), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(1152), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv66[v1 // T.int64(784), v2 // T.int64(9), v1 // T.int64(28) % T.int64(28) + v2 // T.int64(3) % T.int64(3) - T.int64(1), v1 % T.int64(28) + v2 % T.int64(3) - T.int64(1)])
                                                        T.writes(pad_temp_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        pad_temp_reindex_shared[v0, v1, v2] = T.if_then_else(T.int64(1) <= v1 // T.int64(28) % T.int64(28) + v2 // T.int64(3) % T.int64(3) and v1 // T.int64(28) % T.int64(28) + v2 // T.int64(3) % T.int64(3) < T.int64(29) and T.int64(1) <= v1 % T.int64(28) + v2 % T.int64(3) and v1 % T.int64(28) + v2 % T.int64(3) < T.int64(29), lv66[v1 // T.int64(784), v2 // T.int64(9), v1 // T.int64(28) % T.int64(28) + v2 // T.int64(3) % T.int64(3) - T.int64(1), v1 % T.int64(28) + v2 % T.int64(3) - T.int64(1)], T.float32(0.0))
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("B_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(128), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(1152), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(B[v1, v2 // T.int64(9), v2 // T.int64(3) % T.int64(3), v2 % T.int64(3)])
                                                        T.writes(B_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        B_reindex_shared[v0, v1, v2] = B[v1, v2 // T.int64(9), v2 // T.int64(3) % T.int64(3), v2 % T.int64(3)]
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with T.block("conv2d_nchw_update"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial(T.int64(12544), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(128), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(1152), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(conv2d_nchw_reindex_local[T.int64(0), v1, v2], pad_temp_reindex_shared[T.int64(0), v1, v3], B_reindex_shared[T.int64(0), v2, v3])
                                                T.writes(conv2d_nchw_reindex_local[T.int64(0), v1, v2])
                                                conv2d_nchw_reindex_local[T.int64(0), v1, v2] = conv2d_nchw_reindex_local[T.int64(0), v1, v2] + pad_temp_reindex_shared[T.int64(0), v1, v3] * B_reindex_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_reindex_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(T.int64(12544), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(128), ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            T.reads(conv2d_nchw_reindex_local[v0, v1, v2])
                                            T.writes(conv2d_nchw[v1 // T.int64(784), v2, v1 // T.int64(28) % T.int64(28), v1 % T.int64(28)])
                                            conv2d_nchw[v1 // T.int64(784), v2, v1 // T.int64(28) % T.int64(28), v1 % T.int64(28)] = conv2d_nchw_reindex_local[v0, v1, v2]

    @T.prim_func
    def conv2d11(lv101: T.Buffer((T.int64(16), T.int64(512), T.int64(28), T.int64(28)), "float32"), B: T.Buffer((T.int64(256), T.int64(512), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(16), T.int64(256), T.int64(28), T.int64(28)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        conv2d_nchw_reindex_local = T.alloc_buffer((T.int64(1), T.int64(12544), T.int64(256)), scope="local")
        pad_temp_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(12544), T.int64(512)), scope="shared")
        B_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(512)), scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(4), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding(T.int64(392), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_init"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(12544), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = T.axis.spatial(T.int64(256), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            T.reads()
                                            T.writes(conv2d_nchw_reindex_local[T.int64(0), v1, v2])
                                            conv2d_nchw_reindex_local[T.int64(0), v1, v2] = T.float32(0.0)
                                for ax3_0 in range(T.int64(32)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("pad_temp_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(12544), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(512), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv101[v1 // T.int64(784), v2, v1 // T.int64(28) % T.int64(28), v1 % T.int64(28)])
                                                        T.writes(pad_temp_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        pad_temp_reindex_shared[v0, v1, v2] = lv101[v1 // T.int64(784), v2, v1 // T.int64(28) % T.int64(28), v1 % T.int64(28)]
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("B_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(256), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(512), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(B[v1, v2, T.int64(0), T.int64(0)])
                                                        T.writes(B_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        B_reindex_shared[v0, v1, v2] = B[v1, v2, T.int64(0), T.int64(0)]
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with T.block("conv2d_nchw_update"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial(T.int64(12544), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(256), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(512), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(conv2d_nchw_reindex_local[T.int64(0), v1, v2], pad_temp_reindex_shared[T.int64(0), v1, v3], B_reindex_shared[T.int64(0), v2, v3])
                                                T.writes(conv2d_nchw_reindex_local[T.int64(0), v1, v2])
                                                conv2d_nchw_reindex_local[T.int64(0), v1, v2] = conv2d_nchw_reindex_local[T.int64(0), v1, v2] + pad_temp_reindex_shared[T.int64(0), v1, v3] * B_reindex_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_reindex_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(T.int64(12544), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(256), ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            T.reads(conv2d_nchw_reindex_local[v0, v1, v2])
                                            T.writes(conv2d_nchw[v1 // T.int64(784), v2, v1 // T.int64(28) % T.int64(28), v1 % T.int64(28)])
                                            conv2d_nchw[v1 // T.int64(784), v2, v1 // T.int64(28) % T.int64(28), v1 % T.int64(28)] = conv2d_nchw_reindex_local[v0, v1, v2]

    @T.prim_func
    def conv2d12(lv105: T.Buffer((T.int64(16), T.int64(256), T.int64(28), T.int64(28)), "float32"), B: T.Buffer((T.int64(256), T.int64(256), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(16), T.int64(256), T.int64(14), T.int64(14)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        conv2d_nchw_reindex_local = T.alloc_buffer((T.int64(1), T.int64(3136), T.int64(256)), scope="local")
        pad_temp_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(3136), T.int64(2304)), scope="shared")
        B_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(2304)), scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(4), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_init"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(3136), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = T.axis.spatial(T.int64(256), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            T.reads()
                                            T.writes(conv2d_nchw_reindex_local[T.int64(0), v1, v2])
                                            conv2d_nchw_reindex_local[T.int64(0), v1, v2] = T.float32(0.0)
                                for ax3_0 in range(T.int64(144)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("pad_temp_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(3136), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(2304), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv105[v1 // T.int64(196), v2 // T.int64(9), v1 // T.int64(14) % T.int64(14) * T.int64(2) + v2 // T.int64(3) % T.int64(3) - T.int64(1), v1 % T.int64(14) * T.int64(2) + v2 % T.int64(3) - T.int64(1)])
                                                        T.writes(pad_temp_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        pad_temp_reindex_shared[v0, v1, v2] = T.if_then_else(T.int64(1) <= v1 // T.int64(14) % T.int64(14) * T.int64(2) + v2 // T.int64(3) % T.int64(3) and v1 // T.int64(14) % T.int64(14) * T.int64(2) + v2 // T.int64(3) % T.int64(3) < T.int64(29) and T.int64(1) <= v1 % T.int64(14) * T.int64(2) + v2 % T.int64(3) and v1 % T.int64(14) * T.int64(2) + v2 % T.int64(3) < T.int64(29), lv105[v1 // T.int64(196), v2 // T.int64(9), v1 // T.int64(14) % T.int64(14) * T.int64(2) + v2 // T.int64(3) % T.int64(3) - T.int64(1), v1 % T.int64(14) * T.int64(2) + v2 % T.int64(3) - T.int64(1)], T.float32(0.0))
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("B_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(256), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(2304), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(B[v1, v2 // T.int64(9), v2 // T.int64(3) % T.int64(3), v2 % T.int64(3)])
                                                        T.writes(B_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        B_reindex_shared[v0, v1, v2] = B[v1, v2 // T.int64(9), v2 // T.int64(3) % T.int64(3), v2 % T.int64(3)]
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with T.block("conv2d_nchw_update"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial(T.int64(3136), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(256), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(2304), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(conv2d_nchw_reindex_local[T.int64(0), v1, v2], pad_temp_reindex_shared[T.int64(0), v1, v3], B_reindex_shared[T.int64(0), v2, v3])
                                                T.writes(conv2d_nchw_reindex_local[T.int64(0), v1, v2])
                                                conv2d_nchw_reindex_local[T.int64(0), v1, v2] = conv2d_nchw_reindex_local[T.int64(0), v1, v2] + pad_temp_reindex_shared[T.int64(0), v1, v3] * B_reindex_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_reindex_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(T.int64(3136), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(256), ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            T.reads(conv2d_nchw_reindex_local[v0, v1, v2])
                                            T.writes(conv2d_nchw[v1 // T.int64(196), v2, v1 // T.int64(14) % T.int64(14), v1 % T.int64(14)])
                                            conv2d_nchw[v1 // T.int64(196), v2, v1 // T.int64(14) % T.int64(14), v1 % T.int64(14)] = conv2d_nchw_reindex_local[v0, v1, v2]

    @T.prim_func
    def conv2d13(lv109: T.Buffer((T.int64(16), T.int64(256), T.int64(14), T.int64(14)), "float32"), B: T.Buffer((T.int64(1024), T.int64(256), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(16), T.int64(1024), T.int64(14), T.int64(14)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        conv2d_nchw_reindex_local = T.alloc_buffer((T.int64(1), T.int64(3136), T.int64(1024)), scope="local")
        pad_temp_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(3136), T.int64(256)), scope="shared")
        B_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(1024), T.int64(256)), scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(16), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_init"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(3136), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = T.axis.spatial(T.int64(1024), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            T.reads()
                                            T.writes(conv2d_nchw_reindex_local[T.int64(0), v1, v2])
                                            conv2d_nchw_reindex_local[T.int64(0), v1, v2] = T.float32(0.0)
                                for ax3_0 in range(T.int64(16)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("pad_temp_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(3136), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(256), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv109[v1 // T.int64(196), v2, v1 // T.int64(14) % T.int64(14), v1 % T.int64(14)])
                                                        T.writes(pad_temp_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        pad_temp_reindex_shared[v0, v1, v2] = lv109[v1 // T.int64(196), v2, v1 // T.int64(14) % T.int64(14), v1 % T.int64(14)]
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("B_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(1024), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(256), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(B[v1, v2, T.int64(0), T.int64(0)])
                                                        T.writes(B_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        B_reindex_shared[v0, v1, v2] = B[v1, v2, T.int64(0), T.int64(0)]
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with T.block("conv2d_nchw_update"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial(T.int64(3136), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(1024), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(256), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(conv2d_nchw_reindex_local[T.int64(0), v1, v2], pad_temp_reindex_shared[T.int64(0), v1, v3], B_reindex_shared[T.int64(0), v2, v3])
                                                T.writes(conv2d_nchw_reindex_local[T.int64(0), v1, v2])
                                                conv2d_nchw_reindex_local[T.int64(0), v1, v2] = conv2d_nchw_reindex_local[T.int64(0), v1, v2] + pad_temp_reindex_shared[T.int64(0), v1, v3] * B_reindex_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_reindex_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(T.int64(3136), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(1024), ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            T.reads(conv2d_nchw_reindex_local[v0, v1, v2])
                                            T.writes(conv2d_nchw[v1 // T.int64(196), v2, v1 // T.int64(14) % T.int64(14), v1 % T.int64(14)])
                                            conv2d_nchw[v1 // T.int64(196), v2, v1 // T.int64(14) % T.int64(14), v1 % T.int64(14)] = conv2d_nchw_reindex_local[v0, v1, v2]

    @T.prim_func
    def conv2d14(lv101: T.Buffer((T.int64(16), T.int64(512), T.int64(28), T.int64(28)), "float32"), B: T.Buffer((T.int64(1024), T.int64(512), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(16), T.int64(1024), T.int64(14), T.int64(14)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        conv2d_nchw_reindex_local = T.alloc_buffer((T.int64(1), T.int64(3136), T.int64(1024)), scope="local")
        pad_temp_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(3136), T.int64(512)), scope="shared")
        B_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(1024), T.int64(512)), scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(16), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_init"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(3136), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = T.axis.spatial(T.int64(1024), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            T.reads()
                                            T.writes(conv2d_nchw_reindex_local[T.int64(0), v1, v2])
                                            conv2d_nchw_reindex_local[T.int64(0), v1, v2] = T.float32(0.0)
                                for ax3_0 in range(T.int64(32)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("pad_temp_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(3136), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(512), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv101[v1 // T.int64(196), v2, v1 // T.int64(14) % T.int64(14) * T.int64(2), v1 % T.int64(14) * T.int64(2)])
                                                        T.writes(pad_temp_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        pad_temp_reindex_shared[v0, v1, v2] = lv101[v1 // T.int64(196), v2, v1 // T.int64(14) % T.int64(14) * T.int64(2), v1 % T.int64(14) * T.int64(2)]
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("B_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(1024), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(512), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(B[v1, v2, T.int64(0), T.int64(0)])
                                                        T.writes(B_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        B_reindex_shared[v0, v1, v2] = B[v1, v2, T.int64(0), T.int64(0)]
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with T.block("conv2d_nchw_update"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial(T.int64(3136), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(1024), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(512), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(conv2d_nchw_reindex_local[T.int64(0), v1, v2], pad_temp_reindex_shared[T.int64(0), v1, v3], B_reindex_shared[T.int64(0), v2, v3])
                                                T.writes(conv2d_nchw_reindex_local[T.int64(0), v1, v2])
                                                conv2d_nchw_reindex_local[T.int64(0), v1, v2] = conv2d_nchw_reindex_local[T.int64(0), v1, v2] + pad_temp_reindex_shared[T.int64(0), v1, v3] * B_reindex_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_reindex_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(T.int64(3136), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(1024), ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            T.reads(conv2d_nchw_reindex_local[v0, v1, v2])
                                            T.writes(conv2d_nchw[v1 // T.int64(196), v2, v1 // T.int64(14) % T.int64(14), v1 % T.int64(14)])
                                            conv2d_nchw[v1 // T.int64(196), v2, v1 // T.int64(14) % T.int64(14), v1 % T.int64(14)] = conv2d_nchw_reindex_local[v0, v1, v2]

    @T.prim_func
    def conv2d15(lv117: T.Buffer((T.int64(16), T.int64(1024), T.int64(14), T.int64(14)), "float32"), B: T.Buffer((T.int64(256), T.int64(1024), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(16), T.int64(256), T.int64(14), T.int64(14)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        conv2d_nchw_reindex_local = T.alloc_buffer((T.int64(1), T.int64(3136), T.int64(256)), scope="local")
        pad_temp_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(3136), T.int64(1024)), scope="shared")
        B_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(1024)), scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(4), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_init"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(3136), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = T.axis.spatial(T.int64(256), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            T.reads()
                                            T.writes(conv2d_nchw_reindex_local[T.int64(0), v1, v2])
                                            conv2d_nchw_reindex_local[T.int64(0), v1, v2] = T.float32(0.0)
                                for ax3_0 in range(T.int64(64)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("pad_temp_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(3136), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(1024), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv117[v1 // T.int64(196), v2, v1 // T.int64(14) % T.int64(14), v1 % T.int64(14)])
                                                        T.writes(pad_temp_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        pad_temp_reindex_shared[v0, v1, v2] = lv117[v1 // T.int64(196), v2, v1 // T.int64(14) % T.int64(14), v1 % T.int64(14)]
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("B_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(256), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(1024), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(B[v1, v2, T.int64(0), T.int64(0)])
                                                        T.writes(B_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        B_reindex_shared[v0, v1, v2] = B[v1, v2, T.int64(0), T.int64(0)]
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with T.block("conv2d_nchw_update"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial(T.int64(3136), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(256), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(1024), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(conv2d_nchw_reindex_local[T.int64(0), v1, v2], pad_temp_reindex_shared[T.int64(0), v1, v3], B_reindex_shared[T.int64(0), v2, v3])
                                                T.writes(conv2d_nchw_reindex_local[T.int64(0), v1, v2])
                                                conv2d_nchw_reindex_local[T.int64(0), v1, v2] = conv2d_nchw_reindex_local[T.int64(0), v1, v2] + pad_temp_reindex_shared[T.int64(0), v1, v3] * B_reindex_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_reindex_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(T.int64(3136), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(256), ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            T.reads(conv2d_nchw_reindex_local[v0, v1, v2])
                                            T.writes(conv2d_nchw[v1 // T.int64(196), v2, v1 // T.int64(14) % T.int64(14), v1 % T.int64(14)])
                                            conv2d_nchw[v1 // T.int64(196), v2, v1 // T.int64(14) % T.int64(14), v1 % T.int64(14)] = conv2d_nchw_reindex_local[v0, v1, v2]

    @T.prim_func
    def conv2d16(lv121: T.Buffer((T.int64(16), T.int64(256), T.int64(14), T.int64(14)), "float32"), B: T.Buffer((T.int64(256), T.int64(256), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(16), T.int64(256), T.int64(14), T.int64(14)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        conv2d_nchw_reindex_local = T.alloc_buffer((T.int64(1), T.int64(3136), T.int64(256)), scope="local")
        pad_temp_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(3136), T.int64(2304)), scope="shared")
        B_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(2304)), scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(4), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_init"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(3136), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = T.axis.spatial(T.int64(256), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            T.reads()
                                            T.writes(conv2d_nchw_reindex_local[T.int64(0), v1, v2])
                                            conv2d_nchw_reindex_local[T.int64(0), v1, v2] = T.float32(0.0)
                                for ax3_0 in range(T.int64(144)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("pad_temp_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(3136), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(2304), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv121[v1 // T.int64(196), v2 // T.int64(9), v1 // T.int64(14) % T.int64(14) + v2 // T.int64(3) % T.int64(3) - T.int64(1), v1 % T.int64(14) + v2 % T.int64(3) - T.int64(1)])
                                                        T.writes(pad_temp_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        pad_temp_reindex_shared[v0, v1, v2] = T.if_then_else(T.int64(1) <= v1 // T.int64(14) % T.int64(14) + v2 // T.int64(3) % T.int64(3) and v1 // T.int64(14) % T.int64(14) + v2 // T.int64(3) % T.int64(3) < T.int64(15) and T.int64(1) <= v1 % T.int64(14) + v2 % T.int64(3) and v1 % T.int64(14) + v2 % T.int64(3) < T.int64(15), lv121[v1 // T.int64(196), v2 // T.int64(9), v1 // T.int64(14) % T.int64(14) + v2 // T.int64(3) % T.int64(3) - T.int64(1), v1 % T.int64(14) + v2 % T.int64(3) - T.int64(1)], T.float32(0.0))
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("B_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(256), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(2304), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(B[v1, v2 // T.int64(9), v2 // T.int64(3) % T.int64(3), v2 % T.int64(3)])
                                                        T.writes(B_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        B_reindex_shared[v0, v1, v2] = B[v1, v2 // T.int64(9), v2 // T.int64(3) % T.int64(3), v2 % T.int64(3)]
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with T.block("conv2d_nchw_update"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial(T.int64(3136), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(256), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(2304), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(conv2d_nchw_reindex_local[T.int64(0), v1, v2], pad_temp_reindex_shared[T.int64(0), v1, v3], B_reindex_shared[T.int64(0), v2, v3])
                                                T.writes(conv2d_nchw_reindex_local[T.int64(0), v1, v2])
                                                conv2d_nchw_reindex_local[T.int64(0), v1, v2] = conv2d_nchw_reindex_local[T.int64(0), v1, v2] + pad_temp_reindex_shared[T.int64(0), v1, v3] * B_reindex_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_reindex_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(T.int64(3136), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(256), ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            T.reads(conv2d_nchw_reindex_local[v0, v1, v2])
                                            T.writes(conv2d_nchw[v1 // T.int64(196), v2, v1 // T.int64(14) % T.int64(14), v1 % T.int64(14)])
                                            conv2d_nchw[v1 // T.int64(196), v2, v1 // T.int64(14) % T.int64(14), v1 % T.int64(14)] = conv2d_nchw_reindex_local[v0, v1, v2]

    @T.prim_func
    def conv2d17(lv182: T.Buffer((T.int64(16), T.int64(1024), T.int64(14), T.int64(14)), "float32"), B: T.Buffer((T.int64(512), T.int64(1024), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(16), T.int64(512), T.int64(14), T.int64(14)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        conv2d_nchw_reindex_local = T.alloc_buffer((T.int64(1), T.int64(3136), T.int64(512)), scope="local")
        pad_temp_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(3136), T.int64(1024)), scope="shared")
        B_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(1024)), scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(8), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding(T.int64(98), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_init"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(3136), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = T.axis.spatial(T.int64(512), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            T.reads()
                                            T.writes(conv2d_nchw_reindex_local[T.int64(0), v1, v2])
                                            conv2d_nchw_reindex_local[T.int64(0), v1, v2] = T.float32(0.0)
                                for ax3_0 in range(T.int64(64)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("pad_temp_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(3136), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(1024), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv182[v1 // T.int64(196), v2, v1 // T.int64(14) % T.int64(14), v1 % T.int64(14)])
                                                        T.writes(pad_temp_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        pad_temp_reindex_shared[v0, v1, v2] = lv182[v1 // T.int64(196), v2, v1 // T.int64(14) % T.int64(14), v1 % T.int64(14)]
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("B_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(512), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(1024), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(B[v1, v2, T.int64(0), T.int64(0)])
                                                        T.writes(B_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        B_reindex_shared[v0, v1, v2] = B[v1, v2, T.int64(0), T.int64(0)]
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with T.block("conv2d_nchw_update"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial(T.int64(3136), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(512), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(1024), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(conv2d_nchw_reindex_local[T.int64(0), v1, v2], pad_temp_reindex_shared[T.int64(0), v1, v3], B_reindex_shared[T.int64(0), v2, v3])
                                                T.writes(conv2d_nchw_reindex_local[T.int64(0), v1, v2])
                                                conv2d_nchw_reindex_local[T.int64(0), v1, v2] = conv2d_nchw_reindex_local[T.int64(0), v1, v2] + pad_temp_reindex_shared[T.int64(0), v1, v3] * B_reindex_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_reindex_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(T.int64(3136), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(512), ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            T.reads(conv2d_nchw_reindex_local[v0, v1, v2])
                                            T.writes(conv2d_nchw[v1 // T.int64(196), v2, v1 // T.int64(14) % T.int64(14), v1 % T.int64(14)])
                                            conv2d_nchw[v1 // T.int64(196), v2, v1 // T.int64(14) % T.int64(14), v1 % T.int64(14)] = conv2d_nchw_reindex_local[v0, v1, v2]

    @T.prim_func
    def conv2d18(lv186: T.Buffer((T.int64(16), T.int64(512), T.int64(14), T.int64(14)), "float32"), B: T.Buffer((T.int64(512), T.int64(512), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(16), T.int64(512), T.int64(7), T.int64(7)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        conv2d_nchw_reindex_pad_local = T.alloc_buffer((T.int64(1), T.int64(800), T.int64(512)), scope="local")
        pad_temp_reindex_pad_shared = T.alloc_buffer((T.int64(1), T.int64(800), T.int64(4608)), scope="shared")
        B_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(4608)), scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(8), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding(T.int64(25), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_init"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(800), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = T.axis.spatial(T.int64(512), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            T.reads()
                                            T.writes(conv2d_nchw_reindex_pad_local[T.int64(0), v1, v2])
                                            conv2d_nchw_reindex_pad_local[T.int64(0), v1, v2] = T.float32(0.0)
                                for ax3_0 in range(T.int64(288)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("pad_temp_reindex_pad_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(800), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(4608), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv186[v1 // T.int64(49), v2 // T.int64(9), v1 // T.int64(7) % T.int64(7) * T.int64(2) + v2 // T.int64(3) % T.int64(3) - T.int64(1), v1 % T.int64(7) * T.int64(2) + v2 % T.int64(3) - T.int64(1)])
                                                        T.writes(pad_temp_reindex_pad_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        pad_temp_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v1 < T.int64(784), T.if_then_else(T.int64(1) <= v1 // T.int64(7) % T.int64(7) * T.int64(2) + v2 // T.int64(3) % T.int64(3) and v1 // T.int64(7) % T.int64(7) * T.int64(2) + v2 // T.int64(3) % T.int64(3) < T.int64(15) and T.int64(1) <= v1 % T.int64(7) * T.int64(2) + v2 % T.int64(3) and v1 % T.int64(7) * T.int64(2) + v2 % T.int64(3) < T.int64(15), lv186[v1 // T.int64(49), v2 // T.int64(9), v1 // T.int64(7) % T.int64(7) * T.int64(2) + v2 // T.int64(3) % T.int64(3) - T.int64(1), v1 % T.int64(7) * T.int64(2) + v2 % T.int64(3) - T.int64(1)], T.float32(0.0)), T.float32(0.0))
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("B_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(512), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(4608), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(B[v1, v2 // T.int64(9), v2 // T.int64(3) % T.int64(3), v2 % T.int64(3)])
                                                        T.writes(B_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        B_reindex_shared[v0, v1, v2] = B[v1, v2 // T.int64(9), v2 // T.int64(3) % T.int64(3), v2 % T.int64(3)]
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with T.block("conv2d_nchw_update"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial(T.int64(800), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(512), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(4608), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(conv2d_nchw_reindex_pad_local[T.int64(0), v1, v2], pad_temp_reindex_pad_shared[T.int64(0), v1, v3], B_reindex_shared[T.int64(0), v2, v3])
                                                T.writes(conv2d_nchw_reindex_pad_local[T.int64(0), v1, v2])
                                                conv2d_nchw_reindex_pad_local[T.int64(0), v1, v2] = conv2d_nchw_reindex_pad_local[T.int64(0), v1, v2] + pad_temp_reindex_pad_shared[T.int64(0), v1, v3] * B_reindex_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_reindex_pad_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(T.int64(800), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(512), ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            T.where(ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1 < T.int64(784))
                                            T.reads(conv2d_nchw_reindex_pad_local[v0, v1, v2])
                                            T.writes(conv2d_nchw[v1 // T.int64(49), v2, v1 // T.int64(7) % T.int64(7), v1 % T.int64(7)])
                                            conv2d_nchw[v1 // T.int64(49), v2, v1 // T.int64(7) % T.int64(7), v1 % T.int64(7)] = conv2d_nchw_reindex_pad_local[v0, v1, v2]

    @T.prim_func
    def conv2d19(lv190: T.Buffer((T.int64(16), T.int64(512), T.int64(7), T.int64(7)), "float32"), B: T.Buffer((T.int64(2048), T.int64(512), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(16), T.int64(2048), T.int64(7), T.int64(7)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        conv2d_nchw_reindex_pad_local = T.alloc_buffer((T.int64(1), T.int64(800), T.int64(2048)), scope="local")
        pad_temp_reindex_pad_shared = T.alloc_buffer((T.int64(1), T.int64(800), T.int64(512)), scope="shared")
        B_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(2048), T.int64(512)), scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(32), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding(T.int64(25), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_init"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(800), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = T.axis.spatial(T.int64(2048), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            T.reads()
                                            T.writes(conv2d_nchw_reindex_pad_local[T.int64(0), v1, v2])
                                            conv2d_nchw_reindex_pad_local[T.int64(0), v1, v2] = T.float32(0.0)
                                for ax3_0 in range(T.int64(32)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("pad_temp_reindex_pad_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(800), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(512), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv190[v1 // T.int64(49), v2, v1 // T.int64(7) % T.int64(7), v1 % T.int64(7)])
                                                        T.writes(pad_temp_reindex_pad_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        pad_temp_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v1 < T.int64(784), lv190[v1 // T.int64(49), v2, v1 // T.int64(7) % T.int64(7), v1 % T.int64(7)], T.float32(0.0))
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("B_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(2048), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(512), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(B[v1, v2, T.int64(0), T.int64(0)])
                                                        T.writes(B_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        B_reindex_shared[v0, v1, v2] = B[v1, v2, T.int64(0), T.int64(0)]
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with T.block("conv2d_nchw_update"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial(T.int64(800), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(2048), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(512), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(conv2d_nchw_reindex_pad_local[T.int64(0), v1, v2], pad_temp_reindex_pad_shared[T.int64(0), v1, v3], B_reindex_shared[T.int64(0), v2, v3])
                                                T.writes(conv2d_nchw_reindex_pad_local[T.int64(0), v1, v2])
                                                conv2d_nchw_reindex_pad_local[T.int64(0), v1, v2] = conv2d_nchw_reindex_pad_local[T.int64(0), v1, v2] + pad_temp_reindex_pad_shared[T.int64(0), v1, v3] * B_reindex_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_reindex_pad_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(T.int64(800), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(2048), ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            T.where(ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1 < T.int64(784))
                                            T.reads(conv2d_nchw_reindex_pad_local[v0, v1, v2])
                                            T.writes(conv2d_nchw[v1 // T.int64(49), v2, v1 // T.int64(7) % T.int64(7), v1 % T.int64(7)])
                                            conv2d_nchw[v1 // T.int64(49), v2, v1 // T.int64(7) % T.int64(7), v1 % T.int64(7)] = conv2d_nchw_reindex_pad_local[v0, v1, v2]

    @T.prim_func
    def conv2d2(lv8: T.Buffer((T.int64(16), T.int64(64), T.int64(56), T.int64(56)), "float32"), B: T.Buffer((T.int64(64), T.int64(64), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(16), T.int64(64), T.int64(56), T.int64(56)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        conv2d_nchw_reindex_local = T.alloc_buffer((T.int64(1), T.int64(50176), T.int64(64)), scope="local")
        pad_temp_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(50176), T.int64(576)), scope="shared")
        B_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(576)), scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(1), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding(T.int64(1568), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_init"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(50176), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = T.axis.spatial(T.int64(64), ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            T.reads()
                                            T.writes(conv2d_nchw_reindex_local[T.int64(0), v1, v2])
                                            conv2d_nchw_reindex_local[T.int64(0), v1, v2] = T.float32(0.0)
                                for ax3_0 in range(T.int64(36)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("pad_temp_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(50176), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(576), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv8[v1 // T.int64(3136), v2 // T.int64(9), v1 // T.int64(56) % T.int64(56) + v2 // T.int64(3) % T.int64(3) - T.int64(1), v1 % T.int64(56) + v2 % T.int64(3) - T.int64(1)])
                                                        T.writes(pad_temp_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        pad_temp_reindex_shared[v0, v1, v2] = T.if_then_else(T.int64(1) <= v1 // T.int64(56) % T.int64(56) + v2 // T.int64(3) % T.int64(3) and v1 // T.int64(56) % T.int64(56) + v2 // T.int64(3) % T.int64(3) < T.int64(57) and T.int64(1) <= v1 % T.int64(56) + v2 % T.int64(3) and v1 % T.int64(56) + v2 % T.int64(3) < T.int64(57), lv8[v1 // T.int64(3136), v2 // T.int64(9), v1 // T.int64(56) % T.int64(56) + v2 // T.int64(3) % T.int64(3) - T.int64(1), v1 % T.int64(56) + v2 % T.int64(3) - T.int64(1)], T.float32(0.0))
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("B_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(576), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(B[v1, v2 // T.int64(9), v2 // T.int64(3) % T.int64(3), v2 % T.int64(3)])
                                                        T.writes(B_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        B_reindex_shared[v0, v1, v2] = B[v1, v2 // T.int64(9), v2 // T.int64(3) % T.int64(3), v2 % T.int64(3)]
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with T.block("conv2d_nchw_update"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial(T.int64(50176), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(64), ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(576), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(conv2d_nchw_reindex_local[T.int64(0), v1, v2], pad_temp_reindex_shared[T.int64(0), v1, v3], B_reindex_shared[T.int64(0), v2, v3])
                                                T.writes(conv2d_nchw_reindex_local[T.int64(0), v1, v2])
                                                conv2d_nchw_reindex_local[T.int64(0), v1, v2] = conv2d_nchw_reindex_local[T.int64(0), v1, v2] + pad_temp_reindex_shared[T.int64(0), v1, v3] * B_reindex_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_reindex_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(T.int64(50176), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(64), ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            T.reads(conv2d_nchw_reindex_local[v0, v1, v2])
                                            T.writes(conv2d_nchw[v1 // T.int64(3136), v2, v1 // T.int64(56) % T.int64(56), v1 % T.int64(56)])
                                            conv2d_nchw[v1 // T.int64(3136), v2, v1 // T.int64(56) % T.int64(56), v1 % T.int64(56)] = conv2d_nchw_reindex_local[v0, v1, v2]

    @T.prim_func
    def conv2d20(lv182: T.Buffer((T.int64(16), T.int64(1024), T.int64(14), T.int64(14)), "float32"), B: T.Buffer((T.int64(2048), T.int64(1024), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(16), T.int64(2048), T.int64(7), T.int64(7)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        conv2d_nchw_reindex_pad_local = T.alloc_buffer((T.int64(1), T.int64(800), T.int64(2048)), scope="local")
        pad_temp_reindex_pad_shared = T.alloc_buffer((T.int64(1), T.int64(800), T.int64(1024)), scope="shared")
        B_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(2048), T.int64(1024)), scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(32), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding(T.int64(25), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_init"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(800), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = T.axis.spatial(T.int64(2048), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            T.reads()
                                            T.writes(conv2d_nchw_reindex_pad_local[T.int64(0), v1, v2])
                                            conv2d_nchw_reindex_pad_local[T.int64(0), v1, v2] = T.float32(0.0)
                                for ax3_0 in range(T.int64(64)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("pad_temp_reindex_pad_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(800), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(1024), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv182[v1 // T.int64(49), v2, v1 // T.int64(7) % T.int64(7) * T.int64(2), v1 % T.int64(7) * T.int64(2)])
                                                        T.writes(pad_temp_reindex_pad_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        pad_temp_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v1 < T.int64(784), lv182[v1 // T.int64(49), v2, v1 // T.int64(7) % T.int64(7) * T.int64(2), v1 % T.int64(7) * T.int64(2)], T.float32(0.0))
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("B_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(2048), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(1024), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(B[v1, v2, T.int64(0), T.int64(0)])
                                                        T.writes(B_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        B_reindex_shared[v0, v1, v2] = B[v1, v2, T.int64(0), T.int64(0)]
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with T.block("conv2d_nchw_update"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial(T.int64(800), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(2048), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(1024), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(conv2d_nchw_reindex_pad_local[T.int64(0), v1, v2], pad_temp_reindex_pad_shared[T.int64(0), v1, v3], B_reindex_shared[T.int64(0), v2, v3])
                                                T.writes(conv2d_nchw_reindex_pad_local[T.int64(0), v1, v2])
                                                conv2d_nchw_reindex_pad_local[T.int64(0), v1, v2] = conv2d_nchw_reindex_pad_local[T.int64(0), v1, v2] + pad_temp_reindex_pad_shared[T.int64(0), v1, v3] * B_reindex_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_reindex_pad_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(T.int64(800), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(2048), ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            T.where(ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1 < T.int64(784))
                                            T.reads(conv2d_nchw_reindex_pad_local[v0, v1, v2])
                                            T.writes(conv2d_nchw[v1 // T.int64(49), v2, v1 // T.int64(7) % T.int64(7), v1 % T.int64(7)])
                                            conv2d_nchw[v1 // T.int64(49), v2, v1 // T.int64(7) % T.int64(7), v1 % T.int64(7)] = conv2d_nchw_reindex_pad_local[v0, v1, v2]

    @T.prim_func
    def conv2d21(lv198: T.Buffer((T.int64(16), T.int64(2048), T.int64(7), T.int64(7)), "float32"), B: T.Buffer((T.int64(512), T.int64(2048), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(16), T.int64(512), T.int64(7), T.int64(7)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        conv2d_nchw_reindex_pad_local = T.alloc_buffer((T.int64(1), T.int64(800), T.int64(512)), scope="local")
        pad_temp_reindex_pad_shared = T.alloc_buffer((T.int64(1), T.int64(800), T.int64(2048)), scope="shared")
        B_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(2048)), scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(8), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding(T.int64(25), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_init"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(800), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = T.axis.spatial(T.int64(512), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            T.reads()
                                            T.writes(conv2d_nchw_reindex_pad_local[T.int64(0), v1, v2])
                                            conv2d_nchw_reindex_pad_local[T.int64(0), v1, v2] = T.float32(0.0)
                                for ax3_0 in range(T.int64(128)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("pad_temp_reindex_pad_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(800), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(2048), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv198[v1 // T.int64(49), v2, v1 // T.int64(7) % T.int64(7), v1 % T.int64(7)])
                                                        T.writes(pad_temp_reindex_pad_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        pad_temp_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v1 < T.int64(784), lv198[v1 // T.int64(49), v2, v1 // T.int64(7) % T.int64(7), v1 % T.int64(7)], T.float32(0.0))
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("B_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(512), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(2048), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(B[v1, v2, T.int64(0), T.int64(0)])
                                                        T.writes(B_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        B_reindex_shared[v0, v1, v2] = B[v1, v2, T.int64(0), T.int64(0)]
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with T.block("conv2d_nchw_update"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial(T.int64(800), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(512), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(2048), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(conv2d_nchw_reindex_pad_local[T.int64(0), v1, v2], pad_temp_reindex_pad_shared[T.int64(0), v1, v3], B_reindex_shared[T.int64(0), v2, v3])
                                                T.writes(conv2d_nchw_reindex_pad_local[T.int64(0), v1, v2])
                                                conv2d_nchw_reindex_pad_local[T.int64(0), v1, v2] = conv2d_nchw_reindex_pad_local[T.int64(0), v1, v2] + pad_temp_reindex_pad_shared[T.int64(0), v1, v3] * B_reindex_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_reindex_pad_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(T.int64(800), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(512), ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            T.where(ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1 < T.int64(784))
                                            T.reads(conv2d_nchw_reindex_pad_local[v0, v1, v2])
                                            T.writes(conv2d_nchw[v1 // T.int64(49), v2, v1 // T.int64(7) % T.int64(7), v1 % T.int64(7)])
                                            conv2d_nchw[v1 // T.int64(49), v2, v1 // T.int64(7) % T.int64(7), v1 % T.int64(7)] = conv2d_nchw_reindex_pad_local[v0, v1, v2]

    @T.prim_func
    def conv2d22(lv202: T.Buffer((T.int64(16), T.int64(512), T.int64(7), T.int64(7)), "float32"), B: T.Buffer((T.int64(512), T.int64(512), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(16), T.int64(512), T.int64(7), T.int64(7)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        conv2d_nchw_reindex_pad_local = T.alloc_buffer((T.int64(1), T.int64(800), T.int64(512)), scope="local")
        pad_temp_reindex_pad_shared = T.alloc_buffer((T.int64(1), T.int64(800), T.int64(4608)), scope="shared")
        B_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(4608)), scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(8), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding(T.int64(25), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_init"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(800), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = T.axis.spatial(T.int64(512), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            T.reads()
                                            T.writes(conv2d_nchw_reindex_pad_local[T.int64(0), v1, v2])
                                            conv2d_nchw_reindex_pad_local[T.int64(0), v1, v2] = T.float32(0.0)
                                for ax3_0 in range(T.int64(288)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("pad_temp_reindex_pad_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(800), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(4608), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv202[v1 // T.int64(49), v2 // T.int64(9), v1 // T.int64(7) % T.int64(7) + v2 // T.int64(3) % T.int64(3) - T.int64(1), v1 % T.int64(7) + v2 % T.int64(3) - T.int64(1)])
                                                        T.writes(pad_temp_reindex_pad_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        pad_temp_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v1 < T.int64(784), T.if_then_else(T.int64(1) <= v1 // T.int64(7) % T.int64(7) + v2 // T.int64(3) % T.int64(3) and v1 // T.int64(7) % T.int64(7) + v2 // T.int64(3) % T.int64(3) < T.int64(8) and T.int64(1) <= v1 % T.int64(7) + v2 % T.int64(3) and v1 % T.int64(7) + v2 % T.int64(3) < T.int64(8), lv202[v1 // T.int64(49), v2 // T.int64(9), v1 // T.int64(7) % T.int64(7) + v2 // T.int64(3) % T.int64(3) - T.int64(1), v1 % T.int64(7) + v2 % T.int64(3) - T.int64(1)], T.float32(0.0)), T.float32(0.0))
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("B_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(512), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(4608), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(B[v1, v2 // T.int64(9), v2 // T.int64(3) % T.int64(3), v2 % T.int64(3)])
                                                        T.writes(B_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        B_reindex_shared[v0, v1, v2] = B[v1, v2 // T.int64(9), v2 // T.int64(3) % T.int64(3), v2 % T.int64(3)]
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with T.block("conv2d_nchw_update"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial(T.int64(800), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(512), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(4608), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(conv2d_nchw_reindex_pad_local[T.int64(0), v1, v2], pad_temp_reindex_pad_shared[T.int64(0), v1, v3], B_reindex_shared[T.int64(0), v2, v3])
                                                T.writes(conv2d_nchw_reindex_pad_local[T.int64(0), v1, v2])
                                                conv2d_nchw_reindex_pad_local[T.int64(0), v1, v2] = conv2d_nchw_reindex_pad_local[T.int64(0), v1, v2] + pad_temp_reindex_pad_shared[T.int64(0), v1, v3] * B_reindex_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_reindex_pad_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(T.int64(800), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(512), ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            T.where(ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1 < T.int64(784))
                                            T.reads(conv2d_nchw_reindex_pad_local[v0, v1, v2])
                                            T.writes(conv2d_nchw[v1 // T.int64(49), v2, v1 // T.int64(7) % T.int64(7), v1 % T.int64(7)])
                                            conv2d_nchw[v1 // T.int64(49), v2, v1 // T.int64(7) % T.int64(7), v1 % T.int64(7)] = conv2d_nchw_reindex_pad_local[v0, v1, v2]

    @T.prim_func
    def conv2d3(lv12: T.Buffer((T.int64(16), T.int64(64), T.int64(56), T.int64(56)), "float32"), B: T.Buffer((T.int64(256), T.int64(64), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(16), T.int64(256), T.int64(56), T.int64(56)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        conv2d_nchw_reindex_local = T.alloc_buffer((T.int64(1), T.int64(50176), T.int64(256)), scope="local")
        pad_temp_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(50176), T.int64(64)), scope="shared")
        B_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(64)), scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(4), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding(T.int64(1568), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_init"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(50176), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = T.axis.spatial(T.int64(256), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            T.reads()
                                            T.writes(conv2d_nchw_reindex_local[T.int64(0), v1, v2])
                                            conv2d_nchw_reindex_local[T.int64(0), v1, v2] = T.float32(0.0)
                                for ax3_0 in range(T.int64(4)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("pad_temp_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(50176), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(64), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv12[v1 // T.int64(3136), v2, v1 // T.int64(56) % T.int64(56), v1 % T.int64(56)])
                                                        T.writes(pad_temp_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        pad_temp_reindex_shared[v0, v1, v2] = lv12[v1 // T.int64(3136), v2, v1 // T.int64(56) % T.int64(56), v1 % T.int64(56)]
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("B_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(256), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(64), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(B[v1, v2, T.int64(0), T.int64(0)])
                                                        T.writes(B_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        B_reindex_shared[v0, v1, v2] = B[v1, v2, T.int64(0), T.int64(0)]
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with T.block("conv2d_nchw_update"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial(T.int64(50176), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(256), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(64), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(conv2d_nchw_reindex_local[T.int64(0), v1, v2], pad_temp_reindex_shared[T.int64(0), v1, v3], B_reindex_shared[T.int64(0), v2, v3])
                                                T.writes(conv2d_nchw_reindex_local[T.int64(0), v1, v2])
                                                conv2d_nchw_reindex_local[T.int64(0), v1, v2] = conv2d_nchw_reindex_local[T.int64(0), v1, v2] + pad_temp_reindex_shared[T.int64(0), v1, v3] * B_reindex_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_reindex_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(T.int64(50176), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(256), ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            T.reads(conv2d_nchw_reindex_local[v0, v1, v2])
                                            T.writes(conv2d_nchw[v1 // T.int64(3136), v2, v1 // T.int64(56) % T.int64(56), v1 % T.int64(56)])
                                            conv2d_nchw[v1 // T.int64(3136), v2, v1 // T.int64(56) % T.int64(56), v1 % T.int64(56)] = conv2d_nchw_reindex_local[v0, v1, v2]

    @T.prim_func
    def conv2d4(lv20: T.Buffer((T.int64(16), T.int64(256), T.int64(56), T.int64(56)), "float32"), B: T.Buffer((T.int64(64), T.int64(256), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(16), T.int64(64), T.int64(56), T.int64(56)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        conv2d_nchw_reindex_local = T.alloc_buffer((T.int64(1), T.int64(50176), T.int64(64)), scope="local")
        pad_temp_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(50176), T.int64(256)), scope="shared")
        B_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(256)), scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(1), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding(T.int64(1568), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_init"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(50176), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = T.axis.spatial(T.int64(64), ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            T.reads()
                                            T.writes(conv2d_nchw_reindex_local[T.int64(0), v1, v2])
                                            conv2d_nchw_reindex_local[T.int64(0), v1, v2] = T.float32(0.0)
                                for ax3_0 in range(T.int64(16)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("pad_temp_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(50176), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(256), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv20[v1 // T.int64(3136), v2, v1 // T.int64(56) % T.int64(56), v1 % T.int64(56)])
                                                        T.writes(pad_temp_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        pad_temp_reindex_shared[v0, v1, v2] = lv20[v1 // T.int64(3136), v2, v1 // T.int64(56) % T.int64(56), v1 % T.int64(56)]
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("B_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(256), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(B[v1, v2, T.int64(0), T.int64(0)])
                                                        T.writes(B_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        B_reindex_shared[v0, v1, v2] = B[v1, v2, T.int64(0), T.int64(0)]
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with T.block("conv2d_nchw_update"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial(T.int64(50176), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(64), ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(256), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(conv2d_nchw_reindex_local[T.int64(0), v1, v2], pad_temp_reindex_shared[T.int64(0), v1, v3], B_reindex_shared[T.int64(0), v2, v3])
                                                T.writes(conv2d_nchw_reindex_local[T.int64(0), v1, v2])
                                                conv2d_nchw_reindex_local[T.int64(0), v1, v2] = conv2d_nchw_reindex_local[T.int64(0), v1, v2] + pad_temp_reindex_shared[T.int64(0), v1, v3] * B_reindex_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_reindex_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(T.int64(50176), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(64), ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            T.reads(conv2d_nchw_reindex_local[v0, v1, v2])
                                            T.writes(conv2d_nchw[v1 // T.int64(3136), v2, v1 // T.int64(56) % T.int64(56), v1 % T.int64(56)])
                                            conv2d_nchw[v1 // T.int64(3136), v2, v1 // T.int64(56) % T.int64(56), v1 % T.int64(56)] = conv2d_nchw_reindex_local[v0, v1, v2]

    @T.prim_func
    def conv2d5(lv46: T.Buffer((T.int64(16), T.int64(256), T.int64(56), T.int64(56)), "float32"), B: T.Buffer((T.int64(128), T.int64(256), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(16), T.int64(128), T.int64(56), T.int64(56)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        conv2d_nchw_reindex_local = T.alloc_buffer((T.int64(1), T.int64(50176), T.int64(128)), scope="local")
        pad_temp_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(50176), T.int64(256)), scope="shared")
        B_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(256)), scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(2), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding(T.int64(1568), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_init"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(50176), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = T.axis.spatial(T.int64(128), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            T.reads()
                                            T.writes(conv2d_nchw_reindex_local[T.int64(0), v1, v2])
                                            conv2d_nchw_reindex_local[T.int64(0), v1, v2] = T.float32(0.0)
                                for ax3_0 in range(T.int64(16)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("pad_temp_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(50176), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(256), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv46[v1 // T.int64(3136), v2, v1 // T.int64(56) % T.int64(56), v1 % T.int64(56)])
                                                        T.writes(pad_temp_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        pad_temp_reindex_shared[v0, v1, v2] = lv46[v1 // T.int64(3136), v2, v1 // T.int64(56) % T.int64(56), v1 % T.int64(56)]
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("B_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(128), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(256), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(B[v1, v2, T.int64(0), T.int64(0)])
                                                        T.writes(B_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        B_reindex_shared[v0, v1, v2] = B[v1, v2, T.int64(0), T.int64(0)]
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with T.block("conv2d_nchw_update"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial(T.int64(50176), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(128), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(256), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(conv2d_nchw_reindex_local[T.int64(0), v1, v2], pad_temp_reindex_shared[T.int64(0), v1, v3], B_reindex_shared[T.int64(0), v2, v3])
                                                T.writes(conv2d_nchw_reindex_local[T.int64(0), v1, v2])
                                                conv2d_nchw_reindex_local[T.int64(0), v1, v2] = conv2d_nchw_reindex_local[T.int64(0), v1, v2] + pad_temp_reindex_shared[T.int64(0), v1, v3] * B_reindex_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_reindex_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(T.int64(50176), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(128), ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            T.reads(conv2d_nchw_reindex_local[v0, v1, v2])
                                            T.writes(conv2d_nchw[v1 // T.int64(3136), v2, v1 // T.int64(56) % T.int64(56), v1 % T.int64(56)])
                                            conv2d_nchw[v1 // T.int64(3136), v2, v1 // T.int64(56) % T.int64(56), v1 % T.int64(56)] = conv2d_nchw_reindex_local[v0, v1, v2]

    @T.prim_func
    def conv2d6(lv50: T.Buffer((T.int64(16), T.int64(128), T.int64(56), T.int64(56)), "float32"), B: T.Buffer((T.int64(128), T.int64(128), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(16), T.int64(128), T.int64(28), T.int64(28)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        conv2d_nchw_reindex_local = T.alloc_buffer((T.int64(1), T.int64(12544), T.int64(128)), scope="local")
        pad_temp_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(12544), T.int64(1152)), scope="shared")
        B_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(1152)), scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(2), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding(T.int64(392), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_init"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(12544), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = T.axis.spatial(T.int64(128), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            T.reads()
                                            T.writes(conv2d_nchw_reindex_local[T.int64(0), v1, v2])
                                            conv2d_nchw_reindex_local[T.int64(0), v1, v2] = T.float32(0.0)
                                for ax3_0 in range(T.int64(72)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("pad_temp_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(12544), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(1152), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv50[v1 // T.int64(784), v2 // T.int64(9), v1 // T.int64(28) % T.int64(28) * T.int64(2) + v2 // T.int64(3) % T.int64(3) - T.int64(1), v1 % T.int64(28) * T.int64(2) + v2 % T.int64(3) - T.int64(1)])
                                                        T.writes(pad_temp_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        pad_temp_reindex_shared[v0, v1, v2] = T.if_then_else(T.int64(1) <= v1 // T.int64(28) % T.int64(28) * T.int64(2) + v2 // T.int64(3) % T.int64(3) and v1 // T.int64(28) % T.int64(28) * T.int64(2) + v2 // T.int64(3) % T.int64(3) < T.int64(57) and T.int64(1) <= v1 % T.int64(28) * T.int64(2) + v2 % T.int64(3) and v1 % T.int64(28) * T.int64(2) + v2 % T.int64(3) < T.int64(57), lv50[v1 // T.int64(784), v2 // T.int64(9), v1 // T.int64(28) % T.int64(28) * T.int64(2) + v2 // T.int64(3) % T.int64(3) - T.int64(1), v1 % T.int64(28) * T.int64(2) + v2 % T.int64(3) - T.int64(1)], T.float32(0.0))
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("B_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(128), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(1152), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(B[v1, v2 // T.int64(9), v2 // T.int64(3) % T.int64(3), v2 % T.int64(3)])
                                                        T.writes(B_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        B_reindex_shared[v0, v1, v2] = B[v1, v2 // T.int64(9), v2 // T.int64(3) % T.int64(3), v2 % T.int64(3)]
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with T.block("conv2d_nchw_update"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial(T.int64(12544), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(128), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(1152), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(conv2d_nchw_reindex_local[T.int64(0), v1, v2], pad_temp_reindex_shared[T.int64(0), v1, v3], B_reindex_shared[T.int64(0), v2, v3])
                                                T.writes(conv2d_nchw_reindex_local[T.int64(0), v1, v2])
                                                conv2d_nchw_reindex_local[T.int64(0), v1, v2] = conv2d_nchw_reindex_local[T.int64(0), v1, v2] + pad_temp_reindex_shared[T.int64(0), v1, v3] * B_reindex_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_reindex_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(T.int64(12544), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(128), ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            T.reads(conv2d_nchw_reindex_local[v0, v1, v2])
                                            T.writes(conv2d_nchw[v1 // T.int64(784), v2, v1 // T.int64(28) % T.int64(28), v1 % T.int64(28)])
                                            conv2d_nchw[v1 // T.int64(784), v2, v1 // T.int64(28) % T.int64(28), v1 % T.int64(28)] = conv2d_nchw_reindex_local[v0, v1, v2]

    @T.prim_func
    def conv2d7(lv54: T.Buffer((T.int64(16), T.int64(128), T.int64(28), T.int64(28)), "float32"), B: T.Buffer((T.int64(512), T.int64(128), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(16), T.int64(512), T.int64(28), T.int64(28)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        conv2d_nchw_reindex_local = T.alloc_buffer((T.int64(1), T.int64(12544), T.int64(512)), scope="local")
        pad_temp_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(12544), T.int64(128)), scope="shared")
        B_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(128)), scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(8), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding(T.int64(392), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_init"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(12544), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = T.axis.spatial(T.int64(512), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            T.reads()
                                            T.writes(conv2d_nchw_reindex_local[T.int64(0), v1, v2])
                                            conv2d_nchw_reindex_local[T.int64(0), v1, v2] = T.float32(0.0)
                                for ax3_0 in range(T.int64(8)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("pad_temp_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(12544), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(128), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv54[v1 // T.int64(784), v2, v1 // T.int64(28) % T.int64(28), v1 % T.int64(28)])
                                                        T.writes(pad_temp_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        pad_temp_reindex_shared[v0, v1, v2] = lv54[v1 // T.int64(784), v2, v1 // T.int64(28) % T.int64(28), v1 % T.int64(28)]
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("B_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(512), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(128), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(B[v1, v2, T.int64(0), T.int64(0)])
                                                        T.writes(B_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        B_reindex_shared[v0, v1, v2] = B[v1, v2, T.int64(0), T.int64(0)]
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with T.block("conv2d_nchw_update"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial(T.int64(12544), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(512), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(128), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(conv2d_nchw_reindex_local[T.int64(0), v1, v2], pad_temp_reindex_shared[T.int64(0), v1, v3], B_reindex_shared[T.int64(0), v2, v3])
                                                T.writes(conv2d_nchw_reindex_local[T.int64(0), v1, v2])
                                                conv2d_nchw_reindex_local[T.int64(0), v1, v2] = conv2d_nchw_reindex_local[T.int64(0), v1, v2] + pad_temp_reindex_shared[T.int64(0), v1, v3] * B_reindex_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_reindex_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(T.int64(12544), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(512), ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            T.reads(conv2d_nchw_reindex_local[v0, v1, v2])
                                            T.writes(conv2d_nchw[v1 // T.int64(784), v2, v1 // T.int64(28) % T.int64(28), v1 % T.int64(28)])
                                            conv2d_nchw[v1 // T.int64(784), v2, v1 // T.int64(28) % T.int64(28), v1 % T.int64(28)] = conv2d_nchw_reindex_local[v0, v1, v2]

    @T.prim_func
    def conv2d8(lv46: T.Buffer((T.int64(16), T.int64(256), T.int64(56), T.int64(56)), "float32"), B: T.Buffer((T.int64(512), T.int64(256), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(16), T.int64(512), T.int64(28), T.int64(28)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        conv2d_nchw_reindex_local = T.alloc_buffer((T.int64(1), T.int64(12544), T.int64(512)), scope="local")
        pad_temp_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(12544), T.int64(256)), scope="shared")
        B_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(256)), scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(8), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding(T.int64(392), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_init"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(12544), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = T.axis.spatial(T.int64(512), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            T.reads()
                                            T.writes(conv2d_nchw_reindex_local[T.int64(0), v1, v2])
                                            conv2d_nchw_reindex_local[T.int64(0), v1, v2] = T.float32(0.0)
                                for ax3_0 in range(T.int64(16)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("pad_temp_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(12544), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(256), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv46[v1 // T.int64(784), v2, v1 // T.int64(28) % T.int64(28) * T.int64(2), v1 % T.int64(28) * T.int64(2)])
                                                        T.writes(pad_temp_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        pad_temp_reindex_shared[v0, v1, v2] = lv46[v1 // T.int64(784), v2, v1 // T.int64(28) % T.int64(28) * T.int64(2), v1 % T.int64(28) * T.int64(2)]
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("B_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(512), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(256), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(B[v1, v2, T.int64(0), T.int64(0)])
                                                        T.writes(B_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        B_reindex_shared[v0, v1, v2] = B[v1, v2, T.int64(0), T.int64(0)]
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with T.block("conv2d_nchw_update"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial(T.int64(12544), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(512), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(256), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(conv2d_nchw_reindex_local[T.int64(0), v1, v2], pad_temp_reindex_shared[T.int64(0), v1, v3], B_reindex_shared[T.int64(0), v2, v3])
                                                T.writes(conv2d_nchw_reindex_local[T.int64(0), v1, v2])
                                                conv2d_nchw_reindex_local[T.int64(0), v1, v2] = conv2d_nchw_reindex_local[T.int64(0), v1, v2] + pad_temp_reindex_shared[T.int64(0), v1, v3] * B_reindex_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_reindex_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(T.int64(12544), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(512), ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            T.reads(conv2d_nchw_reindex_local[v0, v1, v2])
                                            T.writes(conv2d_nchw[v1 // T.int64(784), v2, v1 // T.int64(28) % T.int64(28), v1 % T.int64(28)])
                                            conv2d_nchw[v1 // T.int64(784), v2, v1 // T.int64(28) % T.int64(28), v1 % T.int64(28)] = conv2d_nchw_reindex_local[v0, v1, v2]

    @T.prim_func
    def conv2d9(lv62: T.Buffer((T.int64(16), T.int64(512), T.int64(28), T.int64(28)), "float32"), B: T.Buffer((T.int64(128), T.int64(512), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(16), T.int64(128), T.int64(28), T.int64(28)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        conv2d_nchw_reindex_local = T.alloc_buffer((T.int64(1), T.int64(12544), T.int64(128)), scope="local")
        pad_temp_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(12544), T.int64(512)), scope="shared")
        B_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(512)), scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(2), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding(T.int64(392), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_init"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(12544), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = T.axis.spatial(T.int64(128), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            T.reads()
                                            T.writes(conv2d_nchw_reindex_local[T.int64(0), v1, v2])
                                            conv2d_nchw_reindex_local[T.int64(0), v1, v2] = T.float32(0.0)
                                for ax3_0 in range(T.int64(32)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("pad_temp_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(12544), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(512), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv62[v1 // T.int64(784), v2, v1 // T.int64(28) % T.int64(28), v1 % T.int64(28)])
                                                        T.writes(pad_temp_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        pad_temp_reindex_shared[v0, v1, v2] = lv62[v1 // T.int64(784), v2, v1 // T.int64(28) % T.int64(28), v1 % T.int64(28)]
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("B_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(128), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(512), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(B[v1, v2, T.int64(0), T.int64(0)])
                                                        T.writes(B_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        B_reindex_shared[v0, v1, v2] = B[v1, v2, T.int64(0), T.int64(0)]
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with T.block("conv2d_nchw_update"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial(T.int64(12544), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(128), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(512), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(conv2d_nchw_reindex_local[T.int64(0), v1, v2], pad_temp_reindex_shared[T.int64(0), v1, v3], B_reindex_shared[T.int64(0), v2, v3])
                                                T.writes(conv2d_nchw_reindex_local[T.int64(0), v1, v2])
                                                conv2d_nchw_reindex_local[T.int64(0), v1, v2] = conv2d_nchw_reindex_local[T.int64(0), v1, v2] + pad_temp_reindex_shared[T.int64(0), v1, v3] * B_reindex_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_reindex_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(T.int64(12544), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(128), ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            T.reads(conv2d_nchw_reindex_local[v0, v1, v2])
                                            T.writes(conv2d_nchw[v1 // T.int64(784), v2, v1 // T.int64(28) % T.int64(28), v1 % T.int64(28)])
                                            conv2d_nchw[v1 // T.int64(784), v2, v1 // T.int64(28) % T.int64(28), v1 % T.int64(28)] = conv2d_nchw_reindex_local[v0, v1, v2]

    @T.prim_func
    def fused_NT_matmul_add4(lv226: T.Buffer((T.int64(16), T.int64(2048)), "float32"), param_0: T.Buffer((T.int64(1000), T.int64(2048)), "float32"), param_1: T.Buffer((T.int64(1000),), "float32"), T_add_intermediate: T.Buffer((T.int64(16), T.int64(1000)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        NT_matmul_intermediate_reindex_pad_local = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1024)), scope="local")
        lv226_reindex_pad_shared = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(2048)), scope="shared")
        param_0_reindex_pad_shared = T.alloc_buffer((T.int64(1), T.int64(1024), T.int64(2048)), scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(16), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with T.block("NT_matmul_init"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(32), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = T.axis.spatial(T.int64(1024), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            T.reads()
                                            T.writes(NT_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2])
                                            NT_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2] = T.float32(0.0)
                                for ax3_0 in range(T.int64(128)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("lv226_reindex_pad_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(32), (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(2048), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv226[v1, v2])
                                                        T.writes(lv226_reindex_pad_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        lv226_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v1 < T.int64(16), lv226[v1, v2], T.float32(0.0))
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("param_0_reindex_pad_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(1024), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(2048), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(param_0[v1, v2])
                                                        T.writes(param_0_reindex_pad_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        param_0_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v1 < T.int64(1000), param_0[v1, v2], T.float32(0.0))
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with T.block("NT_matmul_update"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial(T.int64(32), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(1024), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(2048), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(NT_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2], lv226_reindex_pad_shared[T.int64(0), v1, v3], param_0_reindex_pad_shared[T.int64(0), v2, v3])
                                                T.writes(NT_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2])
                                                NT_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2] = NT_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2] + lv226_reindex_pad_shared[T.int64(0), v1, v3] * param_0_reindex_pad_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("NT_matmul_intermediate_reindex_pad_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(T.int64(32), ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(1024), ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            T.where(ax1_2 * T.int64(4) + ax1 < T.int64(16) and ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1 < T.int64(1000))
                                            T.reads(NT_matmul_intermediate_reindex_pad_local[v0, v1, v2], param_1[v2])
                                            T.writes(T_add_intermediate[v1, v2])
                                            T_add_intermediate[v1, v2] = NT_matmul_intermediate_reindex_pad_local[v0, v1, v2] + param_1[v2]

    @T.prim_func
    def fused_add1_relu5(lv56_0: T.Buffer((T.int64(16), T.int64(512), T.int64(28), T.int64(28)), "float32"), lv59_0: T.Buffer((T.int64(16), T.int64(512), T.int64(28), T.int64(28)), "float32"), compute_intermediate: T.Buffer((T.int64(16), T.int64(512), T.int64(28), T.int64(28)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(6272), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(401408))
                    v1 = T.axis.spatial(T.int64(512), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(401408) // T.int64(784))
                    v2 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(784) // T.int64(28))
                    v3 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(28))
                    T.reads(lv56_0[v0, v1, v2, v3], lv59_0[v0, v1, v2, v3])
                    T.writes(compute_intermediate[v0, v1, v2, v3])
                    compute_intermediate[v0, v1, v2, v3] = T.max(lv56_0[v0, v1, v2, v3] + lv59_0[v0, v1, v2, v3], T.float32(0.0))

    @T.prim_func
    def fused_add2_relu8(lv111_0: T.Buffer((T.int64(16), T.int64(1024), T.int64(14), T.int64(14)), "float32"), lv114_0: T.Buffer((T.int64(16), T.int64(1024), T.int64(14), T.int64(14)), "float32"), compute_intermediate: T.Buffer((T.int64(16), T.int64(1024), T.int64(14), T.int64(14)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(3136), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(200704))
                    v1 = T.axis.spatial(T.int64(1024), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(200704) // T.int64(196))
                    v2 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(196) // T.int64(14))
                    v3 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(14))
                    T.reads(lv111_0[v0, v1, v2, v3], lv114_0[v0, v1, v2, v3])
                    T.writes(compute_intermediate[v0, v1, v2, v3])
                    compute_intermediate[v0, v1, v2, v3] = T.max(lv111_0[v0, v1, v2, v3] + lv114_0[v0, v1, v2, v3], T.float32(0.0))

    @T.prim_func
    def fused_add3_relu11(lv192_0: T.Buffer((T.int64(16), T.int64(2048), T.int64(7), T.int64(7)), "float32"), lv195_0: T.Buffer((T.int64(16), T.int64(2048), T.int64(7), T.int64(7)), "float32"), compute_intermediate: T.Buffer((T.int64(16), T.int64(2048), T.int64(7), T.int64(7)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1568), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(100352))
                    v1 = T.axis.spatial(T.int64(2048), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(100352) // T.int64(49))
                    v2 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(49) // T.int64(7))
                    v3 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(7))
                    T.reads(lv192_0[v0, v1, v2, v3], lv195_0[v0, v1, v2, v3])
                    T.writes(compute_intermediate[v0, v1, v2, v3])
                    compute_intermediate[v0, v1, v2, v3] = T.max(lv192_0[v0, v1, v2, v3] + lv195_0[v0, v1, v2, v3], T.float32(0.0))

    @T.prim_func
    def fused_add_relu2(lv14_0: T.Buffer((T.int64(16), T.int64(256), T.int64(56), T.int64(56)), "float32"), lv17_0: T.Buffer((T.int64(16), T.int64(256), T.int64(56), T.int64(56)), "float32"), compute_intermediate: T.Buffer((T.int64(16), T.int64(256), T.int64(56), T.int64(56)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(12544), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(802816))
                    v1 = T.axis.spatial(T.int64(256), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(802816) // T.int64(3136))
                    v2 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(3136) // T.int64(56))
                    v3 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(56))
                    T.reads(lv14_0[v0, v1, v2, v3], lv17_0[v0, v1, v2, v3])
                    T.writes(compute_intermediate[v0, v1, v2, v3])
                    compute_intermediate[v0, v1, v2, v3] = T.max(lv14_0[v0, v1, v2, v3] + lv17_0[v0, v1, v2, v3], T.float32(0.0))

    @T.prim_func
    def fused_relu(lv1_0: T.Buffer((T.int64(16), T.int64(64), T.int64(112), T.int64(112)), "float32"), compute_intermediate: T.Buffer((T.int64(16), T.int64(64), T.int64(112), T.int64(112)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(12544), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(802816))
                    v1 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(802816) // T.int64(12544))
                    v2 = T.axis.spatial(T.int64(112), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(12544) // T.int64(112))
                    v3 = T.axis.spatial(T.int64(112), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(112))
                    T.reads(lv1_0[v0, v1, v2, v3])
                    T.writes(compute_intermediate[v0, v1, v2, v3])
                    compute_intermediate[v0, v1, v2, v3] = T.max(lv1_0[v0, v1, v2, v3], T.float32(0.0))

    @T.prim_func
    def fused_relu1(lv6_0: T.Buffer((T.int64(16), T.int64(64), T.int64(56), T.int64(56)), "float32"), compute_intermediate: T.Buffer((T.int64(16), T.int64(64), T.int64(56), T.int64(56)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(3136), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(200704))
                    v1 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(200704) // T.int64(3136))
                    v2 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(3136) // T.int64(56))
                    v3 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(56))
                    T.reads(lv6_0[v0, v1, v2, v3])
                    T.writes(compute_intermediate[v0, v1, v2, v3])
                    compute_intermediate[v0, v1, v2, v3] = T.max(lv6_0[v0, v1, v2, v3], T.float32(0.0))

    @T.prim_func
    def fused_relu10(lv188_0: T.Buffer((T.int64(16), T.int64(512), T.int64(7), T.int64(7)), "float32"), compute_intermediate: T.Buffer((T.int64(16), T.int64(512), T.int64(7), T.int64(7)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(392), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(25088))
                    v1 = T.axis.spatial(T.int64(512), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(25088) // T.int64(49))
                    v2 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(49) // T.int64(7))
                    v3 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(7))
                    T.reads(lv188_0[v0, v1, v2, v3])
                    T.writes(compute_intermediate[v0, v1, v2, v3])
                    compute_intermediate[v0, v1, v2, v3] = T.max(lv188_0[v0, v1, v2, v3], T.float32(0.0))

    @T.prim_func
    def fused_relu3(lv48_0: T.Buffer((T.int64(16), T.int64(128), T.int64(56), T.int64(56)), "float32"), compute_intermediate: T.Buffer((T.int64(16), T.int64(128), T.int64(56), T.int64(56)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(6272), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(401408))
                    v1 = T.axis.spatial(T.int64(128), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(401408) // T.int64(3136))
                    v2 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(3136) // T.int64(56))
                    v3 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(56))
                    T.reads(lv48_0[v0, v1, v2, v3])
                    T.writes(compute_intermediate[v0, v1, v2, v3])
                    compute_intermediate[v0, v1, v2, v3] = T.max(lv48_0[v0, v1, v2, v3], T.float32(0.0))

    @T.prim_func
    def fused_relu4(lv52_0: T.Buffer((T.int64(16), T.int64(128), T.int64(28), T.int64(28)), "float32"), compute_intermediate: T.Buffer((T.int64(16), T.int64(128), T.int64(28), T.int64(28)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1568), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(100352))
                    v1 = T.axis.spatial(T.int64(128), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(100352) // T.int64(784))
                    v2 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(784) // T.int64(28))
                    v3 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(28))
                    T.reads(lv52_0[v0, v1, v2, v3])
                    T.writes(compute_intermediate[v0, v1, v2, v3])
                    compute_intermediate[v0, v1, v2, v3] = T.max(lv52_0[v0, v1, v2, v3], T.float32(0.0))

    @T.prim_func
    def fused_relu6(lv103_0: T.Buffer((T.int64(16), T.int64(256), T.int64(28), T.int64(28)), "float32"), compute_intermediate: T.Buffer((T.int64(16), T.int64(256), T.int64(28), T.int64(28)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(3136), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(200704))
                    v1 = T.axis.spatial(T.int64(256), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(200704) // T.int64(784))
                    v2 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(784) // T.int64(28))
                    v3 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(28))
                    T.reads(lv103_0[v0, v1, v2, v3])
                    T.writes(compute_intermediate[v0, v1, v2, v3])
                    compute_intermediate[v0, v1, v2, v3] = T.max(lv103_0[v0, v1, v2, v3], T.float32(0.0))

    @T.prim_func
    def fused_relu7(lv107_0: T.Buffer((T.int64(16), T.int64(256), T.int64(14), T.int64(14)), "float32"), compute_intermediate: T.Buffer((T.int64(16), T.int64(256), T.int64(14), T.int64(14)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(784), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(50176))
                    v1 = T.axis.spatial(T.int64(256), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(50176) // T.int64(196))
                    v2 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(196) // T.int64(14))
                    v3 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(14))
                    T.reads(lv107_0[v0, v1, v2, v3])
                    T.writes(compute_intermediate[v0, v1, v2, v3])
                    compute_intermediate[v0, v1, v2, v3] = T.max(lv107_0[v0, v1, v2, v3], T.float32(0.0))

    @T.prim_func
    def fused_relu9(lv184_0: T.Buffer((T.int64(16), T.int64(512), T.int64(14), T.int64(14)), "float32"), compute_intermediate: T.Buffer((T.int64(16), T.int64(512), T.int64(14), T.int64(14)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1568), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(100352))
                    v1 = T.axis.spatial(T.int64(512), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(100352) // T.int64(196))
                    v2 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(196) // T.int64(14))
                    v3 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(14))
                    T.reads(lv184_0[v0, v1, v2, v3])
                    T.writes(compute_intermediate[v0, v1, v2, v3])
                    compute_intermediate[v0, v1, v2, v3] = T.max(lv184_0[v0, v1, v2, v3], T.float32(0.0))

    @T.prim_func
    def max_pool2d(lv3: T.Buffer((T.int64(16), T.int64(64), T.int64(112), T.int64(112)), "float32"), pool_max: T.Buffer((T.int64(16), T.int64(64), T.int64(56), T.int64(56)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(3136), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("pool_max_init"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(200704))
                    v1 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(200704) // T.int64(3136))
                    v2 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(3136) // T.int64(56))
                    v3 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(56))
                    T.reads()
                    T.writes(pool_max[v0, v1, v2, v3])
                    T.block_attr({"schedule_rule": "meta_schedule.pool_max"})
                    pool_max[v0, v1, v2, v3] = T.float32(-340282346638528859811704183484516925440.0)
                for ax4, ax5 in T.grid(T.int64(3), T.int64(3)):
                    with T.block("pool_max_update"):
                        v0 = T.axis.spatial(T.int64(16), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(200704))
                        v1 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(200704) // T.int64(3136))
                        v2 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(3136) // T.int64(56))
                        v3 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(56))
                        v4, v5 = T.axis.remap("RR", [ax4, ax5])
                        T.reads(pool_max[v0, v1, v2, v3], lv3[v0, v1, v2 * T.int64(2) + v4 - T.int64(1), v3 * T.int64(2) + v5 - T.int64(1)])
                        T.writes(pool_max[v0, v1, v2, v3])
                        T.block_attr({"schedule_rule": "meta_schedule.pool_max"})
                        pool_max[v0, v1, v2, v3] = T.max(pool_max[v0, v1, v2, v3], T.if_then_else(T.int64(1) <= v2 * T.int64(2) + v4 and v2 * T.int64(2) + v4 < T.int64(113) and T.int64(1) <= v3 * T.int64(2) + v5 and v3 * T.int64(2) + v5 < T.int64(113), lv3[v0, v1, v2 * T.int64(2) + v4 - T.int64(1), v3 * T.int64(2) + v5 - T.int64(1)], T.float32(-340282346638528859811704183484516925440.0)))

    @T.prim_func
    def reshape(lv225: T.Buffer((T.int64(16), T.int64(2048), T.int64(1), T.int64(1)), "float32"), T_reshape: T.Buffer((T.int64(16), T.int64(2048)), "float32")):
        T.func_attr({"op_pattern": 2, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(32), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_reshape"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(2048))
                    v1 = T.axis.spatial(T.int64(2048), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(2048))
                    T.reads(lv225[v0, v1, T.int64(0), T.int64(0)])
                    T.writes(T_reshape[v0, v1])
                    T_reshape[v0, v1] = lv225[v0, v1, T.int64(0), T.int64(0)]

    @R.function
    def main(inp_0: R.Tensor((16, 3, 224, 224), dtype="float32")) -> R.Tensor((16, 1000), dtype="float32"):
        R.func_attr({"relax.force_pure": True})
        cls = Module
        storage: R.Object = R.vm.alloc_storage(R.shape([51380224]), R.prim_value(0), R.dtype("uint8"), R.str("global"))
        alloc: R.Tensor((16, 64, 112, 112), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 64, 112, 112]), R.dtype("float32"))
        cls.conv2d(inp_0, metadata["relax.expr.Constant"][0], alloc)
        storage1: R.Object = R.vm.alloc_storage(R.shape([51380224]), R.prim_value(0), R.dtype("uint8"), R.str("global"))
        alloc1: R.Tensor((16, 64, 112, 112), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 64, 112, 112]), R.dtype("float32"))
        storage2: R.Object = R.vm.alloc_storage(R.shape([8192]), R.prim_value(0), R.dtype("uint8"), R.str("global"))
        alloc2: R.Tensor((64,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([64]), R.dtype("float32"))
        storage3: R.Object = R.vm.alloc_storage(R.shape([8192]), R.prim_value(0), R.dtype("uint8"), R.str("global"))
        alloc3: R.Tensor((64,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([64]), R.dtype("float32"))
        cls.batch_norm(alloc, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2], metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4], alloc1, alloc2, alloc3)
        R.vm.kill_object(alloc)
        lv1: R.Tuple(R.Tensor((16, 64, 112, 112), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = alloc1, alloc2, alloc3
        R.vm.kill_object(alloc2)
        R.vm.kill_object(alloc3)
        alloc4: R.Tensor((16, 64, 112, 112), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 64, 112, 112]), R.dtype("float32"))
        cls.fused_relu(alloc1, alloc4)
        R.vm.kill_object(alloc1)
        alloc5: R.Tensor((16, 64, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 64, 56, 56]), R.dtype("float32"))
        cls.max_pool2d(alloc4, alloc5)
        R.vm.kill_object(alloc4)
        alloc6: R.Tensor((16, 64, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 64, 56, 56]), R.dtype("float32"))
        cls.conv2d1(alloc5, metadata["relax.expr.Constant"][5], alloc6)
        storage4: R.Object = R.vm.alloc_storage(R.shape([51380224]), R.prim_value(0), R.dtype("uint8"), R.str("global"))
        alloc7: R.Tensor((16, 64, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 64, 56, 56]), R.dtype("float32"))
        alloc8: R.Tensor((64,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([64]), R.dtype("float32"))
        alloc9: R.Tensor((64,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([64]), R.dtype("float32"))
        cls.batch_norm1(alloc6, metadata["relax.expr.Constant"][6], metadata["relax.expr.Constant"][7], metadata["relax.expr.Constant"][8], metadata["relax.expr.Constant"][9], alloc7, alloc8, alloc9)
        R.vm.kill_object(alloc6)
        lv6: R.Tuple(R.Tensor((16, 64, 56, 56), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = alloc7, alloc8, alloc9
        R.vm.kill_object(alloc8)
        R.vm.kill_object(alloc9)
        alloc10: R.Tensor((16, 64, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 64, 56, 56]), R.dtype("float32"))
        cls.fused_relu1(alloc7, alloc10)
        R.vm.kill_object(alloc7)
        alloc11: R.Tensor((16, 64, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 64, 56, 56]), R.dtype("float32"))
        cls.conv2d2(alloc10, metadata["relax.expr.Constant"][10], alloc11)
        R.vm.kill_object(alloc10)
        alloc12: R.Tensor((16, 64, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 64, 56, 56]), R.dtype("float32"))
        alloc13: R.Tensor((64,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([64]), R.dtype("float32"))
        alloc14: R.Tensor((64,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([64]), R.dtype("float32"))
        cls.batch_norm1(alloc11, metadata["relax.expr.Constant"][11], metadata["relax.expr.Constant"][12], metadata["relax.expr.Constant"][13], metadata["relax.expr.Constant"][14], alloc12, alloc13, alloc14)
        R.vm.kill_object(alloc11)
        lv10: R.Tuple(R.Tensor((16, 64, 56, 56), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = alloc12, alloc13, alloc14
        R.vm.kill_object(alloc13)
        R.vm.kill_object(alloc14)
        alloc15: R.Tensor((16, 64, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 64, 56, 56]), R.dtype("float32"))
        cls.fused_relu1(alloc12, alloc15)
        R.vm.kill_object(alloc12)
        alloc16: R.Tensor((16, 256, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 256, 56, 56]), R.dtype("float32"))
        cls.conv2d3(alloc15, metadata["relax.expr.Constant"][15], alloc16)
        R.vm.kill_object(alloc15)
        alloc17: R.Tensor((16, 256, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 256, 56, 56]), R.dtype("float32"))
        alloc18: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        alloc19: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        cls.batch_norm2(alloc16, metadata["relax.expr.Constant"][16], metadata["relax.expr.Constant"][17], metadata["relax.expr.Constant"][18], metadata["relax.expr.Constant"][19], alloc17, alloc18, alloc19)
        R.vm.kill_object(alloc16)
        lv14: R.Tuple(R.Tensor((16, 256, 56, 56), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = alloc17, alloc18, alloc19
        R.vm.kill_object(alloc18)
        R.vm.kill_object(alloc19)
        alloc20: R.Tensor((16, 256, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 256, 56, 56]), R.dtype("float32"))
        cls.conv2d3(alloc5, metadata["relax.expr.Constant"][20], alloc20)
        R.vm.kill_object(alloc5)
        alloc21: R.Tensor((16, 256, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 256, 56, 56]), R.dtype("float32"))
        alloc22: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        alloc23: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        cls.batch_norm2(alloc20, metadata["relax.expr.Constant"][21], metadata["relax.expr.Constant"][22], metadata["relax.expr.Constant"][23], metadata["relax.expr.Constant"][24], alloc21, alloc22, alloc23)
        R.vm.kill_object(alloc20)
        lv17: R.Tuple(R.Tensor((16, 256, 56, 56), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = alloc21, alloc22, alloc23
        R.vm.kill_object(alloc22)
        R.vm.kill_object(alloc23)
        alloc24: R.Tensor((16, 256, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 256, 56, 56]), R.dtype("float32"))
        cls.fused_add_relu2(alloc17, alloc21, alloc24)
        R.vm.kill_object(alloc17)
        R.vm.kill_object(alloc21)
        alloc25: R.Tensor((16, 64, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 64, 56, 56]), R.dtype("float32"))
        cls.conv2d4(alloc24, metadata["relax.expr.Constant"][25], alloc25)
        alloc26: R.Tensor((16, 64, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 64, 56, 56]), R.dtype("float32"))
        alloc27: R.Tensor((64,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([64]), R.dtype("float32"))
        alloc28: R.Tensor((64,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([64]), R.dtype("float32"))
        cls.batch_norm1(alloc25, metadata["relax.expr.Constant"][26], metadata["relax.expr.Constant"][27], metadata["relax.expr.Constant"][28], metadata["relax.expr.Constant"][29], alloc26, alloc27, alloc28)
        R.vm.kill_object(alloc25)
        lv22: R.Tuple(R.Tensor((16, 64, 56, 56), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = alloc26, alloc27, alloc28
        R.vm.kill_object(alloc27)
        R.vm.kill_object(alloc28)
        alloc29: R.Tensor((16, 64, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 64, 56, 56]), R.dtype("float32"))
        cls.fused_relu1(alloc26, alloc29)
        R.vm.kill_object(alloc26)
        alloc30: R.Tensor((16, 64, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 64, 56, 56]), R.dtype("float32"))
        cls.conv2d2(alloc29, metadata["relax.expr.Constant"][30], alloc30)
        R.vm.kill_object(alloc29)
        alloc31: R.Tensor((16, 64, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 64, 56, 56]), R.dtype("float32"))
        alloc32: R.Tensor((64,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([64]), R.dtype("float32"))
        alloc33: R.Tensor((64,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([64]), R.dtype("float32"))
        cls.batch_norm1(alloc30, metadata["relax.expr.Constant"][31], metadata["relax.expr.Constant"][32], metadata["relax.expr.Constant"][33], metadata["relax.expr.Constant"][34], alloc31, alloc32, alloc33)
        R.vm.kill_object(alloc30)
        lv26: R.Tuple(R.Tensor((16, 64, 56, 56), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = alloc31, alloc32, alloc33
        R.vm.kill_object(alloc32)
        R.vm.kill_object(alloc33)
        alloc34: R.Tensor((16, 64, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 64, 56, 56]), R.dtype("float32"))
        cls.fused_relu1(alloc31, alloc34)
        R.vm.kill_object(alloc31)
        alloc35: R.Tensor((16, 256, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 256, 56, 56]), R.dtype("float32"))
        cls.conv2d3(alloc34, metadata["relax.expr.Constant"][35], alloc35)
        R.vm.kill_object(alloc34)
        alloc36: R.Tensor((16, 256, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 256, 56, 56]), R.dtype("float32"))
        alloc37: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        alloc38: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        cls.batch_norm2(alloc35, metadata["relax.expr.Constant"][36], metadata["relax.expr.Constant"][37], metadata["relax.expr.Constant"][38], metadata["relax.expr.Constant"][39], alloc36, alloc37, alloc38)
        R.vm.kill_object(alloc35)
        lv30: R.Tuple(R.Tensor((16, 256, 56, 56), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = alloc36, alloc37, alloc38
        R.vm.kill_object(alloc37)
        R.vm.kill_object(alloc38)
        alloc39: R.Tensor((16, 256, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 256, 56, 56]), R.dtype("float32"))
        cls.fused_add_relu2(alloc36, alloc24, alloc39)
        R.vm.kill_object(alloc24)
        R.vm.kill_object(alloc36)
        alloc40: R.Tensor((16, 64, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 64, 56, 56]), R.dtype("float32"))
        cls.conv2d4(alloc39, metadata["relax.expr.Constant"][40], alloc40)
        alloc41: R.Tensor((16, 64, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 64, 56, 56]), R.dtype("float32"))
        alloc42: R.Tensor((64,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([64]), R.dtype("float32"))
        alloc43: R.Tensor((64,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([64]), R.dtype("float32"))
        cls.batch_norm1(alloc40, metadata["relax.expr.Constant"][41], metadata["relax.expr.Constant"][42], metadata["relax.expr.Constant"][43], metadata["relax.expr.Constant"][44], alloc41, alloc42, alloc43)
        R.vm.kill_object(alloc40)
        lv35: R.Tuple(R.Tensor((16, 64, 56, 56), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = alloc41, alloc42, alloc43
        R.vm.kill_object(alloc42)
        R.vm.kill_object(alloc43)
        alloc44: R.Tensor((16, 64, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 64, 56, 56]), R.dtype("float32"))
        cls.fused_relu1(alloc41, alloc44)
        R.vm.kill_object(alloc41)
        alloc45: R.Tensor((16, 64, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 64, 56, 56]), R.dtype("float32"))
        cls.conv2d2(alloc44, metadata["relax.expr.Constant"][45], alloc45)
        R.vm.kill_object(alloc44)
        alloc46: R.Tensor((16, 64, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 64, 56, 56]), R.dtype("float32"))
        alloc47: R.Tensor((64,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([64]), R.dtype("float32"))
        alloc48: R.Tensor((64,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([64]), R.dtype("float32"))
        cls.batch_norm1(alloc45, metadata["relax.expr.Constant"][46], metadata["relax.expr.Constant"][47], metadata["relax.expr.Constant"][48], metadata["relax.expr.Constant"][49], alloc46, alloc47, alloc48)
        R.vm.kill_object(alloc45)
        lv39: R.Tuple(R.Tensor((16, 64, 56, 56), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = alloc46, alloc47, alloc48
        R.vm.kill_object(alloc47)
        R.vm.kill_object(alloc48)
        alloc49: R.Tensor((16, 64, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 64, 56, 56]), R.dtype("float32"))
        cls.fused_relu1(alloc46, alloc49)
        R.vm.kill_object(alloc46)
        alloc50: R.Tensor((16, 256, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 256, 56, 56]), R.dtype("float32"))
        cls.conv2d3(alloc49, metadata["relax.expr.Constant"][50], alloc50)
        R.vm.kill_object(alloc49)
        alloc51: R.Tensor((16, 256, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 256, 56, 56]), R.dtype("float32"))
        alloc52: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        alloc53: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        cls.batch_norm2(alloc50, metadata["relax.expr.Constant"][51], metadata["relax.expr.Constant"][52], metadata["relax.expr.Constant"][53], metadata["relax.expr.Constant"][54], alloc51, alloc52, alloc53)
        R.vm.kill_object(alloc50)
        lv43: R.Tuple(R.Tensor((16, 256, 56, 56), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = alloc51, alloc52, alloc53
        R.vm.kill_object(alloc52)
        R.vm.kill_object(alloc53)
        alloc54: R.Tensor((16, 256, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 256, 56, 56]), R.dtype("float32"))
        cls.fused_add_relu2(alloc51, alloc39, alloc54)
        R.vm.kill_object(alloc39)
        R.vm.kill_object(alloc51)
        alloc55: R.Tensor((16, 128, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 128, 56, 56]), R.dtype("float32"))
        cls.conv2d5(alloc54, metadata["relax.expr.Constant"][55], alloc55)
        alloc56: R.Tensor((16, 128, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 128, 56, 56]), R.dtype("float32"))
        alloc57: R.Tensor((128,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([128]), R.dtype("float32"))
        alloc58: R.Tensor((128,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([128]), R.dtype("float32"))
        cls.batch_norm3(alloc55, metadata["relax.expr.Constant"][56], metadata["relax.expr.Constant"][57], metadata["relax.expr.Constant"][58], metadata["relax.expr.Constant"][59], alloc56, alloc57, alloc58)
        R.vm.kill_object(alloc55)
        lv48: R.Tuple(R.Tensor((16, 128, 56, 56), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = alloc56, alloc57, alloc58
        R.vm.kill_object(alloc57)
        R.vm.kill_object(alloc58)
        alloc59: R.Tensor((16, 128, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 128, 56, 56]), R.dtype("float32"))
        cls.fused_relu3(alloc56, alloc59)
        R.vm.kill_object(alloc56)
        alloc60: R.Tensor((16, 128, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 128, 28, 28]), R.dtype("float32"))
        cls.conv2d6(alloc59, metadata["relax.expr.Constant"][60], alloc60)
        R.vm.kill_object(alloc59)
        alloc61: R.Tensor((16, 128, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 128, 28, 28]), R.dtype("float32"))
        alloc62: R.Tensor((128,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([128]), R.dtype("float32"))
        alloc63: R.Tensor((128,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([128]), R.dtype("float32"))
        cls.batch_norm4(alloc60, metadata["relax.expr.Constant"][61], metadata["relax.expr.Constant"][62], metadata["relax.expr.Constant"][63], metadata["relax.expr.Constant"][64], alloc61, alloc62, alloc63)
        R.vm.kill_object(alloc60)
        lv52: R.Tuple(R.Tensor((16, 128, 28, 28), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = alloc61, alloc62, alloc63
        R.vm.kill_object(alloc62)
        R.vm.kill_object(alloc63)
        alloc64: R.Tensor((16, 128, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 128, 28, 28]), R.dtype("float32"))
        cls.fused_relu4(alloc61, alloc64)
        R.vm.kill_object(alloc61)
        alloc65: R.Tensor((16, 512, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 512, 28, 28]), R.dtype("float32"))
        cls.conv2d7(alloc64, metadata["relax.expr.Constant"][65], alloc65)
        R.vm.kill_object(alloc64)
        alloc66: R.Tensor((16, 512, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 512, 28, 28]), R.dtype("float32"))
        alloc67: R.Tensor((512,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([512]), R.dtype("float32"))
        alloc68: R.Tensor((512,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([512]), R.dtype("float32"))
        cls.batch_norm5(alloc65, metadata["relax.expr.Constant"][66], metadata["relax.expr.Constant"][67], metadata["relax.expr.Constant"][68], metadata["relax.expr.Constant"][69], alloc66, alloc67, alloc68)
        R.vm.kill_object(alloc65)
        lv56: R.Tuple(R.Tensor((16, 512, 28, 28), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = alloc66, alloc67, alloc68
        R.vm.kill_object(alloc67)
        R.vm.kill_object(alloc68)
        alloc69: R.Tensor((16, 512, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 512, 28, 28]), R.dtype("float32"))
        cls.conv2d8(alloc54, metadata["relax.expr.Constant"][70], alloc69)
        R.vm.kill_object(alloc54)
        alloc70: R.Tensor((16, 512, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 512, 28, 28]), R.dtype("float32"))
        alloc71: R.Tensor((512,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([512]), R.dtype("float32"))
        alloc72: R.Tensor((512,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([512]), R.dtype("float32"))
        cls.batch_norm5(alloc69, metadata["relax.expr.Constant"][71], metadata["relax.expr.Constant"][72], metadata["relax.expr.Constant"][73], metadata["relax.expr.Constant"][74], alloc70, alloc71, alloc72)
        R.vm.kill_object(alloc69)
        lv59: R.Tuple(R.Tensor((16, 512, 28, 28), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = alloc70, alloc71, alloc72
        R.vm.kill_object(alloc71)
        R.vm.kill_object(alloc72)
        alloc73: R.Tensor((16, 512, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 512, 28, 28]), R.dtype("float32"))
        cls.fused_add1_relu5(alloc66, alloc70, alloc73)
        R.vm.kill_object(alloc66)
        R.vm.kill_object(alloc70)
        alloc74: R.Tensor((16, 128, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 128, 28, 28]), R.dtype("float32"))
        cls.conv2d9(alloc73, metadata["relax.expr.Constant"][75], alloc74)
        alloc75: R.Tensor((16, 128, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 128, 28, 28]), R.dtype("float32"))
        alloc76: R.Tensor((128,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([128]), R.dtype("float32"))
        alloc77: R.Tensor((128,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([128]), R.dtype("float32"))
        cls.batch_norm4(alloc74, metadata["relax.expr.Constant"][76], metadata["relax.expr.Constant"][77], metadata["relax.expr.Constant"][78], metadata["relax.expr.Constant"][79], alloc75, alloc76, alloc77)
        R.vm.kill_object(alloc74)
        lv64: R.Tuple(R.Tensor((16, 128, 28, 28), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = alloc75, alloc76, alloc77
        R.vm.kill_object(alloc76)
        R.vm.kill_object(alloc77)
        alloc78: R.Tensor((16, 128, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 128, 28, 28]), R.dtype("float32"))
        cls.fused_relu4(alloc75, alloc78)
        R.vm.kill_object(alloc75)
        alloc79: R.Tensor((16, 128, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 128, 28, 28]), R.dtype("float32"))
        cls.conv2d10(alloc78, metadata["relax.expr.Constant"][80], alloc79)
        R.vm.kill_object(alloc78)
        alloc80: R.Tensor((16, 128, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 128, 28, 28]), R.dtype("float32"))
        alloc81: R.Tensor((128,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([128]), R.dtype("float32"))
        alloc82: R.Tensor((128,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([128]), R.dtype("float32"))
        cls.batch_norm4(alloc79, metadata["relax.expr.Constant"][81], metadata["relax.expr.Constant"][82], metadata["relax.expr.Constant"][83], metadata["relax.expr.Constant"][84], alloc80, alloc81, alloc82)
        R.vm.kill_object(alloc79)
        lv68: R.Tuple(R.Tensor((16, 128, 28, 28), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = alloc80, alloc81, alloc82
        R.vm.kill_object(alloc81)
        R.vm.kill_object(alloc82)
        alloc83: R.Tensor((16, 128, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 128, 28, 28]), R.dtype("float32"))
        cls.fused_relu4(alloc80, alloc83)
        R.vm.kill_object(alloc80)
        alloc84: R.Tensor((16, 512, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 512, 28, 28]), R.dtype("float32"))
        cls.conv2d7(alloc83, metadata["relax.expr.Constant"][85], alloc84)
        R.vm.kill_object(alloc83)
        alloc85: R.Tensor((16, 512, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 512, 28, 28]), R.dtype("float32"))
        alloc86: R.Tensor((512,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([512]), R.dtype("float32"))
        alloc87: R.Tensor((512,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([512]), R.dtype("float32"))
        cls.batch_norm5(alloc84, metadata["relax.expr.Constant"][86], metadata["relax.expr.Constant"][87], metadata["relax.expr.Constant"][88], metadata["relax.expr.Constant"][89], alloc85, alloc86, alloc87)
        R.vm.kill_object(alloc84)
        lv72: R.Tuple(R.Tensor((16, 512, 28, 28), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = alloc85, alloc86, alloc87
        R.vm.kill_object(alloc86)
        R.vm.kill_object(alloc87)
        alloc88: R.Tensor((16, 512, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 512, 28, 28]), R.dtype("float32"))
        cls.fused_add1_relu5(alloc85, alloc73, alloc88)
        R.vm.kill_object(alloc73)
        R.vm.kill_object(alloc85)
        alloc89: R.Tensor((16, 128, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 128, 28, 28]), R.dtype("float32"))
        cls.conv2d9(alloc88, metadata["relax.expr.Constant"][90], alloc89)
        alloc90: R.Tensor((16, 128, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 128, 28, 28]), R.dtype("float32"))
        alloc91: R.Tensor((128,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([128]), R.dtype("float32"))
        alloc92: R.Tensor((128,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([128]), R.dtype("float32"))
        cls.batch_norm4(alloc89, metadata["relax.expr.Constant"][91], metadata["relax.expr.Constant"][92], metadata["relax.expr.Constant"][93], metadata["relax.expr.Constant"][94], alloc90, alloc91, alloc92)
        R.vm.kill_object(alloc89)
        lv77: R.Tuple(R.Tensor((16, 128, 28, 28), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = alloc90, alloc91, alloc92
        R.vm.kill_object(alloc91)
        R.vm.kill_object(alloc92)
        alloc93: R.Tensor((16, 128, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 128, 28, 28]), R.dtype("float32"))
        cls.fused_relu4(alloc90, alloc93)
        R.vm.kill_object(alloc90)
        alloc94: R.Tensor((16, 128, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 128, 28, 28]), R.dtype("float32"))
        cls.conv2d10(alloc93, metadata["relax.expr.Constant"][95], alloc94)
        R.vm.kill_object(alloc93)
        alloc95: R.Tensor((16, 128, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 128, 28, 28]), R.dtype("float32"))
        alloc96: R.Tensor((128,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([128]), R.dtype("float32"))
        alloc97: R.Tensor((128,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([128]), R.dtype("float32"))
        cls.batch_norm4(alloc94, metadata["relax.expr.Constant"][96], metadata["relax.expr.Constant"][97], metadata["relax.expr.Constant"][98], metadata["relax.expr.Constant"][99], alloc95, alloc96, alloc97)
        R.vm.kill_object(alloc94)
        lv81: R.Tuple(R.Tensor((16, 128, 28, 28), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = alloc95, alloc96, alloc97
        R.vm.kill_object(alloc96)
        R.vm.kill_object(alloc97)
        alloc98: R.Tensor((16, 128, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 128, 28, 28]), R.dtype("float32"))
        cls.fused_relu4(alloc95, alloc98)
        R.vm.kill_object(alloc95)
        alloc99: R.Tensor((16, 512, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 512, 28, 28]), R.dtype("float32"))
        cls.conv2d7(alloc98, metadata["relax.expr.Constant"][100], alloc99)
        R.vm.kill_object(alloc98)
        alloc100: R.Tensor((16, 512, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 512, 28, 28]), R.dtype("float32"))
        alloc101: R.Tensor((512,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([512]), R.dtype("float32"))
        alloc102: R.Tensor((512,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([512]), R.dtype("float32"))
        cls.batch_norm5(alloc99, metadata["relax.expr.Constant"][101], metadata["relax.expr.Constant"][102], metadata["relax.expr.Constant"][103], metadata["relax.expr.Constant"][104], alloc100, alloc101, alloc102)
        R.vm.kill_object(alloc99)
        lv85: R.Tuple(R.Tensor((16, 512, 28, 28), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = alloc100, alloc101, alloc102
        R.vm.kill_object(alloc101)
        R.vm.kill_object(alloc102)
        alloc103: R.Tensor((16, 512, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 512, 28, 28]), R.dtype("float32"))
        cls.fused_add1_relu5(alloc100, alloc88, alloc103)
        R.vm.kill_object(alloc88)
        R.vm.kill_object(alloc100)
        alloc104: R.Tensor((16, 128, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 128, 28, 28]), R.dtype("float32"))
        cls.conv2d9(alloc103, metadata["relax.expr.Constant"][105], alloc104)
        alloc105: R.Tensor((16, 128, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 128, 28, 28]), R.dtype("float32"))
        alloc106: R.Tensor((128,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([128]), R.dtype("float32"))
        alloc107: R.Tensor((128,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([128]), R.dtype("float32"))
        cls.batch_norm4(alloc104, metadata["relax.expr.Constant"][106], metadata["relax.expr.Constant"][107], metadata["relax.expr.Constant"][108], metadata["relax.expr.Constant"][109], alloc105, alloc106, alloc107)
        R.vm.kill_object(alloc104)
        lv90: R.Tuple(R.Tensor((16, 128, 28, 28), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = alloc105, alloc106, alloc107
        R.vm.kill_object(alloc106)
        R.vm.kill_object(alloc107)
        alloc108: R.Tensor((16, 128, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 128, 28, 28]), R.dtype("float32"))
        cls.fused_relu4(alloc105, alloc108)
        R.vm.kill_object(alloc105)
        alloc109: R.Tensor((16, 128, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 128, 28, 28]), R.dtype("float32"))
        cls.conv2d10(alloc108, metadata["relax.expr.Constant"][110], alloc109)
        R.vm.kill_object(alloc108)
        alloc110: R.Tensor((16, 128, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 128, 28, 28]), R.dtype("float32"))
        alloc111: R.Tensor((128,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([128]), R.dtype("float32"))
        alloc112: R.Tensor((128,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([128]), R.dtype("float32"))
        cls.batch_norm4(alloc109, metadata["relax.expr.Constant"][111], metadata["relax.expr.Constant"][112], metadata["relax.expr.Constant"][113], metadata["relax.expr.Constant"][114], alloc110, alloc111, alloc112)
        R.vm.kill_object(alloc109)
        lv94: R.Tuple(R.Tensor((16, 128, 28, 28), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = alloc110, alloc111, alloc112
        R.vm.kill_object(alloc111)
        R.vm.kill_object(alloc112)
        alloc113: R.Tensor((16, 128, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 128, 28, 28]), R.dtype("float32"))
        cls.fused_relu4(alloc110, alloc113)
        R.vm.kill_object(alloc110)
        alloc114: R.Tensor((16, 512, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 512, 28, 28]), R.dtype("float32"))
        cls.conv2d7(alloc113, metadata["relax.expr.Constant"][115], alloc114)
        R.vm.kill_object(alloc113)
        alloc115: R.Tensor((16, 512, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 512, 28, 28]), R.dtype("float32"))
        alloc116: R.Tensor((512,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([512]), R.dtype("float32"))
        alloc117: R.Tensor((512,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([512]), R.dtype("float32"))
        cls.batch_norm5(alloc114, metadata["relax.expr.Constant"][116], metadata["relax.expr.Constant"][117], metadata["relax.expr.Constant"][118], metadata["relax.expr.Constant"][119], alloc115, alloc116, alloc117)
        R.vm.kill_object(alloc114)
        lv98: R.Tuple(R.Tensor((16, 512, 28, 28), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = alloc115, alloc116, alloc117
        R.vm.kill_object(alloc116)
        R.vm.kill_object(alloc117)
        alloc118: R.Tensor((16, 512, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 512, 28, 28]), R.dtype("float32"))
        cls.fused_add1_relu5(alloc115, alloc103, alloc118)
        R.vm.kill_object(alloc103)
        R.vm.kill_object(alloc115)
        alloc119: R.Tensor((16, 256, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 256, 28, 28]), R.dtype("float32"))
        cls.conv2d11(alloc118, metadata["relax.expr.Constant"][120], alloc119)
        alloc120: R.Tensor((16, 256, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 256, 28, 28]), R.dtype("float32"))
        alloc121: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        alloc122: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        cls.batch_norm6(alloc119, metadata["relax.expr.Constant"][121], metadata["relax.expr.Constant"][122], metadata["relax.expr.Constant"][123], metadata["relax.expr.Constant"][124], alloc120, alloc121, alloc122)
        R.vm.kill_object(alloc119)
        lv103: R.Tuple(R.Tensor((16, 256, 28, 28), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = alloc120, alloc121, alloc122
        R.vm.kill_object(alloc121)
        R.vm.kill_object(alloc122)
        alloc123: R.Tensor((16, 256, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 256, 28, 28]), R.dtype("float32"))
        cls.fused_relu6(alloc120, alloc123)
        R.vm.kill_object(alloc120)
        alloc124: R.Tensor((16, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 256, 14, 14]), R.dtype("float32"))
        cls.conv2d12(alloc123, metadata["relax.expr.Constant"][125], alloc124)
        R.vm.kill_object(alloc123)
        alloc125: R.Tensor((16, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 256, 14, 14]), R.dtype("float32"))
        alloc126: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        alloc127: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        cls.batch_norm7(alloc124, metadata["relax.expr.Constant"][126], metadata["relax.expr.Constant"][127], metadata["relax.expr.Constant"][128], metadata["relax.expr.Constant"][129], alloc125, alloc126, alloc127)
        R.vm.kill_object(alloc124)
        lv107: R.Tuple(R.Tensor((16, 256, 14, 14), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = alloc125, alloc126, alloc127
        R.vm.kill_object(alloc126)
        R.vm.kill_object(alloc127)
        alloc128: R.Tensor((16, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 256, 14, 14]), R.dtype("float32"))
        cls.fused_relu7(alloc125, alloc128)
        R.vm.kill_object(alloc125)
        alloc129: R.Tensor((16, 1024, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 1024, 14, 14]), R.dtype("float32"))
        cls.conv2d13(alloc128, metadata["relax.expr.Constant"][130], alloc129)
        R.vm.kill_object(alloc128)
        alloc130: R.Tensor((16, 1024, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 1024, 14, 14]), R.dtype("float32"))
        alloc131: R.Tensor((1024,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([1024]), R.dtype("float32"))
        alloc132: R.Tensor((1024,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([1024]), R.dtype("float32"))
        cls.batch_norm8(alloc129, metadata["relax.expr.Constant"][131], metadata["relax.expr.Constant"][132], metadata["relax.expr.Constant"][133], metadata["relax.expr.Constant"][134], alloc130, alloc131, alloc132)
        R.vm.kill_object(alloc129)
        lv111: R.Tuple(R.Tensor((16, 1024, 14, 14), dtype="float32"), R.Tensor((1024,), dtype="float32"), R.Tensor((1024,), dtype="float32")) = alloc130, alloc131, alloc132
        R.vm.kill_object(alloc131)
        R.vm.kill_object(alloc132)
        alloc133: R.Tensor((16, 1024, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 1024, 14, 14]), R.dtype("float32"))
        cls.conv2d14(alloc118, metadata["relax.expr.Constant"][135], alloc133)
        R.vm.kill_object(alloc118)
        alloc134: R.Tensor((16, 1024, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 1024, 14, 14]), R.dtype("float32"))
        alloc135: R.Tensor((1024,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([1024]), R.dtype("float32"))
        alloc136: R.Tensor((1024,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([1024]), R.dtype("float32"))
        cls.batch_norm8(alloc133, metadata["relax.expr.Constant"][136], metadata["relax.expr.Constant"][137], metadata["relax.expr.Constant"][138], metadata["relax.expr.Constant"][139], alloc134, alloc135, alloc136)
        R.vm.kill_object(alloc133)
        lv114: R.Tuple(R.Tensor((16, 1024, 14, 14), dtype="float32"), R.Tensor((1024,), dtype="float32"), R.Tensor((1024,), dtype="float32")) = alloc134, alloc135, alloc136
        R.vm.kill_object(alloc135)
        R.vm.kill_object(alloc136)
        alloc137: R.Tensor((16, 1024, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 1024, 14, 14]), R.dtype("float32"))
        cls.fused_add2_relu8(alloc130, alloc134, alloc137)
        R.vm.kill_object(alloc130)
        R.vm.kill_object(alloc134)
        alloc138: R.Tensor((16, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 256, 14, 14]), R.dtype("float32"))
        cls.conv2d15(alloc137, metadata["relax.expr.Constant"][140], alloc138)
        alloc139: R.Tensor((16, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 256, 14, 14]), R.dtype("float32"))
        alloc140: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        alloc141: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        cls.batch_norm7(alloc138, metadata["relax.expr.Constant"][141], metadata["relax.expr.Constant"][142], metadata["relax.expr.Constant"][143], metadata["relax.expr.Constant"][144], alloc139, alloc140, alloc141)
        R.vm.kill_object(alloc138)
        lv119: R.Tuple(R.Tensor((16, 256, 14, 14), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = alloc139, alloc140, alloc141
        R.vm.kill_object(alloc140)
        R.vm.kill_object(alloc141)
        alloc142: R.Tensor((16, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 256, 14, 14]), R.dtype("float32"))
        cls.fused_relu7(alloc139, alloc142)
        R.vm.kill_object(alloc139)
        alloc143: R.Tensor((16, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 256, 14, 14]), R.dtype("float32"))
        cls.conv2d16(alloc142, metadata["relax.expr.Constant"][145], alloc143)
        R.vm.kill_object(alloc142)
        alloc144: R.Tensor((16, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 256, 14, 14]), R.dtype("float32"))
        alloc145: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        alloc146: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        cls.batch_norm7(alloc143, metadata["relax.expr.Constant"][146], metadata["relax.expr.Constant"][147], metadata["relax.expr.Constant"][148], metadata["relax.expr.Constant"][149], alloc144, alloc145, alloc146)
        R.vm.kill_object(alloc143)
        lv123: R.Tuple(R.Tensor((16, 256, 14, 14), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = alloc144, alloc145, alloc146
        R.vm.kill_object(alloc145)
        R.vm.kill_object(alloc146)
        alloc147: R.Tensor((16, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 256, 14, 14]), R.dtype("float32"))
        cls.fused_relu7(alloc144, alloc147)
        R.vm.kill_object(alloc144)
        alloc148: R.Tensor((16, 1024, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 1024, 14, 14]), R.dtype("float32"))
        cls.conv2d13(alloc147, metadata["relax.expr.Constant"][150], alloc148)
        R.vm.kill_object(alloc147)
        alloc149: R.Tensor((16, 1024, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 1024, 14, 14]), R.dtype("float32"))
        alloc150: R.Tensor((1024,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([1024]), R.dtype("float32"))
        alloc151: R.Tensor((1024,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([1024]), R.dtype("float32"))
        cls.batch_norm8(alloc148, metadata["relax.expr.Constant"][151], metadata["relax.expr.Constant"][152], metadata["relax.expr.Constant"][153], metadata["relax.expr.Constant"][154], alloc149, alloc150, alloc151)
        R.vm.kill_object(alloc148)
        lv127: R.Tuple(R.Tensor((16, 1024, 14, 14), dtype="float32"), R.Tensor((1024,), dtype="float32"), R.Tensor((1024,), dtype="float32")) = alloc149, alloc150, alloc151
        R.vm.kill_object(alloc150)
        R.vm.kill_object(alloc151)
        alloc152: R.Tensor((16, 1024, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 1024, 14, 14]), R.dtype("float32"))
        cls.fused_add2_relu8(alloc149, alloc137, alloc152)
        R.vm.kill_object(alloc137)
        R.vm.kill_object(alloc149)
        alloc153: R.Tensor((16, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 256, 14, 14]), R.dtype("float32"))
        cls.conv2d15(alloc152, metadata["relax.expr.Constant"][155], alloc153)
        alloc154: R.Tensor((16, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 256, 14, 14]), R.dtype("float32"))
        alloc155: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        alloc156: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        cls.batch_norm7(alloc153, metadata["relax.expr.Constant"][156], metadata["relax.expr.Constant"][157], metadata["relax.expr.Constant"][158], metadata["relax.expr.Constant"][159], alloc154, alloc155, alloc156)
        R.vm.kill_object(alloc153)
        lv132: R.Tuple(R.Tensor((16, 256, 14, 14), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = alloc154, alloc155, alloc156
        R.vm.kill_object(alloc155)
        R.vm.kill_object(alloc156)
        alloc157: R.Tensor((16, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 256, 14, 14]), R.dtype("float32"))
        cls.fused_relu7(alloc154, alloc157)
        R.vm.kill_object(alloc154)
        alloc158: R.Tensor((16, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 256, 14, 14]), R.dtype("float32"))
        cls.conv2d16(alloc157, metadata["relax.expr.Constant"][160], alloc158)
        R.vm.kill_object(alloc157)
        alloc159: R.Tensor((16, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 256, 14, 14]), R.dtype("float32"))
        alloc160: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        alloc161: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        cls.batch_norm7(alloc158, metadata["relax.expr.Constant"][161], metadata["relax.expr.Constant"][162], metadata["relax.expr.Constant"][163], metadata["relax.expr.Constant"][164], alloc159, alloc160, alloc161)
        R.vm.kill_object(alloc158)
        lv136: R.Tuple(R.Tensor((16, 256, 14, 14), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = alloc159, alloc160, alloc161
        R.vm.kill_object(alloc160)
        R.vm.kill_object(alloc161)
        alloc162: R.Tensor((16, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 256, 14, 14]), R.dtype("float32"))
        cls.fused_relu7(alloc159, alloc162)
        R.vm.kill_object(alloc159)
        alloc163: R.Tensor((16, 1024, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 1024, 14, 14]), R.dtype("float32"))
        cls.conv2d13(alloc162, metadata["relax.expr.Constant"][165], alloc163)
        R.vm.kill_object(alloc162)
        alloc164: R.Tensor((16, 1024, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 1024, 14, 14]), R.dtype("float32"))
        alloc165: R.Tensor((1024,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([1024]), R.dtype("float32"))
        alloc166: R.Tensor((1024,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([1024]), R.dtype("float32"))
        cls.batch_norm8(alloc163, metadata["relax.expr.Constant"][166], metadata["relax.expr.Constant"][167], metadata["relax.expr.Constant"][168], metadata["relax.expr.Constant"][169], alloc164, alloc165, alloc166)
        R.vm.kill_object(alloc163)
        lv140: R.Tuple(R.Tensor((16, 1024, 14, 14), dtype="float32"), R.Tensor((1024,), dtype="float32"), R.Tensor((1024,), dtype="float32")) = alloc164, alloc165, alloc166
        R.vm.kill_object(alloc165)
        R.vm.kill_object(alloc166)
        alloc167: R.Tensor((16, 1024, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 1024, 14, 14]), R.dtype("float32"))
        cls.fused_add2_relu8(alloc164, alloc152, alloc167)
        R.vm.kill_object(alloc152)
        R.vm.kill_object(alloc164)
        alloc168: R.Tensor((16, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 256, 14, 14]), R.dtype("float32"))
        cls.conv2d15(alloc167, metadata["relax.expr.Constant"][170], alloc168)
        alloc169: R.Tensor((16, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 256, 14, 14]), R.dtype("float32"))
        alloc170: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        alloc171: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        cls.batch_norm7(alloc168, metadata["relax.expr.Constant"][171], metadata["relax.expr.Constant"][172], metadata["relax.expr.Constant"][173], metadata["relax.expr.Constant"][174], alloc169, alloc170, alloc171)
        R.vm.kill_object(alloc168)
        lv145: R.Tuple(R.Tensor((16, 256, 14, 14), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = alloc169, alloc170, alloc171
        R.vm.kill_object(alloc170)
        R.vm.kill_object(alloc171)
        alloc172: R.Tensor((16, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 256, 14, 14]), R.dtype("float32"))
        cls.fused_relu7(alloc169, alloc172)
        R.vm.kill_object(alloc169)
        alloc173: R.Tensor((16, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 256, 14, 14]), R.dtype("float32"))
        cls.conv2d16(alloc172, metadata["relax.expr.Constant"][175], alloc173)
        R.vm.kill_object(alloc172)
        alloc174: R.Tensor((16, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 256, 14, 14]), R.dtype("float32"))
        alloc175: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        alloc176: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        cls.batch_norm7(alloc173, metadata["relax.expr.Constant"][176], metadata["relax.expr.Constant"][177], metadata["relax.expr.Constant"][178], metadata["relax.expr.Constant"][179], alloc174, alloc175, alloc176)
        R.vm.kill_object(alloc173)
        lv149: R.Tuple(R.Tensor((16, 256, 14, 14), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = alloc174, alloc175, alloc176
        R.vm.kill_object(alloc175)
        R.vm.kill_object(alloc176)
        alloc177: R.Tensor((16, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 256, 14, 14]), R.dtype("float32"))
        cls.fused_relu7(alloc174, alloc177)
        R.vm.kill_object(alloc174)
        alloc178: R.Tensor((16, 1024, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 1024, 14, 14]), R.dtype("float32"))
        cls.conv2d13(alloc177, metadata["relax.expr.Constant"][180], alloc178)
        R.vm.kill_object(alloc177)
        alloc179: R.Tensor((16, 1024, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 1024, 14, 14]), R.dtype("float32"))
        alloc180: R.Tensor((1024,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([1024]), R.dtype("float32"))
        alloc181: R.Tensor((1024,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([1024]), R.dtype("float32"))
        cls.batch_norm8(alloc178, metadata["relax.expr.Constant"][181], metadata["relax.expr.Constant"][182], metadata["relax.expr.Constant"][183], metadata["relax.expr.Constant"][184], alloc179, alloc180, alloc181)
        R.vm.kill_object(alloc178)
        lv153: R.Tuple(R.Tensor((16, 1024, 14, 14), dtype="float32"), R.Tensor((1024,), dtype="float32"), R.Tensor((1024,), dtype="float32")) = alloc179, alloc180, alloc181
        R.vm.kill_object(alloc180)
        R.vm.kill_object(alloc181)
        alloc182: R.Tensor((16, 1024, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 1024, 14, 14]), R.dtype("float32"))
        cls.fused_add2_relu8(alloc179, alloc167, alloc182)
        R.vm.kill_object(alloc167)
        R.vm.kill_object(alloc179)
        alloc183: R.Tensor((16, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 256, 14, 14]), R.dtype("float32"))
        cls.conv2d15(alloc182, metadata["relax.expr.Constant"][185], alloc183)
        alloc184: R.Tensor((16, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 256, 14, 14]), R.dtype("float32"))
        alloc185: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        alloc186: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        cls.batch_norm7(alloc183, metadata["relax.expr.Constant"][186], metadata["relax.expr.Constant"][187], metadata["relax.expr.Constant"][188], metadata["relax.expr.Constant"][189], alloc184, alloc185, alloc186)
        R.vm.kill_object(alloc183)
        lv158: R.Tuple(R.Tensor((16, 256, 14, 14), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = alloc184, alloc185, alloc186
        R.vm.kill_object(alloc185)
        R.vm.kill_object(alloc186)
        alloc187: R.Tensor((16, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 256, 14, 14]), R.dtype("float32"))
        cls.fused_relu7(alloc184, alloc187)
        R.vm.kill_object(alloc184)
        alloc188: R.Tensor((16, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 256, 14, 14]), R.dtype("float32"))
        cls.conv2d16(alloc187, metadata["relax.expr.Constant"][190], alloc188)
        R.vm.kill_object(alloc187)
        alloc189: R.Tensor((16, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 256, 14, 14]), R.dtype("float32"))
        alloc190: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        alloc191: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        cls.batch_norm7(alloc188, metadata["relax.expr.Constant"][191], metadata["relax.expr.Constant"][192], metadata["relax.expr.Constant"][193], metadata["relax.expr.Constant"][194], alloc189, alloc190, alloc191)
        R.vm.kill_object(alloc188)
        lv162: R.Tuple(R.Tensor((16, 256, 14, 14), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = alloc189, alloc190, alloc191
        R.vm.kill_object(alloc190)
        R.vm.kill_object(alloc191)
        alloc192: R.Tensor((16, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 256, 14, 14]), R.dtype("float32"))
        cls.fused_relu7(alloc189, alloc192)
        R.vm.kill_object(alloc189)
        alloc193: R.Tensor((16, 1024, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 1024, 14, 14]), R.dtype("float32"))
        cls.conv2d13(alloc192, metadata["relax.expr.Constant"][195], alloc193)
        R.vm.kill_object(alloc192)
        alloc194: R.Tensor((16, 1024, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 1024, 14, 14]), R.dtype("float32"))
        alloc195: R.Tensor((1024,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([1024]), R.dtype("float32"))
        alloc196: R.Tensor((1024,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([1024]), R.dtype("float32"))
        cls.batch_norm8(alloc193, metadata["relax.expr.Constant"][196], metadata["relax.expr.Constant"][197], metadata["relax.expr.Constant"][198], metadata["relax.expr.Constant"][199], alloc194, alloc195, alloc196)
        R.vm.kill_object(alloc193)
        lv166: R.Tuple(R.Tensor((16, 1024, 14, 14), dtype="float32"), R.Tensor((1024,), dtype="float32"), R.Tensor((1024,), dtype="float32")) = alloc194, alloc195, alloc196
        R.vm.kill_object(alloc195)
        R.vm.kill_object(alloc196)
        alloc197: R.Tensor((16, 1024, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 1024, 14, 14]), R.dtype("float32"))
        cls.fused_add2_relu8(alloc194, alloc182, alloc197)
        R.vm.kill_object(alloc182)
        R.vm.kill_object(alloc194)
        alloc198: R.Tensor((16, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 256, 14, 14]), R.dtype("float32"))
        cls.conv2d15(alloc197, metadata["relax.expr.Constant"][200], alloc198)
        alloc199: R.Tensor((16, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 256, 14, 14]), R.dtype("float32"))
        alloc200: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        alloc201: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        cls.batch_norm7(alloc198, metadata["relax.expr.Constant"][201], metadata["relax.expr.Constant"][202], metadata["relax.expr.Constant"][203], metadata["relax.expr.Constant"][204], alloc199, alloc200, alloc201)
        R.vm.kill_object(alloc198)
        lv171: R.Tuple(R.Tensor((16, 256, 14, 14), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = alloc199, alloc200, alloc201
        R.vm.kill_object(alloc200)
        R.vm.kill_object(alloc201)
        alloc202: R.Tensor((16, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 256, 14, 14]), R.dtype("float32"))
        cls.fused_relu7(alloc199, alloc202)
        R.vm.kill_object(alloc199)
        alloc203: R.Tensor((16, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 256, 14, 14]), R.dtype("float32"))
        cls.conv2d16(alloc202, metadata["relax.expr.Constant"][205], alloc203)
        R.vm.kill_object(alloc202)
        alloc204: R.Tensor((16, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 256, 14, 14]), R.dtype("float32"))
        alloc205: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        alloc206: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        cls.batch_norm7(alloc203, metadata["relax.expr.Constant"][206], metadata["relax.expr.Constant"][207], metadata["relax.expr.Constant"][208], metadata["relax.expr.Constant"][209], alloc204, alloc205, alloc206)
        R.vm.kill_object(alloc203)
        lv175: R.Tuple(R.Tensor((16, 256, 14, 14), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = alloc204, alloc205, alloc206
        R.vm.kill_object(alloc205)
        R.vm.kill_object(alloc206)
        alloc207: R.Tensor((16, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 256, 14, 14]), R.dtype("float32"))
        cls.fused_relu7(alloc204, alloc207)
        R.vm.kill_object(alloc204)
        alloc208: R.Tensor((16, 1024, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 1024, 14, 14]), R.dtype("float32"))
        cls.conv2d13(alloc207, metadata["relax.expr.Constant"][210], alloc208)
        R.vm.kill_object(alloc207)
        alloc209: R.Tensor((16, 1024, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 1024, 14, 14]), R.dtype("float32"))
        alloc210: R.Tensor((1024,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([1024]), R.dtype("float32"))
        alloc211: R.Tensor((1024,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([1024]), R.dtype("float32"))
        cls.batch_norm8(alloc208, metadata["relax.expr.Constant"][211], metadata["relax.expr.Constant"][212], metadata["relax.expr.Constant"][213], metadata["relax.expr.Constant"][214], alloc209, alloc210, alloc211)
        R.vm.kill_object(alloc208)
        lv179: R.Tuple(R.Tensor((16, 1024, 14, 14), dtype="float32"), R.Tensor((1024,), dtype="float32"), R.Tensor((1024,), dtype="float32")) = alloc209, alloc210, alloc211
        R.vm.kill_object(alloc210)
        R.vm.kill_object(alloc211)
        alloc212: R.Tensor((16, 1024, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 1024, 14, 14]), R.dtype("float32"))
        cls.fused_add2_relu8(alloc209, alloc197, alloc212)
        R.vm.kill_object(alloc197)
        R.vm.kill_object(alloc209)
        alloc213: R.Tensor((16, 512, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 512, 14, 14]), R.dtype("float32"))
        cls.conv2d17(alloc212, metadata["relax.expr.Constant"][215], alloc213)
        alloc214: R.Tensor((16, 512, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 512, 14, 14]), R.dtype("float32"))
        alloc215: R.Tensor((512,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([512]), R.dtype("float32"))
        alloc216: R.Tensor((512,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([512]), R.dtype("float32"))
        cls.batch_norm9(alloc213, metadata["relax.expr.Constant"][216], metadata["relax.expr.Constant"][217], metadata["relax.expr.Constant"][218], metadata["relax.expr.Constant"][219], alloc214, alloc215, alloc216)
        R.vm.kill_object(alloc213)
        lv184: R.Tuple(R.Tensor((16, 512, 14, 14), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = alloc214, alloc215, alloc216
        R.vm.kill_object(alloc215)
        R.vm.kill_object(alloc216)
        alloc217: R.Tensor((16, 512, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 512, 14, 14]), R.dtype("float32"))
        cls.fused_relu9(alloc214, alloc217)
        R.vm.kill_object(alloc214)
        storage5: R.Object = R.vm.alloc_storage(R.shape([1605632]), R.prim_value(0), R.dtype("uint8"), R.str("global"))
        alloc218: R.Tensor((16, 512, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage5, R.prim_value(0), R.shape([16, 512, 7, 7]), R.dtype("float32"))
        cls.conv2d18(alloc217, metadata["relax.expr.Constant"][220], alloc218)
        R.vm.kill_object(alloc217)
        storage6: R.Object = R.vm.alloc_storage(R.shape([1605632]), R.prim_value(0), R.dtype("uint8"), R.str("global"))
        alloc219: R.Tensor((16, 512, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage6, R.prim_value(0), R.shape([16, 512, 7, 7]), R.dtype("float32"))
        alloc220: R.Tensor((512,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([512]), R.dtype("float32"))
        alloc221: R.Tensor((512,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([512]), R.dtype("float32"))
        cls.batch_norm10(alloc218, metadata["relax.expr.Constant"][221], metadata["relax.expr.Constant"][222], metadata["relax.expr.Constant"][223], metadata["relax.expr.Constant"][224], alloc219, alloc220, alloc221)
        R.vm.kill_object(alloc218)
        lv188: R.Tuple(R.Tensor((16, 512, 7, 7), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = alloc219, alloc220, alloc221
        R.vm.kill_object(alloc220)
        R.vm.kill_object(alloc221)
        alloc222: R.Tensor((16, 512, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage5, R.prim_value(0), R.shape([16, 512, 7, 7]), R.dtype("float32"))
        cls.fused_relu10(alloc219, alloc222)
        R.vm.kill_object(alloc219)
        alloc223: R.Tensor((16, 2048, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 2048, 7, 7]), R.dtype("float32"))
        cls.conv2d19(alloc222, metadata["relax.expr.Constant"][225], alloc223)
        R.vm.kill_object(alloc222)
        alloc224: R.Tensor((16, 2048, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 2048, 7, 7]), R.dtype("float32"))
        alloc225: R.Tensor((2048,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([2048]), R.dtype("float32"))
        alloc226: R.Tensor((2048,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([2048]), R.dtype("float32"))
        cls.batch_norm11(alloc223, metadata["relax.expr.Constant"][226], metadata["relax.expr.Constant"][227], metadata["relax.expr.Constant"][228], metadata["relax.expr.Constant"][229], alloc224, alloc225, alloc226)
        R.vm.kill_object(alloc223)
        lv192: R.Tuple(R.Tensor((16, 2048, 7, 7), dtype="float32"), R.Tensor((2048,), dtype="float32"), R.Tensor((2048,), dtype="float32")) = alloc224, alloc225, alloc226
        R.vm.kill_object(alloc225)
        R.vm.kill_object(alloc226)
        alloc227: R.Tensor((16, 2048, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 2048, 7, 7]), R.dtype("float32"))
        cls.conv2d20(alloc212, metadata["relax.expr.Constant"][230], alloc227)
        R.vm.kill_object(alloc212)
        alloc228: R.Tensor((16, 2048, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 2048, 7, 7]), R.dtype("float32"))
        alloc229: R.Tensor((2048,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([2048]), R.dtype("float32"))
        alloc230: R.Tensor((2048,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([2048]), R.dtype("float32"))
        cls.batch_norm11(alloc227, metadata["relax.expr.Constant"][231], metadata["relax.expr.Constant"][232], metadata["relax.expr.Constant"][233], metadata["relax.expr.Constant"][234], alloc228, alloc229, alloc230)
        R.vm.kill_object(alloc227)
        lv195: R.Tuple(R.Tensor((16, 2048, 7, 7), dtype="float32"), R.Tensor((2048,), dtype="float32"), R.Tensor((2048,), dtype="float32")) = alloc228, alloc229, alloc230
        R.vm.kill_object(alloc229)
        R.vm.kill_object(alloc230)
        alloc231: R.Tensor((16, 2048, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 2048, 7, 7]), R.dtype("float32"))
        cls.fused_add3_relu11(alloc224, alloc228, alloc231)
        R.vm.kill_object(alloc224)
        R.vm.kill_object(alloc228)
        alloc232: R.Tensor((16, 512, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage6, R.prim_value(0), R.shape([16, 512, 7, 7]), R.dtype("float32"))
        cls.conv2d21(alloc231, metadata["relax.expr.Constant"][235], alloc232)
        alloc233: R.Tensor((16, 512, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage5, R.prim_value(0), R.shape([16, 512, 7, 7]), R.dtype("float32"))
        alloc234: R.Tensor((512,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([512]), R.dtype("float32"))
        alloc235: R.Tensor((512,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([512]), R.dtype("float32"))
        cls.batch_norm10(alloc232, metadata["relax.expr.Constant"][236], metadata["relax.expr.Constant"][237], metadata["relax.expr.Constant"][238], metadata["relax.expr.Constant"][239], alloc233, alloc234, alloc235)
        R.vm.kill_object(alloc232)
        lv200: R.Tuple(R.Tensor((16, 512, 7, 7), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = alloc233, alloc234, alloc235
        R.vm.kill_object(alloc234)
        R.vm.kill_object(alloc235)
        alloc236: R.Tensor((16, 512, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage6, R.prim_value(0), R.shape([16, 512, 7, 7]), R.dtype("float32"))
        cls.fused_relu10(alloc233, alloc236)
        R.vm.kill_object(alloc233)
        alloc237: R.Tensor((16, 512, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage5, R.prim_value(0), R.shape([16, 512, 7, 7]), R.dtype("float32"))
        cls.conv2d22(alloc236, metadata["relax.expr.Constant"][240], alloc237)
        R.vm.kill_object(alloc236)
        alloc238: R.Tensor((16, 512, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage6, R.prim_value(0), R.shape([16, 512, 7, 7]), R.dtype("float32"))
        alloc239: R.Tensor((512,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([512]), R.dtype("float32"))
        alloc240: R.Tensor((512,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([512]), R.dtype("float32"))
        cls.batch_norm10(alloc237, metadata["relax.expr.Constant"][241], metadata["relax.expr.Constant"][242], metadata["relax.expr.Constant"][243], metadata["relax.expr.Constant"][244], alloc238, alloc239, alloc240)
        R.vm.kill_object(alloc237)
        lv204: R.Tuple(R.Tensor((16, 512, 7, 7), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = alloc238, alloc239, alloc240
        R.vm.kill_object(alloc239)
        R.vm.kill_object(alloc240)
        alloc241: R.Tensor((16, 512, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage5, R.prim_value(0), R.shape([16, 512, 7, 7]), R.dtype("float32"))
        cls.fused_relu10(alloc238, alloc241)
        R.vm.kill_object(alloc238)
        alloc242: R.Tensor((16, 2048, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 2048, 7, 7]), R.dtype("float32"))
        cls.conv2d19(alloc241, metadata["relax.expr.Constant"][245], alloc242)
        R.vm.kill_object(alloc241)
        alloc243: R.Tensor((16, 2048, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 2048, 7, 7]), R.dtype("float32"))
        alloc244: R.Tensor((2048,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([2048]), R.dtype("float32"))
        alloc245: R.Tensor((2048,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([2048]), R.dtype("float32"))
        cls.batch_norm11(alloc242, metadata["relax.expr.Constant"][246], metadata["relax.expr.Constant"][247], metadata["relax.expr.Constant"][248], metadata["relax.expr.Constant"][249], alloc243, alloc244, alloc245)
        R.vm.kill_object(alloc242)
        lv208: R.Tuple(R.Tensor((16, 2048, 7, 7), dtype="float32"), R.Tensor((2048,), dtype="float32"), R.Tensor((2048,), dtype="float32")) = alloc243, alloc244, alloc245
        R.vm.kill_object(alloc244)
        R.vm.kill_object(alloc245)
        alloc246: R.Tensor((16, 2048, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([16, 2048, 7, 7]), R.dtype("float32"))
        R.vm.kill_object(storage4)
        cls.fused_add3_relu11(alloc243, alloc231, alloc246)
        R.vm.kill_object(alloc231)
        R.vm.kill_object(alloc243)
        alloc247: R.Tensor((16, 512, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage6, R.prim_value(0), R.shape([16, 512, 7, 7]), R.dtype("float32"))
        cls.conv2d21(alloc246, metadata["relax.expr.Constant"][250], alloc247)
        alloc248: R.Tensor((16, 512, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage5, R.prim_value(0), R.shape([16, 512, 7, 7]), R.dtype("float32"))
        alloc249: R.Tensor((512,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([512]), R.dtype("float32"))
        alloc250: R.Tensor((512,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([512]), R.dtype("float32"))
        cls.batch_norm10(alloc247, metadata["relax.expr.Constant"][251], metadata["relax.expr.Constant"][252], metadata["relax.expr.Constant"][253], metadata["relax.expr.Constant"][254], alloc248, alloc249, alloc250)
        R.vm.kill_object(alloc247)
        lv213: R.Tuple(R.Tensor((16, 512, 7, 7), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = alloc248, alloc249, alloc250
        R.vm.kill_object(alloc249)
        R.vm.kill_object(alloc250)
        alloc251: R.Tensor((16, 512, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage6, R.prim_value(0), R.shape([16, 512, 7, 7]), R.dtype("float32"))
        cls.fused_relu10(alloc248, alloc251)
        R.vm.kill_object(alloc248)
        alloc252: R.Tensor((16, 512, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage5, R.prim_value(0), R.shape([16, 512, 7, 7]), R.dtype("float32"))
        cls.conv2d22(alloc251, metadata["relax.expr.Constant"][255], alloc252)
        R.vm.kill_object(alloc251)
        alloc253: R.Tensor((16, 512, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage6, R.prim_value(0), R.shape([16, 512, 7, 7]), R.dtype("float32"))
        alloc254: R.Tensor((512,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([512]), R.dtype("float32"))
        alloc255: R.Tensor((512,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([512]), R.dtype("float32"))
        cls.batch_norm10(alloc252, metadata["relax.expr.Constant"][256], metadata["relax.expr.Constant"][257], metadata["relax.expr.Constant"][258], metadata["relax.expr.Constant"][259], alloc253, alloc254, alloc255)
        R.vm.kill_object(alloc252)
        lv217: R.Tuple(R.Tensor((16, 512, 7, 7), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = alloc253, alloc254, alloc255
        R.vm.kill_object(alloc254)
        R.vm.kill_object(alloc255)
        alloc256: R.Tensor((16, 512, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage5, R.prim_value(0), R.shape([16, 512, 7, 7]), R.dtype("float32"))
        R.vm.kill_object(storage5)
        cls.fused_relu10(alloc253, alloc256)
        R.vm.kill_object(alloc253)
        alloc257: R.Tensor((16, 2048, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 2048, 7, 7]), R.dtype("float32"))
        cls.conv2d19(alloc256, metadata["relax.expr.Constant"][260], alloc257)
        R.vm.kill_object(alloc256)
        alloc258: R.Tensor((16, 2048, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([16, 2048, 7, 7]), R.dtype("float32"))
        R.vm.kill_object(storage1)
        alloc259: R.Tensor((2048,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([2048]), R.dtype("float32"))
        R.vm.kill_object(storage2)
        alloc260: R.Tensor((2048,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([2048]), R.dtype("float32"))
        R.vm.kill_object(storage3)
        cls.batch_norm11(alloc257, metadata["relax.expr.Constant"][261], metadata["relax.expr.Constant"][262], metadata["relax.expr.Constant"][263], metadata["relax.expr.Constant"][264], alloc258, alloc259, alloc260)
        R.vm.kill_object(alloc257)
        lv221: R.Tuple(R.Tensor((16, 2048, 7, 7), dtype="float32"), R.Tensor((2048,), dtype="float32"), R.Tensor((2048,), dtype="float32")) = alloc258, alloc259, alloc260
        R.vm.kill_object(alloc259)
        R.vm.kill_object(alloc260)
        alloc261: R.Tensor((16, 2048, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([16, 2048, 7, 7]), R.dtype("float32"))
        R.vm.kill_object(storage)
        cls.fused_add3_relu11(alloc258, alloc246, alloc261)
        R.vm.kill_object(alloc246)
        R.vm.kill_object(alloc258)
        alloc262: R.Tensor((16, 2048, 1, 1), dtype="float32") = R.vm.alloc_tensor(storage6, R.prim_value(0), R.shape([16, 2048, 1, 1]), R.dtype("float32"))
        R.vm.kill_object(storage6)
        cls.adaptive_avg_pool2d(alloc261, alloc262)
        R.vm.kill_object(alloc261)
        lv226: R.Tensor((16, 2048), dtype="float32") = R.call_packed("vm.builtin.reshape", alloc262, R.shape([16, 2048]), sinfo_args=(R.Tensor((16, 2048), dtype="float32"),))
        R.vm.kill_object(alloc262)
        storage_1: R.Object = R.vm.alloc_storage(R.shape([64000]), R.prim_value(0), R.dtype("uint8"), R.str("global"))
        alloc263: R.Tensor((16, 1000), dtype="float32") = R.vm.alloc_tensor(storage_1, R.prim_value(0), R.shape([16, 1000]), R.dtype("float32"))
        R.vm.kill_object(storage_1)
        cls.fused_NT_matmul_add4(lv226, metadata["relax.expr.Constant"][265], metadata["relax.expr.Constant"][266], alloc263)
        R.vm.kill_object(lv226)
        return alloc263

# Metadata omitted. Use show_meta=True in script() method to show it.