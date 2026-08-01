# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def adaptive_avg_pool2d(lv85: T.Buffer((T.int64(64), T.int64(512), T.int64(7), T.int64(7)), "float32"), adaptive_pool_avg: T.Buffer((T.int64(64), T.int64(512), T.int64(1), T.int64(1)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        adaptive_pool_sum_local = T.alloc_buffer((T.int64(64), T.int64(512), T.int64(1), T.int64(1)), scope="local")
        adaptive_pool_sum_rf_local = T.alloc_buffer((T.int64(32), T.int64(64), T.int64(512), T.int64(1), T.int64(1)), scope="local")
        for ax0_ax1_fused in T.thread_binding(T.int64(32768), thread="blockIdx.x"):
            for ax2_ax3_fused_1 in T.thread_binding(T.int64(32), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                with T.block("adaptive_pool_sum_rf_init"):
                    vax2_ax3_fused_1 = T.axis.spatial(T.int64(32), ax2_ax3_fused_1)
                    v0 = T.axis.spatial(T.int64(64), ax0_ax1_fused // T.int64(512))
                    v1 = T.axis.spatial(T.int64(512), ax0_ax1_fused % T.int64(512))
                    T.reads()
                    T.writes(adaptive_pool_sum_rf_local[vax2_ax3_fused_1, v0, v1, T.int64(0), T.int64(0)])
                    adaptive_pool_sum_rf_local[vax2_ax3_fused_1, v0, v1, T.int64(0), T.int64(0)] = T.float32(0.0)
                for ax2_ax3_fused_0, u in T.grid(T.int64(2), 1):
                    with T.block("adaptive_pool_sum_rf_update"):
                        vax2_ax3_fused_1 = T.axis.spatial(T.int64(32), ax2_ax3_fused_1)
                        v0 = T.axis.spatial(T.int64(64), ax0_ax1_fused // T.int64(512))
                        v1 = T.axis.spatial(T.int64(512), ax0_ax1_fused % T.int64(512))
                        vax2_ax3_fused_0 = T.axis.reduce(T.int64(2), ax2_ax3_fused_0)
                        T.where(ax2_ax3_fused_0 * T.int64(32) + ax2_ax3_fused_1 < T.int64(49))
                        T.reads(adaptive_pool_sum_rf_local[vax2_ax3_fused_1, v0, v1, T.int64(0), T.int64(0)], lv85[v0, v1, (vax2_ax3_fused_0 * T.int64(32) + vax2_ax3_fused_1) // T.int64(7), (vax2_ax3_fused_0 * T.int64(32) + vax2_ax3_fused_1) % T.int64(7)])
                        T.writes(adaptive_pool_sum_rf_local[vax2_ax3_fused_1, v0, v1, T.int64(0), T.int64(0)])
                        adaptive_pool_sum_rf_local[vax2_ax3_fused_1, v0, v1, T.int64(0), T.int64(0)] = adaptive_pool_sum_rf_local[vax2_ax3_fused_1, v0, v1, T.int64(0), T.int64(0)] + lv85[v0, v1, (vax2_ax3_fused_0 * T.int64(32) + vax2_ax3_fused_1) // T.int64(7), (vax2_ax3_fused_0 * T.int64(32) + vax2_ax3_fused_1) % T.int64(7)]
            for ax1_ax2_fused in range(T.int64(1)):
                for ax0 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                    with T.block("adaptive_pool_sum"):
                        vax2_ax3_fused_1 = T.axis.reduce(T.int64(32), ax0)
                        v0 = T.axis.spatial(T.int64(64), ax0_ax1_fused // T.int64(512))
                        v1 = T.axis.spatial(T.int64(512), ax0_ax1_fused % T.int64(512))
                        T.reads(adaptive_pool_sum_rf_local[vax2_ax3_fused_1, v0, v1, T.int64(0), T.int64(0)])
                        T.writes(adaptive_pool_sum_local[v0, v1, T.int64(0), T.int64(0)])
                        with T.init():
                            adaptive_pool_sum_local[v0, v1, T.int64(0), T.int64(0)] = T.float32(0.0)
                        adaptive_pool_sum_local[v0, v1, T.int64(0), T.int64(0)] = adaptive_pool_sum_local[v0, v1, T.int64(0), T.int64(0)] + adaptive_pool_sum_rf_local[vax2_ax3_fused_1, v0, v1, T.int64(0), T.int64(0)]
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                with T.block("adaptive_pool_avg"):
                    v0 = T.axis.spatial(T.int64(64), ax0_ax1_fused // T.int64(512) + ax0)
                    v1 = T.axis.spatial(T.int64(512), ax0_ax1_fused % T.int64(512) + ax1)
                    T.reads(adaptive_pool_sum_local[v0, v1, T.int64(0), T.int64(0)])
                    T.writes(adaptive_pool_avg[v0, v1, T.int64(0), T.int64(0)])
                    T.block_attr({"schedule_rule": "meta_schedule.adaptive_pool_avg"})
                    adaptive_pool_avg[v0, v1, T.int64(0), T.int64(0)] = adaptive_pool_sum_local[v0, v1, T.int64(0), T.int64(0)] * T.float32(0.020408163265306121)

    @T.prim_func
    def batch_norm(lv: T.Buffer((T.int64(64), T.int64(64), T.int64(112), T.int64(112)), "float32"), B: T.Buffer((T.int64(64),), "float32"), C: T.Buffer((T.int64(64),), "float32"), D: T.Buffer((T.int64(64),), "float32"), E: T.Buffer((T.int64(64),), "float32"), T_add: T.Buffer((T.int64(64), T.int64(64), T.int64(112), T.int64(112)), "float32"), T_add_1: T.Buffer((T.int64(64),), "float32"), T_add_2: T.Buffer((T.int64(64),), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        lv_red = T.alloc_buffer((T.int64(64),))
        T_multiply_red = T.alloc_buffer((T.int64(64),))
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(50176), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_1"):
                    v0 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(802816))
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
                for ax1, ax2, ax3 in T.grid(T.int64(64), T.int64(112), T.int64(112)):
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
                    T_add_1[v0] = T.float32(0.90000000000000002) * D[v0] + T.float32(0.10000000000000001) * (lv_red[v0] * T.float32(1.2456154336734693e-06))
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply_red_init"):
                    v0 = T.axis.spatial(T.int64(64), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(64))
                    T.reads()
                    T.writes(T_multiply_red[v0])
                    T_multiply_red[v0] = T.float32(0.0)
                for ax1, ax2, ax3 in T.grid(T.int64(64), T.int64(112), T.int64(112)):
                    with T.block("T_multiply_red_update"):
                        v0 = T.axis.spatial(T.int64(64), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                        v1, v2, v3 = T.axis.remap("RRR", [ax1, ax2, ax3])
                        T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(64))
                        T.reads(T_multiply_red[v0], lv[v1, v0, v2, v3], lv_red[v0])
                        T.writes(T_multiply_red[v0])
                        T_multiply_red[v0] = T_multiply_red[v0] + (lv[v1, v0, v2, v3] - lv_red[v0] * T.float32(1.2456154336734693e-06)) * (lv[v1, v0, v2, v3] - lv_red[v0] * T.float32(1.2456154336734693e-06))
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_3"):
                    v0 = T.axis.spatial(T.int64(64), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(64))
                    T.reads(E[v0], T_multiply_red[v0])
                    T.writes(T_add_2[v0])
                    T_add_2[v0] = T.float32(0.90000000000000002) * E[v0] + T.float32(0.10000000000000001) * (T_multiply_red[v0] * T.float32(1.2456154336734693e-06))

    @T.prim_func
    def batch_norm1(lv5: T.Buffer((T.int64(64), T.int64(64), T.int64(56), T.int64(56)), "float32"), B: T.Buffer((T.int64(64),), "float32"), C: T.Buffer((T.int64(64),), "float32"), D: T.Buffer((T.int64(64),), "float32"), E: T.Buffer((T.int64(64),), "float32"), T_add: T.Buffer((T.int64(64), T.int64(64), T.int64(56), T.int64(56)), "float32"), T_add_1: T.Buffer((T.int64(64),), "float32"), T_add_2: T.Buffer((T.int64(64),), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        lv5_red = T.alloc_buffer((T.int64(64),))
        T_multiply_red = T.alloc_buffer((T.int64(64),))
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(12544), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_1"):
                    v0 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(200704))
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
                for ax1, ax2, ax3 in T.grid(T.int64(64), T.int64(56), T.int64(56)):
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
                    T_add_1[v0] = T.float32(0.90000000000000002) * D[v0] + T.float32(0.10000000000000001) * (lv5_red[v0] * T.float32(4.9824617346938772e-06))
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply_red_init"):
                    v0 = T.axis.spatial(T.int64(64), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(64))
                    T.reads()
                    T.writes(T_multiply_red[v0])
                    T_multiply_red[v0] = T.float32(0.0)
                for ax1, ax2, ax3 in T.grid(T.int64(64), T.int64(56), T.int64(56)):
                    with T.block("T_multiply_red_update"):
                        v0 = T.axis.spatial(T.int64(64), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                        v1, v2, v3 = T.axis.remap("RRR", [ax1, ax2, ax3])
                        T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(64))
                        T.reads(T_multiply_red[v0], lv5[v1, v0, v2, v3], lv5_red[v0])
                        T.writes(T_multiply_red[v0])
                        T_multiply_red[v0] = T_multiply_red[v0] + (lv5[v1, v0, v2, v3] - lv5_red[v0] * T.float32(4.9824617346938772e-06)) * (lv5[v1, v0, v2, v3] - lv5_red[v0] * T.float32(4.9824617346938772e-06))
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_3"):
                    v0 = T.axis.spatial(T.int64(64), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(64))
                    T.reads(E[v0], T_multiply_red[v0])
                    T.writes(T_add_2[v0])
                    T_add_2[v0] = T.float32(0.90000000000000002) * E[v0] + T.float32(0.10000000000000001) * (T_multiply_red[v0] * T.float32(4.9824617346938772e-06))

    @T.prim_func
    def batch_norm2(lv23: T.Buffer((T.int64(64), T.int64(128), T.int64(28), T.int64(28)), "float32"), B: T.Buffer((T.int64(128),), "float32"), C: T.Buffer((T.int64(128),), "float32"), D: T.Buffer((T.int64(128),), "float32"), E: T.Buffer((T.int64(128),), "float32"), T_add: T.Buffer((T.int64(64), T.int64(128), T.int64(28), T.int64(28)), "float32"), T_add_1: T.Buffer((T.int64(128),), "float32"), T_add_2: T.Buffer((T.int64(128),), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        lv23_red = T.alloc_buffer((T.int64(128),))
        T_multiply_red = T.alloc_buffer((T.int64(128),))
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(6272), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_1"):
                    v0 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(100352))
                    v1 = T.axis.spatial(T.int64(128), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(100352) // T.int64(784))
                    v2 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(784) // T.int64(28))
                    v3 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(28))
                    T.reads(lv23[v0, v1, v2, v3], D[v1], E[v1], B[v1], C[v1])
                    T.writes(T_add[v0, v1, v2, v3])
                    T_add[v0, v1, v2, v3] = (lv23[v0, v1, v2, v3] - D[v1]) / T.sqrt(E[v1] + T.float32(1.0000000000000001e-05)) * B[v1] + C[v1]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("lv23_red_init"):
                    v0 = T.axis.spatial(T.int64(128), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(128))
                    T.reads()
                    T.writes(lv23_red[v0])
                    lv23_red[v0] = T.float32(0.0)
                for ax1, ax2, ax3 in T.grid(T.int64(64), T.int64(28), T.int64(28)):
                    with T.block("lv23_red_update"):
                        v0 = T.axis.spatial(T.int64(128), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                        v1, v2, v3 = T.axis.remap("RRR", [ax1, ax2, ax3])
                        T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(128))
                        T.reads(lv23_red[v0], lv23[v1, v0, v2, v3])
                        T.writes(lv23_red[v0])
                        lv23_red[v0] = lv23_red[v0] + lv23[v1, v0, v2, v3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_2"):
                    v0 = T.axis.spatial(T.int64(128), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(128))
                    T.reads(D[v0], lv23_red[v0])
                    T.writes(T_add_1[v0])
                    T_add_1[v0] = T.float32(0.90000000000000002) * D[v0] + T.float32(0.10000000000000001) * (lv23_red[v0] * T.float32(1.9929846938775509e-05))
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply_red_init"):
                    v0 = T.axis.spatial(T.int64(128), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(128))
                    T.reads()
                    T.writes(T_multiply_red[v0])
                    T_multiply_red[v0] = T.float32(0.0)
                for ax1, ax2, ax3 in T.grid(T.int64(64), T.int64(28), T.int64(28)):
                    with T.block("T_multiply_red_update"):
                        v0 = T.axis.spatial(T.int64(128), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                        v1, v2, v3 = T.axis.remap("RRR", [ax1, ax2, ax3])
                        T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(128))
                        T.reads(T_multiply_red[v0], lv23[v1, v0, v2, v3], lv23_red[v0])
                        T.writes(T_multiply_red[v0])
                        T_multiply_red[v0] = T_multiply_red[v0] + (lv23[v1, v0, v2, v3] - lv23_red[v0] * T.float32(1.9929846938775509e-05)) * (lv23[v1, v0, v2, v3] - lv23_red[v0] * T.float32(1.9929846938775509e-05))
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_3"):
                    v0 = T.axis.spatial(T.int64(128), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(128))
                    T.reads(E[v0], T_multiply_red[v0])
                    T.writes(T_add_2[v0])
                    T_add_2[v0] = T.float32(0.90000000000000002) * E[v0] + T.float32(0.10000000000000001) * (T_multiply_red[v0] * T.float32(1.9929846938775509e-05))

    @T.prim_func
    def batch_norm3(lv44: T.Buffer((T.int64(64), T.int64(256), T.int64(14), T.int64(14)), "float32"), B: T.Buffer((T.int64(256),), "float32"), C: T.Buffer((T.int64(256),), "float32"), D: T.Buffer((T.int64(256),), "float32"), E: T.Buffer((T.int64(256),), "float32"), T_add: T.Buffer((T.int64(64), T.int64(256), T.int64(14), T.int64(14)), "float32"), T_add_1: T.Buffer((T.int64(256),), "float32"), T_add_2: T.Buffer((T.int64(256),), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        lv44_red = T.alloc_buffer((T.int64(256),))
        T_multiply_red = T.alloc_buffer((T.int64(256),))
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(3136), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_1"):
                    v0 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(50176))
                    v1 = T.axis.spatial(T.int64(256), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(50176) // T.int64(196))
                    v2 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(196) // T.int64(14))
                    v3 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(14))
                    T.reads(lv44[v0, v1, v2, v3], D[v1], E[v1], B[v1], C[v1])
                    T.writes(T_add[v0, v1, v2, v3])
                    T_add[v0, v1, v2, v3] = (lv44[v0, v1, v2, v3] - D[v1]) / T.sqrt(E[v1] + T.float32(1.0000000000000001e-05)) * B[v1] + C[v1]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("lv44_red_init"):
                    v0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(256))
                    T.reads()
                    T.writes(lv44_red[v0])
                    lv44_red[v0] = T.float32(0.0)
                for ax1, ax2, ax3 in T.grid(T.int64(64), T.int64(14), T.int64(14)):
                    with T.block("lv44_red_update"):
                        v0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                        v1, v2, v3 = T.axis.remap("RRR", [ax1, ax2, ax3])
                        T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(256))
                        T.reads(lv44_red[v0], lv44[v1, v0, v2, v3])
                        T.writes(lv44_red[v0])
                        lv44_red[v0] = lv44_red[v0] + lv44[v1, v0, v2, v3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_2"):
                    v0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(256))
                    T.reads(D[v0], lv44_red[v0])
                    T.writes(T_add_1[v0])
                    T_add_1[v0] = T.float32(0.90000000000000002) * D[v0] + T.float32(0.10000000000000001) * (lv44_red[v0] * T.float32(7.9719387755102034e-05))
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply_red_init"):
                    v0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(256))
                    T.reads()
                    T.writes(T_multiply_red[v0])
                    T_multiply_red[v0] = T.float32(0.0)
                for ax1, ax2, ax3 in T.grid(T.int64(64), T.int64(14), T.int64(14)):
                    with T.block("T_multiply_red_update"):
                        v0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                        v1, v2, v3 = T.axis.remap("RRR", [ax1, ax2, ax3])
                        T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(256))
                        T.reads(T_multiply_red[v0], lv44[v1, v0, v2, v3], lv44_red[v0])
                        T.writes(T_multiply_red[v0])
                        T_multiply_red[v0] = T_multiply_red[v0] + (lv44[v1, v0, v2, v3] - lv44_red[v0] * T.float32(7.9719387755102034e-05)) * (lv44[v1, v0, v2, v3] - lv44_red[v0] * T.float32(7.9719387755102034e-05))
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_3"):
                    v0 = T.axis.spatial(T.int64(256), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(256))
                    T.reads(E[v0], T_multiply_red[v0])
                    T.writes(T_add_2[v0])
                    T_add_2[v0] = T.float32(0.90000000000000002) * E[v0] + T.float32(0.10000000000000001) * (T_multiply_red[v0] * T.float32(7.9719387755102034e-05))

    @T.prim_func
    def batch_norm4(lv65: T.Buffer((T.int64(64), T.int64(512), T.int64(7), T.int64(7)), "float32"), B: T.Buffer((T.int64(512),), "float32"), C: T.Buffer((T.int64(512),), "float32"), D: T.Buffer((T.int64(512),), "float32"), E: T.Buffer((T.int64(512),), "float32"), T_add: T.Buffer((T.int64(64), T.int64(512), T.int64(7), T.int64(7)), "float32"), T_add_1: T.Buffer((T.int64(512),), "float32"), T_add_2: T.Buffer((T.int64(512),), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        lv65_red = T.alloc_buffer((T.int64(512),))
        T_multiply_red = T.alloc_buffer((T.int64(512),))
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1568), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_1"):
                    v0 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(25088))
                    v1 = T.axis.spatial(T.int64(512), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(25088) // T.int64(49))
                    v2 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(49) // T.int64(7))
                    v3 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(7))
                    T.reads(lv65[v0, v1, v2, v3], D[v1], E[v1], B[v1], C[v1])
                    T.writes(T_add[v0, v1, v2, v3])
                    T_add[v0, v1, v2, v3] = (lv65[v0, v1, v2, v3] - D[v1]) / T.sqrt(E[v1] + T.float32(1.0000000000000001e-05)) * B[v1] + C[v1]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("lv65_red_init"):
                    v0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(512))
                    T.reads()
                    T.writes(lv65_red[v0])
                    lv65_red[v0] = T.float32(0.0)
                for ax1, ax2, ax3 in T.grid(T.int64(64), T.int64(7), T.int64(7)):
                    with T.block("lv65_red_update"):
                        v0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                        v1, v2, v3 = T.axis.remap("RRR", [ax1, ax2, ax3])
                        T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(512))
                        T.reads(lv65_red[v0], lv65[v1, v0, v2, v3])
                        T.writes(lv65_red[v0])
                        lv65_red[v0] = lv65_red[v0] + lv65[v1, v0, v2, v3]
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_2"):
                    v0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(512))
                    T.reads(D[v0], lv65_red[v0])
                    T.writes(T_add_1[v0])
                    T_add_1[v0] = T.float32(0.90000000000000002) * D[v0] + T.float32(0.10000000000000001) * (lv65_red[v0] * T.float32(0.00031887755102040814))
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply_red_init"):
                    v0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(512))
                    T.reads()
                    T.writes(T_multiply_red[v0])
                    T_multiply_red[v0] = T.float32(0.0)
                for ax1, ax2, ax3 in T.grid(T.int64(64), T.int64(7), T.int64(7)):
                    with T.block("T_multiply_red_update"):
                        v0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                        v1, v2, v3 = T.axis.remap("RRR", [ax1, ax2, ax3])
                        T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(512))
                        T.reads(T_multiply_red[v0], lv65[v1, v0, v2, v3], lv65_red[v0])
                        T.writes(T_multiply_red[v0])
                        T_multiply_red[v0] = T_multiply_red[v0] + (lv65[v1, v0, v2, v3] - lv65_red[v0] * T.float32(0.00031887755102040814)) * (lv65[v1, v0, v2, v3] - lv65_red[v0] * T.float32(0.00031887755102040814))
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add_3"):
                    v0 = T.axis.spatial(T.int64(512), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(512))
                    T.reads(E[v0], T_multiply_red[v0])
                    T.writes(T_add_2[v0])
                    T_add_2[v0] = T.float32(0.90000000000000002) * E[v0] + T.float32(0.10000000000000001) * (T_multiply_red[v0] * T.float32(0.00031887755102040814))

    @T.prim_func
    def conv2d(inp_0: T.Buffer((T.int64(64), T.int64(3), T.int64(224), T.int64(224)), "float32"), B: T.Buffer((T.int64(64), T.int64(3), T.int64(7), T.int64(7)), "float32"), conv2d_nchw: T.Buffer((T.int64(64), T.int64(64), T.int64(112), T.int64(112)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        conv2d_nchw_reindex_local = T.alloc_buffer((T.int64(1), T.int64(802816), T.int64(64)), scope="local")
        pad_temp_reindex_pad_shared = T.alloc_buffer((T.int64(1), T.int64(802816), T.int64(160)), scope="shared")
        B_reindex_pad_shared = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(160)), scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(1), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding(T.int64(25088), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_init"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(802816), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
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
                                                        v1 = T.axis.spatial(T.int64(802816), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
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
                                                v1 = T.axis.spatial(T.int64(802816), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(64), ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(160), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(conv2d_nchw_reindex_local[T.int64(0), v1, v2], pad_temp_reindex_pad_shared[T.int64(0), v1, v3], B_reindex_pad_shared[T.int64(0), v2, v3])
                                                T.writes(conv2d_nchw_reindex_local[T.int64(0), v1, v2])
                                                conv2d_nchw_reindex_local[T.int64(0), v1, v2] = conv2d_nchw_reindex_local[T.int64(0), v1, v2] + pad_temp_reindex_pad_shared[T.int64(0), v1, v3] * B_reindex_pad_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_reindex_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(T.int64(802816), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(64), ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            T.reads(conv2d_nchw_reindex_local[v0, v1, v2])
                                            T.writes(conv2d_nchw[v1 // T.int64(12544), v2, v1 // T.int64(112) % T.int64(112), v1 % T.int64(112)])
                                            conv2d_nchw[v1 // T.int64(12544), v2, v1 // T.int64(112) % T.int64(112), v1 % T.int64(112)] = conv2d_nchw_reindex_local[v0, v1, v2]

    @T.prim_func
    def conv2d1(lv4: T.Buffer((T.int64(64), T.int64(64), T.int64(56), T.int64(56)), "float32"), B: T.Buffer((T.int64(64), T.int64(64), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(64), T.int64(64), T.int64(56), T.int64(56)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        conv2d_nchw_reindex_local = T.alloc_buffer((T.int64(1), T.int64(200704), T.int64(64)), scope="local")
        pad_temp_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(200704), T.int64(576)), scope="shared")
        B_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(576)), scope="shared")
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
                                for ax3_0 in range(T.int64(36)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("pad_temp_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(200704), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(576), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv4[v1 // T.int64(3136), v2 // T.int64(9), v1 // T.int64(56) % T.int64(56) + v2 // T.int64(3) % T.int64(3) - T.int64(1), v1 % T.int64(56) + v2 % T.int64(3) - T.int64(1)])
                                                        T.writes(pad_temp_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        pad_temp_reindex_shared[v0, v1, v2] = T.if_then_else(T.int64(1) <= v1 // T.int64(56) % T.int64(56) + v2 // T.int64(3) % T.int64(3) and v1 // T.int64(56) % T.int64(56) + v2 // T.int64(3) % T.int64(3) < T.int64(57) and T.int64(1) <= v1 % T.int64(56) + v2 % T.int64(3) and v1 % T.int64(56) + v2 % T.int64(3) < T.int64(57), lv4[v1 // T.int64(3136), v2 // T.int64(9), v1 // T.int64(56) % T.int64(56) + v2 // T.int64(3) % T.int64(3) - T.int64(1), v1 % T.int64(56) + v2 % T.int64(3) - T.int64(1)], T.float32(0.0))
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
                                                v1 = T.axis.spatial(T.int64(200704), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(64), ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(576), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(conv2d_nchw_reindex_local[T.int64(0), v1, v2], pad_temp_reindex_shared[T.int64(0), v1, v3], B_reindex_shared[T.int64(0), v2, v3])
                                                T.writes(conv2d_nchw_reindex_local[T.int64(0), v1, v2])
                                                conv2d_nchw_reindex_local[T.int64(0), v1, v2] = conv2d_nchw_reindex_local[T.int64(0), v1, v2] + pad_temp_reindex_shared[T.int64(0), v1, v3] * B_reindex_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("conv2d_nchw_reindex_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(T.int64(200704), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(64), ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            T.reads(conv2d_nchw_reindex_local[v0, v1, v2])
                                            T.writes(conv2d_nchw[v1 // T.int64(3136), v2, v1 // T.int64(56) % T.int64(56), v1 % T.int64(56)])
                                            conv2d_nchw[v1 // T.int64(3136), v2, v1 // T.int64(56) % T.int64(56), v1 % T.int64(56)] = conv2d_nchw_reindex_local[v0, v1, v2]

    @T.prim_func
    def conv2d10(lv64: T.Buffer((T.int64(64), T.int64(256), T.int64(14), T.int64(14)), "float32"), B: T.Buffer((T.int64(512), T.int64(256), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(64), T.int64(512), T.int64(7), T.int64(7)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        conv2d_nchw_reindex_local = T.alloc_buffer((T.int64(1), T.int64(3136), T.int64(512)), scope="local")
        pad_temp_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(3136), T.int64(256)), scope="shared")
        B_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(256)), scope="shared")
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
                                for ax3_0 in range(T.int64(16)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("pad_temp_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(3136), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(256), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv64[v1 // T.int64(49), v2, v1 // T.int64(7) % T.int64(7) * T.int64(2), v1 % T.int64(7) * T.int64(2)])
                                                        T.writes(pad_temp_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        pad_temp_reindex_shared[v0, v1, v2] = lv64[v1 // T.int64(49), v2, v1 // T.int64(7) % T.int64(7) * T.int64(2), v1 % T.int64(7) * T.int64(2)]
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
                                                v1 = T.axis.spatial(T.int64(3136), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(512), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(256), ax3_0 * T.int64(16) + ax3_1)
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
                                            T.writes(conv2d_nchw[v1 // T.int64(49), v2, v1 // T.int64(7) % T.int64(7), v1 % T.int64(7)])
                                            conv2d_nchw[v1 // T.int64(49), v2, v1 // T.int64(7) % T.int64(7), v1 % T.int64(7)] = conv2d_nchw_reindex_local[v0, v1, v2]

    @T.prim_func
    def conv2d2(lv22: T.Buffer((T.int64(64), T.int64(64), T.int64(56), T.int64(56)), "float32"), B: T.Buffer((T.int64(128), T.int64(64), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(64), T.int64(128), T.int64(28), T.int64(28)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        conv2d_nchw_reindex_local = T.alloc_buffer((T.int64(1), T.int64(50176), T.int64(128)), scope="local")
        pad_temp_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(50176), T.int64(576)), scope="shared")
        B_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(576)), scope="shared")
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
                                for ax3_0 in range(T.int64(36)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("pad_temp_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(50176), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(576), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv22[v1 // T.int64(784), v2 // T.int64(9), v1 // T.int64(28) % T.int64(28) * T.int64(2) + v2 // T.int64(3) % T.int64(3) - T.int64(1), v1 % T.int64(28) * T.int64(2) + v2 % T.int64(3) - T.int64(1)])
                                                        T.writes(pad_temp_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        pad_temp_reindex_shared[v0, v1, v2] = T.if_then_else(T.int64(1) <= v1 // T.int64(28) % T.int64(28) * T.int64(2) + v2 // T.int64(3) % T.int64(3) and v1 // T.int64(28) % T.int64(28) * T.int64(2) + v2 // T.int64(3) % T.int64(3) < T.int64(57) and T.int64(1) <= v1 % T.int64(28) * T.int64(2) + v2 % T.int64(3) and v1 % T.int64(28) * T.int64(2) + v2 % T.int64(3) < T.int64(57), lv22[v1 // T.int64(784), v2 // T.int64(9), v1 // T.int64(28) % T.int64(28) * T.int64(2) + v2 // T.int64(3) % T.int64(3) - T.int64(1), v1 % T.int64(28) * T.int64(2) + v2 % T.int64(3) - T.int64(1)], T.float32(0.0))
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("B_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(128), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
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
                                                v2 = T.axis.spatial(T.int64(128), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(576), ax3_0 * T.int64(16) + ax3_1)
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
                                            T.writes(conv2d_nchw[v1 // T.int64(784), v2, v1 // T.int64(28) % T.int64(28), v1 % T.int64(28)])
                                            conv2d_nchw[v1 // T.int64(784), v2, v1 // T.int64(28) % T.int64(28), v1 % T.int64(28)] = conv2d_nchw_reindex_local[v0, v1, v2]

    @T.prim_func
    def conv2d3(lv26: T.Buffer((T.int64(64), T.int64(128), T.int64(28), T.int64(28)), "float32"), B: T.Buffer((T.int64(128), T.int64(128), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(64), T.int64(128), T.int64(28), T.int64(28)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        conv2d_nchw_reindex_local = T.alloc_buffer((T.int64(1), T.int64(50176), T.int64(128)), scope="local")
        pad_temp_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(50176), T.int64(1152)), scope="shared")
        B_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(1152)), scope="shared")
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
                                for ax3_0 in range(T.int64(72)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("pad_temp_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(50176), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(1152), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv26[v1 // T.int64(784), v2 // T.int64(9), v1 // T.int64(28) % T.int64(28) + v2 // T.int64(3) % T.int64(3) - T.int64(1), v1 % T.int64(28) + v2 % T.int64(3) - T.int64(1)])
                                                        T.writes(pad_temp_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        pad_temp_reindex_shared[v0, v1, v2] = T.if_then_else(T.int64(1) <= v1 // T.int64(28) % T.int64(28) + v2 // T.int64(3) % T.int64(3) and v1 // T.int64(28) % T.int64(28) + v2 // T.int64(3) % T.int64(3) < T.int64(29) and T.int64(1) <= v1 % T.int64(28) + v2 % T.int64(3) and v1 % T.int64(28) + v2 % T.int64(3) < T.int64(29), lv26[v1 // T.int64(784), v2 // T.int64(9), v1 // T.int64(28) % T.int64(28) + v2 // T.int64(3) % T.int64(3) - T.int64(1), v1 % T.int64(28) + v2 % T.int64(3) - T.int64(1)], T.float32(0.0))
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
                                                v1 = T.axis.spatial(T.int64(50176), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(128), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(1152), ax3_0 * T.int64(16) + ax3_1)
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
                                            T.writes(conv2d_nchw[v1 // T.int64(784), v2, v1 // T.int64(28) % T.int64(28), v1 % T.int64(28)])
                                            conv2d_nchw[v1 // T.int64(784), v2, v1 // T.int64(28) % T.int64(28), v1 % T.int64(28)] = conv2d_nchw_reindex_local[v0, v1, v2]

    @T.prim_func
    def conv2d4(lv22: T.Buffer((T.int64(64), T.int64(64), T.int64(56), T.int64(56)), "float32"), B: T.Buffer((T.int64(128), T.int64(64), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(64), T.int64(128), T.int64(28), T.int64(28)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        conv2d_nchw_reindex_local = T.alloc_buffer((T.int64(1), T.int64(50176), T.int64(128)), scope="local")
        pad_temp_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(50176), T.int64(64)), scope="shared")
        B_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(64)), scope="shared")
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
                                for ax3_0 in range(T.int64(4)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("pad_temp_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(50176), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(64), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv22[v1 // T.int64(784), v2, v1 // T.int64(28) % T.int64(28) * T.int64(2), v1 % T.int64(28) * T.int64(2)])
                                                        T.writes(pad_temp_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        pad_temp_reindex_shared[v0, v1, v2] = lv22[v1 // T.int64(784), v2, v1 // T.int64(28) % T.int64(28) * T.int64(2), v1 % T.int64(28) * T.int64(2)]
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("B_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(128), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
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
                                                v2 = T.axis.spatial(T.int64(128), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(64), ax3_0 * T.int64(16) + ax3_1)
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
                                            T.writes(conv2d_nchw[v1 // T.int64(784), v2, v1 // T.int64(28) % T.int64(28), v1 % T.int64(28)])
                                            conv2d_nchw[v1 // T.int64(784), v2, v1 // T.int64(28) % T.int64(28), v1 % T.int64(28)] = conv2d_nchw_reindex_local[v0, v1, v2]

    @T.prim_func
    def conv2d5(lv43: T.Buffer((T.int64(64), T.int64(128), T.int64(28), T.int64(28)), "float32"), B: T.Buffer((T.int64(256), T.int64(128), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(64), T.int64(256), T.int64(14), T.int64(14)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        conv2d_nchw_reindex_local = T.alloc_buffer((T.int64(1), T.int64(12544), T.int64(256)), scope="local")
        pad_temp_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(12544), T.int64(1152)), scope="shared")
        B_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(1152)), scope="shared")
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
                                for ax3_0 in range(T.int64(72)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("pad_temp_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(12544), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(1152), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv43[v1 // T.int64(196), v2 // T.int64(9), v1 // T.int64(14) % T.int64(14) * T.int64(2) + v2 // T.int64(3) % T.int64(3) - T.int64(1), v1 % T.int64(14) * T.int64(2) + v2 % T.int64(3) - T.int64(1)])
                                                        T.writes(pad_temp_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        pad_temp_reindex_shared[v0, v1, v2] = T.if_then_else(T.int64(1) <= v1 // T.int64(14) % T.int64(14) * T.int64(2) + v2 // T.int64(3) % T.int64(3) and v1 // T.int64(14) % T.int64(14) * T.int64(2) + v2 // T.int64(3) % T.int64(3) < T.int64(29) and T.int64(1) <= v1 % T.int64(14) * T.int64(2) + v2 % T.int64(3) and v1 % T.int64(14) * T.int64(2) + v2 % T.int64(3) < T.int64(29), lv43[v1 // T.int64(196), v2 // T.int64(9), v1 // T.int64(14) % T.int64(14) * T.int64(2) + v2 // T.int64(3) % T.int64(3) - T.int64(1), v1 % T.int64(14) * T.int64(2) + v2 % T.int64(3) - T.int64(1)], T.float32(0.0))
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("B_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(256), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
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
                                                v2 = T.axis.spatial(T.int64(256), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(1152), ax3_0 * T.int64(16) + ax3_1)
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
                                            T.writes(conv2d_nchw[v1 // T.int64(196), v2, v1 // T.int64(14) % T.int64(14), v1 % T.int64(14)])
                                            conv2d_nchw[v1 // T.int64(196), v2, v1 // T.int64(14) % T.int64(14), v1 % T.int64(14)] = conv2d_nchw_reindex_local[v0, v1, v2]

    @T.prim_func
    def conv2d6(lv47: T.Buffer((T.int64(64), T.int64(256), T.int64(14), T.int64(14)), "float32"), B: T.Buffer((T.int64(256), T.int64(256), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(64), T.int64(256), T.int64(14), T.int64(14)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        conv2d_nchw_reindex_local = T.alloc_buffer((T.int64(1), T.int64(12544), T.int64(256)), scope="local")
        pad_temp_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(12544), T.int64(2304)), scope="shared")
        B_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(2304)), scope="shared")
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
                                for ax3_0 in range(T.int64(144)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("pad_temp_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(12544), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(2304), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv47[v1 // T.int64(196), v2 // T.int64(9), v1 // T.int64(14) % T.int64(14) + v2 // T.int64(3) % T.int64(3) - T.int64(1), v1 % T.int64(14) + v2 % T.int64(3) - T.int64(1)])
                                                        T.writes(pad_temp_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        pad_temp_reindex_shared[v0, v1, v2] = T.if_then_else(T.int64(1) <= v1 // T.int64(14) % T.int64(14) + v2 // T.int64(3) % T.int64(3) and v1 // T.int64(14) % T.int64(14) + v2 // T.int64(3) % T.int64(3) < T.int64(15) and T.int64(1) <= v1 % T.int64(14) + v2 % T.int64(3) and v1 % T.int64(14) + v2 % T.int64(3) < T.int64(15), lv47[v1 // T.int64(196), v2 // T.int64(9), v1 // T.int64(14) % T.int64(14) + v2 // T.int64(3) % T.int64(3) - T.int64(1), v1 % T.int64(14) + v2 % T.int64(3) - T.int64(1)], T.float32(0.0))
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
                                                v1 = T.axis.spatial(T.int64(12544), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(256), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(2304), ax3_0 * T.int64(16) + ax3_1)
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
                                            T.writes(conv2d_nchw[v1 // T.int64(196), v2, v1 // T.int64(14) % T.int64(14), v1 % T.int64(14)])
                                            conv2d_nchw[v1 // T.int64(196), v2, v1 // T.int64(14) % T.int64(14), v1 % T.int64(14)] = conv2d_nchw_reindex_local[v0, v1, v2]

    @T.prim_func
    def conv2d7(lv43: T.Buffer((T.int64(64), T.int64(128), T.int64(28), T.int64(28)), "float32"), B: T.Buffer((T.int64(256), T.int64(128), T.int64(1), T.int64(1)), "float32"), conv2d_nchw: T.Buffer((T.int64(64), T.int64(256), T.int64(14), T.int64(14)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        conv2d_nchw_reindex_local = T.alloc_buffer((T.int64(1), T.int64(12544), T.int64(256)), scope="local")
        pad_temp_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(12544), T.int64(128)), scope="shared")
        B_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(128)), scope="shared")
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
                                for ax3_0 in range(T.int64(8)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("pad_temp_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(12544), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(128), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv43[v1 // T.int64(196), v2, v1 // T.int64(14) % T.int64(14) * T.int64(2), v1 % T.int64(14) * T.int64(2)])
                                                        T.writes(pad_temp_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        pad_temp_reindex_shared[v0, v1, v2] = lv43[v1 // T.int64(196), v2, v1 // T.int64(14) % T.int64(14) * T.int64(2), v1 % T.int64(14) * T.int64(2)]
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("B_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(256), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
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
                                                v2 = T.axis.spatial(T.int64(256), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(128), ax3_0 * T.int64(16) + ax3_1)
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
                                            T.writes(conv2d_nchw[v1 // T.int64(196), v2, v1 // T.int64(14) % T.int64(14), v1 % T.int64(14)])
                                            conv2d_nchw[v1 // T.int64(196), v2, v1 // T.int64(14) % T.int64(14), v1 % T.int64(14)] = conv2d_nchw_reindex_local[v0, v1, v2]

    @T.prim_func
    def conv2d8(lv64: T.Buffer((T.int64(64), T.int64(256), T.int64(14), T.int64(14)), "float32"), B: T.Buffer((T.int64(512), T.int64(256), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(64), T.int64(512), T.int64(7), T.int64(7)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        conv2d_nchw_reindex_local = T.alloc_buffer((T.int64(1), T.int64(3136), T.int64(512)), scope="local")
        pad_temp_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(3136), T.int64(2304)), scope="shared")
        B_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(2304)), scope="shared")
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
                                for ax3_0 in range(T.int64(144)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("pad_temp_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(3136), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(2304), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv64[v1 // T.int64(49), v2 // T.int64(9), v1 // T.int64(7) % T.int64(7) * T.int64(2) + v2 // T.int64(3) % T.int64(3) - T.int64(1), v1 % T.int64(7) * T.int64(2) + v2 % T.int64(3) - T.int64(1)])
                                                        T.writes(pad_temp_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        pad_temp_reindex_shared[v0, v1, v2] = T.if_then_else(T.int64(1) <= v1 // T.int64(7) % T.int64(7) * T.int64(2) + v2 // T.int64(3) % T.int64(3) and v1 // T.int64(7) % T.int64(7) * T.int64(2) + v2 // T.int64(3) % T.int64(3) < T.int64(15) and T.int64(1) <= v1 % T.int64(7) * T.int64(2) + v2 % T.int64(3) and v1 % T.int64(7) * T.int64(2) + v2 % T.int64(3) < T.int64(15), lv64[v1 // T.int64(49), v2 // T.int64(9), v1 // T.int64(7) % T.int64(7) * T.int64(2) + v2 // T.int64(3) % T.int64(3) - T.int64(1), v1 % T.int64(7) * T.int64(2) + v2 % T.int64(3) - T.int64(1)], T.float32(0.0))
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("B_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(512), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
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
                                                v2 = T.axis.spatial(T.int64(512), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(2304), ax3_0 * T.int64(16) + ax3_1)
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
                                            T.writes(conv2d_nchw[v1 // T.int64(49), v2, v1 // T.int64(7) % T.int64(7), v1 % T.int64(7)])
                                            conv2d_nchw[v1 // T.int64(49), v2, v1 // T.int64(7) % T.int64(7), v1 % T.int64(7)] = conv2d_nchw_reindex_local[v0, v1, v2]

    @T.prim_func
    def conv2d9(lv68: T.Buffer((T.int64(64), T.int64(512), T.int64(7), T.int64(7)), "float32"), B: T.Buffer((T.int64(512), T.int64(512), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(64), T.int64(512), T.int64(7), T.int64(7)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        conv2d_nchw_reindex_local = T.alloc_buffer((T.int64(1), T.int64(3136), T.int64(512)), scope="local")
        pad_temp_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(3136), T.int64(4608)), scope="shared")
        B_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(4608)), scope="shared")
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
                                for ax3_0 in range(T.int64(288)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("pad_temp_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(3136), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(4608), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv68[v1 // T.int64(49), v2 // T.int64(9), v1 // T.int64(7) % T.int64(7) + v2 // T.int64(3) % T.int64(3) - T.int64(1), v1 % T.int64(7) + v2 % T.int64(3) - T.int64(1)])
                                                        T.writes(pad_temp_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        pad_temp_reindex_shared[v0, v1, v2] = T.if_then_else(T.int64(1) <= v1 // T.int64(7) % T.int64(7) + v2 // T.int64(3) % T.int64(3) and v1 // T.int64(7) % T.int64(7) + v2 // T.int64(3) % T.int64(3) < T.int64(8) and T.int64(1) <= v1 % T.int64(7) + v2 % T.int64(3) and v1 % T.int64(7) + v2 % T.int64(3) < T.int64(8), lv68[v1 // T.int64(49), v2 // T.int64(9), v1 // T.int64(7) % T.int64(7) + v2 // T.int64(3) % T.int64(3) - T.int64(1), v1 % T.int64(7) + v2 % T.int64(3) - T.int64(1)], T.float32(0.0))
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
                                                v1 = T.axis.spatial(T.int64(3136), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(512), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(4608), ax3_0 * T.int64(16) + ax3_1)
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
                                            T.writes(conv2d_nchw[v1 // T.int64(49), v2, v1 // T.int64(7) % T.int64(7), v1 % T.int64(7)])
                                            conv2d_nchw[v1 // T.int64(49), v2, v1 // T.int64(7) % T.int64(7), v1 % T.int64(7)] = conv2d_nchw_reindex_local[v0, v1, v2]

    @T.prim_func
    def fused_NT_matmul_add4(lv87: T.Buffer((T.int64(64), T.int64(512)), "float32"), param_0: T.Buffer((T.int64(1000), T.int64(512)), "float32"), param_1: T.Buffer((T.int64(1000),), "float32"), T_add_intermediate: T.Buffer((T.int64(64), T.int64(1000)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        NT_matmul_intermediate_reindex_pad_local = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(1024)), scope="local")
        lv87_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(512)), scope="shared")
        param_0_reindex_pad_shared = T.alloc_buffer((T.int64(1), T.int64(1024), T.int64(512)), scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(16), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding(T.int64(2), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with T.block("NT_matmul_init"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(64), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = T.axis.spatial(T.int64(1024), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            T.reads()
                                            T.writes(NT_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2])
                                            NT_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2] = T.float32(0.0)
                                for ax3_0 in range(T.int64(32)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("lv87_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(64), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(512), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv87[v1, v2])
                                                        T.writes(lv87_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        lv87_reindex_shared[v0, v1, v2] = lv87[v1, v2]
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("param_0_reindex_pad_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(1024), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(512), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(param_0[v1, v2])
                                                        T.writes(param_0_reindex_pad_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        param_0_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v1 < T.int64(1000), param_0[v1, v2], T.float32(0.0))
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with T.block("NT_matmul_update"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial(T.int64(64), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(1024), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(512), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(NT_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2], lv87_reindex_shared[T.int64(0), v1, v3], param_0_reindex_pad_shared[T.int64(0), v2, v3])
                                                T.writes(NT_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2])
                                                NT_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2] = NT_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2] + lv87_reindex_shared[T.int64(0), v1, v3] * param_0_reindex_pad_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("NT_matmul_intermediate_reindex_pad_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(T.int64(64), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(1024), ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            T.where(ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1 < T.int64(1000))
                                            T.reads(NT_matmul_intermediate_reindex_pad_local[v0, v1, v2], param_1[v2])
                                            T.writes(T_add_intermediate[v1, v2])
                                            T_add_intermediate[v1, v2] = NT_matmul_intermediate_reindex_pad_local[v0, v1, v2] + param_1[v2]

    @T.prim_func
    def fused_add1_relu2(lv28_0: T.Buffer((T.int64(64), T.int64(128), T.int64(28), T.int64(28)), "float32"), lv31_0: T.Buffer((T.int64(64), T.int64(128), T.int64(28), T.int64(28)), "float32"), compute_intermediate: T.Buffer((T.int64(64), T.int64(128), T.int64(28), T.int64(28)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(6272), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v0 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(100352))
                    v1 = T.axis.spatial(T.int64(128), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(100352) // T.int64(784))
                    v2 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(784) // T.int64(28))
                    v3 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(28))
                    T.reads(lv28_0[v0, v1, v2, v3], lv31_0[v0, v1, v2, v3])
                    T.writes(compute_intermediate[v0, v1, v2, v3])
                    compute_intermediate[v0, v1, v2, v3] = T.max(lv28_0[v0, v1, v2, v3] + lv31_0[v0, v1, v2, v3], T.float32(0.0))

    @T.prim_func
    def fused_add2_relu3(lv49_0: T.Buffer((T.int64(64), T.int64(256), T.int64(14), T.int64(14)), "float32"), lv52_0: T.Buffer((T.int64(64), T.int64(256), T.int64(14), T.int64(14)), "float32"), compute_intermediate: T.Buffer((T.int64(64), T.int64(256), T.int64(14), T.int64(14)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(3136), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v0 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(50176))
                    v1 = T.axis.spatial(T.int64(256), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(50176) // T.int64(196))
                    v2 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(196) // T.int64(14))
                    v3 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(14))
                    T.reads(lv49_0[v0, v1, v2, v3], lv52_0[v0, v1, v2, v3])
                    T.writes(compute_intermediate[v0, v1, v2, v3])
                    compute_intermediate[v0, v1, v2, v3] = T.max(lv49_0[v0, v1, v2, v3] + lv52_0[v0, v1, v2, v3], T.float32(0.0))

    @T.prim_func
    def fused_add3_relu4(lv70_0: T.Buffer((T.int64(64), T.int64(512), T.int64(7), T.int64(7)), "float32"), lv73_0: T.Buffer((T.int64(64), T.int64(512), T.int64(7), T.int64(7)), "float32"), compute_intermediate: T.Buffer((T.int64(64), T.int64(512), T.int64(7), T.int64(7)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1568), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v0 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(25088))
                    v1 = T.axis.spatial(T.int64(512), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(25088) // T.int64(49))
                    v2 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(49) // T.int64(7))
                    v3 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(7))
                    T.reads(lv70_0[v0, v1, v2, v3], lv73_0[v0, v1, v2, v3])
                    T.writes(compute_intermediate[v0, v1, v2, v3])
                    compute_intermediate[v0, v1, v2, v3] = T.max(lv70_0[v0, v1, v2, v3] + lv73_0[v0, v1, v2, v3], T.float32(0.0))

    @T.prim_func
    def fused_add_relu1(lv10_0: T.Buffer((T.int64(64), T.int64(64), T.int64(56), T.int64(56)), "float32"), lv4: T.Buffer((T.int64(64), T.int64(64), T.int64(56), T.int64(56)), "float32"), compute_intermediate: T.Buffer((T.int64(64), T.int64(64), T.int64(56), T.int64(56)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(12544), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v0 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(200704))
                    v1 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(200704) // T.int64(3136))
                    v2 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(3136) // T.int64(56))
                    v3 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(56))
                    T.reads(lv10_0[v0, v1, v2, v3], lv4[v0, v1, v2, v3])
                    T.writes(compute_intermediate[v0, v1, v2, v3])
                    compute_intermediate[v0, v1, v2, v3] = T.max(lv10_0[v0, v1, v2, v3] + lv4[v0, v1, v2, v3], T.float32(0.0))

    @T.prim_func
    def fused_relu(lv1_0: T.Buffer((T.int64(64), T.int64(64), T.int64(112), T.int64(112)), "float32"), compute_intermediate: T.Buffer((T.int64(64), T.int64(64), T.int64(112), T.int64(112)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(50176), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v0 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(802816))
                    v1 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(802816) // T.int64(12544))
                    v2 = T.axis.spatial(T.int64(112), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(12544) // T.int64(112))
                    v3 = T.axis.spatial(T.int64(112), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(112))
                    T.reads(lv1_0[v0, v1, v2, v3])
                    T.writes(compute_intermediate[v0, v1, v2, v3])
                    compute_intermediate[v0, v1, v2, v3] = T.max(lv1_0[v0, v1, v2, v3], T.float32(0.0))

    @T.prim_func
    def fused_relu1(lv6_0: T.Buffer((T.int64(64), T.int64(64), T.int64(56), T.int64(56)), "float32"), compute_intermediate: T.Buffer((T.int64(64), T.int64(64), T.int64(56), T.int64(56)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(12544), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v0 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(200704))
                    v1 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(200704) // T.int64(3136))
                    v2 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(3136) // T.int64(56))
                    v3 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(56))
                    T.reads(lv6_0[v0, v1, v2, v3])
                    T.writes(compute_intermediate[v0, v1, v2, v3])
                    compute_intermediate[v0, v1, v2, v3] = T.max(lv6_0[v0, v1, v2, v3], T.float32(0.0))

    @T.prim_func
    def fused_relu2(lv24_0: T.Buffer((T.int64(64), T.int64(128), T.int64(28), T.int64(28)), "float32"), compute_intermediate: T.Buffer((T.int64(64), T.int64(128), T.int64(28), T.int64(28)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(6272), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v0 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(100352))
                    v1 = T.axis.spatial(T.int64(128), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(100352) // T.int64(784))
                    v2 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(784) // T.int64(28))
                    v3 = T.axis.spatial(T.int64(28), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(28))
                    T.reads(lv24_0[v0, v1, v2, v3])
                    T.writes(compute_intermediate[v0, v1, v2, v3])
                    compute_intermediate[v0, v1, v2, v3] = T.max(lv24_0[v0, v1, v2, v3], T.float32(0.0))

    @T.prim_func
    def fused_relu3(lv45_0: T.Buffer((T.int64(64), T.int64(256), T.int64(14), T.int64(14)), "float32"), compute_intermediate: T.Buffer((T.int64(64), T.int64(256), T.int64(14), T.int64(14)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(3136), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v0 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(50176))
                    v1 = T.axis.spatial(T.int64(256), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(50176) // T.int64(196))
                    v2 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(196) // T.int64(14))
                    v3 = T.axis.spatial(T.int64(14), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(14))
                    T.reads(lv45_0[v0, v1, v2, v3])
                    T.writes(compute_intermediate[v0, v1, v2, v3])
                    compute_intermediate[v0, v1, v2, v3] = T.max(lv45_0[v0, v1, v2, v3], T.float32(0.0))

    @T.prim_func
    def fused_relu4(lv66_0: T.Buffer((T.int64(64), T.int64(512), T.int64(7), T.int64(7)), "float32"), compute_intermediate: T.Buffer((T.int64(64), T.int64(512), T.int64(7), T.int64(7)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(1568), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v0 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(25088))
                    v1 = T.axis.spatial(T.int64(512), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(25088) // T.int64(49))
                    v2 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(49) // T.int64(7))
                    v3 = T.axis.spatial(T.int64(7), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(7))
                    T.reads(lv66_0[v0, v1, v2, v3])
                    T.writes(compute_intermediate[v0, v1, v2, v3])
                    compute_intermediate[v0, v1, v2, v3] = T.max(lv66_0[v0, v1, v2, v3], T.float32(0.0))

    @T.prim_func
    def max_pool2d(lv3: T.Buffer((T.int64(64), T.int64(64), T.int64(112), T.int64(112)), "float32"), pool_max: T.Buffer((T.int64(64), T.int64(64), T.int64(56), T.int64(56)), "float32")):
        T.func_attr({"op_pattern": 4, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_ax2_ax3_fused_0 in T.thread_binding(T.int64(12544), thread="blockIdx.x"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("pool_max_init"):
                    v0 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(200704))
                    v1 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(200704) // T.int64(3136))
                    v2 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(3136) // T.int64(56))
                    v3 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(56))
                    T.reads()
                    T.writes(pool_max[v0, v1, v2, v3])
                    T.block_attr({"schedule_rule": "meta_schedule.pool_max"})
                    pool_max[v0, v1, v2, v3] = T.float32(-340282346638528859811704183484516925440.0)
                for ax4, ax5 in T.grid(T.int64(3), T.int64(3)):
                    with T.block("pool_max_update"):
                        v0 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) // T.int64(200704))
                        v1 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(200704) // T.int64(3136))
                        v2 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(3136) // T.int64(56))
                        v3 = T.axis.spatial(T.int64(56), (ax0_ax1_ax2_ax3_fused_0 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_1) % T.int64(56))
                        v4, v5 = T.axis.remap("RR", [ax4, ax5])
                        T.reads(pool_max[v0, v1, v2, v3], lv3[v0, v1, v2 * T.int64(2) + v4 - T.int64(1), v3 * T.int64(2) + v5 - T.int64(1)])
                        T.writes(pool_max[v0, v1, v2, v3])
                        T.block_attr({"schedule_rule": "meta_schedule.pool_max"})
                        pool_max[v0, v1, v2, v3] = T.max(pool_max[v0, v1, v2, v3], T.if_then_else(T.int64(1) <= v2 * T.int64(2) + v4 and v2 * T.int64(2) + v4 < T.int64(113) and T.int64(1) <= v3 * T.int64(2) + v5 and v3 * T.int64(2) + v5 < T.int64(113), lv3[v0, v1, v2 * T.int64(2) + v4 - T.int64(1), v3 * T.int64(2) + v5 - T.int64(1)], T.float32(-340282346638528859811704183484516925440.0)))

    @T.prim_func
    def reshape(lv86: T.Buffer((T.int64(64), T.int64(512), T.int64(1), T.int64(1)), "float32"), T_reshape: T.Buffer((T.int64(64), T.int64(512)), "float32")):
        T.func_attr({"op_pattern": 2, "tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(32), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_reshape"):
                    v0 = T.axis.spatial(T.int64(64), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(512))
                    v1 = T.axis.spatial(T.int64(512), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(512))
                    T.reads(lv86[v0, v1, T.int64(0), T.int64(0)])
                    T.writes(T_reshape[v0, v1])
                    T_reshape[v0, v1] = lv86[v0, v1, T.int64(0), T.int64(0)]

    @R.function
    def main(inp_0: R.Tensor((64, 3, 224, 224), dtype="float32")) -> R.Tensor((64, 1000), dtype="float32"):
        R.func_attr({"relax.force_pure": True})
        cls = Module
        storage: R.Object = R.vm.alloc_storage(R.shape([205520896]), R.prim_value(0), R.dtype("uint8"), R.str("global"))
        alloc: R.Tensor((64, 64, 112, 112), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([64, 64, 112, 112]), R.dtype("float32"))
        cls.conv2d(inp_0, metadata["relax.expr.Constant"][0], alloc)
        storage1: R.Object = R.vm.alloc_storage(R.shape([205520896]), R.prim_value(0), R.dtype("uint8"), R.str("global"))
        alloc1: R.Tensor((64, 64, 112, 112), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([64, 64, 112, 112]), R.dtype("float32"))
        storage2: R.Object = R.vm.alloc_storage(R.shape([2048]), R.prim_value(0), R.dtype("uint8"), R.str("global"))
        alloc2: R.Tensor((64,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([64]), R.dtype("float32"))
        storage3: R.Object = R.vm.alloc_storage(R.shape([2048]), R.prim_value(0), R.dtype("uint8"), R.str("global"))
        alloc3: R.Tensor((64,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([64]), R.dtype("float32"))
        cls.batch_norm(alloc, metadata["relax.expr.Constant"][1], metadata["relax.expr.Constant"][2], metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4], alloc1, alloc2, alloc3)
        R.vm.kill_object(alloc)
        lv1: R.Tuple(R.Tensor((64, 64, 112, 112), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = alloc1, alloc2, alloc3
        R.vm.kill_object(alloc2)
        R.vm.kill_object(alloc3)
        alloc4: R.Tensor((64, 64, 112, 112), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([64, 64, 112, 112]), R.dtype("float32"))
        cls.fused_relu(alloc1, alloc4)
        R.vm.kill_object(alloc1)
        alloc5: R.Tensor((64, 64, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([64, 64, 56, 56]), R.dtype("float32"))
        cls.max_pool2d(alloc4, alloc5)
        R.vm.kill_object(alloc4)
        alloc6: R.Tensor((64, 64, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([64, 64, 56, 56]), R.dtype("float32"))
        cls.conv2d1(alloc5, metadata["relax.expr.Constant"][5], alloc6)
        storage4: R.Object = R.vm.alloc_storage(R.shape([51380224]), R.prim_value(0), R.dtype("uint8"), R.str("global"))
        alloc7: R.Tensor((64, 64, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([64, 64, 56, 56]), R.dtype("float32"))
        alloc8: R.Tensor((64,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([64]), R.dtype("float32"))
        alloc9: R.Tensor((64,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([64]), R.dtype("float32"))
        cls.batch_norm1(alloc6, metadata["relax.expr.Constant"][6], metadata["relax.expr.Constant"][7], metadata["relax.expr.Constant"][8], metadata["relax.expr.Constant"][9], alloc7, alloc8, alloc9)
        R.vm.kill_object(alloc6)
        lv6: R.Tuple(R.Tensor((64, 64, 56, 56), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = alloc7, alloc8, alloc9
        R.vm.kill_object(alloc8)
        R.vm.kill_object(alloc9)
        alloc10: R.Tensor((64, 64, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([64, 64, 56, 56]), R.dtype("float32"))
        cls.fused_relu1(alloc7, alloc10)
        R.vm.kill_object(alloc7)
        alloc11: R.Tensor((64, 64, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([64, 64, 56, 56]), R.dtype("float32"))
        cls.conv2d1(alloc10, metadata["relax.expr.Constant"][10], alloc11)
        R.vm.kill_object(alloc10)
        alloc12: R.Tensor((64, 64, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([64, 64, 56, 56]), R.dtype("float32"))
        alloc13: R.Tensor((64,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([64]), R.dtype("float32"))
        alloc14: R.Tensor((64,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([64]), R.dtype("float32"))
        cls.batch_norm1(alloc11, metadata["relax.expr.Constant"][11], metadata["relax.expr.Constant"][12], metadata["relax.expr.Constant"][13], metadata["relax.expr.Constant"][14], alloc12, alloc13, alloc14)
        R.vm.kill_object(alloc11)
        lv10: R.Tuple(R.Tensor((64, 64, 56, 56), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = alloc12, alloc13, alloc14
        R.vm.kill_object(alloc13)
        R.vm.kill_object(alloc14)
        alloc15: R.Tensor((64, 64, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([64, 64, 56, 56]), R.dtype("float32"))
        cls.fused_add_relu1(alloc12, alloc5, alloc15)
        R.vm.kill_object(alloc5)
        R.vm.kill_object(alloc12)
        alloc16: R.Tensor((64, 64, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([64, 64, 56, 56]), R.dtype("float32"))
        cls.conv2d1(alloc15, metadata["relax.expr.Constant"][15], alloc16)
        alloc17: R.Tensor((64, 64, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([64, 64, 56, 56]), R.dtype("float32"))
        alloc18: R.Tensor((64,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([64]), R.dtype("float32"))
        alloc19: R.Tensor((64,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([64]), R.dtype("float32"))
        cls.batch_norm1(alloc16, metadata["relax.expr.Constant"][16], metadata["relax.expr.Constant"][17], metadata["relax.expr.Constant"][18], metadata["relax.expr.Constant"][19], alloc17, alloc18, alloc19)
        R.vm.kill_object(alloc16)
        lv15: R.Tuple(R.Tensor((64, 64, 56, 56), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = alloc17, alloc18, alloc19
        R.vm.kill_object(alloc18)
        R.vm.kill_object(alloc19)
        alloc20: R.Tensor((64, 64, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([64, 64, 56, 56]), R.dtype("float32"))
        cls.fused_relu1(alloc17, alloc20)
        R.vm.kill_object(alloc17)
        alloc21: R.Tensor((64, 64, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([64, 64, 56, 56]), R.dtype("float32"))
        cls.conv2d1(alloc20, metadata["relax.expr.Constant"][20], alloc21)
        R.vm.kill_object(alloc20)
        alloc22: R.Tensor((64, 64, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([64, 64, 56, 56]), R.dtype("float32"))
        alloc23: R.Tensor((64,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([64]), R.dtype("float32"))
        alloc24: R.Tensor((64,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([64]), R.dtype("float32"))
        cls.batch_norm1(alloc21, metadata["relax.expr.Constant"][21], metadata["relax.expr.Constant"][22], metadata["relax.expr.Constant"][23], metadata["relax.expr.Constant"][24], alloc22, alloc23, alloc24)
        R.vm.kill_object(alloc21)
        lv19: R.Tuple(R.Tensor((64, 64, 56, 56), dtype="float32"), R.Tensor((64,), dtype="float32"), R.Tensor((64,), dtype="float32")) = alloc22, alloc23, alloc24
        R.vm.kill_object(alloc23)
        R.vm.kill_object(alloc24)
        alloc25: R.Tensor((64, 64, 56, 56), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([64, 64, 56, 56]), R.dtype("float32"))
        cls.fused_add_relu1(alloc22, alloc15, alloc25)
        R.vm.kill_object(alloc15)
        R.vm.kill_object(alloc22)
        alloc26: R.Tensor((64, 128, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([64, 128, 28, 28]), R.dtype("float32"))
        cls.conv2d2(alloc25, metadata["relax.expr.Constant"][25], alloc26)
        alloc27: R.Tensor((64, 128, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([64, 128, 28, 28]), R.dtype("float32"))
        alloc28: R.Tensor((128,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([128]), R.dtype("float32"))
        alloc29: R.Tensor((128,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([128]), R.dtype("float32"))
        cls.batch_norm2(alloc26, metadata["relax.expr.Constant"][26], metadata["relax.expr.Constant"][27], metadata["relax.expr.Constant"][28], metadata["relax.expr.Constant"][29], alloc27, alloc28, alloc29)
        R.vm.kill_object(alloc26)
        lv24: R.Tuple(R.Tensor((64, 128, 28, 28), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = alloc27, alloc28, alloc29
        R.vm.kill_object(alloc28)
        R.vm.kill_object(alloc29)
        alloc30: R.Tensor((64, 128, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([64, 128, 28, 28]), R.dtype("float32"))
        cls.fused_relu2(alloc27, alloc30)
        R.vm.kill_object(alloc27)
        alloc31: R.Tensor((64, 128, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([64, 128, 28, 28]), R.dtype("float32"))
        cls.conv2d3(alloc30, metadata["relax.expr.Constant"][30], alloc31)
        R.vm.kill_object(alloc30)
        alloc32: R.Tensor((64, 128, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([64, 128, 28, 28]), R.dtype("float32"))
        alloc33: R.Tensor((128,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([128]), R.dtype("float32"))
        alloc34: R.Tensor((128,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([128]), R.dtype("float32"))
        cls.batch_norm2(alloc31, metadata["relax.expr.Constant"][31], metadata["relax.expr.Constant"][32], metadata["relax.expr.Constant"][33], metadata["relax.expr.Constant"][34], alloc32, alloc33, alloc34)
        R.vm.kill_object(alloc31)
        lv28: R.Tuple(R.Tensor((64, 128, 28, 28), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = alloc32, alloc33, alloc34
        R.vm.kill_object(alloc33)
        R.vm.kill_object(alloc34)
        alloc35: R.Tensor((64, 128, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([64, 128, 28, 28]), R.dtype("float32"))
        cls.conv2d4(alloc25, metadata["relax.expr.Constant"][35], alloc35)
        R.vm.kill_object(alloc25)
        alloc36: R.Tensor((64, 128, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([64, 128, 28, 28]), R.dtype("float32"))
        alloc37: R.Tensor((128,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([128]), R.dtype("float32"))
        alloc38: R.Tensor((128,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([128]), R.dtype("float32"))
        cls.batch_norm2(alloc35, metadata["relax.expr.Constant"][36], metadata["relax.expr.Constant"][37], metadata["relax.expr.Constant"][38], metadata["relax.expr.Constant"][39], alloc36, alloc37, alloc38)
        R.vm.kill_object(alloc35)
        lv31: R.Tuple(R.Tensor((64, 128, 28, 28), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = alloc36, alloc37, alloc38
        R.vm.kill_object(alloc37)
        R.vm.kill_object(alloc38)
        alloc39: R.Tensor((64, 128, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([64, 128, 28, 28]), R.dtype("float32"))
        cls.fused_add1_relu2(alloc32, alloc36, alloc39)
        R.vm.kill_object(alloc32)
        R.vm.kill_object(alloc36)
        alloc40: R.Tensor((64, 128, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([64, 128, 28, 28]), R.dtype("float32"))
        cls.conv2d3(alloc39, metadata["relax.expr.Constant"][40], alloc40)
        alloc41: R.Tensor((64, 128, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([64, 128, 28, 28]), R.dtype("float32"))
        alloc42: R.Tensor((128,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([128]), R.dtype("float32"))
        alloc43: R.Tensor((128,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([128]), R.dtype("float32"))
        cls.batch_norm2(alloc40, metadata["relax.expr.Constant"][41], metadata["relax.expr.Constant"][42], metadata["relax.expr.Constant"][43], metadata["relax.expr.Constant"][44], alloc41, alloc42, alloc43)
        R.vm.kill_object(alloc40)
        lv36: R.Tuple(R.Tensor((64, 128, 28, 28), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = alloc41, alloc42, alloc43
        R.vm.kill_object(alloc42)
        R.vm.kill_object(alloc43)
        alloc44: R.Tensor((64, 128, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([64, 128, 28, 28]), R.dtype("float32"))
        cls.fused_relu2(alloc41, alloc44)
        R.vm.kill_object(alloc41)
        alloc45: R.Tensor((64, 128, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([64, 128, 28, 28]), R.dtype("float32"))
        cls.conv2d3(alloc44, metadata["relax.expr.Constant"][45], alloc45)
        R.vm.kill_object(alloc44)
        alloc46: R.Tensor((64, 128, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([64, 128, 28, 28]), R.dtype("float32"))
        alloc47: R.Tensor((128,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([128]), R.dtype("float32"))
        alloc48: R.Tensor((128,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([128]), R.dtype("float32"))
        cls.batch_norm2(alloc45, metadata["relax.expr.Constant"][46], metadata["relax.expr.Constant"][47], metadata["relax.expr.Constant"][48], metadata["relax.expr.Constant"][49], alloc46, alloc47, alloc48)
        R.vm.kill_object(alloc45)
        lv40: R.Tuple(R.Tensor((64, 128, 28, 28), dtype="float32"), R.Tensor((128,), dtype="float32"), R.Tensor((128,), dtype="float32")) = alloc46, alloc47, alloc48
        R.vm.kill_object(alloc47)
        R.vm.kill_object(alloc48)
        alloc49: R.Tensor((64, 128, 28, 28), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([64, 128, 28, 28]), R.dtype("float32"))
        cls.fused_add1_relu2(alloc46, alloc39, alloc49)
        R.vm.kill_object(alloc39)
        R.vm.kill_object(alloc46)
        alloc50: R.Tensor((64, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([64, 256, 14, 14]), R.dtype("float32"))
        cls.conv2d5(alloc49, metadata["relax.expr.Constant"][50], alloc50)
        alloc51: R.Tensor((64, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([64, 256, 14, 14]), R.dtype("float32"))
        alloc52: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        alloc53: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        cls.batch_norm3(alloc50, metadata["relax.expr.Constant"][51], metadata["relax.expr.Constant"][52], metadata["relax.expr.Constant"][53], metadata["relax.expr.Constant"][54], alloc51, alloc52, alloc53)
        R.vm.kill_object(alloc50)
        lv45: R.Tuple(R.Tensor((64, 256, 14, 14), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = alloc51, alloc52, alloc53
        R.vm.kill_object(alloc52)
        R.vm.kill_object(alloc53)
        alloc54: R.Tensor((64, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([64, 256, 14, 14]), R.dtype("float32"))
        cls.fused_relu3(alloc51, alloc54)
        R.vm.kill_object(alloc51)
        alloc55: R.Tensor((64, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([64, 256, 14, 14]), R.dtype("float32"))
        cls.conv2d6(alloc54, metadata["relax.expr.Constant"][55], alloc55)
        R.vm.kill_object(alloc54)
        alloc56: R.Tensor((64, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([64, 256, 14, 14]), R.dtype("float32"))
        alloc57: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        alloc58: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        cls.batch_norm3(alloc55, metadata["relax.expr.Constant"][56], metadata["relax.expr.Constant"][57], metadata["relax.expr.Constant"][58], metadata["relax.expr.Constant"][59], alloc56, alloc57, alloc58)
        R.vm.kill_object(alloc55)
        lv49: R.Tuple(R.Tensor((64, 256, 14, 14), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = alloc56, alloc57, alloc58
        R.vm.kill_object(alloc57)
        R.vm.kill_object(alloc58)
        alloc59: R.Tensor((64, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([64, 256, 14, 14]), R.dtype("float32"))
        cls.conv2d7(alloc49, metadata["relax.expr.Constant"][60], alloc59)
        R.vm.kill_object(alloc49)
        alloc60: R.Tensor((64, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([64, 256, 14, 14]), R.dtype("float32"))
        alloc61: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        alloc62: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        cls.batch_norm3(alloc59, metadata["relax.expr.Constant"][61], metadata["relax.expr.Constant"][62], metadata["relax.expr.Constant"][63], metadata["relax.expr.Constant"][64], alloc60, alloc61, alloc62)
        R.vm.kill_object(alloc59)
        lv52: R.Tuple(R.Tensor((64, 256, 14, 14), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = alloc60, alloc61, alloc62
        R.vm.kill_object(alloc61)
        R.vm.kill_object(alloc62)
        alloc63: R.Tensor((64, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([64, 256, 14, 14]), R.dtype("float32"))
        R.vm.kill_object(storage)
        cls.fused_add2_relu3(alloc56, alloc60, alloc63)
        R.vm.kill_object(alloc56)
        R.vm.kill_object(alloc60)
        alloc64: R.Tensor((64, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([64, 256, 14, 14]), R.dtype("float32"))
        cls.conv2d6(alloc63, metadata["relax.expr.Constant"][65], alloc64)
        alloc65: R.Tensor((64, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([64, 256, 14, 14]), R.dtype("float32"))
        alloc66: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        alloc67: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        cls.batch_norm3(alloc64, metadata["relax.expr.Constant"][66], metadata["relax.expr.Constant"][67], metadata["relax.expr.Constant"][68], metadata["relax.expr.Constant"][69], alloc65, alloc66, alloc67)
        R.vm.kill_object(alloc64)
        lv57: R.Tuple(R.Tensor((64, 256, 14, 14), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = alloc65, alloc66, alloc67
        R.vm.kill_object(alloc66)
        R.vm.kill_object(alloc67)
        alloc68: R.Tensor((64, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([64, 256, 14, 14]), R.dtype("float32"))
        cls.fused_relu3(alloc65, alloc68)
        R.vm.kill_object(alloc65)
        alloc69: R.Tensor((64, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([64, 256, 14, 14]), R.dtype("float32"))
        cls.conv2d6(alloc68, metadata["relax.expr.Constant"][70], alloc69)
        R.vm.kill_object(alloc68)
        alloc70: R.Tensor((64, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([64, 256, 14, 14]), R.dtype("float32"))
        alloc71: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        alloc72: R.Tensor((256,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([256]), R.dtype("float32"))
        cls.batch_norm3(alloc69, metadata["relax.expr.Constant"][71], metadata["relax.expr.Constant"][72], metadata["relax.expr.Constant"][73], metadata["relax.expr.Constant"][74], alloc70, alloc71, alloc72)
        R.vm.kill_object(alloc69)
        lv61: R.Tuple(R.Tensor((64, 256, 14, 14), dtype="float32"), R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="float32")) = alloc70, alloc71, alloc72
        R.vm.kill_object(alloc71)
        R.vm.kill_object(alloc72)
        alloc73: R.Tensor((64, 256, 14, 14), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([64, 256, 14, 14]), R.dtype("float32"))
        R.vm.kill_object(storage1)
        cls.fused_add2_relu3(alloc70, alloc63, alloc73)
        R.vm.kill_object(alloc63)
        R.vm.kill_object(alloc70)
        alloc74: R.Tensor((64, 512, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([64, 512, 7, 7]), R.dtype("float32"))
        cls.conv2d8(alloc73, metadata["relax.expr.Constant"][75], alloc74)
        storage5: R.Object = R.vm.alloc_storage(R.shape([6422528]), R.prim_value(0), R.dtype("uint8"), R.str("global"))
        alloc75: R.Tensor((64, 512, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage5, R.prim_value(0), R.shape([64, 512, 7, 7]), R.dtype("float32"))
        alloc76: R.Tensor((512,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([512]), R.dtype("float32"))
        alloc77: R.Tensor((512,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([512]), R.dtype("float32"))
        cls.batch_norm4(alloc74, metadata["relax.expr.Constant"][76], metadata["relax.expr.Constant"][77], metadata["relax.expr.Constant"][78], metadata["relax.expr.Constant"][79], alloc75, alloc76, alloc77)
        R.vm.kill_object(alloc74)
        lv66: R.Tuple(R.Tensor((64, 512, 7, 7), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = alloc75, alloc76, alloc77
        R.vm.kill_object(alloc76)
        R.vm.kill_object(alloc77)
        alloc78: R.Tensor((64, 512, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([64, 512, 7, 7]), R.dtype("float32"))
        cls.fused_relu4(alloc75, alloc78)
        R.vm.kill_object(alloc75)
        alloc79: R.Tensor((64, 512, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage5, R.prim_value(0), R.shape([64, 512, 7, 7]), R.dtype("float32"))
        cls.conv2d9(alloc78, metadata["relax.expr.Constant"][80], alloc79)
        R.vm.kill_object(alloc78)
        alloc80: R.Tensor((64, 512, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([64, 512, 7, 7]), R.dtype("float32"))
        alloc81: R.Tensor((512,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([512]), R.dtype("float32"))
        alloc82: R.Tensor((512,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([512]), R.dtype("float32"))
        cls.batch_norm4(alloc79, metadata["relax.expr.Constant"][81], metadata["relax.expr.Constant"][82], metadata["relax.expr.Constant"][83], metadata["relax.expr.Constant"][84], alloc80, alloc81, alloc82)
        R.vm.kill_object(alloc79)
        lv70: R.Tuple(R.Tensor((64, 512, 7, 7), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = alloc80, alloc81, alloc82
        R.vm.kill_object(alloc81)
        R.vm.kill_object(alloc82)
        alloc83: R.Tensor((64, 512, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage5, R.prim_value(0), R.shape([64, 512, 7, 7]), R.dtype("float32"))
        cls.conv2d10(alloc73, metadata["relax.expr.Constant"][85], alloc83)
        R.vm.kill_object(alloc73)
        storage6: R.Object = R.vm.alloc_storage(R.shape([6422528]), R.prim_value(0), R.dtype("uint8"), R.str("global"))
        alloc84: R.Tensor((64, 512, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage6, R.prim_value(0), R.shape([64, 512, 7, 7]), R.dtype("float32"))
        alloc85: R.Tensor((512,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([512]), R.dtype("float32"))
        alloc86: R.Tensor((512,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([512]), R.dtype("float32"))
        cls.batch_norm4(alloc83, metadata["relax.expr.Constant"][86], metadata["relax.expr.Constant"][87], metadata["relax.expr.Constant"][88], metadata["relax.expr.Constant"][89], alloc84, alloc85, alloc86)
        R.vm.kill_object(alloc83)
        lv73: R.Tuple(R.Tensor((64, 512, 7, 7), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = alloc84, alloc85, alloc86
        R.vm.kill_object(alloc85)
        R.vm.kill_object(alloc86)
        alloc87: R.Tensor((64, 512, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage5, R.prim_value(0), R.shape([64, 512, 7, 7]), R.dtype("float32"))
        R.vm.kill_object(storage5)
        cls.fused_add3_relu4(alloc80, alloc84, alloc87)
        R.vm.kill_object(alloc80)
        R.vm.kill_object(alloc84)
        alloc88: R.Tensor((64, 512, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage6, R.prim_value(0), R.shape([64, 512, 7, 7]), R.dtype("float32"))
        cls.conv2d9(alloc87, metadata["relax.expr.Constant"][90], alloc88)
        alloc89: R.Tensor((64, 512, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([64, 512, 7, 7]), R.dtype("float32"))
        alloc90: R.Tensor((512,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([512]), R.dtype("float32"))
        alloc91: R.Tensor((512,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([512]), R.dtype("float32"))
        cls.batch_norm4(alloc88, metadata["relax.expr.Constant"][91], metadata["relax.expr.Constant"][92], metadata["relax.expr.Constant"][93], metadata["relax.expr.Constant"][94], alloc89, alloc90, alloc91)
        R.vm.kill_object(alloc88)
        lv78: R.Tuple(R.Tensor((64, 512, 7, 7), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = alloc89, alloc90, alloc91
        R.vm.kill_object(alloc90)
        R.vm.kill_object(alloc91)
        alloc92: R.Tensor((64, 512, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage6, R.prim_value(0), R.shape([64, 512, 7, 7]), R.dtype("float32"))
        cls.fused_relu4(alloc89, alloc92)
        R.vm.kill_object(alloc89)
        alloc93: R.Tensor((64, 512, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([64, 512, 7, 7]), R.dtype("float32"))
        cls.conv2d9(alloc92, metadata["relax.expr.Constant"][95], alloc93)
        R.vm.kill_object(alloc92)
        alloc94: R.Tensor((64, 512, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage6, R.prim_value(0), R.shape([64, 512, 7, 7]), R.dtype("float32"))
        R.vm.kill_object(storage6)
        alloc95: R.Tensor((512,), dtype="float32") = R.vm.alloc_tensor(storage3, R.prim_value(0), R.shape([512]), R.dtype("float32"))
        R.vm.kill_object(storage3)
        alloc96: R.Tensor((512,), dtype="float32") = R.vm.alloc_tensor(storage2, R.prim_value(0), R.shape([512]), R.dtype("float32"))
        R.vm.kill_object(storage2)
        cls.batch_norm4(alloc93, metadata["relax.expr.Constant"][96], metadata["relax.expr.Constant"][97], metadata["relax.expr.Constant"][98], metadata["relax.expr.Constant"][99], alloc94, alloc95, alloc96)
        R.vm.kill_object(alloc93)
        lv82: R.Tuple(R.Tensor((64, 512, 7, 7), dtype="float32"), R.Tensor((512,), dtype="float32"), R.Tensor((512,), dtype="float32")) = alloc94, alloc95, alloc96
        R.vm.kill_object(alloc95)
        R.vm.kill_object(alloc96)
        alloc97: R.Tensor((64, 512, 7, 7), dtype="float32") = R.vm.alloc_tensor(storage4, R.prim_value(0), R.shape([64, 512, 7, 7]), R.dtype("float32"))
        R.vm.kill_object(storage4)
        cls.fused_add3_relu4(alloc94, alloc87, alloc97)
        R.vm.kill_object(alloc87)
        R.vm.kill_object(alloc94)
        storage7: R.Object = R.vm.alloc_storage(R.shape([131072]), R.prim_value(0), R.dtype("uint8"), R.str("global"))
        alloc98: R.Tensor((64, 512, 1, 1), dtype="float32") = R.vm.alloc_tensor(storage7, R.prim_value(0), R.shape([64, 512, 1, 1]), R.dtype("float32"))
        R.vm.kill_object(storage7)
        cls.adaptive_avg_pool2d(alloc97, alloc98)
        R.vm.kill_object(alloc97)
        lv87: R.Tensor((64, 512), dtype="float32") = R.call_packed("vm.builtin.reshape", alloc98, R.shape([64, 512]), sinfo_args=(R.Tensor((64, 512), dtype="float32"),))
        R.vm.kill_object(alloc98)
        storage_1: R.Object = R.vm.alloc_storage(R.shape([256000]), R.prim_value(0), R.dtype("uint8"), R.str("global"))
        alloc99: R.Tensor((64, 1000), dtype="float32") = R.vm.alloc_tensor(storage_1, R.prim_value(0), R.shape([64, 1000]), R.dtype("float32"))
        R.vm.kill_object(storage_1)
        cls.fused_NT_matmul_add4(lv87, metadata["relax.expr.Constant"][100], metadata["relax.expr.Constant"][101], alloc99)
        R.vm.kill_object(lv87)
        return alloc99

# Metadata omitted. Use show_meta=True in script() method to show it.