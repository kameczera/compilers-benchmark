# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def add(lv8: T.Buffer((T.int64(1), T.int64(128), T.int64(768)), "float32"), lv12: T.Buffer((T.int64(1), T.int64(128), T.int64(768)), "float32"), T_add: T.Buffer((T.int64(1), T.int64(128), T.int64(768)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(96), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add"):
                    v0 = T.axis.spatial(T.int64(128), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(768))
                    v1 = T.axis.spatial(T.int64(768), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(768))
                    T.reads(lv8[T.int64(0), v0, v1], lv12[T.int64(0), v0, v1])
                    T.writes(T_add[T.int64(0), v0, v1])
                    T_add[T.int64(0), v0, v1] = lv8[T.int64(0), v0, v1] + lv12[T.int64(0), v0, v1]

    @T.prim_func(private=True)
    def add1(lv32: T.Buffer((T.int64(1), T.int64(128), T.int64(768)), "float32"), B: T.Buffer((T.int64(768),), "float32"), T_add: T.Buffer((T.int64(1), T.int64(128), T.int64(768)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(96), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add"):
                    v0 = T.axis.spatial(T.int64(128), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(768))
                    v1 = T.axis.spatial(T.int64(768), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(768))
                    T.reads(lv32[T.int64(0), v0, v1], B[v1])
                    T.writes(T_add[T.int64(0), v0, v1])
                    T_add[T.int64(0), v0, v1] = lv32[T.int64(0), v0, v1] + B[v1]

    @T.prim_func(private=True)
    def add2(lv59: T.Buffer((T.int64(1), T.int64(128), T.int64(3072)), "float32"), B: T.Buffer((T.int64(3072),), "float32"), T_add: T.Buffer((T.int64(1), T.int64(128), T.int64(3072)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(384), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add"):
                    v0 = T.axis.spatial(T.int64(128), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(3072))
                    v1 = T.axis.spatial(T.int64(3072), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(3072))
                    T.reads(lv59[T.int64(0), v0, v1], B[v1])
                    T.writes(T_add[T.int64(0), v0, v1])
                    T_add[T.int64(0), v0, v1] = lv59[T.int64(0), v0, v1] + B[v1]

    @T.prim_func(private=True)
    def attention_bias(lv46: T.Buffer((T.int64(1), T.int64(128), T.int64(12), T.int64(64)), "float32"), lv47: T.Buffer((T.int64(1), T.int64(128), T.int64(12), T.int64(64)), "float32"), lv48: T.Buffer((T.int64(1), T.int64(128), T.int64(12), T.int64(64)), "float32"), lv30: T.Buffer((T.int64(1), T.int64(1), T.int64(128), T.int64(128)), "float32"), T_transpose: T.Buffer((T.int64(1), T.int64(128), T.int64(12), T.int64(64)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        T_batch_matmul_NT = T.alloc_buffer((T.int64(12), T.int64(128), T.int64(128)))
        T_softmax_maxelem = T.alloc_buffer((T.int64(12), T.int64(128)))
        T_softmax_expsum = T.alloc_buffer((T.int64(12), T.int64(128)))
        T_batch_matmul_NN = T.alloc_buffer((T.int64(12), T.int64(128), T.int64(64)))
        for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(192), thread="blockIdx.x"):
            for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_batch_matmul_NT_init"):
                    v0 = T.axis.spatial(T.int64(12), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) // T.int64(16384))
                    v1 = T.axis.spatial(T.int64(128), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(16384) // T.int64(128))
                    v2 = T.axis.spatial(T.int64(128), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(128))
                    T.reads()
                    T.writes(T_batch_matmul_NT[v0, v1, v2])
                    T_reshape = T.Buffer((T.int64(12), T.int64(128), T.int64(64)))
                    T.block_attr({"layout_free_placeholders": [T_reshape]})
                    T_batch_matmul_NT[v0, v1, v2] = T.float32(0.0)
                T_reshape = T.Buffer((T.int64(12), T.int64(128), T.int64(64)))
                for ax3 in range(T.int64(64)):
                    with T.block("T_batch_matmul_NT_update"):
                        v0 = T.axis.spatial(T.int64(12), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) // T.int64(16384))
                        v1 = T.axis.spatial(T.int64(128), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(16384) // T.int64(128))
                        v2 = T.axis.spatial(T.int64(128), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(128))
                        v3 = T.axis.reduce(T.int64(64), ax3)
                        T.reads(T_batch_matmul_NT[v0, v1, v2], lv46[T.int64(0), v1, v0, v3], lv47[T.int64(0), v2, v0, v3])
                        T.writes(T_batch_matmul_NT[v0, v1, v2])
                        T.block_attr({"layout_free_placeholders": [T_reshape]})
                        T_batch_matmul_NT[v0, v1, v2] = T_batch_matmul_NT[v0, v1, v2] + lv46[T.int64(0), v1, v0, v3] * lv47[T.int64(0), v2, v0, v3]
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(2), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_softmax_maxelem_init"):
                    v0 = T.axis.spatial(T.int64(12), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(128))
                    v1 = T.axis.spatial(T.int64(128), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(128))
                    T.where(ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 < T.int64(1536))
                    T.reads()
                    T.writes(T_softmax_maxelem[v0, v1])
                    T_softmax_maxelem[v0, v1] = T.float32(-340282346638528859811704183484516925440.0)
                for ax2 in range(T.int64(128)):
                    with T.block("T_softmax_maxelem_update"):
                        v0 = T.axis.spatial(T.int64(12), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(128))
                        v1 = T.axis.spatial(T.int64(128), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(128))
                        v2 = T.axis.reduce(T.int64(128), ax2)
                        T.where(ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 < T.int64(1536))
                        T.reads(T_softmax_maxelem[v0, v1], T_batch_matmul_NT[v0, v1, v2], lv30[T.int64(0), T.int64(0), v1, v2])
                        T.writes(T_softmax_maxelem[v0, v1])
                        T_softmax_maxelem[v0, v1] = T.max(T_softmax_maxelem[v0, v1], T_batch_matmul_NT[v0, v1, v2] / T.sqrt(T.float32(64.0)) + lv30[T.int64(0), T.int64(0), v1, v2])
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(2), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_softmax_expsum_init"):
                    v0 = T.axis.spatial(T.int64(12), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(128))
                    v1 = T.axis.spatial(T.int64(128), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(128))
                    T.where(ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 < T.int64(1536))
                    T.reads()
                    T.writes(T_softmax_expsum[v0, v1])
                    T_softmax_expsum[v0, v1] = T.float32(0.0)
                for ax2 in range(T.int64(128)):
                    with T.block("T_softmax_expsum_update"):
                        v0 = T.axis.spatial(T.int64(12), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(128))
                        v1 = T.axis.spatial(T.int64(128), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(128))
                        v2 = T.axis.reduce(T.int64(128), ax2)
                        T.where(ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 < T.int64(1536))
                        T.reads(T_softmax_expsum[v0, v1], T_batch_matmul_NT[v0, v1, v2], lv30[T.int64(0), T.int64(0), v1, v2], T_softmax_maxelem[v0, v1])
                        T.writes(T_softmax_expsum[v0, v1])
                        T_softmax_expsum[v0, v1] = T_softmax_expsum[v0, v1] + T.exp(T_batch_matmul_NT[v0, v1, v2] / T.sqrt(T.float32(64.0)) + lv30[T.int64(0), T.int64(0), v1, v2] - T_softmax_maxelem[v0, v1])
        for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(96), thread="blockIdx.x"):
            for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_batch_matmul_NN_init"):
                    v0 = T.axis.spatial(T.int64(12), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) // T.int64(8192))
                    v1 = T.axis.spatial(T.int64(128), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(8192) // T.int64(64))
                    v2 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(64))
                    T.reads()
                    T.writes(T_batch_matmul_NN[v0, v1, v2])
                    T_reshape = T.Buffer((T.int64(12), T.int64(128), T.int64(64)))
                    T.block_attr({"layout_free_placeholders": [T_reshape]})
                    T_batch_matmul_NN[v0, v1, v2] = T.float32(0.0)
                T_reshape = T.Buffer((T.int64(12), T.int64(128), T.int64(64)))
                for ax3 in range(T.int64(128)):
                    with T.block("T_batch_matmul_NN_update"):
                        v0 = T.axis.spatial(T.int64(12), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) // T.int64(8192))
                        v1 = T.axis.spatial(T.int64(128), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(8192) // T.int64(64))
                        v2 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(64))
                        v3 = T.axis.reduce(T.int64(128), ax3)
                        T.reads(T_batch_matmul_NN[v0, v1, v2], T_batch_matmul_NT[v0, v1, v3], lv30[T.int64(0), T.int64(0), v1, v3], T_softmax_maxelem[v0, v1], T_softmax_expsum[v0, v1], lv48[T.int64(0), v3, v0, v2])
                        T.writes(T_batch_matmul_NN[v0, v1, v2])
                        T.block_attr({"layout_free_placeholders": [T_reshape]})
                        T_batch_matmul_NN[v0, v1, v2] = T_batch_matmul_NN[v0, v1, v2] + T.exp(T_batch_matmul_NT[v0, v1, v3] / T.sqrt(T.float32(64.0)) + lv30[T.int64(0), T.int64(0), v1, v3] - T_softmax_maxelem[v0, v1]) / T_softmax_expsum[v0, v1] * lv48[T.int64(0), v3, v0, v2]
        for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(96), thread="blockIdx.x"):
            for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_transpose_3"):
                    v0 = T.axis.spatial(T.int64(128), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) // T.int64(768))
                    v1 = T.axis.spatial(T.int64(12), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(768) // T.int64(64))
                    v2 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(64))
                    T.reads(T_batch_matmul_NN[v1, v0, v2])
                    T.writes(T_transpose[T.int64(0), v0, v1, v2])
                    T_transpose[T.int64(0), v0, v1, v2] = T_batch_matmul_NN[v1, v0, v2]

    @T.prim_func(private=True)
    def cast(input_ids: T.Buffer((T.int64(1), T.int64(128)), "int64"), compute: T.Buffer((T.int64(1), T.int64(128)), "int32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v0 = T.axis.spatial(T.int64(128), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(128))
                    T.reads(input_ids[T.int64(0), v0])
                    T.writes(compute[T.int64(0), v0])
                    compute[T.int64(0), v0] = T.Cast("int32", input_ids[T.int64(0), v0])

    @T.prim_func(private=True)
    def gelu(lv60: T.Buffer((T.int64(1), T.int64(128), T.int64(3072)), "float32"), T_multiply: T.Buffer((T.int64(1), T.int64(128), T.int64(3072)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(384), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply_2"):
                    v0 = T.axis.spatial(T.int64(128), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(3072))
                    v1 = T.axis.spatial(T.int64(3072), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(3072))
                    T.reads(lv60[T.int64(0), v0, v1])
                    T.writes(T_multiply[T.int64(0), v0, v1])
                    T_multiply[T.int64(0), v0, v1] = lv60[T.int64(0), v0, v1] * (T.float32(0.5) + T.erf(lv60[T.int64(0), v0, v1] * T.float32(0.70710678118654757)) * T.float32(0.5))

    @T.prim_func(private=True)
    def layer_norm(lv18: T.Buffer((T.int64(1), T.int64(128), T.int64(768)), "float32"), B: T.Buffer((T.int64(768),), "float32"), C: T.Buffer((T.int64(768),), "float32"), T_layer_norm: T.Buffer((T.int64(1), T.int64(128), T.int64(768)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        lv18_red_temp_v0_shared = T.alloc_buffer((T.int64(1), T.int64(128)), scope="shared")
        lv18_red_temp_v1_shared = T.alloc_buffer((T.int64(1), T.int64(128)), scope="shared")
        for ax0_fused in T.thread_binding(T.int64(128), thread="blockIdx.x"):
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(3), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("lv18_red_temp"):
                            v0 = T.axis.spatial(T.int64(128), ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(768), ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.reads(lv18[T.int64(0), v0, v1])
                            T.writes(lv18_red_temp_v0_shared[T.int64(0), v0], lv18_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                lv18_red_temp_v0_shared[T.int64(0), v0] = T.float32(0.0)
                                lv18_red_temp_v1_shared[T.int64(0), v0] = T.float32(0.0)
                            v_lv18_red_temp_v0: T.float32 = lv18_red_temp_v0_shared[T.int64(0), v0] + lv18[T.int64(0), v0, v1]
                            v_lv18_red_temp_v1: T.float32 = lv18_red_temp_v1_shared[T.int64(0), v0] + lv18[T.int64(0), v0, v1] * lv18[T.int64(0), v0, v1]
                            lv18_red_temp_v0_shared[T.int64(0), v0] = v_lv18_red_temp_v0
                            lv18_red_temp_v1_shared[T.int64(0), v0] = v_lv18_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(3), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_layer_norm"):
                        v0 = T.axis.spatial(T.int64(128), ax0_fused)
                        v1 = T.axis.spatial(T.int64(768), ax1_0 * T.int64(256) + ax1_1)
                        T.reads(lv18[T.int64(0), v0, v1], lv18_red_temp_v0_shared[T.int64(0), v0], lv18_red_temp_v1_shared[T.int64(0), v0], B[v1], C[v1])
                        T.writes(T_layer_norm[T.int64(0), v0, v1])
                        T_layer_norm[T.int64(0), v0, v1] = (lv18[T.int64(0), v0, v1] - lv18_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.0013020833333333333)) * T.rsqrt(lv18_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.0013020833333333333) - lv18_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.0013020833333333333) * (lv18_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.0013020833333333333)) + T.float32(9.9999999999999998e-13)) * B[v1] + C[v1]

    @T.prim_func(private=True)
    def matmul(lv19: T.Buffer((T.int64(1), T.int64(128), T.int64(768)), "float32"), lv31: T.Buffer((T.int64(768), T.int64(768)), "float32"), matmul: T.Buffer((T.int64(1), T.int64(128), T.int64(768)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        matmul_reindex_local = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(768)), scope="local")
        lv19_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(768)), scope="shared")
        lv31_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(768), T.int64(768)), scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(12), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding(T.int64(4), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with T.block("matmul_init"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(128), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = T.axis.spatial(T.int64(768), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            T.reads()
                                            T.writes(matmul_reindex_local[T.int64(0), v1, v2])
                                            matmul_reindex_local[T.int64(0), v1, v2] = T.float32(0.0)
                                for ax3_0 in range(T.int64(48)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("lv19_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(128), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(768), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv19[v0, v1, v2])
                                                        T.writes(lv19_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        lv19_reindex_shared[v0, v1, v2] = lv19[v0, v1, v2]
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("lv31_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(768), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(768), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv31[v2, v1])
                                                        T.writes(lv31_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        lv31_reindex_shared[v0, v1, v2] = lv31[v2, v1]
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with T.block("matmul_update"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial(T.int64(128), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(768), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(768), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(matmul_reindex_local[T.int64(0), v1, v2], lv19_reindex_shared[T.int64(0), v1, v3], lv31_reindex_shared[T.int64(0), v2, v3])
                                                T.writes(matmul_reindex_local[T.int64(0), v1, v2])
                                                matmul_reindex_local[T.int64(0), v1, v2] = matmul_reindex_local[T.int64(0), v1, v2] + lv19_reindex_shared[T.int64(0), v1, v3] * lv31_reindex_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("matmul_reindex_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(T.int64(128), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(768), ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            T.reads(matmul_reindex_local[v0, v1, v2])
                                            T.writes(matmul[T.int64(0), v1, v2])
                                            matmul[T.int64(0), v1, v2] = matmul_reindex_local[v0, v1, v2]

    @T.prim_func(private=True)
    def matmul1(lv57: T.Buffer((T.int64(1), T.int64(128), T.int64(768)), "float32"), lv58: T.Buffer((T.int64(768), T.int64(3072)), "float32"), matmul: T.Buffer((T.int64(1), T.int64(128), T.int64(3072)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        matmul_reindex_local = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(3072)), scope="local")
        lv57_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(768)), scope="shared")
        lv58_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(3072), T.int64(768)), scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(48), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding(T.int64(4), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with T.block("matmul_init"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(128), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = T.axis.spatial(T.int64(3072), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            T.reads()
                                            T.writes(matmul_reindex_local[T.int64(0), v1, v2])
                                            matmul_reindex_local[T.int64(0), v1, v2] = T.float32(0.0)
                                for ax3_0 in range(T.int64(48)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("lv57_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(128), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(768), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv57[v0, v1, v2])
                                                        T.writes(lv57_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        lv57_reindex_shared[v0, v1, v2] = lv57[v0, v1, v2]
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("lv58_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(3072), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(768), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv58[v2, v1])
                                                        T.writes(lv58_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        lv58_reindex_shared[v0, v1, v2] = lv58[v2, v1]
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with T.block("matmul_update"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial(T.int64(128), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(3072), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(768), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(matmul_reindex_local[T.int64(0), v1, v2], lv57_reindex_shared[T.int64(0), v1, v3], lv58_reindex_shared[T.int64(0), v2, v3])
                                                T.writes(matmul_reindex_local[T.int64(0), v1, v2])
                                                matmul_reindex_local[T.int64(0), v1, v2] = matmul_reindex_local[T.int64(0), v1, v2] + lv57_reindex_shared[T.int64(0), v1, v3] * lv58_reindex_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("matmul_reindex_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(T.int64(128), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(3072), ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            T.reads(matmul_reindex_local[v0, v1, v2])
                                            T.writes(matmul[T.int64(0), v1, v2])
                                            matmul[T.int64(0), v1, v2] = matmul_reindex_local[v0, v1, v2]

    @T.prim_func(private=True)
    def matmul2(lv61: T.Buffer((T.int64(1), T.int64(128), T.int64(3072)), "float32"), lv62: T.Buffer((T.int64(3072), T.int64(768)), "float32"), matmul: T.Buffer((T.int64(1), T.int64(128), T.int64(768)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        matmul_reindex_local = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(768)), scope="local")
        lv61_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(3072)), scope="shared")
        lv62_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(768), T.int64(3072)), scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(12), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding(T.int64(4), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with T.block("matmul_init"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(128), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = T.axis.spatial(T.int64(768), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            T.reads()
                                            T.writes(matmul_reindex_local[T.int64(0), v1, v2])
                                            matmul_reindex_local[T.int64(0), v1, v2] = T.float32(0.0)
                                for ax3_0 in range(T.int64(192)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("lv61_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(128), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(3072), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv61[v0, v1, v2])
                                                        T.writes(lv61_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        lv61_reindex_shared[v0, v1, v2] = lv61[v0, v1, v2]
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("lv62_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(768), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(3072), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv62[v2, v1])
                                                        T.writes(lv62_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        lv62_reindex_shared[v0, v1, v2] = lv62[v2, v1]
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with T.block("matmul_update"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial(T.int64(128), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(768), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(3072), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(matmul_reindex_local[T.int64(0), v1, v2], lv61_reindex_shared[T.int64(0), v1, v3], lv62_reindex_shared[T.int64(0), v2, v3])
                                                T.writes(matmul_reindex_local[T.int64(0), v1, v2])
                                                matmul_reindex_local[T.int64(0), v1, v2] = matmul_reindex_local[T.int64(0), v1, v2] + lv61_reindex_shared[T.int64(0), v1, v3] * lv62_reindex_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("matmul_reindex_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(T.int64(128), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(768), ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            T.reads(matmul_reindex_local[v0, v1, v2])
                                            T.writes(matmul[T.int64(0), v1, v2])
                                            matmul[T.int64(0), v1, v2] = matmul_reindex_local[v0, v1, v2]

    @T.prim_func(private=True)
    def reshape(lv5: T.Buffer((T.int64(1), T.int64(128)), "int32"), T_reshape: T.Buffer((T.int64(128),), "int32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_reshape"):
                    v0 = T.axis.spatial(T.int64(128), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(128))
                    T.reads(lv5[T.int64(0), v0])
                    T.writes(T_reshape[v0])
                    T_reshape[v0] = lv5[T.int64(0), v0]

    @T.prim_func(private=True)
    def reshape1(lv7: T.Buffer((T.int64(128), T.int64(768)), "float32"), T_reshape: T.Buffer((T.int64(1), T.int64(128), T.int64(768)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(96), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_reshape"):
                    v0 = T.axis.spatial(T.int64(128), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(768))
                    v1 = T.axis.spatial(T.int64(768), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(768))
                    T.reads(lv7[v0, v1])
                    T.writes(T_reshape[T.int64(0), v0, v1])
                    T_reshape[T.int64(0), v0, v1] = lv7[v0, v1]

    @T.prim_func(private=True)
    def reshape2(lv33: T.Buffer((T.int64(1), T.int64(128), T.int64(768)), "float32"), T_reshape: T.Buffer((T.int64(1), T.int64(128), T.int64(12), T.int64(64)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(96), thread="blockIdx.x"):
            for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_reshape"):
                    v0 = T.axis.spatial(T.int64(128), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) // T.int64(768))
                    v1 = T.axis.spatial(T.int64(12), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(768) // T.int64(64))
                    v2 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(64))
                    T.reads(lv33[T.int64(0), v0, v1 * T.int64(64) + v2])
                    T.writes(T_reshape[T.int64(0), v0, v1, v2])
                    T_reshape[T.int64(0), v0, v1, v2] = lv33[T.int64(0), v0, v1 * T.int64(64) + v2]

    @T.prim_func(private=True)
    def reshape3(lv51: T.Buffer((T.int64(1), T.int64(128), T.int64(12), T.int64(64)), "float32"), T_reshape: T.Buffer((T.int64(1), T.int64(128), T.int64(768)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(96), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_reshape"):
                    v0 = T.axis.spatial(T.int64(128), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(768))
                    v1 = T.axis.spatial(T.int64(768), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(768))
                    T.reads(lv51[T.int64(0), v0, v1 // T.int64(64), v1 % T.int64(64)])
                    T.writes(T_reshape[T.int64(0), v0, v1])
                    T_reshape[T.int64(0), v0, v1] = lv51[T.int64(0), v0, v1 // T.int64(64), v1 % T.int64(64)]

    @T.prim_func(private=True)
    def take(A: T.Buffer((T.int64(30522), T.int64(768)), "float32"), lv6: T.Buffer((T.int64(128),), "int32"), T_take: T.Buffer((T.int64(128), T.int64(768)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(96), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_take"):
                    v0 = T.axis.spatial(T.int64(128), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(768))
                    v1 = T.axis.spatial(T.int64(768), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(768))
                    T.reads(A[lv6[v0], v1], lv6[v0])
                    T.writes(T_take[v0, v1])
                    T_take[v0, v1] = A[lv6[v0], v1]

    @T.prim_func(private=True)
    def transpose1(lv34: T.Buffer((T.int64(1), T.int64(128), T.int64(12), T.int64(64)), "float32"), T_transpose: T.Buffer((T.int64(1), T.int64(12), T.int64(128), T.int64(64)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(96), thread="blockIdx.x"):
            for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_transpose"):
                    v0 = T.axis.spatial(T.int64(12), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) // T.int64(8192))
                    v1 = T.axis.spatial(T.int64(128), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(8192) // T.int64(64))
                    v2 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(64))
                    T.reads(lv34[T.int64(0), v1, v0, v2])
                    T.writes(T_transpose[T.int64(0), v0, v1, v2])
                    T_transpose[T.int64(0), v0, v1, v2] = lv34[T.int64(0), v1, v0, v2]

    @T.prim_func(private=True)
    def transpose2(lv35: T.Buffer((T.int64(1), T.int64(12), T.int64(128), T.int64(64)), "float32"), T_transpose: T.Buffer((T.int64(1), T.int64(128), T.int64(12), T.int64(64)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(96), thread="blockIdx.x"):
            for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_transpose"):
                    v0 = T.axis.spatial(T.int64(128), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) // T.int64(768))
                    v1 = T.axis.spatial(T.int64(12), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(768) // T.int64(64))
                    v2 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(64))
                    T.reads(lv35[T.int64(0), v1, v0, v2])
                    T.writes(T_transpose[T.int64(0), v0, v1, v2])
                    T_transpose[T.int64(0), v0, v1, v2] = lv35[T.int64(0), v1, v0, v2]

    @R.function
    def main(input_ids: R.Tensor((1, 128), dtype="int64")) -> R.Tuple(R.Tensor((1, 128, 768), dtype="float32")):
        cls = Module
        with R.dataflow():
            lv5 = R.call_tir(cls.cast, (input_ids,), out_sinfo=R.Tensor((1, 128), dtype="int32"))
            lv6 = R.call_tir(cls.reshape, (lv5,), out_sinfo=R.Tensor((128,), dtype="int32"))
            lv7 = R.call_tir(cls.take, (metadata["relax.expr.Constant"][0], lv6), out_sinfo=R.Tensor((128, 768), dtype="float32"))
            lv8 = R.call_tir(cls.reshape1, (lv7,), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv13 = R.call_tir(cls.add, (lv8, metadata["relax.expr.Constant"][1]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv18 = R.call_tir(cls.add, (lv13, metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv19 = R.call_tir(cls.layer_norm, (lv18, metadata["relax.expr.Constant"][3], metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv32 = R.call_tir(cls.matmul, (lv19, metadata["relax.expr.Constant"][5]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv33 = R.call_tir(cls.add1, (lv32, metadata["relax.expr.Constant"][6]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv34 = R.call_tir(cls.reshape2, (lv33,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv35 = R.call_tir(cls.transpose1, (lv34,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv37 = R.call_tir(cls.matmul, (lv19, metadata["relax.expr.Constant"][7]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv38 = R.call_tir(cls.add1, (lv37, metadata["relax.expr.Constant"][8]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv39 = R.call_tir(cls.reshape2, (lv38,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv40 = R.call_tir(cls.transpose1, (lv39,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv42 = R.call_tir(cls.matmul, (lv19, metadata["relax.expr.Constant"][9]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv43 = R.call_tir(cls.add1, (lv42, metadata["relax.expr.Constant"][10]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv44 = R.call_tir(cls.reshape2, (lv43,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv45 = R.call_tir(cls.transpose1, (lv44,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv46 = R.call_tir(cls.transpose2, (lv35,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv47 = R.call_tir(cls.transpose2, (lv40,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv48 = R.call_tir(cls.transpose2, (lv45,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv49 = R.call_tir(cls.attention_bias, (lv46, lv47, lv48, metadata["relax.expr.Constant"][11]), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv50 = R.call_tir(cls.transpose1, (lv49,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv51 = R.call_tir(cls.transpose2, (lv50,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv52 = R.call_tir(cls.reshape3, (lv51,), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv54 = R.call_tir(cls.matmul, (lv52, metadata["relax.expr.Constant"][12]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv55 = R.call_tir(cls.add1, (lv54, metadata["relax.expr.Constant"][13]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv56 = R.call_tir(cls.add, (lv55, lv19), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv57 = R.call_tir(cls.layer_norm, (lv56, metadata["relax.expr.Constant"][14], metadata["relax.expr.Constant"][15]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv59 = R.call_tir(cls.matmul1, (lv57, metadata["relax.expr.Constant"][16]), out_sinfo=R.Tensor((1, 128, 3072), dtype="float32"))
            lv60 = R.call_tir(cls.add2, (lv59, metadata["relax.expr.Constant"][17]), out_sinfo=R.Tensor((1, 128, 3072), dtype="float32"))
            lv61 = R.call_tir(cls.gelu, (lv60,), out_sinfo=R.Tensor((1, 128, 3072), dtype="float32"))
            lv63 = R.call_tir(cls.matmul2, (lv61, metadata["relax.expr.Constant"][18]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv64 = R.call_tir(cls.add1, (lv63, metadata["relax.expr.Constant"][19]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv65 = R.call_tir(cls.add, (lv64, lv57), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv66 = R.call_tir(cls.layer_norm, (lv65, metadata["relax.expr.Constant"][20], metadata["relax.expr.Constant"][21]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv68 = R.call_tir(cls.matmul, (lv66, metadata["relax.expr.Constant"][22]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv69 = R.call_tir(cls.add1, (lv68, metadata["relax.expr.Constant"][23]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv70 = R.call_tir(cls.reshape2, (lv69,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv71 = R.call_tir(cls.transpose1, (lv70,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv73 = R.call_tir(cls.matmul, (lv66, metadata["relax.expr.Constant"][24]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv74 = R.call_tir(cls.add1, (lv73, metadata["relax.expr.Constant"][25]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv75 = R.call_tir(cls.reshape2, (lv74,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv76 = R.call_tir(cls.transpose1, (lv75,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv78 = R.call_tir(cls.matmul, (lv66, metadata["relax.expr.Constant"][26]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv79 = R.call_tir(cls.add1, (lv78, metadata["relax.expr.Constant"][27]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv80 = R.call_tir(cls.reshape2, (lv79,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv81 = R.call_tir(cls.transpose1, (lv80,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv82 = R.call_tir(cls.transpose2, (lv71,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv83 = R.call_tir(cls.transpose2, (lv76,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv84 = R.call_tir(cls.transpose2, (lv81,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv85 = R.call_tir(cls.attention_bias, (lv82, lv83, lv84, metadata["relax.expr.Constant"][11]), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv86 = R.call_tir(cls.transpose1, (lv85,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv87 = R.call_tir(cls.transpose2, (lv86,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv88 = R.call_tir(cls.reshape3, (lv87,), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv90 = R.call_tir(cls.matmul, (lv88, metadata["relax.expr.Constant"][28]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv91 = R.call_tir(cls.add1, (lv90, metadata["relax.expr.Constant"][29]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv92 = R.call_tir(cls.add, (lv91, lv66), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv93 = R.call_tir(cls.layer_norm, (lv92, metadata["relax.expr.Constant"][30], metadata["relax.expr.Constant"][31]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv95 = R.call_tir(cls.matmul1, (lv93, metadata["relax.expr.Constant"][32]), out_sinfo=R.Tensor((1, 128, 3072), dtype="float32"))
            lv96 = R.call_tir(cls.add2, (lv95, metadata["relax.expr.Constant"][33]), out_sinfo=R.Tensor((1, 128, 3072), dtype="float32"))
            lv97 = R.call_tir(cls.gelu, (lv96,), out_sinfo=R.Tensor((1, 128, 3072), dtype="float32"))
            lv99 = R.call_tir(cls.matmul2, (lv97, metadata["relax.expr.Constant"][34]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv100 = R.call_tir(cls.add1, (lv99, metadata["relax.expr.Constant"][35]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv101 = R.call_tir(cls.add, (lv100, lv93), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv102 = R.call_tir(cls.layer_norm, (lv101, metadata["relax.expr.Constant"][36], metadata["relax.expr.Constant"][37]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv104 = R.call_tir(cls.matmul, (lv102, metadata["relax.expr.Constant"][38]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv105 = R.call_tir(cls.add1, (lv104, metadata["relax.expr.Constant"][39]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv106 = R.call_tir(cls.reshape2, (lv105,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv107 = R.call_tir(cls.transpose1, (lv106,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv109 = R.call_tir(cls.matmul, (lv102, metadata["relax.expr.Constant"][40]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv110 = R.call_tir(cls.add1, (lv109, metadata["relax.expr.Constant"][41]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv111 = R.call_tir(cls.reshape2, (lv110,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv112 = R.call_tir(cls.transpose1, (lv111,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv114 = R.call_tir(cls.matmul, (lv102, metadata["relax.expr.Constant"][42]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv115 = R.call_tir(cls.add1, (lv114, metadata["relax.expr.Constant"][43]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv116 = R.call_tir(cls.reshape2, (lv115,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv117 = R.call_tir(cls.transpose1, (lv116,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv118 = R.call_tir(cls.transpose2, (lv107,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv119 = R.call_tir(cls.transpose2, (lv112,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv120 = R.call_tir(cls.transpose2, (lv117,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv121 = R.call_tir(cls.attention_bias, (lv118, lv119, lv120, metadata["relax.expr.Constant"][11]), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv122 = R.call_tir(cls.transpose1, (lv121,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv123 = R.call_tir(cls.transpose2, (lv122,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv124 = R.call_tir(cls.reshape3, (lv123,), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv126 = R.call_tir(cls.matmul, (lv124, metadata["relax.expr.Constant"][44]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv127 = R.call_tir(cls.add1, (lv126, metadata["relax.expr.Constant"][45]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv128 = R.call_tir(cls.add, (lv127, lv102), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv129 = R.call_tir(cls.layer_norm, (lv128, metadata["relax.expr.Constant"][46], metadata["relax.expr.Constant"][47]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv131 = R.call_tir(cls.matmul1, (lv129, metadata["relax.expr.Constant"][48]), out_sinfo=R.Tensor((1, 128, 3072), dtype="float32"))
            lv132 = R.call_tir(cls.add2, (lv131, metadata["relax.expr.Constant"][49]), out_sinfo=R.Tensor((1, 128, 3072), dtype="float32"))
            lv133 = R.call_tir(cls.gelu, (lv132,), out_sinfo=R.Tensor((1, 128, 3072), dtype="float32"))
            lv135 = R.call_tir(cls.matmul2, (lv133, metadata["relax.expr.Constant"][50]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv136 = R.call_tir(cls.add1, (lv135, metadata["relax.expr.Constant"][51]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv137 = R.call_tir(cls.add, (lv136, lv129), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv138 = R.call_tir(cls.layer_norm, (lv137, metadata["relax.expr.Constant"][52], metadata["relax.expr.Constant"][53]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv140 = R.call_tir(cls.matmul, (lv138, metadata["relax.expr.Constant"][54]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv141 = R.call_tir(cls.add1, (lv140, metadata["relax.expr.Constant"][55]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv142 = R.call_tir(cls.reshape2, (lv141,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv143 = R.call_tir(cls.transpose1, (lv142,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv145 = R.call_tir(cls.matmul, (lv138, metadata["relax.expr.Constant"][56]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv146 = R.call_tir(cls.add1, (lv145, metadata["relax.expr.Constant"][57]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv147 = R.call_tir(cls.reshape2, (lv146,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv148 = R.call_tir(cls.transpose1, (lv147,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv150 = R.call_tir(cls.matmul, (lv138, metadata["relax.expr.Constant"][58]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv151 = R.call_tir(cls.add1, (lv150, metadata["relax.expr.Constant"][59]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv152 = R.call_tir(cls.reshape2, (lv151,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv153 = R.call_tir(cls.transpose1, (lv152,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv154 = R.call_tir(cls.transpose2, (lv143,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv155 = R.call_tir(cls.transpose2, (lv148,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv156 = R.call_tir(cls.transpose2, (lv153,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv157 = R.call_tir(cls.attention_bias, (lv154, lv155, lv156, metadata["relax.expr.Constant"][11]), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv158 = R.call_tir(cls.transpose1, (lv157,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv159 = R.call_tir(cls.transpose2, (lv158,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv160 = R.call_tir(cls.reshape3, (lv159,), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv162 = R.call_tir(cls.matmul, (lv160, metadata["relax.expr.Constant"][60]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv163 = R.call_tir(cls.add1, (lv162, metadata["relax.expr.Constant"][61]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv164 = R.call_tir(cls.add, (lv163, lv138), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv165 = R.call_tir(cls.layer_norm, (lv164, metadata["relax.expr.Constant"][62], metadata["relax.expr.Constant"][63]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv167 = R.call_tir(cls.matmul1, (lv165, metadata["relax.expr.Constant"][64]), out_sinfo=R.Tensor((1, 128, 3072), dtype="float32"))
            lv168 = R.call_tir(cls.add2, (lv167, metadata["relax.expr.Constant"][65]), out_sinfo=R.Tensor((1, 128, 3072), dtype="float32"))
            lv169 = R.call_tir(cls.gelu, (lv168,), out_sinfo=R.Tensor((1, 128, 3072), dtype="float32"))
            lv171 = R.call_tir(cls.matmul2, (lv169, metadata["relax.expr.Constant"][66]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv172 = R.call_tir(cls.add1, (lv171, metadata["relax.expr.Constant"][67]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv173 = R.call_tir(cls.add, (lv172, lv165), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv174 = R.call_tir(cls.layer_norm, (lv173, metadata["relax.expr.Constant"][68], metadata["relax.expr.Constant"][69]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv176 = R.call_tir(cls.matmul, (lv174, metadata["relax.expr.Constant"][70]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv177 = R.call_tir(cls.add1, (lv176, metadata["relax.expr.Constant"][71]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv178 = R.call_tir(cls.reshape2, (lv177,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv179 = R.call_tir(cls.transpose1, (lv178,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv181 = R.call_tir(cls.matmul, (lv174, metadata["relax.expr.Constant"][72]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv182 = R.call_tir(cls.add1, (lv181, metadata["relax.expr.Constant"][73]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv183 = R.call_tir(cls.reshape2, (lv182,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv184 = R.call_tir(cls.transpose1, (lv183,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv186 = R.call_tir(cls.matmul, (lv174, metadata["relax.expr.Constant"][74]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv187 = R.call_tir(cls.add1, (lv186, metadata["relax.expr.Constant"][75]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv188 = R.call_tir(cls.reshape2, (lv187,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv189 = R.call_tir(cls.transpose1, (lv188,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv190 = R.call_tir(cls.transpose2, (lv179,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv191 = R.call_tir(cls.transpose2, (lv184,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv192 = R.call_tir(cls.transpose2, (lv189,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv193 = R.call_tir(cls.attention_bias, (lv190, lv191, lv192, metadata["relax.expr.Constant"][11]), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv194 = R.call_tir(cls.transpose1, (lv193,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv195 = R.call_tir(cls.transpose2, (lv194,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv196 = R.call_tir(cls.reshape3, (lv195,), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv198 = R.call_tir(cls.matmul, (lv196, metadata["relax.expr.Constant"][76]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv199 = R.call_tir(cls.add1, (lv198, metadata["relax.expr.Constant"][77]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv200 = R.call_tir(cls.add, (lv199, lv174), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv201 = R.call_tir(cls.layer_norm, (lv200, metadata["relax.expr.Constant"][78], metadata["relax.expr.Constant"][79]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv203 = R.call_tir(cls.matmul1, (lv201, metadata["relax.expr.Constant"][80]), out_sinfo=R.Tensor((1, 128, 3072), dtype="float32"))
            lv204 = R.call_tir(cls.add2, (lv203, metadata["relax.expr.Constant"][81]), out_sinfo=R.Tensor((1, 128, 3072), dtype="float32"))
            lv205 = R.call_tir(cls.gelu, (lv204,), out_sinfo=R.Tensor((1, 128, 3072), dtype="float32"))
            lv207 = R.call_tir(cls.matmul2, (lv205, metadata["relax.expr.Constant"][82]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv208 = R.call_tir(cls.add1, (lv207, metadata["relax.expr.Constant"][83]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv209 = R.call_tir(cls.add, (lv208, lv201), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv210 = R.call_tir(cls.layer_norm, (lv209, metadata["relax.expr.Constant"][84], metadata["relax.expr.Constant"][85]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv212 = R.call_tir(cls.matmul, (lv210, metadata["relax.expr.Constant"][86]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv213 = R.call_tir(cls.add1, (lv212, metadata["relax.expr.Constant"][87]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv214 = R.call_tir(cls.reshape2, (lv213,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv215 = R.call_tir(cls.transpose1, (lv214,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv217 = R.call_tir(cls.matmul, (lv210, metadata["relax.expr.Constant"][88]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv218 = R.call_tir(cls.add1, (lv217, metadata["relax.expr.Constant"][89]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv219 = R.call_tir(cls.reshape2, (lv218,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv220 = R.call_tir(cls.transpose1, (lv219,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv222 = R.call_tir(cls.matmul, (lv210, metadata["relax.expr.Constant"][90]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv223 = R.call_tir(cls.add1, (lv222, metadata["relax.expr.Constant"][91]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv224 = R.call_tir(cls.reshape2, (lv223,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv225 = R.call_tir(cls.transpose1, (lv224,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv226 = R.call_tir(cls.transpose2, (lv215,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv227 = R.call_tir(cls.transpose2, (lv220,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv228 = R.call_tir(cls.transpose2, (lv225,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv229 = R.call_tir(cls.attention_bias, (lv226, lv227, lv228, metadata["relax.expr.Constant"][11]), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv230 = R.call_tir(cls.transpose1, (lv229,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv231 = R.call_tir(cls.transpose2, (lv230,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv232 = R.call_tir(cls.reshape3, (lv231,), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv234 = R.call_tir(cls.matmul, (lv232, metadata["relax.expr.Constant"][92]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv235 = R.call_tir(cls.add1, (lv234, metadata["relax.expr.Constant"][93]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv236 = R.call_tir(cls.add, (lv235, lv210), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv237 = R.call_tir(cls.layer_norm, (lv236, metadata["relax.expr.Constant"][94], metadata["relax.expr.Constant"][95]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv239 = R.call_tir(cls.matmul1, (lv237, metadata["relax.expr.Constant"][96]), out_sinfo=R.Tensor((1, 128, 3072), dtype="float32"))
            lv240 = R.call_tir(cls.add2, (lv239, metadata["relax.expr.Constant"][97]), out_sinfo=R.Tensor((1, 128, 3072), dtype="float32"))
            lv241 = R.call_tir(cls.gelu, (lv240,), out_sinfo=R.Tensor((1, 128, 3072), dtype="float32"))
            lv243 = R.call_tir(cls.matmul2, (lv241, metadata["relax.expr.Constant"][98]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv244 = R.call_tir(cls.add1, (lv243, metadata["relax.expr.Constant"][99]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv245 = R.call_tir(cls.add, (lv244, lv237), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv246 = R.call_tir(cls.layer_norm, (lv245, metadata["relax.expr.Constant"][100], metadata["relax.expr.Constant"][101]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv248 = R.call_tir(cls.matmul, (lv246, metadata["relax.expr.Constant"][102]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv249 = R.call_tir(cls.add1, (lv248, metadata["relax.expr.Constant"][103]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv250 = R.call_tir(cls.reshape2, (lv249,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv251 = R.call_tir(cls.transpose1, (lv250,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv253 = R.call_tir(cls.matmul, (lv246, metadata["relax.expr.Constant"][104]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv254 = R.call_tir(cls.add1, (lv253, metadata["relax.expr.Constant"][105]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv255 = R.call_tir(cls.reshape2, (lv254,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv256 = R.call_tir(cls.transpose1, (lv255,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv258 = R.call_tir(cls.matmul, (lv246, metadata["relax.expr.Constant"][106]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv259 = R.call_tir(cls.add1, (lv258, metadata["relax.expr.Constant"][107]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv260 = R.call_tir(cls.reshape2, (lv259,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv261 = R.call_tir(cls.transpose1, (lv260,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv262 = R.call_tir(cls.transpose2, (lv251,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv263 = R.call_tir(cls.transpose2, (lv256,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv264 = R.call_tir(cls.transpose2, (lv261,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv265 = R.call_tir(cls.attention_bias, (lv262, lv263, lv264, metadata["relax.expr.Constant"][11]), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv266 = R.call_tir(cls.transpose1, (lv265,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv267 = R.call_tir(cls.transpose2, (lv266,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv268 = R.call_tir(cls.reshape3, (lv267,), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv270 = R.call_tir(cls.matmul, (lv268, metadata["relax.expr.Constant"][108]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv271 = R.call_tir(cls.add1, (lv270, metadata["relax.expr.Constant"][109]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv272 = R.call_tir(cls.add, (lv271, lv246), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv273 = R.call_tir(cls.layer_norm, (lv272, metadata["relax.expr.Constant"][110], metadata["relax.expr.Constant"][111]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv275 = R.call_tir(cls.matmul1, (lv273, metadata["relax.expr.Constant"][112]), out_sinfo=R.Tensor((1, 128, 3072), dtype="float32"))
            lv276 = R.call_tir(cls.add2, (lv275, metadata["relax.expr.Constant"][113]), out_sinfo=R.Tensor((1, 128, 3072), dtype="float32"))
            lv277 = R.call_tir(cls.gelu, (lv276,), out_sinfo=R.Tensor((1, 128, 3072), dtype="float32"))
            lv279 = R.call_tir(cls.matmul2, (lv277, metadata["relax.expr.Constant"][114]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv280 = R.call_tir(cls.add1, (lv279, metadata["relax.expr.Constant"][115]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv281 = R.call_tir(cls.add, (lv280, lv273), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv282 = R.call_tir(cls.layer_norm, (lv281, metadata["relax.expr.Constant"][116], metadata["relax.expr.Constant"][117]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv284 = R.call_tir(cls.matmul, (lv282, metadata["relax.expr.Constant"][118]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv285 = R.call_tir(cls.add1, (lv284, metadata["relax.expr.Constant"][119]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv286 = R.call_tir(cls.reshape2, (lv285,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv287 = R.call_tir(cls.transpose1, (lv286,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv289 = R.call_tir(cls.matmul, (lv282, metadata["relax.expr.Constant"][120]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv290 = R.call_tir(cls.add1, (lv289, metadata["relax.expr.Constant"][121]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv291 = R.call_tir(cls.reshape2, (lv290,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv292 = R.call_tir(cls.transpose1, (lv291,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv294 = R.call_tir(cls.matmul, (lv282, metadata["relax.expr.Constant"][122]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv295 = R.call_tir(cls.add1, (lv294, metadata["relax.expr.Constant"][123]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv296 = R.call_tir(cls.reshape2, (lv295,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv297 = R.call_tir(cls.transpose1, (lv296,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv298 = R.call_tir(cls.transpose2, (lv287,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv299 = R.call_tir(cls.transpose2, (lv292,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv300 = R.call_tir(cls.transpose2, (lv297,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv301 = R.call_tir(cls.attention_bias, (lv298, lv299, lv300, metadata["relax.expr.Constant"][11]), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv302 = R.call_tir(cls.transpose1, (lv301,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv303 = R.call_tir(cls.transpose2, (lv302,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv304 = R.call_tir(cls.reshape3, (lv303,), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv306 = R.call_tir(cls.matmul, (lv304, metadata["relax.expr.Constant"][124]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv307 = R.call_tir(cls.add1, (lv306, metadata["relax.expr.Constant"][125]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv308 = R.call_tir(cls.add, (lv307, lv282), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv309 = R.call_tir(cls.layer_norm, (lv308, metadata["relax.expr.Constant"][126], metadata["relax.expr.Constant"][127]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv311 = R.call_tir(cls.matmul1, (lv309, metadata["relax.expr.Constant"][128]), out_sinfo=R.Tensor((1, 128, 3072), dtype="float32"))
            lv312 = R.call_tir(cls.add2, (lv311, metadata["relax.expr.Constant"][129]), out_sinfo=R.Tensor((1, 128, 3072), dtype="float32"))
            lv313 = R.call_tir(cls.gelu, (lv312,), out_sinfo=R.Tensor((1, 128, 3072), dtype="float32"))
            lv315 = R.call_tir(cls.matmul2, (lv313, metadata["relax.expr.Constant"][130]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv316 = R.call_tir(cls.add1, (lv315, metadata["relax.expr.Constant"][131]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv317 = R.call_tir(cls.add, (lv316, lv309), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv318 = R.call_tir(cls.layer_norm, (lv317, metadata["relax.expr.Constant"][132], metadata["relax.expr.Constant"][133]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv320 = R.call_tir(cls.matmul, (lv318, metadata["relax.expr.Constant"][134]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv321 = R.call_tir(cls.add1, (lv320, metadata["relax.expr.Constant"][135]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv322 = R.call_tir(cls.reshape2, (lv321,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv323 = R.call_tir(cls.transpose1, (lv322,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv325 = R.call_tir(cls.matmul, (lv318, metadata["relax.expr.Constant"][136]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv326 = R.call_tir(cls.add1, (lv325, metadata["relax.expr.Constant"][137]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv327 = R.call_tir(cls.reshape2, (lv326,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv328 = R.call_tir(cls.transpose1, (lv327,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv330 = R.call_tir(cls.matmul, (lv318, metadata["relax.expr.Constant"][138]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv331 = R.call_tir(cls.add1, (lv330, metadata["relax.expr.Constant"][139]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv332 = R.call_tir(cls.reshape2, (lv331,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv333 = R.call_tir(cls.transpose1, (lv332,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv334 = R.call_tir(cls.transpose2, (lv323,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv335 = R.call_tir(cls.transpose2, (lv328,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv336 = R.call_tir(cls.transpose2, (lv333,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv337 = R.call_tir(cls.attention_bias, (lv334, lv335, lv336, metadata["relax.expr.Constant"][11]), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv338 = R.call_tir(cls.transpose1, (lv337,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv339 = R.call_tir(cls.transpose2, (lv338,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv340 = R.call_tir(cls.reshape3, (lv339,), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv342 = R.call_tir(cls.matmul, (lv340, metadata["relax.expr.Constant"][140]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv343 = R.call_tir(cls.add1, (lv342, metadata["relax.expr.Constant"][141]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv344 = R.call_tir(cls.add, (lv343, lv318), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv345 = R.call_tir(cls.layer_norm, (lv344, metadata["relax.expr.Constant"][142], metadata["relax.expr.Constant"][143]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv347 = R.call_tir(cls.matmul1, (lv345, metadata["relax.expr.Constant"][144]), out_sinfo=R.Tensor((1, 128, 3072), dtype="float32"))
            lv348 = R.call_tir(cls.add2, (lv347, metadata["relax.expr.Constant"][145]), out_sinfo=R.Tensor((1, 128, 3072), dtype="float32"))
            lv349 = R.call_tir(cls.gelu, (lv348,), out_sinfo=R.Tensor((1, 128, 3072), dtype="float32"))
            lv351 = R.call_tir(cls.matmul2, (lv349, metadata["relax.expr.Constant"][146]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv352 = R.call_tir(cls.add1, (lv351, metadata["relax.expr.Constant"][147]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv353 = R.call_tir(cls.add, (lv352, lv345), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv354 = R.call_tir(cls.layer_norm, (lv353, metadata["relax.expr.Constant"][148], metadata["relax.expr.Constant"][149]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv356 = R.call_tir(cls.matmul, (lv354, metadata["relax.expr.Constant"][150]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv357 = R.call_tir(cls.add1, (lv356, metadata["relax.expr.Constant"][151]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv358 = R.call_tir(cls.reshape2, (lv357,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv359 = R.call_tir(cls.transpose1, (lv358,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv361 = R.call_tir(cls.matmul, (lv354, metadata["relax.expr.Constant"][152]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv362 = R.call_tir(cls.add1, (lv361, metadata["relax.expr.Constant"][153]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv363 = R.call_tir(cls.reshape2, (lv362,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv364 = R.call_tir(cls.transpose1, (lv363,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv366 = R.call_tir(cls.matmul, (lv354, metadata["relax.expr.Constant"][154]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv367 = R.call_tir(cls.add1, (lv366, metadata["relax.expr.Constant"][155]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv368 = R.call_tir(cls.reshape2, (lv367,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv369 = R.call_tir(cls.transpose1, (lv368,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv370 = R.call_tir(cls.transpose2, (lv359,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv371 = R.call_tir(cls.transpose2, (lv364,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv372 = R.call_tir(cls.transpose2, (lv369,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv373 = R.call_tir(cls.attention_bias, (lv370, lv371, lv372, metadata["relax.expr.Constant"][11]), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv374 = R.call_tir(cls.transpose1, (lv373,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv375 = R.call_tir(cls.transpose2, (lv374,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv376 = R.call_tir(cls.reshape3, (lv375,), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv378 = R.call_tir(cls.matmul, (lv376, metadata["relax.expr.Constant"][156]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv379 = R.call_tir(cls.add1, (lv378, metadata["relax.expr.Constant"][157]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv380 = R.call_tir(cls.add, (lv379, lv354), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv381 = R.call_tir(cls.layer_norm, (lv380, metadata["relax.expr.Constant"][158], metadata["relax.expr.Constant"][159]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv383 = R.call_tir(cls.matmul1, (lv381, metadata["relax.expr.Constant"][160]), out_sinfo=R.Tensor((1, 128, 3072), dtype="float32"))
            lv384 = R.call_tir(cls.add2, (lv383, metadata["relax.expr.Constant"][161]), out_sinfo=R.Tensor((1, 128, 3072), dtype="float32"))
            lv385 = R.call_tir(cls.gelu, (lv384,), out_sinfo=R.Tensor((1, 128, 3072), dtype="float32"))
            lv387 = R.call_tir(cls.matmul2, (lv385, metadata["relax.expr.Constant"][162]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv388 = R.call_tir(cls.add1, (lv387, metadata["relax.expr.Constant"][163]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv389 = R.call_tir(cls.add, (lv388, lv381), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv390 = R.call_tir(cls.layer_norm, (lv389, metadata["relax.expr.Constant"][164], metadata["relax.expr.Constant"][165]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv392 = R.call_tir(cls.matmul, (lv390, metadata["relax.expr.Constant"][166]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv393 = R.call_tir(cls.add1, (lv392, metadata["relax.expr.Constant"][167]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv394 = R.call_tir(cls.reshape2, (lv393,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv395 = R.call_tir(cls.transpose1, (lv394,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv397 = R.call_tir(cls.matmul, (lv390, metadata["relax.expr.Constant"][168]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv398 = R.call_tir(cls.add1, (lv397, metadata["relax.expr.Constant"][169]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv399 = R.call_tir(cls.reshape2, (lv398,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv400 = R.call_tir(cls.transpose1, (lv399,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv402 = R.call_tir(cls.matmul, (lv390, metadata["relax.expr.Constant"][170]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv403 = R.call_tir(cls.add1, (lv402, metadata["relax.expr.Constant"][171]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv404 = R.call_tir(cls.reshape2, (lv403,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv405 = R.call_tir(cls.transpose1, (lv404,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv406 = R.call_tir(cls.transpose2, (lv395,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv407 = R.call_tir(cls.transpose2, (lv400,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv408 = R.call_tir(cls.transpose2, (lv405,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv409 = R.call_tir(cls.attention_bias, (lv406, lv407, lv408, metadata["relax.expr.Constant"][11]), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv410 = R.call_tir(cls.transpose1, (lv409,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv411 = R.call_tir(cls.transpose2, (lv410,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv412 = R.call_tir(cls.reshape3, (lv411,), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv414 = R.call_tir(cls.matmul, (lv412, metadata["relax.expr.Constant"][172]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv415 = R.call_tir(cls.add1, (lv414, metadata["relax.expr.Constant"][173]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv416 = R.call_tir(cls.add, (lv415, lv390), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv417 = R.call_tir(cls.layer_norm, (lv416, metadata["relax.expr.Constant"][174], metadata["relax.expr.Constant"][175]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv419 = R.call_tir(cls.matmul1, (lv417, metadata["relax.expr.Constant"][176]), out_sinfo=R.Tensor((1, 128, 3072), dtype="float32"))
            lv420 = R.call_tir(cls.add2, (lv419, metadata["relax.expr.Constant"][177]), out_sinfo=R.Tensor((1, 128, 3072), dtype="float32"))
            lv421 = R.call_tir(cls.gelu, (lv420,), out_sinfo=R.Tensor((1, 128, 3072), dtype="float32"))
            lv423 = R.call_tir(cls.matmul2, (lv421, metadata["relax.expr.Constant"][178]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv424 = R.call_tir(cls.add1, (lv423, metadata["relax.expr.Constant"][179]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv425 = R.call_tir(cls.add, (lv424, lv417), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv426 = R.call_tir(cls.layer_norm, (lv425, metadata["relax.expr.Constant"][180], metadata["relax.expr.Constant"][181]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv428 = R.call_tir(cls.matmul, (lv426, metadata["relax.expr.Constant"][182]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv429 = R.call_tir(cls.add1, (lv428, metadata["relax.expr.Constant"][183]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv430 = R.call_tir(cls.reshape2, (lv429,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv431 = R.call_tir(cls.transpose1, (lv430,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv433 = R.call_tir(cls.matmul, (lv426, metadata["relax.expr.Constant"][184]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv434 = R.call_tir(cls.add1, (lv433, metadata["relax.expr.Constant"][185]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv435 = R.call_tir(cls.reshape2, (lv434,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv436 = R.call_tir(cls.transpose1, (lv435,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv438 = R.call_tir(cls.matmul, (lv426, metadata["relax.expr.Constant"][186]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv439 = R.call_tir(cls.add1, (lv438, metadata["relax.expr.Constant"][187]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv440 = R.call_tir(cls.reshape2, (lv439,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv441 = R.call_tir(cls.transpose1, (lv440,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv442 = R.call_tir(cls.transpose2, (lv431,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv443 = R.call_tir(cls.transpose2, (lv436,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv444 = R.call_tir(cls.transpose2, (lv441,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv445 = R.call_tir(cls.attention_bias, (lv442, lv443, lv444, metadata["relax.expr.Constant"][11]), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv446 = R.call_tir(cls.transpose1, (lv445,), out_sinfo=R.Tensor((1, 12, 128, 64), dtype="float32"))
            lv447 = R.call_tir(cls.transpose2, (lv446,), out_sinfo=R.Tensor((1, 128, 12, 64), dtype="float32"))
            lv448 = R.call_tir(cls.reshape3, (lv447,), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv450 = R.call_tir(cls.matmul, (lv448, metadata["relax.expr.Constant"][188]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv451 = R.call_tir(cls.add1, (lv450, metadata["relax.expr.Constant"][189]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv452 = R.call_tir(cls.add, (lv451, lv426), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv453 = R.call_tir(cls.layer_norm, (lv452, metadata["relax.expr.Constant"][190], metadata["relax.expr.Constant"][191]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv455 = R.call_tir(cls.matmul1, (lv453, metadata["relax.expr.Constant"][192]), out_sinfo=R.Tensor((1, 128, 3072), dtype="float32"))
            lv456 = R.call_tir(cls.add2, (lv455, metadata["relax.expr.Constant"][193]), out_sinfo=R.Tensor((1, 128, 3072), dtype="float32"))
            lv457 = R.call_tir(cls.gelu, (lv456,), out_sinfo=R.Tensor((1, 128, 3072), dtype="float32"))
            lv459 = R.call_tir(cls.matmul2, (lv457, metadata["relax.expr.Constant"][194]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv460 = R.call_tir(cls.add1, (lv459, metadata["relax.expr.Constant"][195]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv461 = R.call_tir(cls.add, (lv460, lv453), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            lv462 = R.call_tir(cls.layer_norm, (lv461, metadata["relax.expr.Constant"][196], metadata["relax.expr.Constant"][197]), out_sinfo=R.Tensor((1, 128, 768), dtype="float32"))
            gv: R.Tuple(R.Tensor((1, 128, 768), dtype="float32")) = (lv462,)
            R.output(gv)
        return gv

# Metadata omitted. Use show_meta=True in script() method to show it.