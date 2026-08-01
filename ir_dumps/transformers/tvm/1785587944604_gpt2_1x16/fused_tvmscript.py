# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def add(lv4: T.Buffer((T.int64(1), T.int64(16), T.int64(768)), "float32"), lv11: T.Buffer((T.int64(1), T.int64(16), T.int64(768)), "float32"), T_add: T.Buffer((T.int64(1), T.int64(16), T.int64(768)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(12), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(768))
                    v1 = T.axis.spatial(T.int64(768), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(768))
                    T.reads(lv4[T.int64(0), v0, v1], lv11[T.int64(0), v0, v1])
                    T.writes(T_add[T.int64(0), v0, v1])
                    T_add[T.int64(0), v0, v1] = lv4[T.int64(0), v0, v1] + lv11[T.int64(0), v0, v1]

    @T.prim_func(private=True)
    def add1(A: T.Buffer((T.int64(2304),), "float32"), lv15: T.Buffer((T.int64(16), T.int64(2304)), "float32"), T_add: T.Buffer((T.int64(16), T.int64(2304)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(36), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(2304))
                    v1 = T.axis.spatial(T.int64(2304), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(2304))
                    T.reads(A[v1], lv15[v0, v1])
                    T.writes(T_add[v0, v1])
                    T_add[v0, v1] = A[v1] + lv15[v0, v1]

    @T.prim_func(private=True)
    def add2(A: T.Buffer((T.int64(768),), "float32"), lv39: T.Buffer((T.int64(16), T.int64(768)), "float32"), T_add: T.Buffer((T.int64(16), T.int64(768)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(12), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(768))
                    v1 = T.axis.spatial(T.int64(768), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(768))
                    T.reads(A[v1], lv39[v0, v1])
                    T.writes(T_add[v0, v1])
                    T_add[v0, v1] = A[v1] + lv39[v0, v1]

    @T.prim_func(private=True)
    def add3(A: T.Buffer((T.int64(3072),), "float32"), lv45: T.Buffer((T.int64(16), T.int64(3072)), "float32"), T_add: T.Buffer((T.int64(16), T.int64(3072)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(48), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(3072))
                    v1 = T.axis.spatial(T.int64(3072), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(3072))
                    T.reads(A[v1], lv45[v0, v1])
                    T.writes(T_add[v0, v1])
                    T_add[v0, v1] = A[v1] + lv45[v0, v1]

    @T.prim_func(private=True)
    def add4(lv47: T.Buffer((T.int64(1), T.int64(16), T.int64(3072)), "float32"), lv50: T.Buffer((T.int64(1), T.int64(16), T.int64(3072)), "float32"), T_add: T.Buffer((T.int64(1), T.int64(16), T.int64(3072)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(48), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(3072))
                    v1 = T.axis.spatial(T.int64(3072), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(3072))
                    T.reads(lv47[T.int64(0), v0, v1], lv50[T.int64(0), v0, v1])
                    T.writes(T_add[T.int64(0), v0, v1])
                    T_add[T.int64(0), v0, v1] = lv47[T.int64(0), v0, v1] + lv50[T.int64(0), v0, v1]

    @T.prim_func(private=True)
    def add5(lv53: T.Buffer((T.int64(1), T.int64(16), T.int64(3072)), "float32"), T_add: T.Buffer((T.int64(1), T.int64(16), T.int64(3072)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(48), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_add"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(3072))
                    v1 = T.axis.spatial(T.int64(3072), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(3072))
                    T.reads(lv53[T.int64(0), v0, v1])
                    T.writes(T_add[T.int64(0), v0, v1])
                    T_add[T.int64(0), v0, v1] = lv53[T.int64(0), v0, v1] + T.float32(1.0)

    @T.prim_func(private=True)
    def attention_bias(lv31: T.Buffer((T.int64(1), T.int64(16), T.int64(12), T.int64(64)), "float32"), lv32: T.Buffer((T.int64(1), T.int64(16), T.int64(12), T.int64(64)), "float32"), lv33: T.Buffer((T.int64(1), T.int64(16), T.int64(12), T.int64(64)), "float32"), lv30: T.Buffer((T.int64(1), T.int64(1), T.int64(16), T.int64(16)), "float32"), T_transpose: T.Buffer((T.int64(1), T.int64(16), T.int64(12), T.int64(64)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        T_batch_matmul_NT = T.alloc_buffer((T.int64(12), T.int64(16), T.int64(16)))
        T_softmax_maxelem = T.alloc_buffer((T.int64(12), T.int64(16)))
        T_softmax_expsum = T.alloc_buffer((T.int64(12), T.int64(16)))
        T_batch_matmul_NN = T.alloc_buffer((T.int64(12), T.int64(16), T.int64(64)))
        for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(3), thread="blockIdx.x"):
            for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_batch_matmul_NT_init"):
                    v0 = T.axis.spatial(T.int64(12), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) // T.int64(256))
                    v1 = T.axis.spatial(T.int64(16), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(256) // T.int64(16))
                    v2 = T.axis.spatial(T.int64(16), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(16))
                    T.reads()
                    T.writes(T_batch_matmul_NT[v0, v1, v2])
                    T_reshape = T.Buffer((T.int64(12), T.int64(16), T.int64(64)))
                    T.block_attr({"layout_free_placeholders": [T_reshape]})
                    T_batch_matmul_NT[v0, v1, v2] = T.float32(0.0)
                T_reshape = T.Buffer((T.int64(12), T.int64(16), T.int64(64)))
                for ax3 in range(T.int64(64)):
                    with T.block("T_batch_matmul_NT_update"):
                        v0 = T.axis.spatial(T.int64(12), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) // T.int64(256))
                        v1 = T.axis.spatial(T.int64(16), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(256) // T.int64(16))
                        v2 = T.axis.spatial(T.int64(16), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(16))
                        v3 = T.axis.reduce(T.int64(64), ax3)
                        T.reads(T_batch_matmul_NT[v0, v1, v2], lv31[T.int64(0), v1, v0, v3], lv32[T.int64(0), v2, v0, v3])
                        T.writes(T_batch_matmul_NT[v0, v1, v2])
                        T.block_attr({"layout_free_placeholders": [T_reshape]})
                        T_batch_matmul_NT[v0, v1, v2] = T_batch_matmul_NT[v0, v1, v2] + lv31[T.int64(0), v1, v0, v3] * lv32[T.int64(0), v2, v0, v3]
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_softmax_maxelem_init"):
                    v0 = T.axis.spatial(T.int64(12), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(16))
                    v1 = T.axis.spatial(T.int64(16), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(16))
                    T.where(ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 < T.int64(192))
                    T.reads()
                    T.writes(T_softmax_maxelem[v0, v1])
                    T_softmax_maxelem[v0, v1] = T.float32(-340282346638528859811704183484516925440.0)
                for ax2 in range(T.int64(16)):
                    with T.block("T_softmax_maxelem_update"):
                        v0 = T.axis.spatial(T.int64(12), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(16))
                        v1 = T.axis.spatial(T.int64(16), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(16))
                        v2 = T.axis.reduce(T.int64(16), ax2)
                        T.where(ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 < T.int64(192))
                        T.reads(T_softmax_maxelem[v0, v1], T_batch_matmul_NT[v0, v1, v2], lv30[T.int64(0), T.int64(0), v1, v2])
                        T.writes(T_softmax_maxelem[v0, v1])
                        T_softmax_maxelem[v0, v1] = T.max(T_softmax_maxelem[v0, v1], T_batch_matmul_NT[v0, v1, v2] / T.sqrt(T.float32(64.0)) + lv30[T.int64(0), T.int64(0), v1, v2])
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_softmax_expsum_init"):
                    v0 = T.axis.spatial(T.int64(12), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(16))
                    v1 = T.axis.spatial(T.int64(16), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(16))
                    T.where(ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 < T.int64(192))
                    T.reads()
                    T.writes(T_softmax_expsum[v0, v1])
                    T_softmax_expsum[v0, v1] = T.float32(0.0)
                for ax2 in range(T.int64(16)):
                    with T.block("T_softmax_expsum_update"):
                        v0 = T.axis.spatial(T.int64(12), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(16))
                        v1 = T.axis.spatial(T.int64(16), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(16))
                        v2 = T.axis.reduce(T.int64(16), ax2)
                        T.where(ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 < T.int64(192))
                        T.reads(T_softmax_expsum[v0, v1], T_batch_matmul_NT[v0, v1, v2], lv30[T.int64(0), T.int64(0), v1, v2], T_softmax_maxelem[v0, v1])
                        T.writes(T_softmax_expsum[v0, v1])
                        T_softmax_expsum[v0, v1] = T_softmax_expsum[v0, v1] + T.exp(T_batch_matmul_NT[v0, v1, v2] / T.sqrt(T.float32(64.0)) + lv30[T.int64(0), T.int64(0), v1, v2] - T_softmax_maxelem[v0, v1])
        for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(12), thread="blockIdx.x"):
            for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_batch_matmul_NN_init"):
                    v0 = T.axis.spatial(T.int64(12), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) // T.int64(1024))
                    v1 = T.axis.spatial(T.int64(16), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(1024) // T.int64(64))
                    v2 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(64))
                    T.reads()
                    T.writes(T_batch_matmul_NN[v0, v1, v2])
                    T_reshape = T.Buffer((T.int64(12), T.int64(16), T.int64(64)))
                    T.block_attr({"layout_free_placeholders": [T_reshape]})
                    T_batch_matmul_NN[v0, v1, v2] = T.float32(0.0)
                T_reshape = T.Buffer((T.int64(12), T.int64(16), T.int64(64)))
                for ax3 in range(T.int64(16)):
                    with T.block("T_batch_matmul_NN_update"):
                        v0 = T.axis.spatial(T.int64(12), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) // T.int64(1024))
                        v1 = T.axis.spatial(T.int64(16), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(1024) // T.int64(64))
                        v2 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(64))
                        v3 = T.axis.reduce(T.int64(16), ax3)
                        T.reads(T_batch_matmul_NN[v0, v1, v2], T_batch_matmul_NT[v0, v1, v3], lv30[T.int64(0), T.int64(0), v1, v3], T_softmax_maxelem[v0, v1], T_softmax_expsum[v0, v1], lv33[T.int64(0), v3, v0, v2])
                        T.writes(T_batch_matmul_NN[v0, v1, v2])
                        T.block_attr({"layout_free_placeholders": [T_reshape]})
                        T_batch_matmul_NN[v0, v1, v2] = T_batch_matmul_NN[v0, v1, v2] + T.exp(T_batch_matmul_NT[v0, v1, v3] / T.sqrt(T.float32(64.0)) + lv30[T.int64(0), T.int64(0), v1, v3] - T_softmax_maxelem[v0, v1]) / T_softmax_expsum[v0, v1] * lv33[T.int64(0), v3, v0, v2]
        for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(12), thread="blockIdx.x"):
            for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_transpose_3"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) // T.int64(768))
                    v1 = T.axis.spatial(T.int64(12), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(768) // T.int64(64))
                    v2 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(64))
                    T.reads(T_batch_matmul_NN[v1, v0, v2])
                    T.writes(T_transpose[T.int64(0), v0, v1, v2])
                    T_transpose[T.int64(0), v0, v1, v2] = T_batch_matmul_NN[v1, v0, v2]

    @T.prim_func(private=True)
    def cast(lv: T.Buffer((T.int64(1), T.int64(16)), "int64"), compute: T.Buffer((T.int64(1), T.int64(16)), "int32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v0 = T.axis.spatial(T.int64(16), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(16))
                    T.reads(lv[T.int64(0), v0])
                    T.writes(compute[T.int64(0), v0])
                    compute[T.int64(0), v0] = T.Cast("int32", lv[T.int64(0), v0])

    @T.prim_func(private=True)
    def layer_norm(lv12: T.Buffer((T.int64(1), T.int64(16), T.int64(768)), "float32"), B: T.Buffer((T.int64(768),), "float32"), C: T.Buffer((T.int64(768),), "float32"), T_layer_norm: T.Buffer((T.int64(1), T.int64(16), T.int64(768)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        lv12_red_temp_v0_shared = T.alloc_buffer((T.int64(1), T.int64(16)), scope="shared")
        lv12_red_temp_v1_shared = T.alloc_buffer((T.int64(1), T.int64(16)), scope="shared")
        for ax0_fused in T.thread_binding(T.int64(16), thread="blockIdx.x"):
            for ax0 in range(T.int64(1)):
                for ax1_fused_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                    for ax1_fused_0 in T.serial(T.int64(3), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                        with T.block("lv12_red_temp"):
                            v0 = T.axis.spatial(T.int64(16), ax0_fused + ax0)
                            v1 = T.axis.reduce(T.int64(768), ax1_fused_0 * T.int64(256) + ax1_fused_1)
                            T.reads(lv12[T.int64(0), v0, v1])
                            T.writes(lv12_red_temp_v0_shared[T.int64(0), v0], lv12_red_temp_v1_shared[T.int64(0), v0])
                            with T.init():
                                lv12_red_temp_v0_shared[T.int64(0), v0] = T.float32(0.0)
                                lv12_red_temp_v1_shared[T.int64(0), v0] = T.float32(0.0)
                            v_lv12_red_temp_v0: T.float32 = lv12_red_temp_v0_shared[T.int64(0), v0] + lv12[T.int64(0), v0, v1]
                            v_lv12_red_temp_v1: T.float32 = lv12_red_temp_v1_shared[T.int64(0), v0] + lv12[T.int64(0), v0, v1] * lv12[T.int64(0), v0, v1]
                            lv12_red_temp_v0_shared[T.int64(0), v0] = v_lv12_red_temp_v0
                            lv12_red_temp_v1_shared[T.int64(0), v0] = v_lv12_red_temp_v1
            for ax1_1 in T.thread_binding(T.int64(256), thread="threadIdx.x"):
                for ax1_0 in T.serial(T.int64(3), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    with T.block("T_layer_norm"):
                        v0 = T.axis.spatial(T.int64(16), ax0_fused)
                        v1 = T.axis.spatial(T.int64(768), ax1_0 * T.int64(256) + ax1_1)
                        T.reads(lv12[T.int64(0), v0, v1], lv12_red_temp_v0_shared[T.int64(0), v0], lv12_red_temp_v1_shared[T.int64(0), v0], B[v1], C[v1])
                        T.writes(T_layer_norm[T.int64(0), v0, v1])
                        T_layer_norm[T.int64(0), v0, v1] = (lv12[T.int64(0), v0, v1] - lv12_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.0013020833333333333)) * T.rsqrt(lv12_red_temp_v1_shared[T.int64(0), v0] * T.float32(0.0013020833333333333) - lv12_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.0013020833333333333) * (lv12_red_temp_v0_shared[T.int64(0), v0] * T.float32(0.0013020833333333333)) + T.float32(1.0000000000000001e-05)) * B[v1] + C[v1]

    @T.prim_func(private=True)
    def matmul(lv14: T.Buffer((T.int64(16), T.int64(768)), "float32"), B: T.Buffer((T.int64(768), T.int64(2304)), "float32"), matmul: T.Buffer((T.int64(16), T.int64(2304)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        matmul_reindex_pad_local = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(2304)), scope="local")
        lv14_reindex_pad_shared = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(768)), scope="shared")
        B_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(2304), T.int64(768)), scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(36), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with T.block("matmul_init"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(32), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = T.axis.spatial(T.int64(2304), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            T.reads()
                                            T.writes(matmul_reindex_pad_local[T.int64(0), v1, v2])
                                            matmul_reindex_pad_local[T.int64(0), v1, v2] = T.float32(0.0)
                                for ax3_0 in range(T.int64(48)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("lv14_reindex_pad_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(32), (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(768), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv14[v1, v2])
                                                        T.writes(lv14_reindex_pad_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        lv14_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v1 < T.int64(16), lv14[v1, v2], T.float32(0.0))
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("B_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(2304), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(768), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(B[v2, v1])
                                                        T.writes(B_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        B_reindex_shared[v0, v1, v2] = B[v2, v1]
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with T.block("matmul_update"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial(T.int64(32), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(2304), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(768), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(matmul_reindex_pad_local[T.int64(0), v1, v2], lv14_reindex_pad_shared[T.int64(0), v1, v3], B_reindex_shared[T.int64(0), v2, v3])
                                                T.writes(matmul_reindex_pad_local[T.int64(0), v1, v2])
                                                matmul_reindex_pad_local[T.int64(0), v1, v2] = matmul_reindex_pad_local[T.int64(0), v1, v2] + lv14_reindex_pad_shared[T.int64(0), v1, v3] * B_reindex_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("matmul_reindex_pad_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(T.int64(32), ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(2304), ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            T.where(ax1_2 * T.int64(4) + ax1 < T.int64(16))
                                            T.reads(matmul_reindex_pad_local[v0, v1, v2])
                                            T.writes(matmul[v1, v2])
                                            matmul[v1, v2] = matmul_reindex_pad_local[v0, v1, v2]

    @T.prim_func(private=True)
    def matmul1(lv38: T.Buffer((T.int64(16), T.int64(768)), "float32"), B: T.Buffer((T.int64(768), T.int64(768)), "float32"), matmul: T.Buffer((T.int64(16), T.int64(768)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        matmul_reindex_pad_local = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(768)), scope="local")
        lv38_reindex_pad_shared = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(768)), scope="shared")
        B_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(768), T.int64(768)), scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(12), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with T.block("matmul_init"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(32), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = T.axis.spatial(T.int64(768), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            T.reads()
                                            T.writes(matmul_reindex_pad_local[T.int64(0), v1, v2])
                                            matmul_reindex_pad_local[T.int64(0), v1, v2] = T.float32(0.0)
                                for ax3_0 in range(T.int64(48)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("lv38_reindex_pad_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(32), (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(768), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv38[v1, v2])
                                                        T.writes(lv38_reindex_pad_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        lv38_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v1 < T.int64(16), lv38[v1, v2], T.float32(0.0))
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("B_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(768), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(768), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(B[v2, v1])
                                                        T.writes(B_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        B_reindex_shared[v0, v1, v2] = B[v2, v1]
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with T.block("matmul_update"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial(T.int64(32), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(768), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(768), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(matmul_reindex_pad_local[T.int64(0), v1, v2], lv38_reindex_pad_shared[T.int64(0), v1, v3], B_reindex_shared[T.int64(0), v2, v3])
                                                T.writes(matmul_reindex_pad_local[T.int64(0), v1, v2])
                                                matmul_reindex_pad_local[T.int64(0), v1, v2] = matmul_reindex_pad_local[T.int64(0), v1, v2] + lv38_reindex_pad_shared[T.int64(0), v1, v3] * B_reindex_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("matmul_reindex_pad_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(T.int64(32), ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(768), ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            T.where(ax1_2 * T.int64(4) + ax1 < T.int64(16))
                                            T.reads(matmul_reindex_pad_local[v0, v1, v2])
                                            T.writes(matmul[v1, v2])
                                            matmul[v1, v2] = matmul_reindex_pad_local[v0, v1, v2]

    @T.prim_func(private=True)
    def matmul2(lv44: T.Buffer((T.int64(16), T.int64(768)), "float32"), B: T.Buffer((T.int64(768), T.int64(3072)), "float32"), matmul: T.Buffer((T.int64(16), T.int64(3072)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        matmul_reindex_pad_local = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(3072)), scope="local")
        lv44_reindex_pad_shared = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(768)), scope="shared")
        B_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(3072), T.int64(768)), scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(48), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with T.block("matmul_init"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(32), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = T.axis.spatial(T.int64(3072), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            T.reads()
                                            T.writes(matmul_reindex_pad_local[T.int64(0), v1, v2])
                                            matmul_reindex_pad_local[T.int64(0), v1, v2] = T.float32(0.0)
                                for ax3_0 in range(T.int64(48)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("lv44_reindex_pad_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(32), (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(768), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv44[v1, v2])
                                                        T.writes(lv44_reindex_pad_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        lv44_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v1 < T.int64(16), lv44[v1, v2], T.float32(0.0))
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("B_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(3072), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(768), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(B[v2, v1])
                                                        T.writes(B_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        B_reindex_shared[v0, v1, v2] = B[v2, v1]
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with T.block("matmul_update"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial(T.int64(32), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(3072), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(768), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(matmul_reindex_pad_local[T.int64(0), v1, v2], lv44_reindex_pad_shared[T.int64(0), v1, v3], B_reindex_shared[T.int64(0), v2, v3])
                                                T.writes(matmul_reindex_pad_local[T.int64(0), v1, v2])
                                                matmul_reindex_pad_local[T.int64(0), v1, v2] = matmul_reindex_pad_local[T.int64(0), v1, v2] + lv44_reindex_pad_shared[T.int64(0), v1, v3] * B_reindex_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("matmul_reindex_pad_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(T.int64(32), ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(3072), ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            T.where(ax1_2 * T.int64(4) + ax1 < T.int64(16))
                                            T.reads(matmul_reindex_pad_local[v0, v1, v2])
                                            T.writes(matmul[v1, v2])
                                            matmul[v1, v2] = matmul_reindex_pad_local[v0, v1, v2]

    @T.prim_func(private=True)
    def matmul3(lv56: T.Buffer((T.int64(16), T.int64(3072)), "float32"), B: T.Buffer((T.int64(3072), T.int64(768)), "float32"), matmul: T.Buffer((T.int64(16), T.int64(768)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        matmul_reindex_pad_local = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(768)), scope="local")
        lv56_reindex_pad_shared = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(3072)), scope="shared")
        B_reindex_shared = T.alloc_buffer((T.int64(1), T.int64(768), T.int64(3072)), scope="shared")
        for ax0_ax2_0_fused in T.thread_binding(T.int64(12), thread="blockIdx.y"):
            for ax1_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for ax2_1 in T.thread_binding(T.int64(1), thread="vthread.y"):
                    for ax1_1 in T.thread_binding(T.int64(1), thread="vthread.x"):
                        for ax2_2 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(T.int64(8), thread="threadIdx.x", annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with T.block("matmul_init"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(32), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = T.axis.spatial(T.int64(768), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            T.reads()
                                            T.writes(matmul_reindex_pad_local[T.int64(0), v1, v2])
                                            matmul_reindex_pad_local[T.int64(0), v1, v2] = T.float32(0.0)
                                for ax3_0 in range(T.int64(192)):
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("lv56_reindex_pad_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(32), (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(3072), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(lv56[v1, v2])
                                                        T.writes(lv56_reindex_pad_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        lv56_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v1 < T.int64(16), lv56[v1, v2], T.float32(0.0))
                                    for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with T.block("B_reindex_shared"):
                                                        v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = T.axis.spatial(T.int64(768), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = T.axis.spatial(T.int64(3072), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        T.reads(B[v2, v1])
                                                        T.writes(B_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"buffer_dim_align": [[0, 1, 8, 2]]})
                                                        B_reindex_shared[v0, v1, v2] = B[v2, v1]
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with T.block("matmul_update"):
                                                v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = T.axis.spatial(T.int64(32), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = T.axis.spatial(T.int64(768), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = T.axis.reduce(T.int64(3072), ax3_0 * T.int64(16) + ax3_1)
                                                T.reads(matmul_reindex_pad_local[T.int64(0), v1, v2], lv56_reindex_pad_shared[T.int64(0), v1, v3], B_reindex_shared[T.int64(0), v2, v3])
                                                T.writes(matmul_reindex_pad_local[T.int64(0), v1, v2])
                                                matmul_reindex_pad_local[T.int64(0), v1, v2] = matmul_reindex_pad_local[T.int64(0), v1, v2] + lv56_reindex_pad_shared[T.int64(0), v1, v3] * B_reindex_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with T.block("matmul_reindex_pad_local"):
                                            v0 = T.axis.spatial(T.int64(1), ax0)
                                            v1 = T.axis.spatial(T.int64(32), ax1_2 * T.int64(4) + ax1)
                                            v2 = T.axis.spatial(T.int64(768), ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            T.where(ax1_2 * T.int64(4) + ax1 < T.int64(16))
                                            T.reads(matmul_reindex_pad_local[v0, v1, v2])
                                            T.writes(matmul[v1, v2])
                                            matmul[v1, v2] = matmul_reindex_pad_local[v0, v1, v2]

    @T.prim_func(private=True)
    def multiply(lv47: T.Buffer((T.int64(1), T.int64(16), T.int64(3072)), "float32"), T_multiply: T.Buffer((T.int64(1), T.int64(16), T.int64(3072)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(48), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(3072))
                    v1 = T.axis.spatial(T.int64(3072), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(3072))
                    T.reads(lv47[T.int64(0), v0, v1])
                    T.writes(T_multiply[T.int64(0), v0, v1])
                    T_multiply[T.int64(0), v0, v1] = lv47[T.int64(0), v0, v1] * T.float32(0.5)

    @T.prim_func(private=True)
    def multiply1(lv49: T.Buffer((T.int64(1), T.int64(16), T.int64(3072)), "float32"), T_multiply: T.Buffer((T.int64(1), T.int64(16), T.int64(3072)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(48), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(3072))
                    v1 = T.axis.spatial(T.int64(3072), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(3072))
                    T.reads(lv49[T.int64(0), v0, v1])
                    T.writes(T_multiply[T.int64(0), v0, v1])
                    T_multiply[T.int64(0), v0, v1] = lv49[T.int64(0), v0, v1] * T.float32(0.044714998453855515)

    @T.prim_func(private=True)
    def multiply2(lv51: T.Buffer((T.int64(1), T.int64(16), T.int64(3072)), "float32"), T_multiply: T.Buffer((T.int64(1), T.int64(16), T.int64(3072)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(48), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(3072))
                    v1 = T.axis.spatial(T.int64(3072), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(3072))
                    T.reads(lv51[T.int64(0), v0, v1])
                    T.writes(T_multiply[T.int64(0), v0, v1])
                    T_multiply[T.int64(0), v0, v1] = lv51[T.int64(0), v0, v1] * T.float32(0.79788458347320557)

    @T.prim_func(private=True)
    def multiply3(lv48: T.Buffer((T.int64(1), T.int64(16), T.int64(3072)), "float32"), lv54: T.Buffer((T.int64(1), T.int64(16), T.int64(3072)), "float32"), T_multiply: T.Buffer((T.int64(1), T.int64(16), T.int64(3072)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(48), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_multiply"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(3072))
                    v1 = T.axis.spatial(T.int64(3072), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(3072))
                    T.reads(lv48[T.int64(0), v0, v1], lv54[T.int64(0), v0, v1])
                    T.writes(T_multiply[T.int64(0), v0, v1])
                    T_multiply[T.int64(0), v0, v1] = lv48[T.int64(0), v0, v1] * lv54[T.int64(0), v0, v1]

    @T.prim_func(private=True)
    def power(lv47: T.Buffer((T.int64(1), T.int64(16), T.int64(3072)), "float32"), T_power: T.Buffer((T.int64(1), T.int64(16), T.int64(3072)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(48), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_power"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(3072))
                    v1 = T.axis.spatial(T.int64(3072), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(3072))
                    T.reads(lv47[T.int64(0), v0, v1])
                    T.writes(T_power[T.int64(0), v0, v1])
                    T_power[T.int64(0), v0, v1] = T.pow(lv47[T.int64(0), v0, v1], T.float32(3.0))

    @T.prim_func(private=True)
    def reshape(input_ids: T.Buffer((T.int64(1), T.int64(16)), "int64"), T_reshape: T.Buffer((T.int64(1), T.int64(16)), "int64")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_reshape"):
                    v0 = T.axis.spatial(T.int64(16), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(16))
                    T.reads(input_ids[T.int64(0), v0])
                    T.writes(T_reshape[T.int64(0), v0])
                    T_reshape[T.int64(0), v0] = input_ids[T.int64(0), v0]

    @T.prim_func(private=True)
    def reshape1(lv1: T.Buffer((T.int64(1), T.int64(16)), "int32"), T_reshape: T.Buffer((T.int64(16),), "int32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
            for ax0_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_reshape"):
                    v0 = T.axis.spatial(T.int64(16), ax0_fused_0 * T.int64(1024) + ax0_fused_1)
                    T.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(16))
                    T.reads(lv1[T.int64(0), v0])
                    T.writes(T_reshape[v0])
                    T_reshape[v0] = lv1[T.int64(0), v0]

    @T.prim_func(private=True)
    def reshape2(lv3: T.Buffer((T.int64(16), T.int64(768)), "float32"), T_reshape: T.Buffer((T.int64(1), T.int64(16), T.int64(768)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(12), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_reshape"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(768))
                    v1 = T.axis.spatial(T.int64(768), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(768))
                    T.reads(lv3[v0, v1])
                    T.writes(T_reshape[T.int64(0), v0, v1])
                    T_reshape[T.int64(0), v0, v1] = lv3[v0, v1]

    @T.prim_func(private=True)
    def reshape3(lv13: T.Buffer((T.int64(1), T.int64(16), T.int64(768)), "float32"), T_reshape: T.Buffer((T.int64(16), T.int64(768)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(12), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_reshape"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(768))
                    v1 = T.axis.spatial(T.int64(768), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(768))
                    T.reads(lv13[T.int64(0), v0, v1])
                    T.writes(T_reshape[v0, v1])
                    T_reshape[v0, v1] = lv13[T.int64(0), v0, v1]

    @T.prim_func(private=True)
    def reshape4(lv16: T.Buffer((T.int64(16), T.int64(2304)), "float32"), T_reshape: T.Buffer((T.int64(1), T.int64(16), T.int64(2304)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(36), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_reshape"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(2304))
                    v1 = T.axis.spatial(T.int64(2304), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(2304))
                    T.reads(lv16[v0, v1])
                    T.writes(T_reshape[T.int64(0), v0, v1])
                    T_reshape[T.int64(0), v0, v1] = lv16[v0, v1]

    @T.prim_func(private=True)
    def reshape5(lv20: T.Buffer((T.int64(1), T.int64(16), T.int64(768)), "float32"), T_reshape: T.Buffer((T.int64(1), T.int64(16), T.int64(12), T.int64(64)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(12), thread="blockIdx.x"):
            for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_reshape"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) // T.int64(768))
                    v1 = T.axis.spatial(T.int64(12), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(768) // T.int64(64))
                    v2 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(64))
                    T.reads(lv20[T.int64(0), v0, v1 * T.int64(64) + v2])
                    T.writes(T_reshape[T.int64(0), v0, v1, v2])
                    T_reshape[T.int64(0), v0, v1, v2] = lv20[T.int64(0), v0, v1 * T.int64(64) + v2]

    @T.prim_func(private=True)
    def reshape6(lv36: T.Buffer((T.int64(1), T.int64(16), T.int64(12), T.int64(64)), "float32"), T_reshape: T.Buffer((T.int64(1), T.int64(16), T.int64(768)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(12), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_reshape"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(768))
                    v1 = T.axis.spatial(T.int64(768), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(768))
                    T.reads(lv36[T.int64(0), v0, v1 // T.int64(64), v1 % T.int64(64)])
                    T.writes(T_reshape[T.int64(0), v0, v1])
                    T_reshape[T.int64(0), v0, v1] = lv36[T.int64(0), v0, v1 // T.int64(64), v1 % T.int64(64)]

    @T.prim_func(private=True)
    def reshape7(lv46: T.Buffer((T.int64(16), T.int64(3072)), "float32"), T_reshape: T.Buffer((T.int64(1), T.int64(16), T.int64(3072)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(48), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_reshape"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(3072))
                    v1 = T.axis.spatial(T.int64(3072), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(3072))
                    T.reads(lv46[v0, v1])
                    T.writes(T_reshape[T.int64(0), v0, v1])
                    T_reshape[T.int64(0), v0, v1] = lv46[v0, v1]

    @T.prim_func(private=True)
    def reshape8(lv55: T.Buffer((T.int64(1), T.int64(16), T.int64(3072)), "float32"), T_reshape: T.Buffer((T.int64(16), T.int64(3072)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(48), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_reshape"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(3072))
                    v1 = T.axis.spatial(T.int64(3072), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(3072))
                    T.reads(lv55[T.int64(0), v0, v1])
                    T.writes(T_reshape[v0, v1])
                    T_reshape[v0, v1] = lv55[T.int64(0), v0, v1]

    @T.prim_func(private=True)
    def reshape9(lv589: T.Buffer((T.int64(1), T.int64(16), T.int64(768)), "float32"), T_reshape: T.Buffer((T.int64(1), T.int64(16), T.int64(768)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(12), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_reshape"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(768))
                    v1 = T.axis.spatial(T.int64(768), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(768))
                    T.reads(lv589[T.int64(0), v0, v1])
                    T.writes(T_reshape[T.int64(0), v0, v1])
                    T_reshape[T.int64(0), v0, v1] = lv589[T.int64(0), v0, v1]

    @T.prim_func(private=True)
    def split(lv17: T.Buffer((T.int64(1), T.int64(16), T.int64(2304)), "float32"), T_split_sections: T.Buffer((T.int64(1), T.int64(16), T.int64(768)), "float32"), T_split_sections_1: T.Buffer((T.int64(1), T.int64(16), T.int64(768)), "float32"), T_split_sections_2: T.Buffer((T.int64(1), T.int64(16), T.int64(768)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(12), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_split_sections"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(768))
                    v1 = T.axis.spatial(T.int64(768), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(768))
                    T.reads(lv17[T.int64(0), v0, v1])
                    T.writes(T_split_sections[T.int64(0), v0, v1])
                    T_split_sections[T.int64(0), v0, v1] = lv17[T.int64(0), v0, v1]
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(12), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_split_sections_1"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(768))
                    v1 = T.axis.spatial(T.int64(768), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(768))
                    T.reads(lv17[T.int64(0), v0, v1 + T.int64(768)])
                    T.writes(T_split_sections_1[T.int64(0), v0, v1])
                    T_split_sections_1[T.int64(0), v0, v1] = lv17[T.int64(0), v0, v1 + T.int64(768)]
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(12), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_split_sections_2"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(768))
                    v1 = T.axis.spatial(T.int64(768), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(768))
                    T.reads(lv17[T.int64(0), v0, v1 + T.int64(1536)])
                    T.writes(T_split_sections_2[T.int64(0), v0, v1])
                    T_split_sections_2[T.int64(0), v0, v1] = lv17[T.int64(0), v0, v1 + T.int64(1536)]

    @T.prim_func(private=True)
    def take(A: T.Buffer((T.int64(50257), T.int64(768)), "float32"), lv2: T.Buffer((T.int64(16),), "int32"), T_take: T.Buffer((T.int64(16), T.int64(768)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(12), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_take"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(768))
                    v1 = T.axis.spatial(T.int64(768), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(768))
                    T.reads(A[lv2[v0], v1], lv2[v0])
                    T.writes(T_take[v0, v1])
                    T_take[v0, v1] = A[lv2[v0], v1]

    @T.prim_func(private=True)
    def tir_tanh(lv52: T.Buffer((T.int64(1), T.int64(16), T.int64(3072)), "float32"), compute: T.Buffer((T.int64(1), T.int64(16), T.int64(3072)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_fused_0 in T.thread_binding(T.int64(48), thread="blockIdx.x"):
            for ax0_ax1_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("compute"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) // T.int64(3072))
                    v1 = T.axis.spatial(T.int64(3072), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1) % T.int64(3072))
                    T.reads(lv52[T.int64(0), v0, v1])
                    T.writes(compute[T.int64(0), v0, v1])
                    compute[T.int64(0), v0, v1] = T.tanh(lv52[T.int64(0), v0, v1])

    @T.prim_func(private=True)
    def transpose(lv22: T.Buffer((T.int64(1), T.int64(16), T.int64(12), T.int64(64)), "float32"), T_transpose: T.Buffer((T.int64(1), T.int64(12), T.int64(16), T.int64(64)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(12), thread="blockIdx.x"):
            for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_transpose"):
                    v0 = T.axis.spatial(T.int64(12), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) // T.int64(1024))
                    v1 = T.axis.spatial(T.int64(16), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(1024) // T.int64(64))
                    v2 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(64))
                    T.reads(lv22[T.int64(0), v1, v0, v2])
                    T.writes(T_transpose[T.int64(0), v0, v1, v2])
                    T_transpose[T.int64(0), v0, v1, v2] = lv22[T.int64(0), v1, v0, v2]

    @T.prim_func(private=True)
    def transpose1(lv27: T.Buffer((T.int64(1), T.int64(12), T.int64(16), T.int64(64)), "float32"), T_transpose: T.Buffer((T.int64(1), T.int64(16), T.int64(12), T.int64(64)), "float32")):
        T.func_attr({"tir.is_scheduled": True, "tir.noalias": True})
        # with T.block("root"):
        for ax0_ax1_ax2_fused_0 in T.thread_binding(T.int64(12), thread="blockIdx.x"):
            for ax0_ax1_ax2_fused_1 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                with T.block("T_transpose"):
                    v0 = T.axis.spatial(T.int64(16), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) // T.int64(768))
                    v1 = T.axis.spatial(T.int64(12), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(768) // T.int64(64))
                    v2 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_fused_0 * T.int64(1024) + ax0_ax1_ax2_fused_1) % T.int64(64))
                    T.reads(lv27[T.int64(0), v1, v0, v2])
                    T.writes(T_transpose[T.int64(0), v0, v1, v2])
                    T_transpose[T.int64(0), v0, v1, v2] = lv27[T.int64(0), v1, v0, v2]

    @R.function
    def main(input_ids: R.Tensor((1, 16), dtype="int64")) -> R.Tuple(R.Tensor((1, 16, 768), dtype="float32")):
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.reshape, (input_ids,), out_sinfo=R.Tensor((1, 16), dtype="int64"))
            lv1 = R.call_tir(cls.cast, (lv,), out_sinfo=R.Tensor((1, 16), dtype="int32"))
            lv2 = R.call_tir(cls.reshape1, (lv1,), out_sinfo=R.Tensor((16,), dtype="int32"))
            lv3 = R.call_tir(cls.take, (metadata["relax.expr.Constant"][0], lv2), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv4 = R.call_tir(cls.reshape2, (lv3,), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv12 = R.call_tir(cls.add, (lv4, metadata["relax.expr.Constant"][1]), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv13 = R.call_tir(cls.layer_norm, (lv12, metadata["relax.expr.Constant"][2], metadata["relax.expr.Constant"][3]), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv14 = R.call_tir(cls.reshape3, (lv13,), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv15 = R.call_tir(cls.matmul, (lv14, metadata["relax.expr.Constant"][4]), out_sinfo=R.Tensor((16, 2304), dtype="float32"))
            lv16 = R.call_tir(cls.add1, (metadata["relax.expr.Constant"][5], lv15), out_sinfo=R.Tensor((16, 2304), dtype="float32"))
            lv17 = R.call_tir(cls.reshape4, (lv16,), out_sinfo=R.Tensor((1, 16, 2304), dtype="float32"))
            lv18 = R.call_tir(cls.split, (lv17,), out_sinfo=[R.Tensor((1, 16, 768), dtype="float32"), R.Tensor((1, 16, 768), dtype="float32"), R.Tensor((1, 16, 768), dtype="float32")])
            lv19: R.Tensor((1, 16, 768), dtype="float32") = lv18[0]
            lv20: R.Tensor((1, 16, 768), dtype="float32") = lv18[1]
            lv21: R.Tensor((1, 16, 768), dtype="float32") = lv18[2]
            lv22 = R.call_tir(cls.reshape5, (lv20,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv23 = R.call_tir(cls.transpose, (lv22,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv24 = R.call_tir(cls.reshape5, (lv21,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv25 = R.call_tir(cls.transpose, (lv24,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv26 = R.call_tir(cls.reshape5, (lv19,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv27 = R.call_tir(cls.transpose, (lv26,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv31 = R.call_tir(cls.transpose1, (lv27,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv32 = R.call_tir(cls.transpose1, (lv23,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv33 = R.call_tir(cls.transpose1, (lv25,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv34 = R.call_tir(cls.attention_bias, (lv31, lv32, lv33, metadata["relax.expr.Constant"][6]), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv35 = R.call_tir(cls.transpose, (lv34,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv36 = R.call_tir(cls.transpose1, (lv35,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv37 = R.call_tir(cls.reshape6, (lv36,), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv38 = R.call_tir(cls.reshape3, (lv37,), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv39 = R.call_tir(cls.matmul1, (lv38, metadata["relax.expr.Constant"][7]), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv40 = R.call_tir(cls.add2, (metadata["relax.expr.Constant"][8], lv39), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv41 = R.call_tir(cls.reshape2, (lv40,), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv42 = R.call_tir(cls.add, (lv41, lv12), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv43 = R.call_tir(cls.layer_norm, (lv42, metadata["relax.expr.Constant"][9], metadata["relax.expr.Constant"][10]), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv44 = R.call_tir(cls.reshape3, (lv43,), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv45 = R.call_tir(cls.matmul2, (lv44, metadata["relax.expr.Constant"][11]), out_sinfo=R.Tensor((16, 3072), dtype="float32"))
            lv46 = R.call_tir(cls.add3, (metadata["relax.expr.Constant"][12], lv45), out_sinfo=R.Tensor((16, 3072), dtype="float32"))
            lv47 = R.call_tir(cls.reshape7, (lv46,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv48 = R.call_tir(cls.multiply, (lv47,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv49 = R.call_tir(cls.power, (lv47,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv50 = R.call_tir(cls.multiply1, (lv49,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv51 = R.call_tir(cls.add4, (lv47, lv50), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv52 = R.call_tir(cls.multiply2, (lv51,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv53 = R.call_tir(cls.tir_tanh, (lv52,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv54 = R.call_tir(cls.add5, (lv53,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv55 = R.call_tir(cls.multiply3, (lv48, lv54), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv56 = R.call_tir(cls.reshape8, (lv55,), out_sinfo=R.Tensor((16, 3072), dtype="float32"))
            lv57 = R.call_tir(cls.matmul3, (lv56, metadata["relax.expr.Constant"][13]), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv58 = R.call_tir(cls.add2, (metadata["relax.expr.Constant"][14], lv57), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv59 = R.call_tir(cls.reshape2, (lv58,), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv60 = R.call_tir(cls.add, (lv42, lv59), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv61 = R.call_tir(cls.layer_norm, (lv60, metadata["relax.expr.Constant"][15], metadata["relax.expr.Constant"][16]), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv62 = R.call_tir(cls.reshape3, (lv61,), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv63 = R.call_tir(cls.matmul, (lv62, metadata["relax.expr.Constant"][17]), out_sinfo=R.Tensor((16, 2304), dtype="float32"))
            lv64 = R.call_tir(cls.add1, (metadata["relax.expr.Constant"][18], lv63), out_sinfo=R.Tensor((16, 2304), dtype="float32"))
            lv65 = R.call_tir(cls.reshape4, (lv64,), out_sinfo=R.Tensor((1, 16, 2304), dtype="float32"))
            lv66 = R.call_tir(cls.split, (lv65,), out_sinfo=[R.Tensor((1, 16, 768), dtype="float32"), R.Tensor((1, 16, 768), dtype="float32"), R.Tensor((1, 16, 768), dtype="float32")])
            lv67: R.Tensor((1, 16, 768), dtype="float32") = lv66[0]
            lv68: R.Tensor((1, 16, 768), dtype="float32") = lv66[1]
            lv69: R.Tensor((1, 16, 768), dtype="float32") = lv66[2]
            lv70 = R.call_tir(cls.reshape5, (lv68,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv71 = R.call_tir(cls.transpose, (lv70,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv72 = R.call_tir(cls.reshape5, (lv69,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv73 = R.call_tir(cls.transpose, (lv72,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv74 = R.call_tir(cls.reshape5, (lv67,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv75 = R.call_tir(cls.transpose, (lv74,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv79 = R.call_tir(cls.transpose1, (lv75,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv80 = R.call_tir(cls.transpose1, (lv71,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv81 = R.call_tir(cls.transpose1, (lv73,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv82 = R.call_tir(cls.attention_bias, (lv79, lv80, lv81, metadata["relax.expr.Constant"][19]), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv83 = R.call_tir(cls.transpose, (lv82,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv84 = R.call_tir(cls.transpose1, (lv83,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv85 = R.call_tir(cls.reshape6, (lv84,), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv86 = R.call_tir(cls.reshape3, (lv85,), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv87 = R.call_tir(cls.matmul1, (lv86, metadata["relax.expr.Constant"][20]), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv88 = R.call_tir(cls.add2, (metadata["relax.expr.Constant"][21], lv87), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv89 = R.call_tir(cls.reshape2, (lv88,), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv90 = R.call_tir(cls.add, (lv89, lv60), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv91 = R.call_tir(cls.layer_norm, (lv90, metadata["relax.expr.Constant"][22], metadata["relax.expr.Constant"][23]), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv92 = R.call_tir(cls.reshape3, (lv91,), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv93 = R.call_tir(cls.matmul2, (lv92, metadata["relax.expr.Constant"][24]), out_sinfo=R.Tensor((16, 3072), dtype="float32"))
            lv94 = R.call_tir(cls.add3, (metadata["relax.expr.Constant"][25], lv93), out_sinfo=R.Tensor((16, 3072), dtype="float32"))
            lv95 = R.call_tir(cls.reshape7, (lv94,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv96 = R.call_tir(cls.multiply, (lv95,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv97 = R.call_tir(cls.power, (lv95,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv98 = R.call_tir(cls.multiply1, (lv97,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv99 = R.call_tir(cls.add4, (lv95, lv98), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv100 = R.call_tir(cls.multiply2, (lv99,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv101 = R.call_tir(cls.tir_tanh, (lv100,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv102 = R.call_tir(cls.add5, (lv101,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv103 = R.call_tir(cls.multiply3, (lv96, lv102), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv104 = R.call_tir(cls.reshape8, (lv103,), out_sinfo=R.Tensor((16, 3072), dtype="float32"))
            lv105 = R.call_tir(cls.matmul3, (lv104, metadata["relax.expr.Constant"][26]), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv106 = R.call_tir(cls.add2, (metadata["relax.expr.Constant"][27], lv105), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv107 = R.call_tir(cls.reshape2, (lv106,), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv108 = R.call_tir(cls.add, (lv90, lv107), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv109 = R.call_tir(cls.layer_norm, (lv108, metadata["relax.expr.Constant"][28], metadata["relax.expr.Constant"][29]), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv110 = R.call_tir(cls.reshape3, (lv109,), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv111 = R.call_tir(cls.matmul, (lv110, metadata["relax.expr.Constant"][30]), out_sinfo=R.Tensor((16, 2304), dtype="float32"))
            lv112 = R.call_tir(cls.add1, (metadata["relax.expr.Constant"][31], lv111), out_sinfo=R.Tensor((16, 2304), dtype="float32"))
            lv113 = R.call_tir(cls.reshape4, (lv112,), out_sinfo=R.Tensor((1, 16, 2304), dtype="float32"))
            lv114 = R.call_tir(cls.split, (lv113,), out_sinfo=[R.Tensor((1, 16, 768), dtype="float32"), R.Tensor((1, 16, 768), dtype="float32"), R.Tensor((1, 16, 768), dtype="float32")])
            lv115: R.Tensor((1, 16, 768), dtype="float32") = lv114[0]
            lv116: R.Tensor((1, 16, 768), dtype="float32") = lv114[1]
            lv117: R.Tensor((1, 16, 768), dtype="float32") = lv114[2]
            lv118 = R.call_tir(cls.reshape5, (lv116,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv119 = R.call_tir(cls.transpose, (lv118,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv120 = R.call_tir(cls.reshape5, (lv117,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv121 = R.call_tir(cls.transpose, (lv120,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv122 = R.call_tir(cls.reshape5, (lv115,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv123 = R.call_tir(cls.transpose, (lv122,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv127 = R.call_tir(cls.transpose1, (lv123,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv128 = R.call_tir(cls.transpose1, (lv119,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv129 = R.call_tir(cls.transpose1, (lv121,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv130 = R.call_tir(cls.attention_bias, (lv127, lv128, lv129, metadata["relax.expr.Constant"][32]), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv131 = R.call_tir(cls.transpose, (lv130,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv132 = R.call_tir(cls.transpose1, (lv131,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv133 = R.call_tir(cls.reshape6, (lv132,), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv134 = R.call_tir(cls.reshape3, (lv133,), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv135 = R.call_tir(cls.matmul1, (lv134, metadata["relax.expr.Constant"][33]), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv136 = R.call_tir(cls.add2, (metadata["relax.expr.Constant"][34], lv135), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv137 = R.call_tir(cls.reshape2, (lv136,), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv138 = R.call_tir(cls.add, (lv137, lv108), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv139 = R.call_tir(cls.layer_norm, (lv138, metadata["relax.expr.Constant"][35], metadata["relax.expr.Constant"][36]), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv140 = R.call_tir(cls.reshape3, (lv139,), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv141 = R.call_tir(cls.matmul2, (lv140, metadata["relax.expr.Constant"][37]), out_sinfo=R.Tensor((16, 3072), dtype="float32"))
            lv142 = R.call_tir(cls.add3, (metadata["relax.expr.Constant"][38], lv141), out_sinfo=R.Tensor((16, 3072), dtype="float32"))
            lv143 = R.call_tir(cls.reshape7, (lv142,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv144 = R.call_tir(cls.multiply, (lv143,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv145 = R.call_tir(cls.power, (lv143,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv146 = R.call_tir(cls.multiply1, (lv145,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv147 = R.call_tir(cls.add4, (lv143, lv146), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv148 = R.call_tir(cls.multiply2, (lv147,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv149 = R.call_tir(cls.tir_tanh, (lv148,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv150 = R.call_tir(cls.add5, (lv149,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv151 = R.call_tir(cls.multiply3, (lv144, lv150), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv152 = R.call_tir(cls.reshape8, (lv151,), out_sinfo=R.Tensor((16, 3072), dtype="float32"))
            lv153 = R.call_tir(cls.matmul3, (lv152, metadata["relax.expr.Constant"][39]), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv154 = R.call_tir(cls.add2, (metadata["relax.expr.Constant"][40], lv153), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv155 = R.call_tir(cls.reshape2, (lv154,), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv156 = R.call_tir(cls.add, (lv138, lv155), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv157 = R.call_tir(cls.layer_norm, (lv156, metadata["relax.expr.Constant"][41], metadata["relax.expr.Constant"][42]), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv158 = R.call_tir(cls.reshape3, (lv157,), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv159 = R.call_tir(cls.matmul, (lv158, metadata["relax.expr.Constant"][43]), out_sinfo=R.Tensor((16, 2304), dtype="float32"))
            lv160 = R.call_tir(cls.add1, (metadata["relax.expr.Constant"][44], lv159), out_sinfo=R.Tensor((16, 2304), dtype="float32"))
            lv161 = R.call_tir(cls.reshape4, (lv160,), out_sinfo=R.Tensor((1, 16, 2304), dtype="float32"))
            lv162 = R.call_tir(cls.split, (lv161,), out_sinfo=[R.Tensor((1, 16, 768), dtype="float32"), R.Tensor((1, 16, 768), dtype="float32"), R.Tensor((1, 16, 768), dtype="float32")])
            lv163: R.Tensor((1, 16, 768), dtype="float32") = lv162[0]
            lv164: R.Tensor((1, 16, 768), dtype="float32") = lv162[1]
            lv165: R.Tensor((1, 16, 768), dtype="float32") = lv162[2]
            lv166 = R.call_tir(cls.reshape5, (lv164,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv167 = R.call_tir(cls.transpose, (lv166,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv168 = R.call_tir(cls.reshape5, (lv165,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv169 = R.call_tir(cls.transpose, (lv168,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv170 = R.call_tir(cls.reshape5, (lv163,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv171 = R.call_tir(cls.transpose, (lv170,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv175 = R.call_tir(cls.transpose1, (lv171,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv176 = R.call_tir(cls.transpose1, (lv167,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv177 = R.call_tir(cls.transpose1, (lv169,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv178 = R.call_tir(cls.attention_bias, (lv175, lv176, lv177, metadata["relax.expr.Constant"][45]), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv179 = R.call_tir(cls.transpose, (lv178,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv180 = R.call_tir(cls.transpose1, (lv179,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv181 = R.call_tir(cls.reshape6, (lv180,), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv182 = R.call_tir(cls.reshape3, (lv181,), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv183 = R.call_tir(cls.matmul1, (lv182, metadata["relax.expr.Constant"][46]), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv184 = R.call_tir(cls.add2, (metadata["relax.expr.Constant"][47], lv183), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv185 = R.call_tir(cls.reshape2, (lv184,), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv186 = R.call_tir(cls.add, (lv185, lv156), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv187 = R.call_tir(cls.layer_norm, (lv186, metadata["relax.expr.Constant"][48], metadata["relax.expr.Constant"][49]), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv188 = R.call_tir(cls.reshape3, (lv187,), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv189 = R.call_tir(cls.matmul2, (lv188, metadata["relax.expr.Constant"][50]), out_sinfo=R.Tensor((16, 3072), dtype="float32"))
            lv190 = R.call_tir(cls.add3, (metadata["relax.expr.Constant"][51], lv189), out_sinfo=R.Tensor((16, 3072), dtype="float32"))
            lv191 = R.call_tir(cls.reshape7, (lv190,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv192 = R.call_tir(cls.multiply, (lv191,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv193 = R.call_tir(cls.power, (lv191,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv194 = R.call_tir(cls.multiply1, (lv193,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv195 = R.call_tir(cls.add4, (lv191, lv194), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv196 = R.call_tir(cls.multiply2, (lv195,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv197 = R.call_tir(cls.tir_tanh, (lv196,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv198 = R.call_tir(cls.add5, (lv197,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv199 = R.call_tir(cls.multiply3, (lv192, lv198), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv200 = R.call_tir(cls.reshape8, (lv199,), out_sinfo=R.Tensor((16, 3072), dtype="float32"))
            lv201 = R.call_tir(cls.matmul3, (lv200, metadata["relax.expr.Constant"][52]), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv202 = R.call_tir(cls.add2, (metadata["relax.expr.Constant"][53], lv201), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv203 = R.call_tir(cls.reshape2, (lv202,), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv204 = R.call_tir(cls.add, (lv186, lv203), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv205 = R.call_tir(cls.layer_norm, (lv204, metadata["relax.expr.Constant"][54], metadata["relax.expr.Constant"][55]), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv206 = R.call_tir(cls.reshape3, (lv205,), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv207 = R.call_tir(cls.matmul, (lv206, metadata["relax.expr.Constant"][56]), out_sinfo=R.Tensor((16, 2304), dtype="float32"))
            lv208 = R.call_tir(cls.add1, (metadata["relax.expr.Constant"][57], lv207), out_sinfo=R.Tensor((16, 2304), dtype="float32"))
            lv209 = R.call_tir(cls.reshape4, (lv208,), out_sinfo=R.Tensor((1, 16, 2304), dtype="float32"))
            lv210 = R.call_tir(cls.split, (lv209,), out_sinfo=[R.Tensor((1, 16, 768), dtype="float32"), R.Tensor((1, 16, 768), dtype="float32"), R.Tensor((1, 16, 768), dtype="float32")])
            lv211: R.Tensor((1, 16, 768), dtype="float32") = lv210[0]
            lv212: R.Tensor((1, 16, 768), dtype="float32") = lv210[1]
            lv213: R.Tensor((1, 16, 768), dtype="float32") = lv210[2]
            lv214 = R.call_tir(cls.reshape5, (lv212,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv215 = R.call_tir(cls.transpose, (lv214,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv216 = R.call_tir(cls.reshape5, (lv213,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv217 = R.call_tir(cls.transpose, (lv216,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv218 = R.call_tir(cls.reshape5, (lv211,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv219 = R.call_tir(cls.transpose, (lv218,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv223 = R.call_tir(cls.transpose1, (lv219,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv224 = R.call_tir(cls.transpose1, (lv215,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv225 = R.call_tir(cls.transpose1, (lv217,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv226 = R.call_tir(cls.attention_bias, (lv223, lv224, lv225, metadata["relax.expr.Constant"][58]), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv227 = R.call_tir(cls.transpose, (lv226,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv228 = R.call_tir(cls.transpose1, (lv227,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv229 = R.call_tir(cls.reshape6, (lv228,), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv230 = R.call_tir(cls.reshape3, (lv229,), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv231 = R.call_tir(cls.matmul1, (lv230, metadata["relax.expr.Constant"][59]), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv232 = R.call_tir(cls.add2, (metadata["relax.expr.Constant"][60], lv231), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv233 = R.call_tir(cls.reshape2, (lv232,), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv234 = R.call_tir(cls.add, (lv233, lv204), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv235 = R.call_tir(cls.layer_norm, (lv234, metadata["relax.expr.Constant"][61], metadata["relax.expr.Constant"][62]), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv236 = R.call_tir(cls.reshape3, (lv235,), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv237 = R.call_tir(cls.matmul2, (lv236, metadata["relax.expr.Constant"][63]), out_sinfo=R.Tensor((16, 3072), dtype="float32"))
            lv238 = R.call_tir(cls.add3, (metadata["relax.expr.Constant"][64], lv237), out_sinfo=R.Tensor((16, 3072), dtype="float32"))
            lv239 = R.call_tir(cls.reshape7, (lv238,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv240 = R.call_tir(cls.multiply, (lv239,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv241 = R.call_tir(cls.power, (lv239,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv242 = R.call_tir(cls.multiply1, (lv241,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv243 = R.call_tir(cls.add4, (lv239, lv242), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv244 = R.call_tir(cls.multiply2, (lv243,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv245 = R.call_tir(cls.tir_tanh, (lv244,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv246 = R.call_tir(cls.add5, (lv245,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv247 = R.call_tir(cls.multiply3, (lv240, lv246), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv248 = R.call_tir(cls.reshape8, (lv247,), out_sinfo=R.Tensor((16, 3072), dtype="float32"))
            lv249 = R.call_tir(cls.matmul3, (lv248, metadata["relax.expr.Constant"][65]), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv250 = R.call_tir(cls.add2, (metadata["relax.expr.Constant"][66], lv249), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv251 = R.call_tir(cls.reshape2, (lv250,), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv252 = R.call_tir(cls.add, (lv234, lv251), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv253 = R.call_tir(cls.layer_norm, (lv252, metadata["relax.expr.Constant"][67], metadata["relax.expr.Constant"][68]), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv254 = R.call_tir(cls.reshape3, (lv253,), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv255 = R.call_tir(cls.matmul, (lv254, metadata["relax.expr.Constant"][69]), out_sinfo=R.Tensor((16, 2304), dtype="float32"))
            lv256 = R.call_tir(cls.add1, (metadata["relax.expr.Constant"][70], lv255), out_sinfo=R.Tensor((16, 2304), dtype="float32"))
            lv257 = R.call_tir(cls.reshape4, (lv256,), out_sinfo=R.Tensor((1, 16, 2304), dtype="float32"))
            lv258 = R.call_tir(cls.split, (lv257,), out_sinfo=[R.Tensor((1, 16, 768), dtype="float32"), R.Tensor((1, 16, 768), dtype="float32"), R.Tensor((1, 16, 768), dtype="float32")])
            lv259: R.Tensor((1, 16, 768), dtype="float32") = lv258[0]
            lv260: R.Tensor((1, 16, 768), dtype="float32") = lv258[1]
            lv261: R.Tensor((1, 16, 768), dtype="float32") = lv258[2]
            lv262 = R.call_tir(cls.reshape5, (lv260,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv263 = R.call_tir(cls.transpose, (lv262,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv264 = R.call_tir(cls.reshape5, (lv261,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv265 = R.call_tir(cls.transpose, (lv264,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv266 = R.call_tir(cls.reshape5, (lv259,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv267 = R.call_tir(cls.transpose, (lv266,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv271 = R.call_tir(cls.transpose1, (lv267,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv272 = R.call_tir(cls.transpose1, (lv263,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv273 = R.call_tir(cls.transpose1, (lv265,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv274 = R.call_tir(cls.attention_bias, (lv271, lv272, lv273, metadata["relax.expr.Constant"][71]), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv275 = R.call_tir(cls.transpose, (lv274,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv276 = R.call_tir(cls.transpose1, (lv275,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv277 = R.call_tir(cls.reshape6, (lv276,), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv278 = R.call_tir(cls.reshape3, (lv277,), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv279 = R.call_tir(cls.matmul1, (lv278, metadata["relax.expr.Constant"][72]), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv280 = R.call_tir(cls.add2, (metadata["relax.expr.Constant"][73], lv279), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv281 = R.call_tir(cls.reshape2, (lv280,), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv282 = R.call_tir(cls.add, (lv281, lv252), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv283 = R.call_tir(cls.layer_norm, (lv282, metadata["relax.expr.Constant"][74], metadata["relax.expr.Constant"][75]), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv284 = R.call_tir(cls.reshape3, (lv283,), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv285 = R.call_tir(cls.matmul2, (lv284, metadata["relax.expr.Constant"][76]), out_sinfo=R.Tensor((16, 3072), dtype="float32"))
            lv286 = R.call_tir(cls.add3, (metadata["relax.expr.Constant"][77], lv285), out_sinfo=R.Tensor((16, 3072), dtype="float32"))
            lv287 = R.call_tir(cls.reshape7, (lv286,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv288 = R.call_tir(cls.multiply, (lv287,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv289 = R.call_tir(cls.power, (lv287,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv290 = R.call_tir(cls.multiply1, (lv289,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv291 = R.call_tir(cls.add4, (lv287, lv290), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv292 = R.call_tir(cls.multiply2, (lv291,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv293 = R.call_tir(cls.tir_tanh, (lv292,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv294 = R.call_tir(cls.add5, (lv293,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv295 = R.call_tir(cls.multiply3, (lv288, lv294), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv296 = R.call_tir(cls.reshape8, (lv295,), out_sinfo=R.Tensor((16, 3072), dtype="float32"))
            lv297 = R.call_tir(cls.matmul3, (lv296, metadata["relax.expr.Constant"][78]), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv298 = R.call_tir(cls.add2, (metadata["relax.expr.Constant"][79], lv297), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv299 = R.call_tir(cls.reshape2, (lv298,), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv300 = R.call_tir(cls.add, (lv282, lv299), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv301 = R.call_tir(cls.layer_norm, (lv300, metadata["relax.expr.Constant"][80], metadata["relax.expr.Constant"][81]), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv302 = R.call_tir(cls.reshape3, (lv301,), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv303 = R.call_tir(cls.matmul, (lv302, metadata["relax.expr.Constant"][82]), out_sinfo=R.Tensor((16, 2304), dtype="float32"))
            lv304 = R.call_tir(cls.add1, (metadata["relax.expr.Constant"][83], lv303), out_sinfo=R.Tensor((16, 2304), dtype="float32"))
            lv305 = R.call_tir(cls.reshape4, (lv304,), out_sinfo=R.Tensor((1, 16, 2304), dtype="float32"))
            lv306 = R.call_tir(cls.split, (lv305,), out_sinfo=[R.Tensor((1, 16, 768), dtype="float32"), R.Tensor((1, 16, 768), dtype="float32"), R.Tensor((1, 16, 768), dtype="float32")])
            lv307: R.Tensor((1, 16, 768), dtype="float32") = lv306[0]
            lv308: R.Tensor((1, 16, 768), dtype="float32") = lv306[1]
            lv309: R.Tensor((1, 16, 768), dtype="float32") = lv306[2]
            lv310 = R.call_tir(cls.reshape5, (lv308,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv311 = R.call_tir(cls.transpose, (lv310,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv312 = R.call_tir(cls.reshape5, (lv309,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv313 = R.call_tir(cls.transpose, (lv312,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv314 = R.call_tir(cls.reshape5, (lv307,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv315 = R.call_tir(cls.transpose, (lv314,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv319 = R.call_tir(cls.transpose1, (lv315,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv320 = R.call_tir(cls.transpose1, (lv311,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv321 = R.call_tir(cls.transpose1, (lv313,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv322 = R.call_tir(cls.attention_bias, (lv319, lv320, lv321, metadata["relax.expr.Constant"][84]), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv323 = R.call_tir(cls.transpose, (lv322,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv324 = R.call_tir(cls.transpose1, (lv323,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv325 = R.call_tir(cls.reshape6, (lv324,), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv326 = R.call_tir(cls.reshape3, (lv325,), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv327 = R.call_tir(cls.matmul1, (lv326, metadata["relax.expr.Constant"][85]), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv328 = R.call_tir(cls.add2, (metadata["relax.expr.Constant"][86], lv327), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv329 = R.call_tir(cls.reshape2, (lv328,), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv330 = R.call_tir(cls.add, (lv329, lv300), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv331 = R.call_tir(cls.layer_norm, (lv330, metadata["relax.expr.Constant"][87], metadata["relax.expr.Constant"][88]), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv332 = R.call_tir(cls.reshape3, (lv331,), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv333 = R.call_tir(cls.matmul2, (lv332, metadata["relax.expr.Constant"][89]), out_sinfo=R.Tensor((16, 3072), dtype="float32"))
            lv334 = R.call_tir(cls.add3, (metadata["relax.expr.Constant"][90], lv333), out_sinfo=R.Tensor((16, 3072), dtype="float32"))
            lv335 = R.call_tir(cls.reshape7, (lv334,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv336 = R.call_tir(cls.multiply, (lv335,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv337 = R.call_tir(cls.power, (lv335,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv338 = R.call_tir(cls.multiply1, (lv337,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv339 = R.call_tir(cls.add4, (lv335, lv338), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv340 = R.call_tir(cls.multiply2, (lv339,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv341 = R.call_tir(cls.tir_tanh, (lv340,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv342 = R.call_tir(cls.add5, (lv341,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv343 = R.call_tir(cls.multiply3, (lv336, lv342), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv344 = R.call_tir(cls.reshape8, (lv343,), out_sinfo=R.Tensor((16, 3072), dtype="float32"))
            lv345 = R.call_tir(cls.matmul3, (lv344, metadata["relax.expr.Constant"][91]), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv346 = R.call_tir(cls.add2, (metadata["relax.expr.Constant"][92], lv345), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv347 = R.call_tir(cls.reshape2, (lv346,), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv348 = R.call_tir(cls.add, (lv330, lv347), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv349 = R.call_tir(cls.layer_norm, (lv348, metadata["relax.expr.Constant"][93], metadata["relax.expr.Constant"][94]), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv350 = R.call_tir(cls.reshape3, (lv349,), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv351 = R.call_tir(cls.matmul, (lv350, metadata["relax.expr.Constant"][95]), out_sinfo=R.Tensor((16, 2304), dtype="float32"))
            lv352 = R.call_tir(cls.add1, (metadata["relax.expr.Constant"][96], lv351), out_sinfo=R.Tensor((16, 2304), dtype="float32"))
            lv353 = R.call_tir(cls.reshape4, (lv352,), out_sinfo=R.Tensor((1, 16, 2304), dtype="float32"))
            lv354 = R.call_tir(cls.split, (lv353,), out_sinfo=[R.Tensor((1, 16, 768), dtype="float32"), R.Tensor((1, 16, 768), dtype="float32"), R.Tensor((1, 16, 768), dtype="float32")])
            lv355: R.Tensor((1, 16, 768), dtype="float32") = lv354[0]
            lv356: R.Tensor((1, 16, 768), dtype="float32") = lv354[1]
            lv357: R.Tensor((1, 16, 768), dtype="float32") = lv354[2]
            lv358 = R.call_tir(cls.reshape5, (lv356,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv359 = R.call_tir(cls.transpose, (lv358,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv360 = R.call_tir(cls.reshape5, (lv357,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv361 = R.call_tir(cls.transpose, (lv360,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv362 = R.call_tir(cls.reshape5, (lv355,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv363 = R.call_tir(cls.transpose, (lv362,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv367 = R.call_tir(cls.transpose1, (lv363,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv368 = R.call_tir(cls.transpose1, (lv359,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv369 = R.call_tir(cls.transpose1, (lv361,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv370 = R.call_tir(cls.attention_bias, (lv367, lv368, lv369, metadata["relax.expr.Constant"][97]), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv371 = R.call_tir(cls.transpose, (lv370,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv372 = R.call_tir(cls.transpose1, (lv371,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv373 = R.call_tir(cls.reshape6, (lv372,), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv374 = R.call_tir(cls.reshape3, (lv373,), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv375 = R.call_tir(cls.matmul1, (lv374, metadata["relax.expr.Constant"][98]), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv376 = R.call_tir(cls.add2, (metadata["relax.expr.Constant"][99], lv375), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv377 = R.call_tir(cls.reshape2, (lv376,), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv378 = R.call_tir(cls.add, (lv377, lv348), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv379 = R.call_tir(cls.layer_norm, (lv378, metadata["relax.expr.Constant"][100], metadata["relax.expr.Constant"][101]), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv380 = R.call_tir(cls.reshape3, (lv379,), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv381 = R.call_tir(cls.matmul2, (lv380, metadata["relax.expr.Constant"][102]), out_sinfo=R.Tensor((16, 3072), dtype="float32"))
            lv382 = R.call_tir(cls.add3, (metadata["relax.expr.Constant"][103], lv381), out_sinfo=R.Tensor((16, 3072), dtype="float32"))
            lv383 = R.call_tir(cls.reshape7, (lv382,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv384 = R.call_tir(cls.multiply, (lv383,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv385 = R.call_tir(cls.power, (lv383,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv386 = R.call_tir(cls.multiply1, (lv385,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv387 = R.call_tir(cls.add4, (lv383, lv386), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv388 = R.call_tir(cls.multiply2, (lv387,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv389 = R.call_tir(cls.tir_tanh, (lv388,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv390 = R.call_tir(cls.add5, (lv389,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv391 = R.call_tir(cls.multiply3, (lv384, lv390), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv392 = R.call_tir(cls.reshape8, (lv391,), out_sinfo=R.Tensor((16, 3072), dtype="float32"))
            lv393 = R.call_tir(cls.matmul3, (lv392, metadata["relax.expr.Constant"][104]), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv394 = R.call_tir(cls.add2, (metadata["relax.expr.Constant"][105], lv393), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv395 = R.call_tir(cls.reshape2, (lv394,), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv396 = R.call_tir(cls.add, (lv378, lv395), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv397 = R.call_tir(cls.layer_norm, (lv396, metadata["relax.expr.Constant"][106], metadata["relax.expr.Constant"][107]), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv398 = R.call_tir(cls.reshape3, (lv397,), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv399 = R.call_tir(cls.matmul, (lv398, metadata["relax.expr.Constant"][108]), out_sinfo=R.Tensor((16, 2304), dtype="float32"))
            lv400 = R.call_tir(cls.add1, (metadata["relax.expr.Constant"][109], lv399), out_sinfo=R.Tensor((16, 2304), dtype="float32"))
            lv401 = R.call_tir(cls.reshape4, (lv400,), out_sinfo=R.Tensor((1, 16, 2304), dtype="float32"))
            lv402 = R.call_tir(cls.split, (lv401,), out_sinfo=[R.Tensor((1, 16, 768), dtype="float32"), R.Tensor((1, 16, 768), dtype="float32"), R.Tensor((1, 16, 768), dtype="float32")])
            lv403: R.Tensor((1, 16, 768), dtype="float32") = lv402[0]
            lv404: R.Tensor((1, 16, 768), dtype="float32") = lv402[1]
            lv405: R.Tensor((1, 16, 768), dtype="float32") = lv402[2]
            lv406 = R.call_tir(cls.reshape5, (lv404,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv407 = R.call_tir(cls.transpose, (lv406,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv408 = R.call_tir(cls.reshape5, (lv405,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv409 = R.call_tir(cls.transpose, (lv408,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv410 = R.call_tir(cls.reshape5, (lv403,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv411 = R.call_tir(cls.transpose, (lv410,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv415 = R.call_tir(cls.transpose1, (lv411,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv416 = R.call_tir(cls.transpose1, (lv407,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv417 = R.call_tir(cls.transpose1, (lv409,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv418 = R.call_tir(cls.attention_bias, (lv415, lv416, lv417, metadata["relax.expr.Constant"][110]), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv419 = R.call_tir(cls.transpose, (lv418,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv420 = R.call_tir(cls.transpose1, (lv419,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv421 = R.call_tir(cls.reshape6, (lv420,), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv422 = R.call_tir(cls.reshape3, (lv421,), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv423 = R.call_tir(cls.matmul1, (lv422, metadata["relax.expr.Constant"][111]), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv424 = R.call_tir(cls.add2, (metadata["relax.expr.Constant"][112], lv423), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv425 = R.call_tir(cls.reshape2, (lv424,), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv426 = R.call_tir(cls.add, (lv425, lv396), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv427 = R.call_tir(cls.layer_norm, (lv426, metadata["relax.expr.Constant"][113], metadata["relax.expr.Constant"][114]), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv428 = R.call_tir(cls.reshape3, (lv427,), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv429 = R.call_tir(cls.matmul2, (lv428, metadata["relax.expr.Constant"][115]), out_sinfo=R.Tensor((16, 3072), dtype="float32"))
            lv430 = R.call_tir(cls.add3, (metadata["relax.expr.Constant"][116], lv429), out_sinfo=R.Tensor((16, 3072), dtype="float32"))
            lv431 = R.call_tir(cls.reshape7, (lv430,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv432 = R.call_tir(cls.multiply, (lv431,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv433 = R.call_tir(cls.power, (lv431,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv434 = R.call_tir(cls.multiply1, (lv433,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv435 = R.call_tir(cls.add4, (lv431, lv434), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv436 = R.call_tir(cls.multiply2, (lv435,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv437 = R.call_tir(cls.tir_tanh, (lv436,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv438 = R.call_tir(cls.add5, (lv437,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv439 = R.call_tir(cls.multiply3, (lv432, lv438), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv440 = R.call_tir(cls.reshape8, (lv439,), out_sinfo=R.Tensor((16, 3072), dtype="float32"))
            lv441 = R.call_tir(cls.matmul3, (lv440, metadata["relax.expr.Constant"][117]), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv442 = R.call_tir(cls.add2, (metadata["relax.expr.Constant"][118], lv441), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv443 = R.call_tir(cls.reshape2, (lv442,), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv444 = R.call_tir(cls.add, (lv426, lv443), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv445 = R.call_tir(cls.layer_norm, (lv444, metadata["relax.expr.Constant"][119], metadata["relax.expr.Constant"][120]), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv446 = R.call_tir(cls.reshape3, (lv445,), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv447 = R.call_tir(cls.matmul, (lv446, metadata["relax.expr.Constant"][121]), out_sinfo=R.Tensor((16, 2304), dtype="float32"))
            lv448 = R.call_tir(cls.add1, (metadata["relax.expr.Constant"][122], lv447), out_sinfo=R.Tensor((16, 2304), dtype="float32"))
            lv449 = R.call_tir(cls.reshape4, (lv448,), out_sinfo=R.Tensor((1, 16, 2304), dtype="float32"))
            lv450 = R.call_tir(cls.split, (lv449,), out_sinfo=[R.Tensor((1, 16, 768), dtype="float32"), R.Tensor((1, 16, 768), dtype="float32"), R.Tensor((1, 16, 768), dtype="float32")])
            lv451: R.Tensor((1, 16, 768), dtype="float32") = lv450[0]
            lv452: R.Tensor((1, 16, 768), dtype="float32") = lv450[1]
            lv453: R.Tensor((1, 16, 768), dtype="float32") = lv450[2]
            lv454 = R.call_tir(cls.reshape5, (lv452,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv455 = R.call_tir(cls.transpose, (lv454,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv456 = R.call_tir(cls.reshape5, (lv453,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv457 = R.call_tir(cls.transpose, (lv456,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv458 = R.call_tir(cls.reshape5, (lv451,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv459 = R.call_tir(cls.transpose, (lv458,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv463 = R.call_tir(cls.transpose1, (lv459,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv464 = R.call_tir(cls.transpose1, (lv455,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv465 = R.call_tir(cls.transpose1, (lv457,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv466 = R.call_tir(cls.attention_bias, (lv463, lv464, lv465, metadata["relax.expr.Constant"][123]), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv467 = R.call_tir(cls.transpose, (lv466,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv468 = R.call_tir(cls.transpose1, (lv467,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv469 = R.call_tir(cls.reshape6, (lv468,), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv470 = R.call_tir(cls.reshape3, (lv469,), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv471 = R.call_tir(cls.matmul1, (lv470, metadata["relax.expr.Constant"][124]), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv472 = R.call_tir(cls.add2, (metadata["relax.expr.Constant"][125], lv471), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv473 = R.call_tir(cls.reshape2, (lv472,), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv474 = R.call_tir(cls.add, (lv473, lv444), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv475 = R.call_tir(cls.layer_norm, (lv474, metadata["relax.expr.Constant"][126], metadata["relax.expr.Constant"][127]), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv476 = R.call_tir(cls.reshape3, (lv475,), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv477 = R.call_tir(cls.matmul2, (lv476, metadata["relax.expr.Constant"][128]), out_sinfo=R.Tensor((16, 3072), dtype="float32"))
            lv478 = R.call_tir(cls.add3, (metadata["relax.expr.Constant"][129], lv477), out_sinfo=R.Tensor((16, 3072), dtype="float32"))
            lv479 = R.call_tir(cls.reshape7, (lv478,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv480 = R.call_tir(cls.multiply, (lv479,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv481 = R.call_tir(cls.power, (lv479,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv482 = R.call_tir(cls.multiply1, (lv481,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv483 = R.call_tir(cls.add4, (lv479, lv482), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv484 = R.call_tir(cls.multiply2, (lv483,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv485 = R.call_tir(cls.tir_tanh, (lv484,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv486 = R.call_tir(cls.add5, (lv485,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv487 = R.call_tir(cls.multiply3, (lv480, lv486), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv488 = R.call_tir(cls.reshape8, (lv487,), out_sinfo=R.Tensor((16, 3072), dtype="float32"))
            lv489 = R.call_tir(cls.matmul3, (lv488, metadata["relax.expr.Constant"][130]), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv490 = R.call_tir(cls.add2, (metadata["relax.expr.Constant"][131], lv489), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv491 = R.call_tir(cls.reshape2, (lv490,), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv492 = R.call_tir(cls.add, (lv474, lv491), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv493 = R.call_tir(cls.layer_norm, (lv492, metadata["relax.expr.Constant"][132], metadata["relax.expr.Constant"][133]), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv494 = R.call_tir(cls.reshape3, (lv493,), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv495 = R.call_tir(cls.matmul, (lv494, metadata["relax.expr.Constant"][134]), out_sinfo=R.Tensor((16, 2304), dtype="float32"))
            lv496 = R.call_tir(cls.add1, (metadata["relax.expr.Constant"][135], lv495), out_sinfo=R.Tensor((16, 2304), dtype="float32"))
            lv497 = R.call_tir(cls.reshape4, (lv496,), out_sinfo=R.Tensor((1, 16, 2304), dtype="float32"))
            lv498 = R.call_tir(cls.split, (lv497,), out_sinfo=[R.Tensor((1, 16, 768), dtype="float32"), R.Tensor((1, 16, 768), dtype="float32"), R.Tensor((1, 16, 768), dtype="float32")])
            lv499: R.Tensor((1, 16, 768), dtype="float32") = lv498[0]
            lv500: R.Tensor((1, 16, 768), dtype="float32") = lv498[1]
            lv501: R.Tensor((1, 16, 768), dtype="float32") = lv498[2]
            lv502 = R.call_tir(cls.reshape5, (lv500,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv503 = R.call_tir(cls.transpose, (lv502,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv504 = R.call_tir(cls.reshape5, (lv501,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv505 = R.call_tir(cls.transpose, (lv504,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv506 = R.call_tir(cls.reshape5, (lv499,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv507 = R.call_tir(cls.transpose, (lv506,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv511 = R.call_tir(cls.transpose1, (lv507,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv512 = R.call_tir(cls.transpose1, (lv503,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv513 = R.call_tir(cls.transpose1, (lv505,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv514 = R.call_tir(cls.attention_bias, (lv511, lv512, lv513, metadata["relax.expr.Constant"][136]), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv515 = R.call_tir(cls.transpose, (lv514,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv516 = R.call_tir(cls.transpose1, (lv515,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv517 = R.call_tir(cls.reshape6, (lv516,), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv518 = R.call_tir(cls.reshape3, (lv517,), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv519 = R.call_tir(cls.matmul1, (lv518, metadata["relax.expr.Constant"][137]), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv520 = R.call_tir(cls.add2, (metadata["relax.expr.Constant"][138], lv519), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv521 = R.call_tir(cls.reshape2, (lv520,), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv522 = R.call_tir(cls.add, (lv521, lv492), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv523 = R.call_tir(cls.layer_norm, (lv522, metadata["relax.expr.Constant"][139], metadata["relax.expr.Constant"][140]), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv524 = R.call_tir(cls.reshape3, (lv523,), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv525 = R.call_tir(cls.matmul2, (lv524, metadata["relax.expr.Constant"][141]), out_sinfo=R.Tensor((16, 3072), dtype="float32"))
            lv526 = R.call_tir(cls.add3, (metadata["relax.expr.Constant"][142], lv525), out_sinfo=R.Tensor((16, 3072), dtype="float32"))
            lv527 = R.call_tir(cls.reshape7, (lv526,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv528 = R.call_tir(cls.multiply, (lv527,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv529 = R.call_tir(cls.power, (lv527,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv530 = R.call_tir(cls.multiply1, (lv529,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv531 = R.call_tir(cls.add4, (lv527, lv530), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv532 = R.call_tir(cls.multiply2, (lv531,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv533 = R.call_tir(cls.tir_tanh, (lv532,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv534 = R.call_tir(cls.add5, (lv533,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv535 = R.call_tir(cls.multiply3, (lv528, lv534), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv536 = R.call_tir(cls.reshape8, (lv535,), out_sinfo=R.Tensor((16, 3072), dtype="float32"))
            lv537 = R.call_tir(cls.matmul3, (lv536, metadata["relax.expr.Constant"][143]), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv538 = R.call_tir(cls.add2, (metadata["relax.expr.Constant"][144], lv537), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv539 = R.call_tir(cls.reshape2, (lv538,), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv540 = R.call_tir(cls.add, (lv522, lv539), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv541 = R.call_tir(cls.layer_norm, (lv540, metadata["relax.expr.Constant"][145], metadata["relax.expr.Constant"][146]), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv542 = R.call_tir(cls.reshape3, (lv541,), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv543 = R.call_tir(cls.matmul, (lv542, metadata["relax.expr.Constant"][147]), out_sinfo=R.Tensor((16, 2304), dtype="float32"))
            lv544 = R.call_tir(cls.add1, (metadata["relax.expr.Constant"][148], lv543), out_sinfo=R.Tensor((16, 2304), dtype="float32"))
            lv545 = R.call_tir(cls.reshape4, (lv544,), out_sinfo=R.Tensor((1, 16, 2304), dtype="float32"))
            lv546 = R.call_tir(cls.split, (lv545,), out_sinfo=[R.Tensor((1, 16, 768), dtype="float32"), R.Tensor((1, 16, 768), dtype="float32"), R.Tensor((1, 16, 768), dtype="float32")])
            lv547: R.Tensor((1, 16, 768), dtype="float32") = lv546[0]
            lv548: R.Tensor((1, 16, 768), dtype="float32") = lv546[1]
            lv549: R.Tensor((1, 16, 768), dtype="float32") = lv546[2]
            lv550 = R.call_tir(cls.reshape5, (lv548,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv551 = R.call_tir(cls.transpose, (lv550,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv552 = R.call_tir(cls.reshape5, (lv549,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv553 = R.call_tir(cls.transpose, (lv552,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv554 = R.call_tir(cls.reshape5, (lv547,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv555 = R.call_tir(cls.transpose, (lv554,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv559 = R.call_tir(cls.transpose1, (lv555,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv560 = R.call_tir(cls.transpose1, (lv551,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv561 = R.call_tir(cls.transpose1, (lv553,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv562 = R.call_tir(cls.attention_bias, (lv559, lv560, lv561, metadata["relax.expr.Constant"][149]), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv563 = R.call_tir(cls.transpose, (lv562,), out_sinfo=R.Tensor((1, 12, 16, 64), dtype="float32"))
            lv564 = R.call_tir(cls.transpose1, (lv563,), out_sinfo=R.Tensor((1, 16, 12, 64), dtype="float32"))
            lv565 = R.call_tir(cls.reshape6, (lv564,), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv566 = R.call_tir(cls.reshape3, (lv565,), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv567 = R.call_tir(cls.matmul1, (lv566, metadata["relax.expr.Constant"][150]), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv568 = R.call_tir(cls.add2, (metadata["relax.expr.Constant"][151], lv567), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv569 = R.call_tir(cls.reshape2, (lv568,), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv570 = R.call_tir(cls.add, (lv569, lv540), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv571 = R.call_tir(cls.layer_norm, (lv570, metadata["relax.expr.Constant"][152], metadata["relax.expr.Constant"][153]), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv572 = R.call_tir(cls.reshape3, (lv571,), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv573 = R.call_tir(cls.matmul2, (lv572, metadata["relax.expr.Constant"][154]), out_sinfo=R.Tensor((16, 3072), dtype="float32"))
            lv574 = R.call_tir(cls.add3, (metadata["relax.expr.Constant"][155], lv573), out_sinfo=R.Tensor((16, 3072), dtype="float32"))
            lv575 = R.call_tir(cls.reshape7, (lv574,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv576 = R.call_tir(cls.multiply, (lv575,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv577 = R.call_tir(cls.power, (lv575,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv578 = R.call_tir(cls.multiply1, (lv577,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv579 = R.call_tir(cls.add4, (lv575, lv578), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv580 = R.call_tir(cls.multiply2, (lv579,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv581 = R.call_tir(cls.tir_tanh, (lv580,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv582 = R.call_tir(cls.add5, (lv581,), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv583 = R.call_tir(cls.multiply3, (lv576, lv582), out_sinfo=R.Tensor((1, 16, 3072), dtype="float32"))
            lv584 = R.call_tir(cls.reshape8, (lv583,), out_sinfo=R.Tensor((16, 3072), dtype="float32"))
            lv585 = R.call_tir(cls.matmul3, (lv584, metadata["relax.expr.Constant"][156]), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv586 = R.call_tir(cls.add2, (metadata["relax.expr.Constant"][157], lv585), out_sinfo=R.Tensor((16, 768), dtype="float32"))
            lv587 = R.call_tir(cls.reshape2, (lv586,), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv588 = R.call_tir(cls.add, (lv570, lv587), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv589 = R.call_tir(cls.layer_norm, (lv588, metadata["relax.expr.Constant"][158], metadata["relax.expr.Constant"][159]), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            lv590 = R.call_tir(cls.reshape9, (lv589,), out_sinfo=R.Tensor((1, 16, 768), dtype="float32"))
            gv: R.Tuple(R.Tensor((1, 16, 768), dtype="float32")) = (lv590,)
            R.output(gv)
        return gv

# Metadata omitted. Use show_meta=True in script() method to show it.