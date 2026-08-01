# -*- coding: utf-8 -*-
from __future__ import annotations
import time, ast
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import torch
from torch import fx
import tvm
from tvm import relax, dlight
from tvm.relax.frontend.torch import from_fx
from tvm.tir import transform as tir_transform

# --- imports robustos ---
try:
    from .common import pretty_shape, estimate_energy_j, stats_ms
except Exception:
    try:
        from backends.common import pretty_shape, estimate_energy_j, stats_ms
    except Exception:
        from cnnbench.backends.common import pretty_shape, estimate_energy_j, stats_ms

try:
    from ..models.resnet_torch import get_resnet
except Exception:
    try:
        from models.resnet_torch import get_resnet
    except Exception:
        from cnnbench.models.resnet_torch import get_resnet


# ===== Kernel counter (TVM) =====
def _extract_name_or_attr(n: ast.AST):
    if isinstance(n, ast.Name):
        return n.id
    if isinstance(n, ast.Attribute):
        return n.attr
    return None

def _is_call_tir(func: ast.AST) -> bool:
    if isinstance(func, ast.Attribute) and func.attr == "call_tir":
        return True
    if isinstance(func, ast.Name) and func.id == "call_tir":
        return True
    return False

class TVMKernelCounter(ast.NodeVisitor):
    def __init__(self):
        self.call_tir_total = 0
        self.call_tir_funcs = {}
        self.cls_total = 0
        self.cls_funcs = {}

    def visit_Call(self, node: ast.Call):
        if _is_call_tir(node.func) and node.args:
            fn = _extract_name_or_attr(node.args[0])
            if fn:
                self.call_tir_total += 1
                self.call_tir_funcs[fn] = self.call_tir_funcs.get(fn, 0) + 1
        if isinstance(node.func, ast.Attribute):
            obj = node.func.value
            if isinstance(obj, ast.Name) and obj.id in {"cls", "Module"}:
                nm = node.func.attr
                self.cls_total += 1
                self.cls_funcs[nm] = self.cls_funcs.get(nm, 0) + 1
        self.generic_visit(node)

def _dump_tvm_ir(mod: tvm.ir.IRModule, out_path: Path) -> Dict[str, Any]:
    txt = str(mod.script())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(txt, encoding="utf-8")
    size_info = {
        "code_size_bytes": len(txt.encode("utf-8")),
        "code_lines": txt.count("\n") + (1 if txt else 0),
    }
    # contar kernels
    try:
        tree = ast.parse(txt)
        c = TVMKernelCounter()
        c.visit(tree)
        return {
            "path": str(out_path),
            "kernel_count": {
                "summary": {
                    "tvm_call_tir_total": c.call_tir_total,
                    "tvm_call_tir_kernels_unicos": len(c.call_tir_funcs),
                    "tvm_cls_total": c.cls_total,
                    "tvm_cls_kernels_unicos": len(c.cls_funcs),
                    **size_info,
                },
                "details": {
                    "tvm_call_tir_functions": c.call_tir_funcs,
                    "tvm_cls_functions": c.cls_funcs,
                }
            }
        }
    except Exception:
        return {"path": str(out_path), "kernel_count": {"summary": size_info, "details": {}}}

def _target_and_dev(device: str):
    if device == "cuda":
        # Target.from_device preenche os atributos da GPU (max_shared_memory_per_block,
        # max_threads_per_block etc.), exigidos pelas regras do dlight (ex.: GEMV).
        # O Target("cuda") genérico não os tem e quebra com KeyError('key is not in Map').
        try:
            return tvm.target.Target.from_device("cuda"), tvm.cuda(0)
        except Exception:
            return tvm.target.Target("cuda"), tvm.cuda(0)
    return tvm.target.Target("llvm"), tvm.cpu(0)

def _sync(dev, device: str):
    if device == "cuda":
        try: dev.sync()
        except Exception: pass

class _CudaSync:
    def __init__(self, dev, device: str):
        self.dev = dev
        self.device = device
    def __enter__(self):
        _sync(self.dev, self.device)
        return self
    def __exit__(self, exc_type, exc, tb):
        _sync(self.dev, self.device)

def schedule_tir_default_gpu(mod: tvm.ir.IRModule, target: tvm.target.Target) -> tvm.ir.IRModule:
    new_funcs = {}
    for gv, f in mod.functions.items():
        if isinstance(f, tvm.tir.PrimFunc):
            if not (getattr(f, "attrs", None) and "target" in f.attrs):
                f = f.with_attr("target", target)
        new_funcs[gv] = f
    mod = tvm.ir.IRModule(new_funcs, attrs=mod.attrs)
    with target:
        try:
            mod = tir_transform.DefaultGPUSchedule()(mod)
        except tvm.TVMError:
            mod = dlight.ApplyDefaultSchedule(
                dlight.gpu.Matmul(),
                dlight.gpu.GEMV(),
                dlight.gpu.Reduction(),
                dlight.gpu.GeneralReduction(),
                dlight.gpu.Fallback(),
            )(mod)
    return mod

def _build_vm(mod: tvm.ir.IRModule, target, dev, params=None):
    if params:
        mod = relax.transform.BindParams("main", params)(mod)
    with tvm.transform.PassContext(opt_level=3):
        ex = relax.build(mod, target=target)
    return relax.VirtualMachine(ex, dev)

def _bench_vm(vm, inp_nd, dev, device: str, warmup: int, iters: int):
    with _CudaSync(dev, device):
        for _ in range(max(1, warmup)):
            _ = vm["main"](inp_nd)
    # A VM lança kernels de forma assíncrona: o sync por iteração garante que
    # cada amostra cubra a execução inteira na GPU (sem ele o cronômetro fecha
    # antes de a GPU terminar) e dá a mesma semântica de medição dos caminhos
    # Inductor (torch.cuda.synchronize) e XLA (block_until_ready).
    _sync(dev, device)
    samples = []
    for _ in range(max(1, iters)):
        t0 = time.perf_counter() * 1000.0
        _ = vm["main"](inp_nd)
        _sync(dev, device)
        samples.append(time.perf_counter() * 1000.0 - t0)
    return stats_ms(samples)

def run_tvm(model_name: str = "resnet18",
            device: str = "cuda",
            dtype: str = "fp32",
            shape_nchw: Tuple[int,int,int,int] = (16,3,224,224),
            warmup: int = 10,
            iters: int = 50,
            power_compile_w: float = 25.0,
            power_exec_w: float = 25.0,
            variant: str = "both") -> Dict[str, Any]:
    if variant not in {"both", "unfused", "fused"}:
        raise ValueError(f"unknown TVM variant: {variant}")

    model_seed = 0
    torch.manual_seed(model_seed)
    model = get_resnet(model_name).eval()
    with torch.no_grad():
        traced = fx.symbolic_trace(model)

    input_info = [(shape_nchw, "float32")]
    mod = from_fx(traced, input_info)
    mod, params = relax.frontend.detach_params(mod)

    target, dev = _target_and_dev(device)
    inp = np.random.default_rng(model_seed).standard_normal(
        shape_nchw
    ).astype("float32")
    inp_nd = tvm.nd.array(inp, dev)

    stamp = int(time.time() * 1000)
    base_dir = Path("ir_dumps") / "tvm" / f"{stamp}_{model_name}_{pretty_shape(shape_nchw)}"

    # ---- UNFUSED ----
    # A janela de compilação inclui os passes de grafo (Legalize/Fuse*/schedule)
    # e o relax.build, para ser isonômica com o compile_ms do Inductor/XLA,
    # cujo proxy engloba captura + fusão + codegen na primeira chamada.
    stats_u = None
    if variant in {"both", "unfused"}:
        tb0 = time.perf_counter() * 1000.0
        with target:
            mod_unfused = tvm.transform.Sequential([
                relax.transform.LegalizeOps(),
                relax.transform.AnnotateTIROpPattern(),
                relax.transform.FuseOps(),
                relax.transform.FuseTIR(),
            ])(mod)
        mod_unfused = schedule_tir_default_gpu(mod_unfused, target)
        vm_u = _build_vm(mod_unfused, target, dev, params)
        tb1 = time.perf_counter() * 1000.0
        compile_u_ms = tb1 - tb0

        # dump IR (fora da janela de tempo)
        dump_unfused = _dump_tvm_ir(
            mod_unfused, base_dir / "unfused_tvmscript.py"
        )

        _sync(dev, device)
        tf0 = time.perf_counter() * 1000.0
        _ = vm_u["main"](inp_nd)
        _sync(dev, device)
        tf1 = time.perf_counter() * 1000.0
        first_runtime_u_ms = tf1 - tf0
        stats_u = _bench_vm(vm_u, inp_nd, dev, device, warmup, iters)
        tpr_u = stats_u["mean"]

    # ---- FUSED ----
    stats_f = None
    if variant in {"both", "fused"}:
        tb0 = time.perf_counter() * 1000.0
        with target:
            mod_fused = tvm.transform.Sequential([
                relax.transform.FuseTransposeMatmul(),
                relax.transform.LegalizeOps(),
                relax.transform.AnnotateTIROpPattern(),
                relax.transform.FoldConstant(),
                relax.transform.FuseOps(),
                relax.transform.FuseTIR(),
                relax.transform.DeadCodeElimination(),
                dlight.ApplyDefaultSchedule(
                    dlight.gpu.Matmul(),
                    dlight.gpu.GEMV(),
                    dlight.gpu.Reduction(),
                    dlight.gpu.GeneralReduction(),
                    dlight.gpu.Fallback(),
                ),
                relax.transform.RewriteDataflowReshape(),
                relax.transform.ToNonDataflow(),
                relax.transform.RemovePurityChecking(),
                relax.transform.CallTIRRewrite(),
                relax.transform.StaticPlanBlockMemory(),
                relax.transform.RewriteCUDAGraph(),
                relax.transform.LowerAllocTensor(),
                relax.transform.KillAfterLastUse(),
                relax.transform.LowerRuntimeBuiltin(),
                relax.transform.AttachGlobalSymbol(),
            ])(mod)
        vm_f = _build_vm(mod_fused, target, dev, params)
        tb1 = time.perf_counter() * 1000.0
        compile_f_ms = tb1 - tb0

        # dump IR (fora da janela de tempo)
        dump_fused = _dump_tvm_ir(
            mod_fused, base_dir / "fused_tvmscript.py"
        )

        _sync(dev, device)
        tf0 = time.perf_counter() * 1000.0
        _ = vm_f["main"](inp_nd)
        _sync(dev, device)
        tf1 = time.perf_counter() * 1000.0
        first_runtime_f_ms = tf1 - tf0
        stats_f = _bench_vm(vm_f, inp_nd, dev, device, warmup, iters)
        tpr_f = stats_f["mean"]

    shape_out = {}
    ir_dump = {}
    if stats_u is not None:
        energy_unfused = estimate_energy_j(
            first_runtime_u_ms, tpr_u, iters, power_compile_w, power_exec_w
        )
        shape_out["unfused"] = {
            "ttfb_ms": float(first_runtime_u_ms),
            "compile_ms": float(compile_u_ms),
            "exec_ms": float(tpr_u),
            "exec_ms_std": float(stats_u["std"]),
            "exec_ms_ci95": float(stats_u["ci95_halfwidth"]),
            "exec_samples_ms": stats_u["samples"],
            "energy_j": float(energy_unfused),
        }
        ir_dump["unfused"] = dump_unfused
    if stats_f is not None:
        energy_fused = estimate_energy_j(
            first_runtime_f_ms, tpr_f, iters, power_compile_w, power_exec_w
        )
        shape_out["fused"] = {
            "ttfb_ms": float(first_runtime_f_ms),
            "compile_ms": float(compile_f_ms),
            "exec_ms": float(tpr_f),
            "exec_ms_std": float(stats_f["std"]),
            "exec_ms_ci95": float(stats_f["ci95_halfwidth"]),
            "exec_samples_ms": stats_f["samples"],
            "energy_j": float(energy_fused),
        }
        ir_dump["fused"] = dump_fused
    if stats_u is not None and stats_f is not None:
        shape_out["speedup_exec_x"] = float(tpr_u / tpr_f) if tpr_f > 0 else None

    out = {
        "meta": {
            "device": device,
            "dtype": dtype,
            "warmup": warmup,
            "iters": iters,
            "variant": variant,
            "model_seed": model_seed,
        },
        "shapes": {pretty_shape(shape_nchw): shape_out},
        "ir_dump": ir_dump,
    }
    return out
