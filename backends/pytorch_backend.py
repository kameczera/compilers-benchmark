# -*- coding: utf-8 -*-
from __future__ import annotations
import os, io, time, math, ast, re
from pathlib import Path
from typing import Dict, Any, Tuple
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- imports robustos ---
try:
    from .common import now_ms, sync_cuda, pretty_shape, estimate_energy_j
except Exception:
    try:
        from backends.common import now_ms, sync_cuda, pretty_shape, estimate_energy_j
    except Exception:
        from cnnbench.backends.common import now_ms, sync_cuda, pretty_shape, estimate_energy_j

try:
    from ..models.resnet_torch import get_resnet, to_dtype
except Exception:
    try:
        from models.resnet_torch import get_resnet, to_dtype
    except Exception:
        from cnnbench.models.resnet_torch import get_resnet, to_dtype


# ====== Kernel counters (adaptados do main.py) ======
class InductorKernelCounter(ast.NodeVisitor):
    def __init__(self, source_text=None):
        self.triton_runs = {}
        self.extern_calls = {}
        self.fused_ops = {}
        self.source_text = source_text
        if source_text:
            self._parse_fused_ops_from_comments(source_text)

    def _parse_fused_ops_from_comments(self, text: str):
        patt = re.compile(r"# Topologically Sorted Source Nodes.*?(?=triton_[\w]+ ?=)", re.DOTALL)
        for block in patt.findall(text):
            m = re.search(r"(triton_\w+)\s*=", text[text.find(block):])
            if not m:
                continue
            kname = m.group(1)
            ops = []
            m2 = re.search(r"Original ATen:\s*\[(.*?)\]", block)
            if m2:
                ops += [o.strip() for o in m2.group(1).split(",")]
            for line in block.splitlines():
                if "=>" in line:
                    rhs = line.split("=>")[1]
                    ops += [o.strip() for o in rhs.split(",")]
            self.fused_ops[kname] = ops

    def visit_Call(self, node: ast.Call):
        # triton kernel launches: triton_X.run(...)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "run":
            base = node.func.value
            if isinstance(base, ast.Name) and base.id.startswith("triton_"):
                self.triton_runs[base.id] = self.triton_runs.get(base.id, 0) + 1
        # extern_kernels.*(...)
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "extern_kernels":
                fn = node.func.attr
                self.extern_calls[fn] = self.extern_calls.get(fn, 0) + 1
        self.generic_visit(node)


def _save_text(path: Path, txt: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(txt, encoding="utf-8")
    return str(path)

def _count_triton_from_text(txt: str) -> Dict[str, Any]:
    # txt é um arquivo .py com kernels gerados (output_code)
    try:
        tree = ast.parse(txt)
    except SyntaxError:
        tree = None
    c = InductorKernelCounter(source_text=txt)
    if tree is not None:
        c.visit(tree)
    total_triton = sum(c.triton_runs.values()) if c.triton_runs else 0
    total_extern = sum(c.extern_calls.values()) if c.extern_calls else 0
    return {
        "summary": {
            "triton_launches_total": total_triton,
            "triton_kernels_unicos": len(c.triton_runs),
            "extern_kernels_total": total_extern,
            "extern_funcs_unicas": len(c.extern_calls),
        },
        "details": {
            "triton_runs": c.triton_runs,
            "extern_calls": c.extern_calls,
            "fused_ops": c.fused_ops,
        }
    }

# ====== BN fold (inferência) ======
def fold_bn_inplace(model: nn.Module) -> nn.Module:
    """
    Dobra BN em Conv para inferência (eval). Suporta padrões comuns de ResNet.
    Mantém ReLU separada; epílogo ReLU será fundido pelo Inductor quando possível.
    """
    model.eval()

    def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
        # similar a torch.nn.utils.fusion.fuse_conv_bn_eval
        w = conv.weight
        b = conv.bias if conv.bias is not None else torch.zeros(w.size(0), device=w.device, dtype=w.dtype)
        bn_w = bn.weight if bn.affine else torch.ones_like(bn.running_var)
        bn_b = bn.bias if bn.affine else torch.zeros_like(bn.running_var)
        invstd = torch.rsqrt(bn.running_var + bn.eps)
        w_f = w * (bn_w * invstd).reshape([-1, 1, 1, 1])
        b_f = (b - bn.running_mean) * invstd * bn_w + bn_b
        conv.weight = nn.Parameter(w_f)
        conv.bias = nn.Parameter(b_f)
        return conv

    # aplica em blocos típicos (conv/bn ou conv->bn->relu em sequenciais)
    for m in model.modules():
        if isinstance(m, nn.Sequential) and len(m) >= 2:
            for i in range(len(m) - 1):
                if isinstance(m[i], nn.Conv2d) and isinstance(m[i+1], nn.BatchNorm2d):
                    _fuse_conv_bn(m[i], m[i+1])
                    # remove BN substituindo por Identity
                    m[i+1] = nn.Identity()
        # padrões da ResNet básica
        if hasattr(m, "conv1") and hasattr(m, "bn1") and isinstance(m.conv1, nn.Conv2d) and isinstance(m.bn1, nn.BatchNorm2d):
            _fuse_conv_bn(m.conv1, m.bn1)
            m.bn1 = nn.Identity()
        if hasattr(m, "conv2") and hasattr(m, "bn2") and isinstance(m.conv2, nn.Conv2d) and isinstance(m.bn2, nn.BatchNorm2d):
            _fuse_conv_bn(m.conv2, m.bn2)
            m.bn2 = nn.Identity()
        if hasattr(m, "downsample") and isinstance(m.downsample, nn.Sequential):
            ds = m.downsample
            for i in range(len(ds) - 1):
                if isinstance(ds[i], nn.Conv2d) and isinstance(ds[i+1], nn.BatchNorm2d):
                    _fuse_conv_bn(ds[i], ds[i+1])
                    ds[i+1] = nn.Identity()
    return model


def _compile_and_dump_code(model: nn.Module, x: torch.Tensor, out_dir: Path, tag: str) -> Tuple[float, float, float, str, Dict[str, Any]]:
    """
    Compila com Inductor, captura 'output_code' (Triton + extern) e mede tempos.
    Retorna: (compile_ms_proxy, ttfb_ms, exec_ms_avg, code_path, kernel_counts)
    """
    # habilita dump de código no stdout
    os.environ["TORCH_LOGS"] = "output_code"
    os.environ["TORCHINDUCTOR_DISABLE_CACHE"] = "1"
    compiled = torch.compile(model, backend="inductor", mode="max-autotune")

    # primeira chamada = compile + run (capturando o output_code)
    buf = io.StringIO()
    with torch.inference_mode(), contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        t0 = now_ms()
        y = compiled(x)
        if hasattr(y, "contiguous"):
            _ = y.contiguous()
        sync_cuda()
        t1 = now_ms()
    ttfb_ms = t1 - t0
    code_txt = buf.getvalue()

    # salva código e conta kernels
    out_dir.mkdir(parents=True, exist_ok=True)
    code_path = out_dir / f"{tag}_inductor_output_code.py"
    _save_text(code_path, code_txt)
    kc = _count_triton_from_text(code_txt)

    # execuções subsequentes
    with torch.inference_mode():
        t0 = now_ms()
        iters = 50
        for _ in range(iters):
            y = compiled(x)
        sync_cuda()
        t1 = now_ms()
    exec_ms = (t1 - t0) / iters

    # proxy de compile_ms: usamos o ttfb_ms (compilação + 1st run) e subtraímos exec médio
    compile_ms = max(0.0, ttfb_ms - exec_ms)
    return compile_ms, ttfb_ms, exec_ms, str(code_path), kc


def run_pytorch(model_name: str = "resnet18",
                device: str = "cuda",
                dtype: str = "fp32",
                shape_nchw: Tuple[int,int,int,int] = (16,3,224,224),
                warmup: int = 10,
                iters: int = 50,
                power_compile_w: float = 25.0,
                power_exec_w: float = 25.0) -> Dict[str, Any]:

    N, C, H, W = shape_nchw
    torch.backends.cudnn.benchmark = True
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    # modelos
    base = get_resnet(model_name).eval()
    base = to_dtype(base, dtype).to(dev)

    x = torch.randn(N, C, H, W, device=dev, dtype=getattr(torch, dtype.replace("fp", "float")))
    for _ in range(max(1, warmup // 2)):
        with torch.inference_mode():
            _ = base(x)
    sync_cuda()

    # ----- Eager baseline -----
    with torch.inference_mode():
        t0 = now_ms()
        for _ in range(iters):
            _ = base(x)
        sync_cuda()
        t1 = now_ms()
    eager_exec_ms = (t1 - t0) / iters
    eager_ttfb_ms = eager_exec_ms

    out = {
        "meta": {"device": str(dev), "dtype": dtype, "warmup": warmup, "iters": iters},
        "shapes": { pretty_shape(shape_nchw): {} },
        "ir_dump": {}
    }

    # diretório para IRs
    stamp = int(time.time() * 1000)
    out_dir = Path("ir_dumps") / "inductor" / f"{stamp}_{model_name}_{pretty_shape(shape_nchw)}"

    # ----- Inductor padrão -----
    compile_ms, ttfb_ms, exec_ms, code_path, kcount = _compile_and_dump_code(base, x, out_dir, "fused")
    energy = estimate_energy_j(ttfb_ms, exec_ms, iters, power_compile_w, power_exec_w)
    out["shapes"][pretty_shape(shape_nchw)]["fused_inductor"] = {
        "ttfb_ms": float(ttfb_ms),
        "compile_ms": float(compile_ms),
        "exec_ms": float(exec_ms),
        "energy_j": float(energy),
    }
    out["ir_dump"]["fused_inductor"] = {"path": code_path, "kernel_count": kcount}

    # ----- Inductor com BN fold -----
    folded = fold_bn_inplace(get_resnet(model_name)).eval().to(dev)
    folded = to_dtype(folded, dtype)
    compile_ms2, ttfb_ms2, exec_ms2, code_path2, kcount2 = _compile_and_dump_code(folded, x, out_dir, "fused_fold")
    energy2 = estimate_energy_j(ttfb_ms2, exec_ms2, iters, power_compile_w, power_exec_w)
    out["shapes"][pretty_shape(shape_nchw)]["fused_fold_inductor"] = {
        "ttfb_ms": float(ttfb_ms2),
        "compile_ms": float(compile_ms2),
        "exec_ms": float(exec_ms2),
        "energy_j": float(energy2),
    }
    out["ir_dump"]["fused_fold_inductor"] = {"path": code_path2, "kernel_count": kcount2}

    # eager no mesmo bloco
    out["shapes"][pretty_shape(shape_nchw)]["eager"] = {
        "ttfb_ms": float(eager_ttfb_ms),
        "exec_ms": float(eager_exec_ms),
    }

    # speedups
    out["shapes"][pretty_shape(shape_nchw)]["speedup_exec_x"] = \
        float(eager_exec_ms / exec_ms) if exec_ms > 0 else None

    return out