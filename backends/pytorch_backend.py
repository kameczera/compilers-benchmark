# -*- coding: utf-8 -*-
from __future__ import annotations
import copy
import os, io, time, math, ast, re
from pathlib import Path
from typing import Dict, Any, Tuple
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- imports robustos ---
try:
    from .common import now_ms, sync_cuda, pretty_shape, estimate_energy_j, stats_ms
except Exception:
    try:
        from backends.common import now_ms, sync_cuda, pretty_shape, estimate_energy_j, stats_ms
    except Exception:
        from cnnbench.backends.common import now_ms, sync_cuda, pretty_shape, estimate_energy_j, stats_ms

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

def _pycodecache_files() -> set:
    """Arquivos .py gerados pelo Inductor já carregados neste processo."""
    try:
        from torch._inductor.codecache import PyCodeCache
    except Exception:
        return set()
    mods = getattr(PyCodeCache, "modules", None)
    if mods is None:
        mods = list(getattr(PyCodeCache, "cache", {}).values())
    files = set()
    for m in mods:
        f = getattr(m, "__file__", None)
        if f:
            files.add(f)
    return files

def _inductor_cache_dir() -> str:
    try:
        from torch._inductor.runtime.runtime_utils import cache_dir
        return cache_dir()
    except Exception:
        pass
    try:
        from torch._inductor.codecache import cache_dir  # versões antigas
        return cache_dir()
    except Exception:
        import getpass, tempfile
        return os.path.join(tempfile.gettempdir(), f"torchinductor_{getpass.getuser()}")

def _new_inductor_files(files_before: set, t_start_s: float) -> list:
    """
    Arquivos .py gerados pelo Inductor desde t_start_s: união do PyCodeCache
    (kernels Triton) com uma varredura por mtime no diretório de cache (o
    módulo wrapper, com call()/extern_kernels, não passa pelo PyCodeCache
    em versões recentes do torch).
    """
    found = set(_pycodecache_files() - files_before)
    try:
        root = Path(_inductor_cache_dir())
        if root.is_dir():
            for p in root.rglob("*.py"):
                try:
                    if p.stat().st_mtime >= t_start_s - 1.0:
                        found.add(str(p))
                except OSError:
                    pass
    except Exception:
        pass
    return sorted(found)

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
    direct_aten_calls = {}
    for opname in re.findall(r"torch\.ops\.aten\.([A-Za-z0-9_.]+)\(", txt):
        direct_aten_calls[opname] = direct_aten_calls.get(opname, 0) + 1
    return {
        "summary": {
            "triton_launches_total": total_triton,
            "triton_kernels_unicos": len(c.triton_runs),
            "extern_kernels_total": total_extern,
            "extern_funcs_unicas": len(c.extern_calls),
            "direct_aten_calls_total": sum(direct_aten_calls.values()),
        },
        "details": {
            "triton_runs": c.triton_runs,
            "extern_calls": c.extern_calls,
            "direct_aten_calls": direct_aten_calls,
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
        # padrões da ResNet: BasicBlock (conv1/bn1, conv2/bn2) e Bottleneck
        # (conv1..conv3/bn1..bn3 — sem o par 3, o ResNet-50 ficava com os 16
        # bn3 por dobrar e o "fold" media um fold incompleto)
        for idx in (1, 2, 3):
            conv = getattr(m, f"conv{idx}", None)
            bn = getattr(m, f"bn{idx}", None)
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                _fuse_conv_bn(conv, bn)
                setattr(m, f"bn{idx}", nn.Identity())
        if hasattr(m, "downsample") and isinstance(m.downsample, nn.Sequential):
            ds = m.downsample
            for i in range(len(ds) - 1):
                if isinstance(ds[i], nn.Conv2d) and isinstance(ds[i+1], nn.BatchNorm2d):
                    _fuse_conv_bn(ds[i], ds[i+1])
                    ds[i+1] = nn.Identity()
    return model


def _compile_and_dump_code(model: nn.Module, x: torch.Tensor, out_dir: Path, tag: str,
                           warmup: int = 10, iters: int = 50,
                           profile_kernel_names: bool = False) -> Tuple[float, float, Dict[str, Any], str, Dict[str, Any]]:
    """
    Compila com Inductor, captura 'output_code' (Triton + extern) e mede tempos.
    Retorna: (compile_ms_proxy, ttfb_ms, exec_stats, code_path, kernel_counts)
    """
    # Caches persistentes (FXGraphCache em /tmp/torchinductor_$USER) pulam o
    # codegen: compile_ms sai subestimado e nenhum output_code é gerado.
    # TORCHINDUCTOR_FORCE_DISABLE_CACHES precisa existir antes do import torch
    # (run_bench.py faz isso); aqui reforçamos via config em runtime.
    try:
        import torch._inductor.config as inductor_config
        inductor_config.force_disable_caches = True
    except Exception:
        pass
    try:
        torch._dynamo.reset()
    except Exception:
        pass

    files_before = _pycodecache_files()
    t_scan0 = time.time()
    compiled = torch.compile(model, backend="inductor", mode="max-autotune")

    # primeira chamada = compile + run
    buf = io.StringIO()
    with torch.inference_mode(), contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        t0 = now_ms()
        y = compiled(x)
        if hasattr(y, "contiguous"):
            _ = y.contiguous()
        sync_cuda()
        t1 = now_ms()
    ttfb_ms = t1 - t0

    # O código gerado é lido direto dos arquivos que o Inductor produziu
    # (wrapper com call()/extern_kernels + kernels Triton), em vez de depender
    # de TORCH_LOGS=output_code no stdout — TORCH_LOGS só tem efeito se setado
    # antes do import torch.
    srcs = []
    for f in _new_inductor_files(files_before, t_scan0):
        try:
            srcs.append((f, Path(f).read_text(encoding="utf-8")))
        except Exception:
            pass
    # wrapper (def call(...)) primeiro, kernels depois
    srcs.sort(key=lambda fs: (0 if "def call(" in fs[1] else 1, fs[0]))
    parts = [f"# ===== inductor generated file: {f} =====\n{src}" for f, src in srcs]
    code_txt = "\n\n".join(parts)
    capture_method = "inductor_cache_files"
    if not code_txt.strip():
        code_txt = buf.getvalue()
        srcs = [("stdout", code_txt)]
        capture_method = "stdout_fallback"

    # salva código e conta kernels (por arquivo, somando os contadores)
    out_dir.mkdir(parents=True, exist_ok=True)
    code_path = out_dir / f"{tag}_inductor_output_code.py"
    _save_text(code_path, code_txt)
    triton_runs, extern_calls, direct_aten_calls, fused_ops = {}, {}, {}, {}
    for _, src in srcs:
        c = _count_triton_from_text(src)
        for k, v in c["details"]["triton_runs"].items():
            triton_runs[k] = triton_runs.get(k, 0) + v
        for k, v in c["details"]["extern_calls"].items():
            extern_calls[k] = extern_calls.get(k, 0) + v
        for k, v in c["details"]["direct_aten_calls"].items():
            direct_aten_calls[k] = direct_aten_calls.get(k, 0) + v
        fused_ops.update(c["details"]["fused_ops"])
    kc = {
        "summary": {
            "triton_launches_total": sum(triton_runs.values()),
            "triton_kernels_unicos": len(triton_runs),
            "extern_kernels_total": sum(extern_calls.values()),
            "extern_funcs_unicas": len(extern_calls),
            "direct_aten_calls_total": sum(direct_aten_calls.values()),
            "code_size_bytes": len(code_txt.encode("utf-8")),
            "code_lines": code_txt.count("\n") + (1 if code_txt else 0),
        },
        "details": {
            "triton_runs": triton_runs,
            "extern_calls": extern_calls,
            "direct_aten_calls": direct_aten_calls,
            "fused_ops": fused_ops,
        },
        "capture_method": capture_method,
    }
    if kc["summary"]["triton_launches_total"] == 0 and kc["summary"]["extern_kernels_total"] == 0:
        kc["warning"] = (
            "Nenhum kernel capturado: provavelmente o cache do TorchInductor foi "
            "reaproveitado e o codegen nao rodou. Rode 'make clean_inductor_cache' "
            "e repita com TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 (ver README)."
        )

    # warmup pós-compilação (protocolo do artigo: 'warmup' iterações compiladas
    # antes da janela medida — a 1ª chamada acima não conta como warmup)
    with torch.inference_mode():
        for _ in range(max(1, warmup)):
            _ = compiled(x)
        sync_cuda()

    # Optional single-forward dynamic census. This complements static code
    # counting for models with graph breaks or reused compiled subgraphs, where
    # a generated wrapper may execute multiple times in one model invocation.
    if profile_kernel_names and torch.cuda.is_available():
        try:
            from collections import Counter
            from torch.profiler import ProfilerActivity, profile

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            ) as prof:
                with torch.inference_mode():
                    _ = compiled(x)
                    sync_cuda()
            cuda_events = [
                event.name
                for event in prof.events()
                if str(event.device_type).lower().endswith("cuda")
            ]
            cpu_aten_events = [
                event.name
                for event in prof.events()
                if (
                    str(event.device_type).lower().endswith("cpu")
                    and event.name.startswith("aten::")
                )
            ]
            kc["dynamic_cuda_kernels"] = {
                "total": len(cuda_events),
                "by_name": dict(Counter(cuda_events).most_common()),
            }
            kc["dynamic_cpu_aten_ops"] = {
                "total": len(cpu_aten_events),
                "by_name": dict(Counter(cpu_aten_events).most_common()),
            }
        except Exception as exc:
            kc["dynamic_cuda_kernels_error"] = f"{type(exc).__name__}: {exc}"

    # janela medida: uma amostra por iteração, com sync por iteração — mesma
    # semântica do block_until_ready do caminho XLA e do dev.sync() do TVM,
    # e é o que permite reportar média ± desvio-padrão
    samples = []
    with torch.inference_mode():
        for _ in range(iters):
            t0 = now_ms()
            _ = compiled(x)
            sync_cuda()
            samples.append(now_ms() - t0)
    exec_stats = stats_ms(samples)

    # proxy de compile_ms: usamos o ttfb_ms (compilação + 1st run) e subtraímos exec médio
    compile_ms = max(0.0, ttfb_ms - exec_stats["mean"])
    return compile_ms, ttfb_ms, exec_stats, str(code_path), kc


def run_pytorch(model_name: str = "resnet18",
                device: str = "cuda",
                dtype: str = "fp32",
                shape_nchw: Tuple[int,int,int,int] = (16,3,224,224),
                warmup: int = 10,
                iters: int = 50,
                power_compile_w: float = 25.0,
                power_exec_w: float = 25.0,
                variant: str = "both") -> Dict[str, Any]:
    if variant not in {"both", "base", "fold"}:
        raise ValueError(f"unknown TorchInductor variant: {variant}")

    N, C, H, W = shape_nchw
    torch.backends.cudnn.benchmark = True
    # Condições padronizadas do artigo: TF32 habilitado e layout channels_last.
    # (cudnn conv já usa TF32 por default; isto habilita também para matmul.)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    # A mesma inicialização alimenta as variantes base e fold, inclusive quando
    # elas são coletadas em filhos independentes pelo protocolo K=5.
    model_seed = 0
    torch.manual_seed(model_seed)
    base = get_resnet(model_name).eval()
    base = to_dtype(base, dtype).to(dev)
    if dev.type == "cuda":
        base = base.to(memory_format=torch.channels_last)

    x = torch.randn(N, C, H, W, device=dev, dtype=getattr(torch, dtype.replace("fp", "float")))
    if dev.type == "cuda":
        x = x.to(memory_format=torch.channels_last)
    for _ in range(max(1, warmup)):
        with torch.inference_mode():
            _ = base(x)
    sync_cuda()

    # ----- Eager baseline (uma amostra por iteração, sync por iteração) -----
    eager_samples = []
    with torch.inference_mode():
        for _ in range(iters):
            t0 = now_ms()
            _ = base(x)
            sync_cuda()
            eager_samples.append(now_ms() - t0)
    eager_stats = stats_ms(eager_samples)
    eager_exec_ms = eager_stats["mean"]
    eager_ttfb_ms = eager_exec_ms

    out = {
        "meta": {
            "device": str(dev),
            "dtype": dtype,
            "warmup": warmup,
            "iters": iters,
            "model_seed": model_seed,
            "base_fold_same_initial_state": True,
        },
        "shapes": { pretty_shape(shape_nchw): {} },
        "ir_dump": {}
    }

    # diretório para IRs
    stamp = int(time.time() * 1000)
    out_dir = Path("ir_dumps") / "inductor" / f"{stamp}_{model_name}_{pretty_shape(shape_nchw)}"

    # ----- Inductor padrão -----
    exec_ms = None
    if variant in {"both", "base"}:
        compile_ms, ttfb_ms, exec_stats, code_path, kcount = _compile_and_dump_code(
            base, x, out_dir, "fused", warmup=warmup, iters=iters
        )
        exec_ms = exec_stats["mean"]
        energy = estimate_energy_j(ttfb_ms, exec_ms, iters, power_compile_w, power_exec_w)
        out["shapes"][pretty_shape(shape_nchw)]["fused_inductor"] = {
            "ttfb_ms": float(ttfb_ms),
            "compile_ms": float(compile_ms),
            "exec_ms": float(exec_ms),
            "exec_ms_std": float(exec_stats["std"]),
            "exec_ms_ci95": float(exec_stats["ci95_halfwidth"]),
            "exec_samples_ms": exec_stats["samples"],
            "energy_j": float(energy),
        }
        out["ir_dump"]["fused_inductor"] = {"path": code_path, "kernel_count": kcount}

    # ----- Inductor com BN fold -----
    if variant in {"both", "fold"}:
        sync_cuda()
        fold_t0 = now_ms()
        folded = copy.deepcopy(base)
        folded = fold_bn_inplace(folded).eval()
        sync_cuda()
        fold_preprocess_ms = now_ms() - fold_t0
        if variant == "fold":
            del base
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        if dev.type == "cuda":
            folded = folded.to(memory_format=torch.channels_last)
        compile_ms2, ttfb_ms2, exec_stats2, code_path2, kcount2 = _compile_and_dump_code(
            folded, x, out_dir, "fused_fold", warmup=warmup, iters=iters
        )
        exec_ms2 = exec_stats2["mean"]
        energy2 = estimate_energy_j(ttfb_ms2, exec_ms2, iters, power_compile_w, power_exec_w)
        out["shapes"][pretty_shape(shape_nchw)]["fused_fold_inductor"] = {
            "ttfb_ms": float(ttfb_ms2),
            "compile_ms": float(compile_ms2),
            "exec_ms": float(exec_ms2),
            "exec_ms_std": float(exec_stats2["std"]),
            "exec_ms_ci95": float(exec_stats2["ci95_halfwidth"]),
            "exec_samples_ms": exec_stats2["samples"],
            "fold_preprocess_ms": float(fold_preprocess_ms),
            "compile_plus_preprocess_ms": float(
                compile_ms2 + fold_preprocess_ms
            ),
            "energy_j": float(energy2),
        }
        out["ir_dump"]["fused_fold_inductor"] = {
            "path": code_path2,
            "kernel_count": kcount2,
        }

    # eager no mesmo bloco
    out["shapes"][pretty_shape(shape_nchw)]["eager"] = {
        "ttfb_ms": float(eager_ttfb_ms),
        "exec_ms": float(eager_exec_ms),
        "exec_ms_std": float(eager_stats["std"]),
        "exec_ms_ci95": float(eager_stats["ci95_halfwidth"]),
        "exec_samples_ms": eager_stats["samples"],
    }

    # speedups
    if exec_ms is not None:
        out["shapes"][pretty_shape(shape_nchw)]["speedup_exec_x"] = (
            float(eager_exec_ms / exec_ms) if exec_ms > 0 else None
        )

    return out
