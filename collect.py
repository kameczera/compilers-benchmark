from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> str | None:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception:
        return None


def main() -> None:
    info = {
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "env": {
            "CONDA_DEFAULT_ENV": os.environ.get("CONDA_DEFAULT_ENV"),
            "CUDA_HOME": os.environ.get("CUDA_HOME"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
            "XLA_FLAGS": os.environ.get("XLA_FLAGS"),
            "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get(
                "XLA_PYTHON_CLIENT_PREALLOCATE"
            ),
            "XLA_PYTHON_CLIENT_MEM_FRACTION": os.environ.get(
                "XLA_PYTHON_CLIENT_MEM_FRACTION"
            ),
        },
        "commands": {
            "nvidia_smi": run_cmd(["nvidia-smi"]),
            "nvcc_version": run_cmd(["nvcc", "--version"]) if shutil.which("nvcc") else None,
            "llvm_config_version": run_cmd(["llvm-config", "--version"]) if shutil.which("llvm-config") else None,
            "gcc_version": run_cmd(["gcc", "--version"]) if shutil.which("gcc") else None,
            "conda_list": run_cmd(["conda", "list"]) if shutil.which("conda") else None,
            "pip_freeze": run_cmd([sys.executable, "-m", "pip", "freeze"]),
        },
        "python_packages": {},
    }

    try:
        import torch

        info["python_packages"]["torch"] = {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
            "devices": [
                torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            ],
        }
    except Exception as e:
        info["python_packages"]["torch_error"] = repr(e)

    try:
        import torchvision

        info["python_packages"]["torchvision"] = {
            "version": torchvision.__version__,
        }
    except Exception as e:
        info["python_packages"]["torchvision_error"] = repr(e)

    try:
        import jax
        import jaxlib
        from jax._src.lib import cuda_versions

        info["python_packages"]["jax"] = {
            "version": jax.__version__,
            "jaxlib_version": jaxlib.__version__,
            "default_backend": jax.default_backend(),
            "devices": [str(d) for d in jax.devices()],
            "cudnn_build_version": int(cuda_versions.cudnn_build_version()),
            "cudnn_runtime_version": int(cuda_versions.cudnn_get_version()),
        }
    except Exception as e:
        info["python_packages"]["jax_error"] = repr(e)

    try:
        import flax

        info["python_packages"]["flax"] = {
            "version": flax.__version__,
        }
    except Exception as e:
        info["python_packages"]["flax_error"] = repr(e)

    try:
        import flaxmodels

        info["python_packages"]["flaxmodels"] = {
            "version": getattr(flaxmodels, "__version__", "unknown"),
        }
    except Exception as e:
        info["python_packages"]["flaxmodels_error"] = repr(e)

    try:
        import triton

        info["python_packages"]["triton"] = {
            "version": triton.__version__,
        }
    except Exception as e:
        info["python_packages"]["triton_error"] = repr(e)

    out_dir = Path("env_reports")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "environment_report.json"
    out_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] saved {out_path}")


if __name__ == "__main__":
    main()
