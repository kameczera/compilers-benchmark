#!/usr/bin/env python3
"""Re-measure BERT/GPT-2 graph and kernel counts used by the paper figures.

The script is intentionally backend-selective so each compiler can run in its
own pinned environment:

  conda run -n tvm-build-venv python scripts/remeasure_transformer_ir.py \
      --backend inductor --model bert

Models use the standard Hugging Face base configurations with random weights:
BERT-base (12 layers, hidden size 768) and GPT-2 small (12 layers, hidden size
768). No network access or pretrained checkpoint is required.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "transformer_ir"


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_result(backend: str, model: str, payload: dict[str, Any]) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{backend}_{model}.json"
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return out


def _graph_summary(graph) -> dict[str, Any]:
    by_kind = Counter(node.op for node in graph.nodes)
    return {
        "nodes_total": sum(by_kind.values()),
        "nodes_by_kind": dict(sorted(by_kind.items())),
    }


def _torch_model(model_name: str):
    import torch
    from torch import nn
    from transformers import BertConfig, BertModel, GPT2Config, GPT2Model

    class TransformerForBenchmark(nn.Module):
        def __init__(self, name: str):
            super().__init__()
            self.name = name
            if name == "bert":
                config = BertConfig(
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                    hidden_dropout_prob=0.1,
                    attention_probs_dropout_prob=0.1,
                    attn_implementation="eager",
                    return_dict=False,
                )
                self.model = BertModel(config, add_pooling_layer=False)
            else:
                config = GPT2Config(
                    n_embd=768,
                    n_layer=12,
                    n_head=12,
                    n_inner=3072,
                    resid_pdrop=0.1,
                    embd_pdrop=0.1,
                    attn_pdrop=0.1,
                    attn_implementation="eager",
                    use_cache=False,
                    return_dict=False,
                )
                self.model = GPT2Model(config)
            self.config_dict = config.to_dict()

        def forward(self, input_ids):
            if self.name == "gpt2":
                return self.model(
                    input_ids=input_ids,
                    use_cache=False,
                    return_dict=False,
                )[0]
            return self.model(input_ids=input_ids, return_dict=False)[0]

    model = TransformerForBenchmark(model_name).eval()
    vocab_size = model.model.config.vocab_size
    return model, vocab_size, torch


def _torch_graph_counts(model, input_ids) -> dict[str, Any]:
    """Capture two graph views; neither count is silently substituted."""
    import torch

    captured: dict[str, Any] = {}
    try:
        from transformers.utils.fx import symbolic_trace

        traced = symbolic_trace(model.model, input_names=["input_ids"])
        captured["huggingface_fx"] = _graph_summary(traced.graph)
    except Exception as exc:
        captured["huggingface_fx_error"] = f"{type(exc).__name__}: {exc}"

    try:
        exported = torch._dynamo.export(model)(input_ids)
        captured["dynamo_fx"] = _graph_summary(exported.graph_module.graph)
    except Exception as exc:
        captured["dynamo_fx_error"] = f"{type(exc).__name__}: {exc}"
    return captured


def run_inductor(model_name: str, batch: int, seq_len: int) -> dict[str, Any]:
    # These must be set before importing torch/Inductor.
    cache_dir = Path(tempfile.mkdtemp(prefix=f"cnnbench-{model_name}-inductor-"))
    os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir / "triton")

    model, vocab_size, torch = _torch_model(model_name)
    import transformers
    from backends.pytorch_backend import _compile_and_dump_code

    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_ids = torch.randint(
        0,
        vocab_size,
        (batch, seq_len),
        dtype=torch.int64,
        device=device,
    )
    graphs = _torch_graph_counts(model, input_ids)

    stamp = int(time.time() * 1000)
    artifact_dir = (
        ROOT
        / "ir_dumps"
        / "transformers"
        / "inductor"
        / f"{stamp}_{model_name}_{batch}x{seq_len}"
    )
    compile_ms, ttfb_ms, exec_stats, code_path, kernel_count = (
        _compile_and_dump_code(
            model,
            input_ids,
            artifact_dir,
            "fused",
            warmup=1,
            iters=3,
            profile_kernel_names=True,
        )
    )
    return {
        "backend": "inductor",
        "model": model_name,
        "input": {"batch": batch, "seq_len": seq_len},
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "model_config": model.config_dict,
        "graph_counts": graphs,
        "kernel_count": kernel_count,
        "timing_sanity_check": {
            "compile_ms": compile_ms,
            "ttfb_ms": ttfb_ms,
            "exec_ms": exec_stats["mean"],
        },
        "artifact": code_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        required=True,
        choices=("inductor",),
        help="Compiler backend; additional backends are added after validation.",
    )
    parser.add_argument("--model", required=True, choices=("bert", "gpt2"))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Sequence length (default: 64 for BERT and 16 for GPT-2).",
    )
    args = parser.parse_args()

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    seq_len = args.seq_len
    if seq_len is None:
        seq_len = 64 if args.model == "bert" else 16
    payload = run_inductor(args.model, args.batch, seq_len)
    path = _write_result(args.backend, args.model, payload)
    summary = payload["kernel_count"]["summary"]
    print(json.dumps({"result": str(path), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
