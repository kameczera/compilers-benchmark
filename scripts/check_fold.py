# -*- coding: utf-8 -*-
"""Valida o fold de BatchNorm em Conv (fold_bn_inplace):

1. Depois do fold NÃO pode restar nenhum nn.BatchNorm2d no modelo
   (ResNet-18 e ResNet-50 — inclui o bn3 dos blocos Bottleneck).
2. A saída do modelo dobrado tem de ser numericamente equivalente à do
   original em eval() (mesmos pesos), dentro de tolerância de FP32.

Uso:  python scripts/check_fold.py            # roda em CPU, não precisa de GPU
Sai com código != 0 se qualquer verificação falhar.
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from backends.pytorch_backend import fold_bn_inplace
from models.resnet_torch import get_resnet


def check_model(model_name: str) -> bool:
    torch.manual_seed(0)
    base = get_resnet(model_name).eval()
    # running stats aleatórias para o teste não passar por acaso com var=1/mean=0
    with torch.no_grad():
        for m in base.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.uniform_(-0.5, 0.5)
                m.running_var.uniform_(0.5, 1.5)

    folded = fold_bn_inplace(copy.deepcopy(base)).eval()

    bn_left = [n for n, m in folded.named_modules() if isinstance(m, nn.BatchNorm2d)]
    x = torch.randn(2, 3, 224, 224)
    with torch.inference_mode():
        y_ref = base(x)
        y_fold = folded(x)
    max_abs = (y_ref - y_fold).abs().max().item()
    max_rel = ((y_ref - y_fold).abs() / y_ref.abs().clamp_min(1e-6)).max().item()
    ok = not bn_left and torch.allclose(y_ref, y_fold, rtol=1e-3, atol=1e-4)

    status = "OK " if ok else "FAIL"
    print(f"[{status}] {model_name}: BN restantes={len(bn_left)} "
          f"max_abs_diff={max_abs:.3e} max_rel_diff={max_rel:.3e}")
    if bn_left:
        print(f"       BNs não dobrados: {bn_left[:8]}{' ...' if len(bn_left) > 8 else ''}")
    return ok


if __name__ == "__main__":
    ok = all([check_model("resnet18"), check_model("resnet50")])
    sys.exit(0 if ok else 1)
