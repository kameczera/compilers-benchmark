
# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50

def get_resnet(model_name: str = "resnet18", num_classes: int = 1000) -> nn.Module:
    if model_name.lower() == "resnet50":
        m = resnet50(weights=None, num_classes=num_classes)
    else:
        m = resnet18(weights=None, num_classes=num_classes)
    m.eval()
    return m

def to_dtype(m: torch.nn.Module, dtype: str = "fp32") -> torch.nn.Module:
    if dtype == "bf16":
        return m.to(torch.bfloat16)
    if dtype == "fp16":
        return m.to(torch.float16)
    return m.to(torch.float32)
