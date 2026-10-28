# -*- coding: utf-8 -*-
"""Shared paths and model loading for tools/ and tests/."""

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CKPT = ROOT / "checkpoints" / "dqn_snake_best.pt"
PLAY = ROOT / "play"
MODEL_JS = PLAY / "model" / "snake_cnn_fp16.js"
RESULT_JSON = ROOT / "results" / "result.json"

# Order matters: this is the byte order inside the exported fp16 blob.
TENSOR_ORDER = ["conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias",
                "conv3.weight", "conv3.bias", "fc1.weight", "fc1.bias",
                "fc2.weight", "fc2.bias"]


def load_checkpoint(path=CKPT):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def build_net(ck, fp16=False):
    """SnakeCNN from a checkpoint dict; fp16=True rounds every weight to float16 and back
    (exactly what the browser sees)."""
    import torch
    from model import SnakeCNN
    net = SnakeCNN(ck["n"], ck["channels"])
    sd = ck["model"]
    if fp16:
        sd = {k: v.to(torch.float16).to(torch.float32) for k, v in sd.items()}
    net.load_state_dict(sd)
    net.eval()
    return net
