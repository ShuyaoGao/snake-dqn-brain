# -*- coding: utf-8 -*-
"""CNN Q 网络：扫描棋盘 → 四方向 Q 值。带激活捕获接口供可视化。"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn


class SnakeCNN(nn.Module):
    def __init__(self, n=12, channels=5):
        super().__init__()
        self.n = n
        self.conv1 = nn.Conv2d(channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * n * n, 256)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x, capture=None):
        """capture 传入 dict 时记录各层激活（可视化用）。"""
        a1 = self.relu(self.conv1(x))
        a2 = self.relu(self.conv2(a1))
        a3 = self.relu(self.conv3(a2))
        h = self.relu(self.fc1(a3.flatten(1)))
        q = self.fc2(h)
        if capture is not None:
            capture["conv1"] = a1
            capture["conv2"] = a2
            capture["conv3"] = a3
            capture["fc1"] = h
        return q
