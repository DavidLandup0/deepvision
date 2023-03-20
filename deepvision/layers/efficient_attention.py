# Copyright 2023 David Landup
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn

"""
Based on: https://github.com/sithu31296/semantic-segmentation/blob/main/semseg/models/backbones/mit.py
"""


class __EfficientAttentionPT(nn.Module):
    def __init__(self, project_dim, num_heads, sr_ratio):
        super().__init__()
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        self.scale = (project_dim // num_heads) ** -0.5
        self.q = nn.Linear(project_dim, project_dim)
        self.kv = nn.Linear(project_dim, project_dim * 2)
        self.proj = nn.Linear(project_dim, project_dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(project_dim, project_dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(project_dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)

        k, v = (
            self.kv(x)
            .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


LAYER_BACKBONES = {
    "tensorflow": None,
    "pytorch": __EfficientAttentionPT,
}


def EfficientAttention(project_dim, num_heads, sr_ratio, backend="pytorch"):

    layer_class = LAYER_BACKBONES.get(backend)
    if layer_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {LAYER_BACKBONES.keys()}"
        )

    layer = layer_class(project_dim=project_dim, num_heads=num_heads, sr_ratio=sr_ratio)

    return layer
