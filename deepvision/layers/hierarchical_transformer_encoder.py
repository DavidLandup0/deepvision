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

from deepvision.layers.efficient_attention import EfficientAttention
from deepvision.layers.mix_ffn import MixFFN
from deepvision.layers.droppath import DropPath


class __HierarchicalTransformerEncoderPT(nn.Module):
    def __init__(self, project_dim, num_heads, sr_ratio=1, drop_prob=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(project_dim)
        self.attn = EfficientAttention(project_dim, num_heads, sr_ratio)
        self.drop_path = (
            DropPath(drop_prob, backend="pytorch") if drop_prob else nn.Identity()
        )
        self.norm2 = nn.LayerNorm(project_dim)
        self.mlp = MixFFN(project_dim, int(project_dim * 4))

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


LAYER_BACKBONES = {
    "tensorflow": None,
    "pytorch": __HierarchicalTransformerEncoderPT,
}


def HierarchicalTransformerEncoder(
    project_dim, num_heads, sr_ratio, drop_prob, backend="pytorch"
):

    layer_class = LAYER_BACKBONES.get(backend)
    if layer_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {LAYER_BACKBONES.keys()}"
        )

    layer = layer_class(
        project_dim=project_dim,
        num_heads=num_heads,
        sr_ratio=sr_ratio,
        drop_prob=drop_prob,
    )

    return layer
