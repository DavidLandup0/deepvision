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


class __OverlappingPatchingAndEmbeddingPT(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, patch_size=7, stride=4):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        self.norm = nn.LayerNorm(dim=out_channels)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


LAYER_BACKBONES = {
    "tensorflow": None,
    "pytorch": __OverlappingPatchingAndEmbeddingPT,
}


def OverlappingPatchingAndEmbedding(
    in_channels=3, out_channels=32, patch_size=7, stride=4, backend="pytorch"
):

    layer_class = LAYER_BACKBONES.get(backend)
    if layer_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {LAYER_BACKBONES.keys()}"
        )

    layer = layer_class(
        in_channels=in_channels,
        out_channels=out_channels,
        patch_size=patch_size,
        stride=stride,
    )

    return layer
