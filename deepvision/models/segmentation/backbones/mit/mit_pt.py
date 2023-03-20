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
from torch import Tensor, nn
from torch.nn import functional as F

from deepvision.layers import (
    HierarchicalTransformerEncoder,
    OverlappingPatchingAndEmbedding,
)

# from semseg.models.layers import DropPath


class __MiTPT(nn.Module):
    def __init__(self, embed_dims, depths):
        super().__init__()
        drop_path_rate = 0.1
        self.channels = embed_dims

        # patch_embed
        self.patch_embed1 = OverlappingPatchingAndEmbedding(
            3, embed_dims[0], 7, 4, backend="pytorch"
        )
        self.patch_embed2 = OverlappingPatchingAndEmbedding(
            embed_dims[0], embed_dims[1], 3, 2, backend="pytorch"
        )
        self.patch_embed3 = OverlappingPatchingAndEmbedding(
            embed_dims[1], embed_dims[2], 3, 2, backend="pytorch"
        )
        self.patch_embed4 = OverlappingPatchingAndEmbedding(
            embed_dims[2], embed_dims[3], 3, 2, backend="pytorch"
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = 0
        self.block1 = nn.ModuleList(
            [
                HierarchicalTransformerEncoder(
                    embed_dims[0], 1, 8, dpr[cur + i], backend="pytorch"
                )
                for i in range(depths[0])
            ]
        )
        self.norm1 = nn.LayerNorm(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList(
            [
                HierarchicalTransformerEncoder(
                    embed_dims[1], 2, 4, dpr[cur + i], backend="pytorch"
                )
                for i in range(depths[1])
            ]
        )
        self.norm2 = nn.LayerNorm(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList(
            [
                HierarchicalTransformerEncoder(
                    embed_dims[2], 5, 2, dpr[cur + i], backend="pytorch"
                )
                for i in range(depths[2])
            ]
        )
        self.norm3 = nn.LayerNorm(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList(
            [
                HierarchicalTransformerEncoder(
                    embed_dims[3], 8, 1, dpr[cur + i], backend="pytorch"
                )
                for i in range(depths[3])
            ]
        )
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        # stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x1 = self.norm1(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # stage 2
        x, H, W = self.patch_embed2(x1)
        for blk in self.block2:
            x = blk(x, H, W)
        x2 = self.norm2(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # stage 3
        x, H, W = self.patch_embed3(x2)
        for blk in self.block3:
            x = blk(x, H, W)
        x3 = self.norm3(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # stage 4
        x, H, W = self.patch_embed4(x3)
        for blk in self.block4:
            x = blk(x, H, W)
        x4 = self.norm4(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        return x1, x2, x3, x4
