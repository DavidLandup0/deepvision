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

from deepvision.layers import (
    HierarchicalTransformerEncoder,
    OverlappingPatchingAndEmbedding,
)


class __MiTPT(torch.nn.Module):
    def __init__(self, input_shape, embed_dims, depths, name):
        super().__init__()
        drop_path_rate = 0.1
        self.channels = embed_dims
        self.num_stages = 4
        self.output_channels = embed_dims

        self.patch_embedding_layers = torch.nn.ModuleList()
        self.transformer_blocks = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()

        blockwise_num_heads = [1, 2, 5, 8]
        blockwise_sr_ratios = [8, 4, 2, 1]
        cur = 0

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        for i in range(self.num_stages):
            patch_embed_layer = OverlappingPatchingAndEmbedding(
                in_channels=3 if i == 0 else embed_dims[i - 1],
                out_channels=embed_dims[0] if i == 0 else embed_dims[i],
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                backend="pytorch",
            )
            self.patch_embedding_layers.append(patch_embed_layer)

            transformer_block = torch.nn.ModuleList(
                [
                    HierarchicalTransformerEncoder(
                        project_dim=embed_dims[i],
                        num_heads=blockwise_num_heads[i],
                        sr_ratio=blockwise_sr_ratios[i],
                        drop_prob=dpr[cur + k],
                        backend="pytorch",
                    )
                    for k in range(depths[i])
                ]
            )
            self.transformer_blocks.append(transformer_block)
            cur += depths[i]

            self.layer_norms.append(torch.nn.LayerNorm(embed_dims[i]))

    def forward(self, x):
        B = x.shape[0]
        out = []
        for i in range(self.num_stages):
            x, H, W = self.patch_embedding_layers[i](x)
            for blk in self.transformer_blocks[i]:
                x = blk(x, H, W)
            x = self.layer_norms[i](x).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            out.append(x)
        return out
