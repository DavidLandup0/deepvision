# Ported and adapted from the original code from Meta Platforms, Inc. and affiliates. Copyright
# Original code Copyright / (c) Meta Platforms, Inc. and affiliates.
# Modifications and adaptations / Copyright 2023 David Landup
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

from typing import Optional
from typing import Tuple
from typing import Type

import torch
import torch.nn as nn

from deepvision.layers import LayerNorm2d
from deepvision.layers import PatchingAndEmbedding
from deepvision.layers import RelativePositionalTransformerEncoder
from deepvision.utils.utils import parse_model_inputs


class ViTDetBackbonePT(nn.Module):
    """
    ViTDet backbone, without the prediction head.
    The class is adapted from the ViTDet implementation at:
        - https://github.com/facebookresearch/deit/blob/main/deit/models/vision_transformer.py

    Currently shipped only as a `torch.nn.Module`, without PyTorch Lightning support.
    """

    def __init__(
        self,
        input_shape=(3, None, None),
        input_tensor=None,
        patch_size: int = 16,
        embed_dim: int = 768,
        transformer_layer_num: int = 12,
        num_heads: int = 12,
        mlp_dim=None,
        project_dim: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            input_shape: Input shape.
            patch_size: Patch size.
            embed_dim: Patch embedding dimension.
            transformer_layer_num: Depth of ViT.
            num_heads: Number of attention heads in each ViT block.
            mlp_dim: MLP hidden dimension
            qkv_bias: If True, add a learnable bias to query, key, value.
            norm_layer: Normalization layer.
            act_layer: Activation layer.
            use_abs_pos: If True, use absolute positional embeddings.
            use_rel_pos: If True, add relative positional embeddings to the attention map.
            window_size: Window size for window attention blocks.
            global_attn_indexes: Indexes for blocks using global attention.
        """
        super().__init__()
        self.input_shape = input_shape

        self.patch_embed = PatchingAndEmbedding(
            patch_size=patch_size,
            input_shape=input_shape,
            project_dim=embed_dim,
            embedding=False,
            padding="valid",
            backend="pytorch",
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrained image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    input_shape[1] // patch_size,
                    input_shape[1] // patch_size,
                    embed_dim,
                )
            )

        self.blocks = nn.ModuleList()
        for i in range(transformer_layer_num):
            self.blocks.append(
                RelativePositionalTransformerEncoder(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    use_rel_pos=use_rel_pos,
                    window_size=window_size if i not in global_attn_indexes else 0,
                    input_size=(
                        input_shape[1] // patch_size,
                        input_shape[1] // patch_size,
                    ),
                )
            )

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                project_dim,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(project_dim),
            nn.Conv2d(
                project_dim,
                project_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(project_dim),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        inputs = parse_model_inputs("pytorch", input_tensor.shape, input_tensor)
        x = inputs
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.permute(0, 3, 1, 2))

        return x
