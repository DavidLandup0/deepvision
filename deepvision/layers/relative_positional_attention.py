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

import tensorflow as tf
import torch
from torch import Tensor
from torch import nn

from deepvision.layers.decomposed_relative_positional_embedding import (
    AddDecomposedRelativePositions,
)


class __RelativePositionalMultiheadAttentionPT(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:

        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = (
            self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        )
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = AddDecomposedRelativePositions(
                self.rel_pos_h, self.rel_pos_w, backend="pytorch"
            )(attn, q, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (
            (attn @ v)
            .view(B, self.num_heads, H, W, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )
        x = self.proj(x)

        return x


class __RelativePositionalMultiheadAttentionTF(tf.keras.layers.Layer):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        embed_dim,
        num_heads=8,
        qkv_bias=True,
        use_rel_pos=False,
        input_size=None,
    ) -> None:
        """
        Args:

        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = tf.keras.layers.Dense(embed_dim * 3, use_bias=qkv_bias)
        self.proj = tf.keras.layers.Dense(embed_dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings

            self.rel_pos_h = self.add_weight(
                shape=[2 * input_size[0], head_dim], name="rel_pos_h", trainable=True
            )
            self.rel_pos_w = self.add_weight(
                shape=[2 * input_size[1], head_dim], name="rel_pos_w", trainable=True
            )

    def call(self, x):
        input_shape = tf.shape(x)
        B, H, W, _ = input_shape
        # qkv with shape (B, H * W, nHead, C, 3)
        qkv = tf.transpose(
            tf.reshape(self.qkv(x), [B, H * W, self.num_heads, -1, 3]), [0, 1, 2, 4, 3]
        )
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = tf.unstack(
            tf.reshape(qkv, [B * self.num_heads, H * W, -1, 3]), axis=3
        )

        attn = tf.matmul(q * self.scale, tf.transpose(k, [0, 2, 1]))

        if self.use_rel_pos:
            attn = AddDecomposedRelativePositions(
                self.rel_pos_h, self.rel_pos_w, backend="tensorflow"
            )(attn, q, (H, W), (H, W))

        attn = tf.nn.softmax(attn, axis=-1)
        x = tf.reshape(
            tf.transpose(
                tf.reshape(tf.matmul(attn, v), [B, H, W, self.num_heads, -1]),
                [0, 3, 1, 2, 4],
            ),
            [B, H, W, -1],
        )
        x = self.proj(x)

        return x


LAYER_BACKBONES = {
    "tensorflow": __RelativePositionalMultiheadAttentionTF,
    "pytorch": __RelativePositionalMultiheadAttentionPT,
}


def RelativePositionalMultiheadAttention(
    embed_dim,
    num_heads=8,
    qkv_bias=True,
    use_rel_pos=False,
    input_size=None,
    backend=None,
):
    """
    Multihead Relative Positional Attention block, as used in MVitV2, Swin and ViTDet:
        - "MViTv2: Improved Multiscale Vision Transformers for Classification and Detection": https://arxiv.org/abs/2112.01526
        - "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows": https://arxiv.org/abs/2103.14030
        - "Exploring Plain Vision Transformer Backbones for Object Detection": https://arxiv.org/abs/2203.16527

    Can be used as a drop-in replacement for MultiheadAttention, but additionally optionally uses relative positional embeddings instead of
    absolute positional embeddings.

    Args:
        embed_dim: The dimensionality of the output.
        num_heads: default 8, Number of attention heads.
        qkv_bias: default True, Whether to add a learnable bias to query, key, value or not.
        use_rel_pos: default False, Whether to add relative positional embeddings to the attention map.
        input_size: default None, Input size
        backend: default None, Which backend to use.
    """

    layer_class = LAYER_BACKBONES.get(backend)
    if layer_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {LAYER_BACKBONES.keys()}"
        )

    layer = layer_class(
        embed_dim=embed_dim,
        num_heads=num_heads,
        qkv_bias=qkv_bias,
        use_rel_pos=use_rel_pos,
        input_size=input_size,
    )

    return layer
