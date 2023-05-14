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
import torch.nn as nn

from deepvision.layers.mlp import MLP
from deepvision.layers.relative_positional_attention import (
    RelativePositionalMultiheadAttention,
)
from deepvision.layers.window_partitioning import WindowPartitioning
from deepvision.layers.window_unpartitioning import WindowUnpartitioning


class __RelativePositionalTransformerEncoderPT(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        project_dim,
        num_heads,
        mlp_dim,
        qkv_bias,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_rel_pos=False,
        window_size=0,
        input_size=None,
    ) -> None:
        """
        Args:
            project_dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_dim (float): MLP dim
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(project_dim)
        self.attn = RelativePositionalMultiheadAttention(
            project_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            input_size=input_size if window_size == 0 else (window_size, window_size),
            backend="pytorch",
        )

        self.norm2 = norm_layer(project_dim)

        self.mlp = _MLPBlock(project_dim=project_dim, mlp_dim=mlp_dim, act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = WindowPartitioning(self.window_size, backend="pytorch")(x)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = WindowUnpartitioning(
                self.window_size, pad_hw, (H, W), backend="pytorch"
            )(x)

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class __RelativePositionalTransformerEncoderTF(tf.keras.layers.Layer):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        project_dim,
        num_heads,
        mlp_dim,
        qkv_bias,
        norm_layer=layers.LayerNormalization,
        act_layer=tf.keras.activations.gelu,
        use_rel_pos=False,
        window_size=0,
        input_size=None,
    ) -> None:
        """
        Args:
            project_dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_dim (float): MLP dim
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer()
        self.attn = RelativePositionalMultiheadAttention(
            project_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            input_size=input_size if window_size == 0 else (window_size, window_size),
            backend="tensorflow",
        )

        self.norm2 = norm_layer()

        self.mlp = MLP(
            output_dim=project_dim,
            embed_dim=mlp_dim,
            activation=act_layer,
            num_layers=2,
            backend="tensorflow",
        )

        self.window_size = window_size

    def call(self, x):
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = WindowPartitioning(self.window_size, backend="tensorflow")(x)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = WindowUnpartitioning(
                self.window_size, pad_hw, (H, W), backend="tensorflow"
            )(x)

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class _MLPBlock(nn.Module):
    """
    Helper class for an MLP block, used instead of the `deepvision.layers.MLP` module to make loading pretrained weights easier.
    """

    def __init__(
        self,
        project_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(project_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, project_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


LAYER_BACKBONES = {
    "tensorflow": __RelativePositionalTransformerEncoderTF,
    "pytorch": __RelativePositionalTransformerEncoderPT,
}


def RelativePositionalTransformerEncoder(
    project_dim,
    mlp_dim,
    num_heads=8,
    qkv_bias=True,
    norm_layer=None,
    act_layer=None,
    use_rel_pos=True,
    window_size=None,
    input_size=None,
    backend=None,
):
    """
    Transformer Encoder Encoder utilizing the RelativePositionalMultiheadAttention block instead of the standard MultiheadAttention.

    Can be used as a drop-in replacement for a TransformerEncoder, but additionally optionally uses relative positional embeddings instead of
    absolute positional embeddings.

    """

    layer_class = LAYER_BACKBONES.get(backend)
    if layer_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {LAYER_BACKBONES.keys()}"
        )

    if act_layer is None:
        act_layer = (
            tf.keras.activations.gelu if backend == "tensorflow" else torch.nn.GELU
        )

    if norm_layer is None:
        norm_layer = (
            tf.keras.layers.LayerNormalization
            if backend == "tensorflow"
            else torch.nn.LayerNorm
        )

    layer = layer_class(
        project_dim=project_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        qkv_bias=qkv_bias,
        norm_layer=norm_layer,
        act_layer=act_layer,
        use_rel_pos=use_rel_pos,
        window_size=window_size,
        input_size=input_size,
    )

    return layer
