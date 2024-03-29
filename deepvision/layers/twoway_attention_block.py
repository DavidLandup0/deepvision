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

from typing import Tuple
from typing import Type

import tensorflow as tf
import torch
from tensorflow.keras import layers
from torch import Tensor
from torch import nn

from deepvision.layers.downscaling_attention import DownscalingMultiheadAttention
from deepvision.layers.mlp import MLP


class __TwoWayAttentionBlockPT(nn.Module):
    def __init__(
        self,
        project_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          project_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = DownscalingMultiheadAttention(
            project_dim, num_heads, backend="pytorch"
        )
        self.norm1 = nn.LayerNorm(project_dim)

        self.cross_attn_token_to_image = DownscalingMultiheadAttention(
            project_dim,
            num_heads,
            downsample_rate=attention_downsample_rate,
            backend="pytorch",
        )
        self.norm2 = nn.LayerNorm(project_dim)

        self.mlp = _MLPBlock(project_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(project_dim)

        self.norm4 = nn.LayerNorm(project_dim)
        self.cross_attn_image_to_token = DownscalingMultiheadAttention(
            project_dim,
            num_heads,
            downsample_rate=attention_downsample_rate,
            backend="pytorch",
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class __TwoWayAttentionBlockTF(layers.Layer):
    def __init__(
        self,
        project_dim,
        num_heads,
        mlp_dim=2048,
        activation=tf.keras.activations.relu,
        attention_downsample_rate=2,
        skip_first_layer_pe=False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          project_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = DownscalingMultiheadAttention(
            project_dim, num_heads, backend="tensorflow"
        )
        self.norm1 = layers.LayerNormalization()

        self.cross_attn_token_to_image = DownscalingMultiheadAttention(
            project_dim,
            num_heads,
            downsample_rate=attention_downsample_rate,
            backend="tensorflow",
        )
        self.norm2 = layers.LayerNormalization()

        self.mlp = MLP(
            output_dim=project_dim,
            embed_dim=mlp_dim,
            activation=activation,
            num_layers=2,
            backend="tensorflow",
        )

        self.norm3 = layers.LayerNormalization()

        self.norm4 = layers.LayerNormalization()
        self.cross_attn_image_to_token = DownscalingMultiheadAttention(
            project_dim,
            num_heads,
            downsample_rate=attention_downsample_rate,
            backend="tensorflow",
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def call(self, queries, keys, query_pe, key_pe):
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class _MLPBlock(nn.Module):
    """
    Helper class for an MLP block, used instead of the `deepvision.layers.MLP` module
    to make loading pretrained weights easier.
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
    "tensorflow": __TwoWayAttentionBlockTF,
    "pytorch": __TwoWayAttentionBlockPT,
}


def TwoWayAttentionBlock(
    project_dim,
    num_heads,
    backend,
    mlp_dim=2048,
    activation=None,
    attention_downsample_rate=2,
    skip_first_layer_pe=False,
):
    layer_class = LAYER_BACKBONES.get(backend)
    if layer_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {LAYER_BACKBONES.keys()}"
        )

    if activation is None:
        activation = (
            tf.keras.activations.gelu if backend == "tensorflow" else torch.nn.GELU
        )

    layer = layer_class(
        project_dim=project_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        activation=activation,
        attention_downsample_rate=attention_downsample_rate,
        skip_first_layer_pe=skip_first_layer_pe,
    )

    return layer
