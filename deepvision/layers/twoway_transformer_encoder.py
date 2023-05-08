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

import math
from typing import Tuple
from typing import Type

import tensorflow as tf
import torch
from torch import Tensor
from torch import nn

from deepvision.layers import DownscalingMultiheadAttention
from deepvision.layers import TwoWayAttentionBlock


class __TwoWayTransformerEncoderPT(nn.Module):
    def __init__(
        self,
        depth: int,
        project_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          project_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide project_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.project_dim = project_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    project_dim=project_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                    backend="pytorch",
                )
            )

        self.final_attn_token_to_image = DownscalingMultiheadAttention(
            project_dim,
            num_heads,
            downsample_rate=attention_downsample_rate,
            backend="pytorch",
        )
        self.norm_final_attn = nn.LayerNorm(project_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x project_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x project_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class __TwoWayTransformerEncoderTF(tf.keras.layers.Layer):
    def __init__(
        self,
        depth: int,
        project_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation=tf.keras.activations.relu,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          project_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide project_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.project_dim = project_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = []

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    project_dim=project_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                    backend="tensorflow",
                )
            )

        self.final_attn_token_to_image = DownscalingMultiheadAttention(
            project_dim,
            num_heads,
            downsample_rate=attention_downsample_rate,
            backend="tensorflow",
        )
        self.norm_final_attn = tf.keras.layers.LayerNormalization()

    def call(
        self,
        image_embedding,
        image_pe,
        point_embedding,
    ):
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x project_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x project_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        input_shape = tf.shape(image_embedding)
        bs, c, h, w = input_shape[0], input_shape[1], input_shape[2], input_shape[3]

        image_embedding = tf.transpose(
            tf.reshape(image_embedding, [bs, c, -1]), perm=[0, 2, 1]
        )

        image_pe = tf.transpose(tf.reshape(image_pe, [bs, c, -1]), perm=[0, 2, 1])

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


LAYER_BACKBONES = {
    "tensorflow": __TwoWayTransformerEncoderTF,
    "pytorch": __TwoWayTransformerEncoderPT,
}


def TwoWayTransformerEncoder(
    depth,
    project_dim,
    num_heads,
    mlp_dim,
    backend,
    activation=None,
    attention_downsample_rate=2,
):
    layer_class = LAYER_BACKBONES.get(backend)
    if layer_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {LAYER_BACKBONES.keys()}"
        )
    if activation is None:
        activation = (
            tf.keras.activations.relu if backend == "tensorflow" else torch.nn.ReLU
        )
    layer = layer_class(
        depth=depth,
        project_dim=project_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        activation=activation,
        attention_downsample_rate=attention_downsample_rate,
    )

    return layer
