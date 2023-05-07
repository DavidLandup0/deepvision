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


class __DownscalingMultiheadAttentionPT(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class __DownscalingMultiheadAttentionTF(tf.keras.layers.Layer):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(self, embedding_dim, num_heads, downsample_rate=1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = tf.keras.layers.Dense(self.internal_dim)
        self.k_proj = tf.keras.layers.Dense(self.internal_dim)
        self.v_proj = tf.keras.layers.Dense(self.internal_dim)
        self.out_proj = tf.keras.layers.Dense(embedding_dim)

    def _separate_heads(self, x: tf.Tensor, num_heads: int) -> tf.Tensor:
        input_shape = tf.shape(x)
        b, n, c = input_shape
        x = tf.reshape(x, [b, n, num_heads, c // num_heads])
        return tf.transpose(x, [0, 2, 1, 3])  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: tf.Tensor) -> tf.Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, [b, n_tokens, n_heads * c_per_head])  # B x N_tokens x C

    def call(self, q: tf.Tensor, k: tf.Tensor, v: tf.Tensor) -> tf.Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = tf.matmul(
            q, tf.transpose(k, [0, 1, 3, 2])
        )  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = tf.nn.softmax(attn, axis=-1)

        # Get output
        out = tf.matmul(attn, v)
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


LAYER_BACKBONES = {
    "tensorflow": __DownscalingMultiheadAttentionTF,
    "pytorch": __DownscalingMultiheadAttentionPT,
}


def DownscalingMultiheadAttention(embedding_dim, num_heads, downsample_rate, backend):
    """
    MultiheadAttention block that downscales the size of the embedding after projection.
    Similar to `deepvision.layers.EfficientMultiheadAttention` which performs a reduction to save computational resources.
    `EfficientMultiheadAttention` performs reduction using a convolutional layer, while DownscalingMultiheadAttention performs reduction
    after projection, using a `match.sqrt()` call.


    """

    layer_class = LAYER_BACKBONES.get(backend)
    if layer_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {LAYER_BACKBONES.keys()}"
        )

    layer = layer_class(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        downsample_rate=downsample_rate,
    )

    return layer
