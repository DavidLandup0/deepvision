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

import tensorflow as tf
import torch
import torch.functional as F
import torch.nn as nn


class __AddDecomposedRelativePositionsPT(nn.Module):
    def __init__(self, rel_pos_h: torch.Tensor, rel_pos_w: torch.Tensor):
        super().__init__()
        self.rel_pos_h = nn.Parameter(rel_pos_h, requires_grad=False)
        self.rel_pos_w = nn.Parameter(rel_pos_w, requires_grad=False)

    def forward(
        self,
        attn: torch.Tensor,
        q: torch.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
        https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py
        Args:
            attn (Tensor): attention map.
            q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
            q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
            k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

        Returns:
            attn (Tensor): attention map with added relative positional embeddings.
        """
        q_h, q_w = q_size
        k_h, k_w = k_size
        Rh = self.__get_rel_pos(q_h, k_h, self.rel_pos_h)
        Rw = self.__get_rel_pos(q_w, k_w, self.rel_pos_w)

        B, _, dim = q.shape
        r_q = q.reshape(B, q_h, q_w, dim)
        rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
        rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

        attn = (
            attn.view(B, q_h, q_w, k_h, k_w)
            + rel_h[:, :, :, :, None]
            + rel_w[:, :, :, None, :]
        ).view(B, q_h * q_w, k_h * k_w)

        return attn

    def __get_rel_pos(
        self, q_size: int, k_size: int, rel_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        Get relative positional embeddings according to the relative positions of
            query and key sizes.
        Args:
            q_size (int): size of query q.
            k_size (int): size of key k.
            rel_pos (Tensor): relative position embeddings (L, C).

        Returns:
            Extracted positional embeddings according to relative positions.
        """
        max_rel_dist = int(2 * max(q_size, k_size) - 1)
        # Interpolate rel pos if needed.
        if rel_pos.shape[0] != max_rel_dist:
            # Interpolate rel pos.
            rel_pos_resized = F.interpolate(
                rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
                size=max_rel_dist,
                mode="linear",
            )
            rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
        else:
            rel_pos_resized = rel_pos

        # Scale the coords with short length if shapes for q and k are different.
        q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
        k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + (k_size - 1) * max(
            q_size / k_size, 1.0
        )

        return rel_pos_resized[relative_coords.long()]


class __AddDecomposedRelativePositionsTF(tf.keras.layers.Layer):
    def __init__(self, rel_pos_h: tf.Tensor, rel_pos_w: tf.Tensor):
        super().__init__()
        self.rel_pos_h = self.add_weight(
            name="rel_pos_h",
            shape=rel_pos_h.shape,
            initializer=tf.keras.initializers.Constant(rel_pos_h),
            trainable=False,
        )
        self.rel_pos_w = self.add_weight(
            name="rel_pos_w",
            shape=rel_pos_w.shape,
            initializer=tf.keras.initializers.Constant(rel_pos_w),
            trainable=False,
        )

    def call(
        self,
        attn: tf.Tensor,
        q: tf.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
    ) -> tf.Tensor:
        """
        Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
        https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py
        Args:
            attn (Tensor): attention map.
            q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
            q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
            k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

        Returns:
            attn (Tensor): attention map with added relative positional embeddings.
        """
        q_h, q_w = q_size
        k_h, k_w = k_size
        Rh = self.__get_rel_pos(q_h, k_h, self.rel_pos_h)
        Rw = self.__get_rel_pos(q_w, k_w, self.rel_pos_w)

        B, _, dim = q.shape
        r_q = tf.reshape(q, (B, q_h, q_w, dim))
        rel_h = tf.einsum("bhwc,hkc->bhwk", r_q, Rh)
        rel_w = tf.einsum("bhwc,wkc->bhwk", r_q, Rw)

        attn = (
            tf.reshape(attn, (B, q_h, q_w, k_h, k_w))
            + tf.expand_dims(rel_h, -1)
            + tf.expand_dims(rel_w, -2)
        )
        attn = tf.reshape(attn, (B, q_h * q_w, k_h * k_w))

        return attn

    def __get_rel_pos(self, q_size: int, k_size: int, rel_pos: tf.Tensor) -> tf.Tensor:
        """
        Get relative positional embeddings according to the relative positions of
            query and key sizes.
        Args:
            q_size (int): size of query q.
            k_size (int): size of key k.
            rel_pos (Tensor): relative position embeddings (L, C).

        Returns:
            Extracted positional embeddings according to relative positions.
        """
        max_rel_dist = int(2 * max(q_size, k_size) - 1)
        if rel_pos.shape[0] != max_rel_dist:
            rel_pos_resized = tf.image.resize(
                tf.transpose(
                    tf.reshape(rel_pos, [1, rel_pos.shape[0], -1]), perm=[0, 2, 1]
                ),
                [max_rel_dist],
                method="linear",
            )
            rel_pos_resized = tf.transpose(
                tf.reshape(rel_pos_resized, [-1, max_rel_dist]), perm=[1, 0]
            )
        else:
            rel_pos_resized = rel_pos

        q_coords = tf.reshape(tf.range(q_size), [q_size, 1]) * max(k_size / q_size, 1.0)
        k_coords = tf.reshape(tf.range(k_size), [1, k_size]) * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + (k_size - 1) * max(
            q_size / k_size, 1.0
        )

        return tf.gather(
            rel_pos_resized, tf.cast(tf.reshape(relative_coords, [-1]), tf.int32)
        )


LAYER_BACKBONES = {
    "tensorflow": __AddDecomposedRelativePositionsTF,
    "pytorch": __AddDecomposedRelativePositionsPT,
}


def AddDecomposedRelativePositions(rel_pos_h, rel_pos_w, backend):
    layer_class = LAYER_BACKBONES.get(backend)
    if layer_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {LAYER_BACKBONES.keys()}"
        )

    layer = layer_class(
        rel_pos_h=rel_pos_h,
        rel_pos_w=rel_pos_w,
    )

    return layer
