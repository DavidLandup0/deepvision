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
import torch.nn as nn
import torch.nn.functional as F


class __WindowPartitioningPT(nn.Module):
    def __init__(self, window_size: int):
        super().__init__()
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Partition into non-overlapping windows with padding if needed.
        Args:
            x (tensor): input tokens with [B, H, W, C].

        Returns:
            windows: windows after partition with [B * num_windows, window_size, window_size, C].
            (Hp, Wp): padded height and width before partition
        """
        B, H, W, C = x.shape

        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w

        x = x.view(
            B,
            Hp // self.window_size,
            self.window_size,
            Wp // self.window_size,
            self.window_size,
            C,
        )
        windows = (
            x.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, self.window_size, self.window_size, C)
        )
        return windows, (Hp, Wp)


class __WindowPartitioningTF(tf.keras.layers.Layer):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size

    def call(self, x):
        """
        Partition into non-overlapping windows with padding if needed.
        Args:
            x (tensor): input tokens with [B, H, W, C].

        Returns:
            windows: windows after partition with [B * num_windows, window_size, window_size, C].
            (Hp, Wp): padded height and width before partition
        """
        input_shape = tf.shape(x)
        B, H, W, C = input_shape

        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            pad_dims = tf.constant([[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
            x = tf.pad(x, pad_dims)
        Hp, Wp = H + pad_h, W + pad_w

        x = tf.reshape(
            x,
            [
                B,
                Hp // self.window_size,
                self.window_size,
                Wp // self.window_size,
                self.window_size,
                C,
            ],
        )
        x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
        windows = tf.reshape(
            x,
            [-1, self.window_size, self.window_size, C],
        )
        return windows, (Hp, Wp)


LAYER_BACKBONES = {
    "tensorflow": __WindowPartitioningTF,
    "pytorch": __WindowPartitioningPT,
}


def WindowPartitioning(
    window_size,
    backend=None,
):
    layer_class = LAYER_BACKBONES.get(backend)
    if layer_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {LAYER_BACKBONES.keys()}"
        )

    layer = layer_class(
        window_size=window_size,
    )

    return layer
