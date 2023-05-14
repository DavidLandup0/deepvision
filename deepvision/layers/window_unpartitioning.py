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


class __WindowUnpartitioningPT(nn.Module):
    def __init__(self, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]):
        super().__init__()
        self.window_size = window_size
        self.pad_hw = pad_hw
        self.hw = hw

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """
        Window unpartition into original sequences and removing padding.
        Args:
            windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].

        Returns:
            x: unpartitioned sequences with [B, H, W, C].
        """
        Hp, Wp = self.pad_hw
        H, W = self.hw
        B = windows.shape[0] // (Hp * Wp // self.window_size // self.window_size)
        x = windows.view(
            B,
            Hp // self.window_size,
            Wp // self.window_size,
            self.window_size,
            self.window_size,
            -1,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

        if Hp > H or Wp > W:
            x = x[:, :H, :W, :].contiguous()
        return x


class __WindowUnpartitioningTF(tf.keras.layers.Layer):
    def __init__(self, window_size, pad_hw, hw):
        super().__init__()
        self.window_size = window_size
        self.pad_hw = pad_hw
        self.hw = hw

    def call(self, windows):
        """
        Window unpartition into original sequences and removing padding.
        Args:
            windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].

        Returns:
            x: unpartitioned sequences with [B, H, W, C].
        """
        Hp, Wp = self.pad_hw
        H, W = self.hw
        B = tf.shape(windows)[0] // (Hp * Wp // self.window_size // self.window_size)
        x = tf.reshape(
            windows,
            [
                B,
                Hp // self.window_size,
                self.window_size,
                Wp // self.window_size,
                self.window_size,
                -1,
            ],
        )
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [B, Hp, Wp, -1])

        if Hp > H or Wp > W:
            x = x[:, :H, :W, :]
        return x


LAYER_BACKBONES = {
    "tensorflow": __WindowUnpartitioningTF,
    "pytorch": __WindowUnpartitioningPT,
}


def WindowUnpartitioning(
    window_size,
    pad_hw,
    hw,
    backend=None,
):
    layer_class = LAYER_BACKBONES.get(backend)
    if layer_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {LAYER_BACKBONES.keys()}"
        )

    layer = layer_class(
        window_size=window_size,
        pad_hw=pad_hw,
        hw=hw,
    )

    return layer
