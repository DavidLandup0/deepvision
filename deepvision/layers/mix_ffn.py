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

import tensorflow as tf
from torch import nn
from torch.nn import functional as F


class __MixFFNPT(nn.Module):
    def __init__(self, channels, mid_channels):
        super().__init__()
        self.fc1 = nn.Linear(channels, mid_channels)
        self.dwconv = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=mid_channels,
        )
        self.fc2 = nn.Linear(mid_channels, channels)

    def forward(self, x, H, W):
        x = self.fc1(x)
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class __MixFFNTF(tf.keras.layers.Layer):
    def __init__(self, channels, mid_channels):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(mid_channels)
        self.dwconv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=3,
            strides=1,
            padding="same",
        )
        self.fc2 = tf.keras.layers.Dense(channels)

    def forward(self, x, H, W):
        x = self.fc1(x)
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


LAYER_BACKBONES = {
    "tensorflow": __MixFFNTF,
    "pytorch": __MixFFNPT,
}


def MixFFN(channels, mid_channels, backend):
    layer_class = LAYER_BACKBONES.get(backend)
    if layer_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {LAYER_BACKBONES.keys()}"
        )

    layer = layer_class(
        channels=channels,
        mid_channels=mid_channels,
    )

    return layer
