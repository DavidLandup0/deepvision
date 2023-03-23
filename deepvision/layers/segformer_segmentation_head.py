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

from typing import Tuple

import tensorflow as tf
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class __SegFormerHeadPT(nn.Module):
    def __init__(self, in_dims, embed_dim=256, num_classes=19, name=None):
        super().__init__()
        self.linear_layers = torch.nn.ModuleList()
        for i, dim in enumerate(in_dims):
            self.linear_layers.append(nn.Linear(dim, embed_dim))

        # To fuse multiple layer outputs into a single feature map using a Conv2d
        self.linear_fuse = torch.nn.Sequential(
            nn.Conv2d(
                in_channels=embed_dim * 4,
                out_channels=embed_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
        )
        # Final segmentation output
        self.seg_out = nn.Conv2d(
            in_channels=embed_dim, out_channels=num_classes, kernel_size=1
        )
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, features: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        B, _, H, W = features[0].shape
        outs = []
        for feature, layer in zip(features, self.linear_layers):
            projected_features = (
                # Flatten and transpose for Linear input
                layer(feature.flatten(2).transpose(1, 2))
                # Permute back
                .permute(0, 2, 1)
                # Reshape into map
                .reshape(B, -1, feature.shape[-2], feature.shape[-1])
            )
            outs.append(
                F.interpolate(
                    projected_features,
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                )
            )

        seg = self.seg_out(self.dropout(self.linear_fuse(torch.cat(outs[::-1], dim=1))))

        return seg


class __SegFormerHeadTF(tf.keras.layers.Layer):
    def __init__(self, in_dims, embed_dim=256, num_classes=19, **kwargs):
        super().__init__(**kwargs)
        self.linear_layers = []

        for i in in_dims:
            self.linear_layers.append(
                tf.keras.layers.Dense(embed_dim, name=f"linear_{i}")
            )

        # To fuse multiple layer outputs into a single feature map using a Conv2d
        self.linear_fuse = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=embed_dim, kernel_size=1, use_bias=False
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("relu"),
            ]
        )
        self.dropout = tf.keras.layers.Dropout(0.1)
        # Final segmentation output
        self.seg_out = tf.keras.layers.Conv2D(filters=num_classes, kernel_size=1)

    def call(self, features):
        B, H, W, _ = features[0].shape
        outs = []

        for feature, layer in zip(features, self.linear_layers):
            feature = layer(feature)
            feature = tf.image.resize(feature, size=(H, W), method="bilinear")
            outs.append(feature)

        seg = self.linear_fuse(tf.concat(outs[::-1], axis=3))
        seg = self.dropout(seg)
        seg = self.seg_out(seg)

        return seg


LAYER_BACKBONES = {
    "tensorflow": __SegFormerHeadTF,
    "pytorch": __SegFormerHeadPT,
}


def SegFormerHead(
    in_dims,
    num_classes,
    backend,
    embed_dim=256,
    name=None,
):
    layer_class = LAYER_BACKBONES.get(backend)
    if layer_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {LAYER_BACKBONES.keys()}"
        )

    layer = layer_class(
        in_dims=in_dims,
        num_classes=num_classes,
        embed_dim=embed_dim,
        name=name,
    )

    return layer
