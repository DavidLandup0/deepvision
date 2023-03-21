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

from deepvision.layers import (
    HierarchicalTransformerEncoder,
    OverlappingPatchingAndEmbedding,
)


class __MiTTF(tf.keras.models.Model):
    def __init__(self, input_shape, embed_dims, depths):
        super().__init__()
        drop_path_rate = 0.1
        self.channels = embed_dims

        # patch_embed
        self.patch_embed1 = OverlappingPatchingAndEmbedding(
            3, embed_dims[0], 7, 4, backend="tensorflow"
        )
        self.patch_embed2 = OverlappingPatchingAndEmbedding(
            embed_dims[0], embed_dims[1], 3, 2, backend="tensorflow"
        )
        self.patch_embed3 = OverlappingPatchingAndEmbedding(
            embed_dims[1], embed_dims[2], 3, 2, backend="tensorflow"
        )
        self.patch_embed4 = OverlappingPatchingAndEmbedding(
            embed_dims[2], embed_dims[3], 3, 2, backend="tensorflow"
        )

        dpr = [x.numpy() for x in tf.linspace(0.0, drop_path_rate, sum(depths))]

        cur = 0
        self.block1 = [
            HierarchicalTransformerEncoder(
                embed_dims[0], 1, 8, dpr[cur + i], backend="tensorflow"
            )
            for i in range(depths[0])
        ]

        self.norm1 = tf.keras.layers.LayerNormalization()

        cur += depths[0]
        self.block2 = [
            HierarchicalTransformerEncoder(
                embed_dims[1], 2, 4, dpr[cur + i], backend="tensorflow"
            )
            for i in range(depths[1])
        ]

        self.norm2 = tf.keras.layers.LayerNormalization()

        cur += depths[1]
        self.block3 = [
            HierarchicalTransformerEncoder(
                embed_dims[2], 5, 2, dpr[cur + i], backend="tensorflow"
            )
            for i in range(depths[2])
        ]

        self.norm3 = tf.keras.layers.LayerNormalization()

        cur += depths[2]
        self.block4 = [
            HierarchicalTransformerEncoder(
                embed_dims[3], 8, 1, dpr[cur + i], backend="tensorflow"
            )
            for i in range(depths[3])
        ]

        self.norm4 = tf.keras.layers.LayerNormalization()

    def call(self, x):
        B = x.shape[0]
        # stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x1 = self.norm1(x)
        x1 = tf.reshape(x1, [B, H, W, -1])

        # stage 2
        x, H, W = self.patch_embed2(x1)
        for blk in self.block2:
            x = blk(x, H, W)
        x2 = self.norm2(x)
        x2 = tf.reshape(x2, [B, H, W, -1])

        # stage 3
        x, H, W = self.patch_embed3(x2)
        for blk in self.block3:
            x = blk(x, H, W)
        x3 = self.norm3(x)
        x3 = tf.reshape(x3, [B, H, W, -1])

        # stage 4
        x, H, W = self.patch_embed4(x3)
        for blk in self.block4:
            x = blk(x, H, W)
        x4 = self.norm4(x)
        x4 = tf.reshape(x4, [B, H, W, -1])

        return x1, x2, x3, x4
