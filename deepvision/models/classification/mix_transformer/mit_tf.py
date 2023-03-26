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
from tensorflow.keras import layers

from deepvision.layers import (
    HierarchicalTransformerEncoder,
    OverlappingPatchingAndEmbedding,
)


class __MiTTF(tf.keras.models.Model):
    def __init__(
        self,
        input_shape=None,
        classes=None,
        include_top=None,
        embed_dims=None,
        depths=None,
        as_backbone=None,
        pooling=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        drop_path_rate = 0.1
        self.channels = embed_dims
        self.num_stages = 4
        self.output_channels = embed_dims
        self.classes = classes
        self.include_top = include_top
        self.as_backbone = as_backbone
        self.pooling = pooling

        self.patch_embedding_layers = []
        self.transformer_blocks = []
        self.layer_norms = []

        if self.include_top and not self.classes:
            raise ValueError(
                "If `include_top` is True, you should specify `classes`. "
                f"Received: classes={self.classes}"
            )

        if self.include_top and self.pooling:
            raise ValueError(
                f"`pooling` must be `None` when `include_top=True`."
                f"Received pooling={self.pooling} and include_top={self.include_top}. "
            )

        if self.include_top and self.as_backbone:
            raise ValueError(
                f"`as_backbone` must be `False` when `include_top=True`."
                f"Received as_backbone={self.as_backbone} and include_top={self.include_top}. "
            )

        dpr = [x.numpy() for x in tf.linspace(0.0, drop_path_rate, sum(depths))]

        blockwise_num_heads = [1, 2, 5, 8]
        blockwise_sr_ratios = [8, 4, 2, 1]

        cur = 0

        for i in range(self.num_stages):
            patch_embed_layer = OverlappingPatchingAndEmbedding(
                in_channels=3 if i == 0 else embed_dims[i - 1],
                out_channels=embed_dims[0] if i == 0 else embed_dims[i],
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                backend="tensorflow",
            )
            self.patch_embedding_layers.append(patch_embed_layer)

            transformer_block = [
                HierarchicalTransformerEncoder(
                    project_dim=embed_dims[i],
                    num_heads=blockwise_num_heads[i],
                    sr_ratio=blockwise_sr_ratios[i],
                    drop_prob=dpr[cur + k],
                    backend="tensorflow",
                )
                for k in range(depths[i])
            ]
            self.transformer_blocks.append(transformer_block)
            cur += depths[i]

            self.layer_norms.append(layers.LayerNormalization())
            self.linear = layers.Dense(
                classes, activation="softmax", name="predictions"
            )

    def call(self, x):
        B = tf.shape(x)[0]
        outputs = []
        for i in range(self.num_stages):
            x, H, W = self.patch_embedding_layers[i](x)
            for blk in self.transformer_blocks[i]:
                x = blk(x, H, W)
            x = self.layer_norms[i](x)
            C = tf.shape(x)[-1]
            x = tf.reshape(x, [B, H, W, C])
            outputs.append(x)

        if self.include_top:
            output = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            return layers.Dense(self.classes, activation="softmax", name="predictions")(
                output
            )
        elif self.as_backbone:
            return outputs
        else:
            if self.pooling == "avg":
                return layers.GlobalAveragePooling2D(name="avg_pool")(x)
            elif self.pooling == "max":
                return layers.GlobalMaxPooling2D(name="max_pool")(x)
