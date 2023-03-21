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

from deepvision.layers.segformer_segmentation_head import SegFormerHead


class __SegFormerTF(tf.keras.Model):
    def __init__(self, num_classes, backbone, embed_dim, softmax_output=None):
        super().__init__()
        self.backbone = backbone
        self.decode_head = SegFormerHead(
            in_dims=backbone.output_channels,
            embed_dim=embed_dim,
            num_classes=num_classes,
            backend="tensorflow",
        )
        self.softmax_output = softmax_output

    def call(self, x):
        y = self.backbone(x)
        y = self.decode_head(y)
        y = tf.image.resize(y, size=[x.shape[1], x.shape[2]], method="bilinear")
        if self.softmax_output:
            y = tf.nn.softmax(y, axis=1)
        return y
