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
from deepvision.utils.utils import parse_model_inputs


class __SegFormerTF(tf.keras.Model):
    def __init__(
        self,
        num_classes=None,
        backbone=None,
        embed_dim=None,
        input_shape=None,
        input_tensor=None,
        softmax_output=None,
        **kwargs
    ):
        inputs = parse_model_inputs("tensorflow", input_shape, input_tensor)
        x = inputs
        y = backbone(x)
        y = SegFormerHead(
            in_dims=backbone.output_channels,
            embed_dim=embed_dim,
            num_classes=num_classes,
            name="segformer_head",
            backend="tensorflow",
        )(y)
        output = tf.keras.layers.Resizing(
            height=x.shape[1], width=x.shape[2], interpolation="bilinear"
        )(y)
        if softmax_output:
            output = tf.keras.layers.Activation("softmax", name="output_activation")(
                output
            )

        super().__init__(
            inputs=inputs,
            outputs=output,
            **kwargs,
        )

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.softmax_output = softmax_output
