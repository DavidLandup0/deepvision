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

from deepvision.layers import Identity
from deepvision.utils.utils import parse_model_inputs


class ResNetV2Block(layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size=3,
        stride=1,
        dilation=1,
        conv_shortcut=False,
        block_type="basic",
    ):
        super().__init__()
        self.block_type = block_type
        self.preact_bn = layers.BatchNormalization(epsilon=1.001e-5)

        s = stride if dilation == 1 else 1
        if conv_shortcut:
            self.shortcut = layers.Conv2D(
                4 * filters if block_type == "bottleneck" else filters,
                1,
                strides=s,
            )
        else:
            self.shortcut = (
                layers.MaxPooling2D(1, strides=stride) if s > 1 else Identity()
            )

        self.conv2 = layers.Conv2D(
            filters,
            1 if block_type == "bottleneck" else kernel_size,
            strides=1,
            use_bias=False,
            padding="same",
        )
        self.bn2 = layers.BatchNormalization(epsilon=1.001e-5)

        self.conv3 = layers.Conv2D(
            filters,
            kernel_size,
            strides=s,
            use_bias=False,
            padding="same",
            dilation_rate=dilation,
        )
        if block_type == "bottleneck":
            self.bn3 = layers.BatchNormalization(epsilon=1.001e-5)
            self.conv4 = layers.Conv2D(4 * filters, 1)

    def call(self, inputs):

        x = self.preact_bn(inputs)
        x = layers.Activation("relu")(x)
        shortcut = self.shortcut(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = layers.Activation("relu")(x)
        x = self.conv3(x)

        if self.block_type == "bottleneck":
            x = self.bn3(x)
            x = layers.Activation("relu")(x)
            x = self.conv4(x)

        x = layers.Add()([shortcut, x])
        return x


class Stack(layers.Layer):
    def __init__(
        self,
        filters,
        blocks,
        stride=2,
        dilations=1,
        block_type=None,
        first_shortcut=True,
    ):
        super().__init__()
        self.block_1 = ResNetV2Block(
            filters,
            conv_shortcut=first_shortcut,
            block_type=block_type,
        )
        self.middle_blocks = []
        for i in range(2, blocks):
            self.middle_blocks.append(
                ResNetV2Block(
                    filters,
                    dilation=dilations,
                    block_type=block_type,
                )
            )

        self.final_block = ResNetV2Block(
            filters,
            stride=stride,
            dilation=dilations,
            block_type=block_type,
        )

    def call(self, inputs):
        x = self.block_1(inputs)
        for block in self.middle_blocks:
            x = block(x)
        x = self.final_block(x)
        return x


class ResNetV2TF(tf.keras.Model):
    def __init__(
        self,
        stackwise_filters,
        stackwise_blocks,
        stackwise_strides,
        include_top,
        input_shape=(None, None, 3),
        input_tensor=None,
        stackwise_dilations=None,
        pooling=None,
        classes=None,
        block_type=None,
        **kwargs,
    ):

        if include_top and not classes:
            raise ValueError(
                "If `include_top` is True, you should specify `classes`. "
                f"Received: classes={classes}"
            )

        if include_top and pooling:
            raise ValueError(
                f"`pooling` must be `None` when `include_top=True`."
                f"Received pooling={pooling} and include_top={include_top}. "
            )

        if not include_top and pooling is None:
            raise ValueError(f"`pooling` must be specified when `include_top=False`.")

        inputs = parse_model_inputs("tensorflow", input_shape, input_tensor)
        x = inputs

        x = layers.Conv2D(
            64,
            7,
            strides=2,
            use_bias=True,
            padding="same",
        )(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        num_stacks = len(stackwise_filters)
        if stackwise_dilations is None:
            stackwise_dilations = [1] * num_stacks

        for stack_index in range(num_stacks):
            x = Stack(
                filters=stackwise_filters[stack_index],
                blocks=stackwise_blocks[stack_index],
                stride=stackwise_strides[stack_index],
                dilations=stackwise_dilations[stack_index],
                block_type=block_type,
                first_shortcut=block_type == "bottleneck" or stack_index > 0,
            )(x)

        x = layers.BatchNormalization(epsilon=1.001e-5)(x)
        x = layers.Activation("relu")(x)

        if include_top:
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            output = layers.Dense(classes, activation="softmax", name="predictions")(x)
        else:
            if pooling == "avg":
                output = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            elif pooling == "max":
                output = layers.GlobalMaxPooling2D(name="max_pool")(x)

        super().__init__(
            inputs={
                "inputs": inputs,
            },
            outputs={
                "output": output,
            },
            **kwargs,
        )
        self.include_top = include_top
        self.pooling = pooling
        self.classes = classes
        self.stackwise_dilations = stackwise_dilations
        self.stackwise_filters = stackwise_filters
        self.stackwise_blocks = stackwise_blocks
        self.stackwise_strides = stackwise_strides
