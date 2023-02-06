import math

import tensorflow as tf
from tensorflow.keras import layers

from deepvision.layers import FusedMBConv
from deepvision.layers import MBConv
from deepvision.utils.utils import parse_model_inputs

"""
Originally implemented by @AdityaKane2001
"""


def round_filters(filters, width_coefficient, min_depth, depth_divisor):
    filters *= width_coefficient
    minimum_depth = min_depth or depth_divisor
    new_filters = max(
        minimum_depth,
        int(filters + depth_divisor / 2) // depth_divisor * depth_divisor,
    )
    return int(new_filters)

"""
From the official MobileNetV2 implementation: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
"""

def _make_divisible(filter_num, depth_divisor, min_depth):
  if min_depth is None:
    min_value = depth_divisor
  new_v = max(min_depth, int(filter_num + depth_divisor / 2) // depth_divisor * depth_divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_v < 0.9 * filter_num:
    new_v += depth_divisor
  return int(new_v)


def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier."""
    return int(math.ceil(depth_coefficient * repeats))


class EfficientNetV2TF(tf.keras.Model):
    def __init__(
        self,
        include_top,
        width_coefficient,
        depth_coefficient,
        input_shape=(None, None, 3),
        input_tensor=None,
        pooling=None,
        classes=None,
        dropout_rate=0.2,
        drop_connect_rate=0.2,
        depth_divisor=8,
        min_depth=8,
        bn_momentum=0.9,
        activation=tf.keras.activations.swish,
        blockwise_kernel_sizes=None,
        blockwise_num_repeat=None,
        blockwise_input_filters=None,
        blockwise_output_filters=None,
        blockwise_expand_ratios=None,
        blockwise_se_ratios=None,
        blockwise_strides=None,
        blockwise_conv_type=None,
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

        inputs = parse_model_inputs("tensorflow", input_shape, input_tensor)
        x = inputs

        stem_filters = round_filters(
            filters=blockwise_input_filters[0],
            width_coefficient=width_coefficient,
            min_depth=min_depth,
            depth_divisor=depth_divisor,
        )
        x = layers.Conv2D(
            filters=stem_filters,
            kernel_size=3,
            strides=2,
            padding="same",
            use_bias=False,
        )(x)
        x = layers.BatchNormalization(momentum=bn_momentum)(x)
        x = layers.Activation(activation)(x)

        block_num = sum(blockwise_num_repeat)
        for block_index in range(len(blockwise_num_repeat)):
            input_filters = round_filters(
                blockwise_input_filters[block_index],
                width_coefficient,
                min_depth,
                depth_divisor,
            )

            output_filters = round_filters(
                blockwise_output_filters[block_index],
                width_coefficient,
                min_depth,
                depth_divisor,
            )
            repeats = round_repeats(
                repeats=blockwise_num_repeat[block_index],
                depth_coefficient=depth_coefficient,
            )

            for repeat in range(repeats):
                if blockwise_conv_type[block_index] == "mbconv":
                    conv_type = MBConv
                    name = f"MBConv_Block{block_index+1}_{repeat+1}"
                else:
                    conv_type = FusedMBConv
                    name = f"FusedMBConv_Block{block_index+1}_{repeat+1}"

                conv_block = conv_type(
                    input_filters=output_filters if repeat > 0 else input_filters,
                    output_filters=output_filters,
                    expand_ratio=blockwise_expand_ratios[block_index],
                    kernel_size=blockwise_kernel_sizes[block_index],
                    strides=1 if repeat > 0 else blockwise_strides[block_index],
                    se_ratio=blockwise_se_ratios[block_index],
                    activation=activation,
                    bn_momentum=bn_momentum,
                    dropout=drop_connect_rate * block_index / block_num,
                    backend="tensorflow",
                    name=name,
                )
                x = conv_block(x)

        top_filters = round_filters(
            filters=1280,
            width_coefficient=width_coefficient,
            min_depth=min_depth,
            depth_divisor=depth_divisor,
        )

        x = layers.Conv2D(
            filters=top_filters,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
        )(x)
        x = layers.BatchNormalization(
            momentum=bn_momentum,
        )(x)
        x = layers.Activation(activation)(x)

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
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.pooling = pooling
        self.classes = classes
        self.dropout_rate = dropout_rate
        self.drop_connect_rate = drop_connect_rate
        self.depth_divisor = depth_divisor
        self.min_depth = min_depth
        self.bn_momentum = bn_momentum
        self.activation = activation
        self.blockwise_kernel_sizes = blockwise_kernel_sizes
        self.blockwise_num_repeat = blockwise_num_repeat
        self.blockwise_input_filters = blockwise_input_filters
        self.blockwise_output_filters = blockwise_output_filters
        self.blockwise_expand_ratios = blockwise_expand_ratios
        self.blockwise_se_ratios = blockwise_se_ratios
        self.blockwise_strides = blockwise_strides
        self.blockwise_conv_type = blockwise_conv_type
