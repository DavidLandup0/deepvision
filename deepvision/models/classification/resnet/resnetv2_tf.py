import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers

from deepvision.utils.utils import parse_model_inputs


def ResNetV2Block(
    filters,
    kernel_size=3,
    stride=1,
    dilation=1,
    conv_shortcut=False,
    type="basic",
):
    def apply(x):
        preact = layers.BatchNormalization(epsilon=1.001e-5)(x)
        preact = layers.Activation("relu")(preact)

        s = stride if dilation == 1 else 1
        if conv_shortcut:
            shortcut = layers.Conv2D(
                4 * filters if type == "bottleneck" else filters,
                1,
                strides=s,
            )(preact)
        else:
            shortcut = layers.MaxPooling2D(1, strides=stride)(x) if s > 1 else x

        x = layers.Conv2D(
            filters,
            1 if type == "bottleneck" else kernel_size,
            strides=1,
            use_bias=False,
            padding="same",
        )(preact)
        x = layers.BatchNormalization(epsilon=1.001e-5)(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(
            filters,
            kernel_size,
            strides=s,
            use_bias=False,
            padding="same",
            dilation_rate=dilation,
        )(x)
        if type == "bottleneck":
            x = layers.BatchNormalization(epsilon=1.001e-5)(x)
            x = layers.Activation("relu")(x)
            x = layers.Conv2D(4 * filters, 1)(x)
        x = shortcut + x
        return x

    return apply


def Stack(
    filters,
    blocks,
    stride=2,
    dilations=1,
    block_type=None,
    first_shortcut=True,
):
    def apply(x):
        x = ResNetV2Block(
            filters,
            conv_shortcut=first_shortcut,
            type=block_type,
        )(x)
        for i in range(2, blocks):
            x = ResNetV2Block(
                filters,
                dilation=dilations,
                type=block_type,
            )(x)
        x = ResNetV2Block(
            filters,
            stride=stride,
            dilation=dilations,
            type=block_type,
        )(x)
        return x

    return apply


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
