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
    name=None,
    type="basic",
):

    if name is None:
        name = f"v2_block_{backend.get_uid('v2_block')}"

    def apply(x):
        use_preactivation = layers.BatchNormalization(
            epsilon=1.001e-5, name=name + "_use_preactivation_bn"
        )(x)

        use_preactivation = layers.Activation(
            "relu", name=name + "_use_preactivation_relu"
        )(use_preactivation)

        s = stride if dilation == 1 else 1
        if conv_shortcut:
            shortcut = layers.Conv2D(
                4 * filters if type == "bottleneck" else filters,
                1,
                strides=s,
                name=name + "_0_conv",
            )(use_preactivation)
        else:
            shortcut = (
                layers.MaxPooling2D(1, strides=stride, name=name + "_0_max_pooling")(x)
                if s > 1
                else x
            )

        x = layers.Conv2D(
            filters,
            1 if type == "bottleneck" else kernel_size,
            strides=1,
            use_bias=False,
            padding="same",
            name=name + "_1_conv",
        )(use_preactivation)
        x = layers.BatchNormalization(epsilon=1.001e-5, name=name + "_1_bn")(x)
        x = layers.Activation("relu", name=name + "_1_relu")(x)

        x = layers.Conv2D(
            filters,
            kernel_size,
            strides=s,
            use_bias=False,
            padding="same",
            dilation_rate=dilation,
            name=name + "_2_conv",
        )(x)
        if type == "bottleneck":
            x = layers.BatchNormalization(epsilon=1.001e-5, name=name + "_2_bn")(x)
            x = layers.Activation("relu", name=name + "_2_relu")(x)
            x = layers.Conv2D(4 * filters, 1, name=name + "_3_conv")(x)
        x = layers.Add(name=name + "_out")([shortcut, x])
        return x

    return apply


def Stack(
    filters,
    blocks,
    stride=2,
    dilations=1,
    name=None,
    block_type=None,
    first_shortcut=True,
    stack_index=1,
):
    if name is None:
        name = f"v2_stack_{stack_index}"

    def apply(x):
        x = ResNetV2Block(
            filters,
            conv_shortcut=first_shortcut,
            name=name + "_block1",
            type=block_type,
        )(x)
        for i in range(2, blocks):
            x = ResNetV2Block(
                filters,
                dilation=dilations,
                name=name + "_block" + str(i),
                type=block_type,
            )(x)
        x = ResNetV2Block(
            filters,
            stride=stride,
            dilation=dilations,
            name=name + "_block" + str(blocks),
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
            name="conv1_conv",
        )(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same", name="pool1_pool")(x)
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
                stack_index=stack_index,
            )(x)

        x = layers.BatchNormalization(epsilon=1.001e-5, name="post_bn")(x)
        x = layers.Activation("relu", name="post_relu")(x)

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
