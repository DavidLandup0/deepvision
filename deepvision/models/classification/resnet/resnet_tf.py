import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers
from deepvision import utils

MODEL_CONFIGS = {
    "ResNet18V2": {
        "stackwise_filters": [64, 128, 256, 512],
        "stackwise_blocks": [2, 2, 2, 2],
        "stackwise_strides": [1, 2, 2, 2],
    },

}

def BasicBlock(
    filters, kernel_size=3, stride=1, dilation=1, conv_shortcut=False, name=None
):

    if name is None:
        name = f"v2_basic_block_{backend.get_uid('v2_basic_block')}"

    def apply(x):
        use_preactivation = layers.BatchNormalization(epsilon=1.001e-5, name=name + "_use_preactivation_bn"
        )(x)

        use_preactivation = layers.Activation(
            "relu", name=name + "_use_preactivation_relu"
        )(use_preactivation)

        s = stride if dilation == 1 else 1
        if conv_shortcut:
            shortcut = layers.Conv2D(filters, 1, strides=s, name=name + "_0_conv")(
                use_preactivation
            )
        else:
            shortcut = (
                layers.MaxPooling2D(1, strides=stride, name=name + "_0_max_pooling")(x)
                if s > 1
                else x
            )

        x = layers.Conv2D(
            filters,
            kernel_size,
            padding="SAME",
            strides=1,
            use_bias=False,
            name=name + "_1_conv",
        )(use_preactivation)
        x = layers.BatchNormalization(epsilon=1.001e-5, name=name + "_1_bn"
        )(x)
        x = layers.Activation("relu", name=name + "_1_relu")(x)

        x = layers.Conv2D(
            filters,
            kernel_size,
            strides=s,
            padding="same",
            dilation_rate=dilation,
            use_bias=False,
            name=name + "_2_conv",
        )(x)

        x = layers.Add(name=name + "_out")([shortcut, x])
        return x

    return apply


def Block(filters, kernel_size=3, stride=1, dilation=1, conv_shortcut=False, name=None):

    if name is None:
        name = f"v2_block_{backend.get_uid('v2_block')}"

    def apply(x):
        use_preactivation = layers.BatchNormalization(epsilon=1.001e-5, name=name + "_use_preactivation_bn"
        )(x)

        use_preactivation = layers.Activation(
            "relu", name=name + "_use_preactivation_relu"
        )(use_preactivation)

        s = stride if dilation == 1 else 1
        if conv_shortcut:
            shortcut = layers.Conv2D(
                4 * filters,
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

        x = layers.Conv2D(filters, 1, strides=1, use_bias=False, name=name + "_1_conv")(
            use_preactivation
        )
        x = layers.BatchNormalization(epsilon=1.001e-5, name=name + "_1_bn"
        )(x)
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
        x = layers.BatchNormalization(epsilon=1.001e-5, name=name + "_2_bn"
        )(x)
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
    block_fn=Block,
    first_shortcut=True,
    stack_index=1,
):
    if name is None:
        name = f"v2_stack_{stack_index}"

    def apply(x):
        x = block_fn(filters, conv_shortcut=first_shortcut, name=name + "_block1")(x)
        for i in range(2, blocks):
            x = block_fn(filters, dilation=dilations, name=name + "_block" + str(i))(x)
        x = block_fn(
            filters,
            stride=stride,
            dilation=dilations,
            name=name + "_block" + str(blocks),
        )(x)
        return x

    return apply


class ResNetV2TF(tf.keras.Model):

    def __init__(self,
    stackwise_filters,
    stackwise_blocks,
    stackwise_strides,
    include_top,
    stackwise_dilations=None,
    pooling=None,
    classes=None,
    block_fn=Block,
    **kwargs,):

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

        self.stackwise_dilations = stackwise_dilations
        self.stackwise_filters = stackwise_filters
        self.stackwise_blocks = stackwise_blocks
        self.stackwise_strides = stackwise_strides
        self.include_top = include_top
        self.classes = classes
        self.pooling = pooling

        self.conv1 = layers.Conv2D(
            64,
            7,
            strides=2,
            use_bias=True,
            padding="same",
            name="conv1_conv",
        )
        self.maxpool = layers.MaxPooling2D(3, strides=2, padding="same", name="pool1_pool")

        self.stacks = []

        num_stacks = len(stackwise_filters)
        if stackwise_dilations is None:
            stackwise_dilations = [1] * num_stacks

        for stack_index in range(num_stacks):
            self.stacks.append(Stack(
                filters=stackwise_filters[stack_index],
                blocks=stackwise_blocks[stack_index],
                stride=stackwise_strides[stack_index],
                dilations=stackwise_dilations[stack_index],
                block_fn=block_fn,
                first_shortcut=block_fn == Block or stack_index > 0,
                stack_index=stack_index,
            ))
        self.batchnorm = layers.BatchNormalization(epsilon=1.001e-5, name="post_bn")
        self.top_dense = layers.Dense(classes, activation='softmax', name="predictions")

    def call(self):

        inputs = utils.parse_model_inputs('tensorflow', self.input_shape, self.input_tensor)
        x = inputs

        x = self.conv1(x)
        x = self.maxpool(x)
        for stack in self.stacks:
            x = stack(x)
        x = self.batchnorm(x)
        x = layers.Activation("relu", name="post_relu")(x)

        if self.include_top:
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            x = self.top_dense(x)
        else:
            if self.pooling == "avg":
                x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            elif self.pooling == "max":
                x = layers.GlobalMaxPooling2D(name="max_pool")(x)