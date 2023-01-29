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


def ResNetV2(
    stackwise_filters,
    stackwise_blocks,
    stackwise_strides,
    include_rescaling,
    include_top,
    stackwise_dilations=None,
    name="ResNetV2",
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classes=None,
    classifier_activation="softmax",
    block_fn=Block,
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

    inputs = utils.parse_model_inputs('tensorflow', input_shape, input_tensor)
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

    stack_level_outputs = {}
    for stack_index in range(num_stacks):
        x = Stack(
            filters=stackwise_filters[stack_index],
            blocks=stackwise_blocks[stack_index],
            stride=stackwise_strides[stack_index],
            dilations=stackwise_dilations[stack_index],
            block_fn=block_fn,
            first_shortcut=block_fn == Block or stack_index > 0,
            stack_index=stack_index,
        )(x)
        stack_level_outputs[stack_index + 2] = x

    x = layers.BatchNormalization(epsilon=1.001e-5, name="post_bn")(x)
    x = layers.Activation("relu", name="post_relu")(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        x = layers.Dense(classes, activation=classifier_activation, name="predictions")(
            x
        )
    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D(name="max_pool")(x)

    # Create model.
    model = tf.keras.Model(inputs, x, name=name, **kwargs)

    if weights is not None:
        model.load_weights(weights)


def ResNet18V2(
    include_rescaling,
    include_top,
    classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="resnet18",
    **kwargs,
):
    """Instantiates the ResNet18 architecture."""

    return ResNetV2(
        stackwise_filters=MODEL_CONFIGS["ResNet18V2"]["stackwise_filters"],
        stackwise_blocks=MODEL_CONFIGS["ResNet18V2"]["stackwise_blocks"],
        stackwise_strides=MODEL_CONFIGS["ResNet18V2"]["stackwise_strides"],
        include_rescaling=include_rescaling,
        include_top=include_top,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        block_fn=BasicBlock,
        **kwargs,
    )