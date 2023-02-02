from deepvision.models.classification.resnet.resnetv2_pt import ResNetV2PT
from deepvision.models.classification.resnet.resnetv2_tf import ResNetV2TF

MODEL_CONFIGS = {
    "ResNet18V2": {
        "stackwise_filters": [64, 128, 256, 512],
        "stackwise_blocks": [2, 2, 2, 2],
        "stackwise_strides": [1, 2, 2, 2],
    },
    "ResNet34V2": {
        "stackwise_filters": [64, 128, 256, 512],
        "stackwise_blocks": [3, 4, 6, 3],
        "stackwise_strides": [1, 2, 2, 2],
    },
    "ResNet50V2": {
        "stackwise_filters": [64, 128, 256, 512],
        "stackwise_blocks": [3, 4, 6, 3],
        "stackwise_strides": [1, 2, 2, 2],
    },
    "ResNet101V2": {
        "stackwise_filters": [64, 128, 256, 512],
        "stackwise_blocks": [3, 4, 23, 3],
        "stackwise_strides": [1, 2, 2, 2],
    },
    "ResNet152V2": {
        "stackwise_filters": [64, 128, 256, 512],
        "stackwise_blocks": [3, 8, 36, 3],
        "stackwise_strides": [1, 2, 2, 2],
    },
}

MODEL_BACKBONES = {"tensorflow": ResNetV2TF, "pytorch": ResNetV2PT}


def ResNet18V2(
    backend,
    include_top,
    classes=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    **kwargs,
):
    model_class = MODEL_BACKBONES.get(backend)
    model = model_class(
        stackwise_filters=MODEL_CONFIGS["ResNet18V2"]["stackwise_filters"],
        stackwise_blocks=MODEL_CONFIGS["ResNet18V2"]["stackwise_blocks"],
        stackwise_strides=MODEL_CONFIGS["ResNet18V2"]["stackwise_strides"],
        include_top=include_top,
        input_shape=input_shape,
        input_tensor=input_tensor,
        block_type="basic",
        pooling=pooling,
        classes=classes,
        **kwargs,
    )
    return model


def ResNet34V2(
    backend,
    include_top,
    classes=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    **kwargs,
):
    model_class = MODEL_BACKBONES.get(backend)
    model = model_class(
        stackwise_filters=MODEL_CONFIGS["ResNet34V2"]["stackwise_filters"],
        stackwise_blocks=MODEL_CONFIGS["ResNet34V2"]["stackwise_blocks"],
        stackwise_strides=MODEL_CONFIGS["ResNet34V2"]["stackwise_strides"],
        include_top=include_top,
        input_shape=input_shape,
        input_tensor=input_tensor,
        block_type="basic",
        pooling=pooling,
        classes=classes,
        **kwargs,
    )
    return model


def ResNet50V2(
    backend,
    include_top,
    classes=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    **kwargs,
):
    model_class = MODEL_BACKBONES.get(backend)
    model = model_class(
        stackwise_filters=MODEL_CONFIGS["ResNet50V2"]["stackwise_filters"],
        stackwise_blocks=MODEL_CONFIGS["ResNet50V2"]["stackwise_blocks"],
        stackwise_strides=MODEL_CONFIGS["ResNet50V2"]["stackwise_strides"],
        include_top=include_top,
        input_shape=input_shape,
        input_tensor=input_tensor,
        block_type="bottleneck",
        pooling=pooling,
        classes=classes,
        **kwargs,
    )
    return model


def ResNet101V2(
    backend,
    include_top,
    classes=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    **kwargs,
):
    model_class = MODEL_BACKBONES.get(backend)
    model = model_class(
        stackwise_filters=MODEL_CONFIGS["ResNet101V2"]["stackwise_filters"],
        stackwise_blocks=MODEL_CONFIGS["ResNet101V2"]["stackwise_blocks"],
        stackwise_strides=MODEL_CONFIGS["ResNet101V2"]["stackwise_strides"],
        include_top=include_top,
        input_shape=input_shape,
        input_tensor=input_tensor,
        block_type="bottleneck",
        pooling=pooling,
        classes=classes,
        **kwargs,
    )
    return model


def ResNet152V2(
    backend,
    include_top,
    classes=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    **kwargs,
):
    model_class = MODEL_BACKBONES.get(backend)
    model = model_class(
        stackwise_filters=MODEL_CONFIGS["ResNet152V2"]["stackwise_filters"],
        stackwise_blocks=MODEL_CONFIGS["ResNet152V2"]["stackwise_blocks"],
        stackwise_strides=MODEL_CONFIGS["ResNet152V2"]["stackwise_strides"],
        include_top=include_top,
        input_shape=input_shape,
        input_tensor=input_tensor,
        block_type="bottleneck",
        pooling=pooling,
        classes=classes,
        **kwargs,
    )
    return model
