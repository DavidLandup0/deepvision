from deepvision.models.classification.resnet.resnet_pt import ResNetV2PT
from deepvision.models.classification.resnet.resnet_tf import ResNetV2TF

MODEL_CONFIGS = {
    "ResNet18V2": {
        "stackwise_filters": [64, 128, 256, 512],
        "stackwise_blocks": [2, 2, 2, 2],
        "stackwise_strides": [1, 2, 2, 2],
    },
}


def ResNet18V2(
    include_top,
    backend,
    classes=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    **kwargs
):
    if backend == "pytorch":
        return ResNetV2PT(
            stackwise_filters=MODEL_CONFIGS["ResNet18V2"]["stackwise_filters"],
            stackwise_blocks=MODEL_CONFIGS["ResNet18V2"]["stackwise_blocks"],
            stackwise_strides=MODEL_CONFIGS["ResNet18V2"]["stackwise_strides"],
            include_top=include_top,
            input_shape=input_shape,
            input_tensor=input_tensor,
            pooling=pooling,
            classes=classes,
            **kwargs
        )
    elif backend == "tensorflow":
        return ResNetV2TF(
            stackwise_filters=MODEL_CONFIGS["ResNet18V2"]["stackwise_filters"],
            stackwise_blocks=MODEL_CONFIGS["ResNet18V2"]["stackwise_blocks"],
            stackwise_strides=MODEL_CONFIGS["ResNet18V2"]["stackwise_strides"],
            include_top=include_top,
            input_shape=input_shape,
            input_tensor=input_tensor,
            pooling=pooling,
            classes=classes,
            **kwargs
        )
    return model
