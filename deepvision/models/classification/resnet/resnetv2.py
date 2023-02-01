from deepvision import config
from deepvision.models.classification.resnet.resnetv2_pt import ResNetV2PT
from deepvision.models.classification.resnet.resnetv2_tf import ResNetV2TF

MODEL_CONFIGS = {
    "ResNet18V2": {
        "stackwise_filters": [64, 128, 256, 512],
        "stackwise_blocks": [2, 2, 2, 2],
        "stackwise_strides": [1, 2, 2, 2],
    },
}

MODEL_BACKBONES = {"tensorflow": ResNetV2TF, "pytorch": ResNetV2PT}


def ResNet18V2(
    include_top,
    classes=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    **kwargs,
):
    model_placeholder = MODEL_BACKBONES.get(config.get_backend())
    model = model_placeholder(
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
