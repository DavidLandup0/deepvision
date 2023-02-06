from deepvision.models.classification.efficientnet.efficientnetv2_pt import (
    EfficientNetV2PT,
)
from deepvision.models.classification.efficientnet.efficientnetv2_tf import (
    EfficientNetV2TF,
)

MODEL_CONFIGS = {
    "EfficientNetV2B0": {
        "blockwise_kernel_sizes": [3, 3, 3, 3, 3, 3],
        "blockwise_num_repeat": [1, 2, 2, 3, 5, 8],
        "blockwise_input_filters": [32, 16, 32, 48, 96, 112],
        "blockwise_output_filters": [16, 32, 48, 96, 112, 192],
        "blockwise_expand_ratios": [1, 4, 4, 4, 6, 6],
        "blockwise_se_ratios": [0, 0, 0, 0.25, 0.25, 0.25],
        "blockwise_strides": [1, 2, 2, 2, 1, 2],
        "blockwise_conv_type": [
            "fused",
            "fused",
            "fused",
            "mbconv",
            "mbconv",
            "mbconv",
        ],
    },
}

MODEL_BACKBONES = {"tensorflow": EfficientNetV2TF, "pytorch": EfficientNetV2PT}


def EfficientNetV2B0(
    backend,
    include_top,
    classes=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    **kwargs,
):
    model_class = MODEL_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )
    model = model_class(
        include_top=include_top,
        classes=classes,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        width_coefficient=1.0,
        depth_coefficient=1.0,
        blockwise_kernel_sizes=MODEL_CONFIGS["EfficientNetV2B0"][
            "blockwise_kernel_sizes"
        ],
        blockwise_num_repeat=MODEL_CONFIGS["EfficientNetV2B0"]["blockwise_num_repeat"],
        blockwise_input_filters=MODEL_CONFIGS["EfficientNetV2B0"][
            "blockwise_input_filters"
        ],
        blockwise_output_filters=MODEL_CONFIGS["EfficientNetV2B0"][
            "blockwise_output_filters"
        ],
        blockwise_expand_ratios=MODEL_CONFIGS["EfficientNetV2B0"][
            "blockwise_expand_ratios"
        ],
        blockwise_se_ratios=MODEL_CONFIGS["EfficientNetV2B0"]["blockwise_se_ratios"],
        blockwise_strides=MODEL_CONFIGS["EfficientNetV2B0"]["blockwise_strides"],
        blockwise_conv_type=MODEL_CONFIGS["EfficientNetV2B0"]["blockwise_conv_type"],
    )
    return model
