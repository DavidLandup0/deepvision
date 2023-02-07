from deepvision.models.classification.efficientnet.efficientnetv2_pt import (
    EfficientNetV2PT,
)
from deepvision.models.classification.efficientnet.efficientnetv2_tf import (
    EfficientNetV2TF,
)

# All configurations are block-wise, shortened to `block_N`
# for cleaner formatting
MODEL_CONFIGS = {
    "EfficientNetV2B0": {
        "block_kernel_sizes": [3, 3, 3, 3, 3, 3],
        "block_num_repeat": [1, 2, 2, 3, 5, 8],
        "block_in_filters": [32, 16, 32, 48, 96, 112],
        "block_out_filters": [16, 32, 48, 96, 112, 192],
        "block_exp_ratios": [1, 4, 4, 4, 6, 6],
        "block_se_ratios": [0, 0, 0, 0.25, 0.25, 0.25],
        "block_strides": [1, 2, 2, 2, 1, 2],
        "block_conv_type": [
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
        blockwise_kernel_sizes=MODEL_CONFIGS["EfficientNetV2B0"]["block_kernel_sizes"],
        blockwise_num_repeat=MODEL_CONFIGS["EfficientNetV2B0"]["block_num_repeat"],
        blockwise_input_filters=MODEL_CONFIGS["EfficientNetV2B0"]["block_in_filters"],
        blockwise_output_filters=MODEL_CONFIGS["EfficientNetV2B0"]["block_out_filters"],
        blockwise_expand_ratios=MODEL_CONFIGS["EfficientNetV2B0"]["block_exp_ratios"],
        blockwise_se_ratios=MODEL_CONFIGS["EfficientNetV2B0"]["block_se_ratios"],
        blockwise_strides=MODEL_CONFIGS["EfficientNetV2B0"]["block_strides"],
        blockwise_conv_type=MODEL_CONFIGS["EfficientNetV2B0"]["block_conv_type"],
    )
    return model
