# Copyright 2023 David Landup
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from deepvision.models.classification.efficientnet.efficientnetv2_pt import (
    EfficientNetV2PT,
)
from deepvision.models.classification.efficientnet.efficientnetv2_tf import (
    EfficientNetV2TF,
)

"""
All configurations are block-wise, shortened to `block_N` for cleaner formatting.
V2B0...V2B1 have the same configurations. Just different width/depth coefficients, so they all
read from "EfficientNetV2Base".
V2S, V2M and V2L have the same width/depth coefficients, but different configurations.
"""
MODEL_CONFIGS = {
    "EfficientNetV2Base": {
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
    "EfficientNetV2S": {
        "block_kernel_sizes": [3, 3, 3, 3, 3, 3],
        "block_num_repeat": [2, 4, 4, 6, 9, 15],
        "block_in_filters": [24, 24, 48, 64, 128, 160],
        "block_out_filters": [24, 48, 64, 128, 160, 256],
        "block_exp_ratios": [1, 4, 4, 4, 6, 6],
        "block_se_ratios": [0.0, 0.0, 0, 0.25, 0.25, 0.25],
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
    "EfficientNetV2M": {
        "block_kernel_sizes": [3, 3, 3, 3, 3, 3, 3],
        "block_num_repeat": [3, 5, 5, 7, 14, 18, 5],
        "block_in_filters": [24, 24, 48, 80, 160, 176, 304],
        "block_out_filters": [24, 48, 80, 160, 176, 304, 512],
        "block_exp_ratios": [1, 4, 4, 4, 6, 6, 6],
        "block_se_ratios": [0, 0, 0, 0.25, 0.25, 0.25, 0.25],
        "block_strides": [1, 2, 2, 2, 1, 2, 1],
        "block_conv_type": [
            "fused",
            "fused",
            "fused",
            "mbconv",
            "mbconv",
            "mbconv",
            "mbconv",
        ],
    },
    "EfficientNetV2L": {
        "block_kernel_sizes": [3, 3, 3, 3, 3, 3, 3],
        "block_num_repeat": [4, 7, 7, 10, 19, 25, 7],
        "block_in_filters": [32, 32, 64, 96, 192, 224, 384],
        "block_out_filters": [
            32,
            64,
            96,
            192,
            224,
            384,
            640,
        ],
        "block_exp_ratios": [1, 4, 4, 4, 6, 6, 6],
        "block_se_ratios": [0, 0, 0, 0.25, 0.25, 0.25, 0.25],
        "block_strides": [1, 2, 2, 2, 1, 2, 1],
        "block_conv_type": [
            "fused",
            "fused",
            "fused",
            "mbconv",
            "mbconv",
            "mbconv",
            "mbconv",
        ],
    },
}

MODEL_BACKBONES = {"tensorflow": EfficientNetV2TF, "pytorch": EfficientNetV2PT}

"""
Returns a TensorFlow or PyTorch model, of the EfficientNetV2 family.

References:
    EfficientNet-EdgeTPU: Creating Accelerator-Optimized Neural Networks with AutoML - https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html.
    EfficientNetV2: Smaller Models and Faster Training - https://arxiv.org/abs/2104.00298v3.

The paper introduces EfficientNetV2S, EfficientNetV2M and EfficientNetV2L only, but the same arguments for building V1
EfficientNetB0, ... EfficientNetB7 can be used with the new architecture changes.

To widen the application of EfficientNets, libraries like KerasCV have adopted V2B0...V2B3, as well as V2S, V2M and V2L.
We've chosen to do the same.
"""


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
        blockwise_kernel_sizes=MODEL_CONFIGS["EfficientNetV2Base"][
            "block_kernel_sizes"
        ],
        blockwise_num_repeat=MODEL_CONFIGS["EfficientNetV2Base"]["block_num_repeat"],
        blockwise_input_filters=MODEL_CONFIGS["EfficientNetV2Base"]["block_in_filters"],
        blockwise_output_filters=MODEL_CONFIGS["EfficientNetV2Base"][
            "block_out_filters"
        ],
        blockwise_expand_ratios=MODEL_CONFIGS["EfficientNetV2Base"]["block_exp_ratios"],
        blockwise_se_ratios=MODEL_CONFIGS["EfficientNetV2Base"]["block_se_ratios"],
        blockwise_strides=MODEL_CONFIGS["EfficientNetV2Base"]["block_strides"],
        blockwise_conv_type=MODEL_CONFIGS["EfficientNetV2Base"]["block_conv_type"],
    )
    return model


def EfficientNetV2B1(
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
        depth_coefficient=1.1,
        blockwise_kernel_sizes=MODEL_CONFIGS["EfficientNetV2Base"][
            "block_kernel_sizes"
        ],
        blockwise_num_repeat=MODEL_CONFIGS["EfficientNetV2Base"]["block_num_repeat"],
        blockwise_input_filters=MODEL_CONFIGS["EfficientNetV2Base"]["block_in_filters"],
        blockwise_output_filters=MODEL_CONFIGS["EfficientNetV2Base"][
            "block_out_filters"
        ],
        blockwise_expand_ratios=MODEL_CONFIGS["EfficientNetV2Base"]["block_exp_ratios"],
        blockwise_se_ratios=MODEL_CONFIGS["EfficientNetV2Base"]["block_se_ratios"],
        blockwise_strides=MODEL_CONFIGS["EfficientNetV2Base"]["block_strides"],
        blockwise_conv_type=MODEL_CONFIGS["EfficientNetV2Base"]["block_conv_type"],
    )
    return model


def EfficientNetV2B2(
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
        width_coefficient=1.1,
        depth_coefficient=1.2,
        blockwise_kernel_sizes=MODEL_CONFIGS["EfficientNetV2Base"][
            "block_kernel_sizes"
        ],
        blockwise_num_repeat=MODEL_CONFIGS["EfficientNetV2Base"]["block_num_repeat"],
        blockwise_input_filters=MODEL_CONFIGS["EfficientNetV2Base"]["block_in_filters"],
        blockwise_output_filters=MODEL_CONFIGS["EfficientNetV2Base"][
            "block_out_filters"
        ],
        blockwise_expand_ratios=MODEL_CONFIGS["EfficientNetV2Base"]["block_exp_ratios"],
        blockwise_se_ratios=MODEL_CONFIGS["EfficientNetV2Base"]["block_se_ratios"],
        blockwise_strides=MODEL_CONFIGS["EfficientNetV2Base"]["block_strides"],
        blockwise_conv_type=MODEL_CONFIGS["EfficientNetV2Base"]["block_conv_type"],
    )
    return model


def EfficientNetV2B3(
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
        width_coefficient=1.2,
        depth_coefficient=1.4,
        blockwise_kernel_sizes=MODEL_CONFIGS["EfficientNetV2Base"][
            "block_kernel_sizes"
        ],
        blockwise_num_repeat=MODEL_CONFIGS["EfficientNetV2Base"]["block_num_repeat"],
        blockwise_input_filters=MODEL_CONFIGS["EfficientNetV2Base"]["block_in_filters"],
        blockwise_output_filters=MODEL_CONFIGS["EfficientNetV2Base"][
            "block_out_filters"
        ],
        blockwise_expand_ratios=MODEL_CONFIGS["EfficientNetV2Base"]["block_exp_ratios"],
        blockwise_se_ratios=MODEL_CONFIGS["EfficientNetV2Base"]["block_se_ratios"],
        blockwise_strides=MODEL_CONFIGS["EfficientNetV2Base"]["block_strides"],
        blockwise_conv_type=MODEL_CONFIGS["EfficientNetV2Base"]["block_conv_type"],
    )
    return model


def EfficientNetV2S(
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
        blockwise_kernel_sizes=MODEL_CONFIGS["EfficientNetV2S"]["block_kernel_sizes"],
        blockwise_num_repeat=MODEL_CONFIGS["EfficientNetV2S"]["block_num_repeat"],
        blockwise_input_filters=MODEL_CONFIGS["EfficientNetV2S"]["block_in_filters"],
        blockwise_output_filters=MODEL_CONFIGS["EfficientNetV2S"]["block_out_filters"],
        blockwise_expand_ratios=MODEL_CONFIGS["EfficientNetV2S"]["block_exp_ratios"],
        blockwise_se_ratios=MODEL_CONFIGS["EfficientNetV2S"]["block_se_ratios"],
        blockwise_strides=MODEL_CONFIGS["EfficientNetV2S"]["block_strides"],
        blockwise_conv_type=MODEL_CONFIGS["EfficientNetV2S"]["block_conv_type"],
    )
    return model


def EfficientNetV2M(
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
        blockwise_kernel_sizes=MODEL_CONFIGS["EfficientNetV2M"]["block_kernel_sizes"],
        blockwise_num_repeat=MODEL_CONFIGS["EfficientNetV2M"]["block_num_repeat"],
        blockwise_input_filters=MODEL_CONFIGS["EfficientNetV2M"]["block_in_filters"],
        blockwise_output_filters=MODEL_CONFIGS["EfficientNetV2M"]["block_out_filters"],
        blockwise_expand_ratios=MODEL_CONFIGS["EfficientNetV2M"]["block_exp_ratios"],
        blockwise_se_ratios=MODEL_CONFIGS["EfficientNetV2M"]["block_se_ratios"],
        blockwise_strides=MODEL_CONFIGS["EfficientNetV2M"]["block_strides"],
        blockwise_conv_type=MODEL_CONFIGS["EfficientNetV2M"]["block_conv_type"],
    )
    return model


def EfficientNetV2L(
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
        blockwise_kernel_sizes=MODEL_CONFIGS["EfficientNetV2L"]["block_kernel_sizes"],
        blockwise_num_repeat=MODEL_CONFIGS["EfficientNetV2L"]["block_num_repeat"],
        blockwise_input_filters=MODEL_CONFIGS["EfficientNetV2L"]["block_in_filters"],
        blockwise_output_filters=MODEL_CONFIGS["EfficientNetV2L"]["block_out_filters"],
        blockwise_expand_ratios=MODEL_CONFIGS["EfficientNetV2L"]["block_exp_ratios"],
        blockwise_se_ratios=MODEL_CONFIGS["EfficientNetV2L"]["block_se_ratios"],
        blockwise_strides=MODEL_CONFIGS["EfficientNetV2L"]["block_strides"],
        blockwise_conv_type=MODEL_CONFIGS["EfficientNetV2L"]["block_conv_type"],
    )
    return model
