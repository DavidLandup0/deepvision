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

from deepvision.models.classification.vision_transformer.vit_pt import ViTPT
from deepvision.models.classification.vision_transformer.vit_tf import ViTTF

MODEL_CONFIGS = {
    "ViTTiny16": {
        "patch_size": 16,
        "transformer_layer_num": 12,
        "project_dim": 192,
        "mlp_dim": 768,
        "num_heads": 3,
        "mlp_dropout": 0.0,
        "attention_dropout": 0.0,
    },
    "ViTS16": {
        "patch_size": 16,
        "transformer_layer_num": 12,
        "project_dim": 384,
        "mlp_dim": 1536,
        "num_heads": 6,
        "mlp_dropout": 0.0,
        "attention_dropout": 0.0,
    },
    "ViTB16": {
        "patch_size": 16,
        "transformer_layer_num": 12,
        "project_dim": 768,
        "mlp_dim": 3072,
        "num_heads": 12,
        "mlp_dropout": 0.0,
        "attention_dropout": 0.0,
    },
    "ViTL16": {
        "patch_size": 16,
        "transformer_layer_num": 24,
        "project_dim": 1024,
        "mlp_dim": 4096,
        "num_heads": 16,
        "mlp_dropout": 0.1,
        "attention_dropout": 0.0,
    },
    "ViTH16": {
        "patch_size": 16,
        "transformer_layer_num": 32,
        "project_dim": 1280,
        "mlp_dim": 5120,
        "num_heads": 16,
        "mlp_dropout": 0.1,
        "attention_dropout": 0.0,
    },
    "ViTTiny32": {
        "patch_size": 32,
        "transformer_layer_num": 12,
        "project_dim": 192,
        "mlp_dim": 768,
        "num_heads": 3,
        "mlp_dropout": 0.0,
        "attention_dropout": 0.0,
    },
    "ViTS32": {
        "patch_size": 32,
        "transformer_layer_num": 12,
        "project_dim": 384,
        "mlp_dim": 1536,
        "num_heads": 6,
        "mlp_dropout": 0.0,
        "attention_dropout": 0.0,
    },
    "ViTB32": {
        "patch_size": 32,
        "transformer_layer_num": 12,
        "project_dim": 768,
        "mlp_dim": 3072,
        "num_heads": 12,
        "mlp_dropout": 0.0,
        "attention_dropout": 0.0,
    },
    "ViTL32": {
        "patch_size": 32,
        "transformer_layer_num": 24,
        "project_dim": 1024,
        "mlp_dim": 4096,
        "num_heads": 16,
        "mlp_dropout": 0.1,
        "attention_dropout": 0.0,
    },
    "ViTH32": {
        "patch_size": 32,
        "transformer_layer_num": 32,
        "project_dim": 1280,
        "mlp_dim": 5120,
        "num_heads": 16,
        "mlp_dropout": 0.1,
        "attention_dropout": 0.0,
    },
}

MODEL_BACKBONES = {"tensorflow": ViTTF, "pytorch": ViTPT}


def ViTTiny16(
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
        include_top,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        patch_size=MODEL_CONFIGS["ViTTiny16"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViTTiny16"]["transformer_layer_num"],
        project_dim=MODEL_CONFIGS["ViTTiny16"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViTTiny16"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViTTiny16"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViTTiny16"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViTTiny16"]["attention_dropout"],
        **kwargs,
    )

    return model


def ViTS16(
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
        include_top,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        patch_size=MODEL_CONFIGS["ViTS16"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViTS16"]["transformer_layer_num"],
        project_dim=MODEL_CONFIGS["ViTS16"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViTS16"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViTS16"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViTS16"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViTS16"]["attention_dropout"],
        **kwargs,
    )

    return model


def ViTB16(
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
        include_top,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        patch_size=MODEL_CONFIGS["ViTB16"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViTB16"]["transformer_layer_num"],
        project_dim=MODEL_CONFIGS["ViTB16"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViTB16"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViTB16"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViTB16"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViTB16"]["attention_dropout"],
        **kwargs,
    )

    return model


def ViTL16(
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
        include_top,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        patch_size=MODEL_CONFIGS["ViTL16"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViTL16"]["transformer_layer_num"],
        project_dim=MODEL_CONFIGS["ViTL16"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViTL16"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViTL16"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViTL16"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViTL16"]["attention_dropout"],
        **kwargs,
    )

    return model


def ViTH16(
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
        include_top,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        patch_size=MODEL_CONFIGS["ViTH16"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViTH16"]["transformer_layer_num"],
        project_dim=MODEL_CONFIGS["ViTH16"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViTH16"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViTH16"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViTH16"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViTH16"]["attention_dropout"],
        **kwargs,
    )

    return model


def ViTTiny32(
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
        include_top,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        patch_size=MODEL_CONFIGS["ViTTiny32"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViTTiny32"]["transformer_layer_num"],
        project_dim=MODEL_CONFIGS["ViTTiny32"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViTTiny32"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViTTiny32"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViTTiny32"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViTTiny32"]["attention_dropout"],
        **kwargs,
    )

    return model


def ViTS32(
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
        include_top,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        patch_size=MODEL_CONFIGS["ViTS32"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViTS32"]["transformer_layer_num"],
        project_dim=MODEL_CONFIGS["ViTS32"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViTS32"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViTS32"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViTS32"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViTS32"]["attention_dropout"],
        **kwargs,
    )

    return model


def ViTB32(
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
        include_top,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        patch_size=MODEL_CONFIGS["ViTB32"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViTB32"]["transformer_layer_num"],
        project_dim=MODEL_CONFIGS["ViTB32"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViTB32"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViTB32"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViTB32"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViTB32"]["attention_dropout"],
        **kwargs,
    )

    return model


def ViTL32(
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
        include_top,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        patch_size=MODEL_CONFIGS["ViTL32"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViTL32"]["transformer_layer_num"],
        project_dim=MODEL_CONFIGS["ViTL32"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViTL32"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViTL32"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViTL32"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViTL32"]["attention_dropout"],
        **kwargs,
    )

    return model


def ViTH32(
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
        include_top,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        patch_size=MODEL_CONFIGS["ViTH32"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViTH32"]["transformer_layer_num"],
        project_dim=MODEL_CONFIGS["ViTH32"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViTH32"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViTH32"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViTH32"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViTH32"]["attention_dropout"],
        **kwargs,
    )

    return model
