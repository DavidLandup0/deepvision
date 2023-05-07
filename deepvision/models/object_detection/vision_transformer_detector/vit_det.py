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

from functools import partial

import torch

from deepvision.models.object_detection.vision_transformer_detector.vit_det_pt import (
    ViTDetBackbonePT,
)

MODEL_CONFIGS = {
    "ViTDetB": {
        "prompt_embed_dim": 256,
        "input_shape": (3, 1024, 1024),
        "vit_patch_size": 16,
        "encoder_embed_dim": 768,
        "encoder_transformer_layer_num": 12,
        "encoder_num_heads": 12,
        "encoder_global_attn_indexes": [2, 5, 8, 11],
        "mlp_dim": 3072,
        "window_size": 14,
    },
    "ViTDetL": {
        "prompt_embed_dim": 256,
        "input_shape": (3, 1024, 1024),
        "vit_patch_size": 16,
        "encoder_embed_dim": 1024,
        "encoder_transformer_layer_num": 24,
        "encoder_num_heads": 16,
        "encoder_global_attn_indexes": [5, 11, 17, 23],
        "mlp_dim": 4096,
        "window_size": 14,
    },
    "ViTDetH": {
        "prompt_embed_dim": 256,
        "input_shape": (3, 1024, 1024),
        "vit_patch_size": 16,
        "encoder_embed_dim": 1280,
        "encoder_transformer_layer_num": 32,
        "encoder_num_heads": 16,
        "encoder_global_attn_indexes": [7, 15, 23, 31],
        "mlp_dim": 5120,
        "window_size": 14,
    },
}

MODEL_BACKBONES = {"tensorflow": None, "pytorch": ViTDetBackbonePT}


def ViTDetB(
    backend,
    **kwargs,
):
    model_class = MODEL_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )

    model = model_class(
        transformer_layer_num=MODEL_CONFIGS["ViTDetB"]["encoder_transformer_layer_num"],
        embed_dim=MODEL_CONFIGS["ViTDetB"]["encoder_embed_dim"],
        input_shape=MODEL_CONFIGS["ViTDetB"]["input_shape"],
        mlp_dim=MODEL_CONFIGS["ViTDetB"]["mlp_dim"],
        norm_layer=torch.nn.LayerNorm,
        num_heads=MODEL_CONFIGS["ViTDetB"]["encoder_num_heads"],
        patch_size=MODEL_CONFIGS["ViTDetB"]["vit_patch_size"],
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=MODEL_CONFIGS["ViTDetB"]["encoder_global_attn_indexes"],
        window_size=MODEL_CONFIGS["ViTDetB"]["window_size"],
        project_dim=MODEL_CONFIGS["ViTDetB"]["prompt_embed_dim"],
    )

    return model


def ViTDetL(
    backend,
    **kwargs,
):
    model_class = MODEL_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )

    model = model_class(
        transformer_layer_num=MODEL_CONFIGS["ViTDetL"]["encoder_transformer_layer_num"],
        embed_dim=MODEL_CONFIGS["ViTDetL"]["encoder_embed_dim"],
        input_shape=MODEL_CONFIGS["ViTDetL"]["input_shape"],
        mlp_dim=MODEL_CONFIGS["ViTDetL"]["mlp_dim"],
        norm_layer=torch.nn.LayerNorm,
        num_heads=MODEL_CONFIGS["ViTDetL"]["encoder_num_heads"],
        patch_size=MODEL_CONFIGS["ViTDetL"]["vit_patch_size"],
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=MODEL_CONFIGS["ViTDetL"]["encoder_global_attn_indexes"],
        window_size=MODEL_CONFIGS["ViTDetL"]["window_size"],
        project_dim=MODEL_CONFIGS["ViTDetL"]["prompt_embed_dim"],
    )

    return model


def ViTDetH(
    backend,
    **kwargs,
):
    model_class = MODEL_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )

    model = model_class(
        transformer_layer_num=MODEL_CONFIGS["ViTDetH"]["encoder_transformer_layer_num"],
        embed_dim=MODEL_CONFIGS["ViTDetH"]["encoder_embed_dim"],
        input_shape=MODEL_CONFIGS["ViTDetH"]["input_shape"],
        mlp_dim=MODEL_CONFIGS["ViTDetH"]["mlp_dim"],
        norm_layer=torch.nn.LayerNorm,
        num_heads=MODEL_CONFIGS["ViTDetH"]["encoder_num_heads"],
        patch_size=MODEL_CONFIGS["ViTDetH"]["vit_patch_size"],
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=MODEL_CONFIGS["ViTDetH"]["encoder_global_attn_indexes"],
        window_size=MODEL_CONFIGS["ViTDetH"]["window_size"],
        project_dim=MODEL_CONFIGS["ViTDetH"]["project_dim"],
    )

    return model
