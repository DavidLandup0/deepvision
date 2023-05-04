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

from deepvision.models.classification.vision_transformer_detector.vit_det_pt import (
    ViTDetBackbonePT,
)

MODEL_CONFIGS = {
    "ViTDetB": {
        "prompt_embed_dim": 256,
        "image_size": 1024,
        "vit_patch_size": 16,
        "encoder_embed_dim": 768,
        "encoder_depth": 12,
        "encoder_num_heads": 12,
        "encoder_global_attn_indexes": [2, 5, 8, 11],
        "mlp_ratio": 4,
        "window_size": 14,
    },
}

"""
L
 encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        
        
H
 encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
"""

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
        depth=MODEL_CONFIGS["ViTDetB"]["encoder_depth"],
        embed_dim=MODEL_CONFIGS["ViTDetB"]["encoder_embed_dim"],
        img_size=MODEL_CONFIGS["ViTDetB"]["image_size"],
        mlp_ratio=MODEL_CONFIGS["ViTDetB"]["mlp_ratio"],
        norm_layer=torch.nn.LayerNorm,
        num_heads=MODEL_CONFIGS["ViTDetB"]["encoder_num_heads"],
        patch_size=MODEL_CONFIGS["ViTDetB"]["vit_patch_size"],
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=MODEL_CONFIGS["ViTDetB"]["encoder_global_attn_indexes"],
        window_size=MODEL_CONFIGS["ViTDetB"]["window_size"],
        out_chans=MODEL_CONFIGS["ViTDetB"]["prompt_embed_dim"],
    )

    return model
