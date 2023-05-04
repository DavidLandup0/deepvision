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
    "ViTDetB": {},
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

    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    encoder_embed_dim = 768
    encoder_depth = 12
    encoder_num_heads = 12
    encoder_global_attn_indexes = [2, 5, 8, 11]

    model = model_class(
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
    )

    return model
