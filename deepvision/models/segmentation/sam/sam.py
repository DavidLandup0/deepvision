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

from deepvision.layers.sam_mask_decoder import MaskDecoder
from deepvision.layers.sam_prompt_encoder import PromptEncoder
from deepvision.layers.twoway_transformer_encoder import TwoWayTransformerEncoder
from deepvision.models.classification.vision_transformer_detector.vit_det import ViTDetB
from deepvision.models.segmentation.sam.sam_pt import SAM_PT

MODEL_CONFIGS = {
    "SAM_B": {},
    "SAM_L": {},
    "SAM_H": {},
}

MODEL_BACKBONES = {"tensorflow": None, "pytorch": SAM_PT}


def SAM_B(
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

    image_embedding_size = image_size // vit_patch_size

    image_encoder = ViTDetB(backend="pytorch")

    prompt_encoder = PromptEncoder(
        embed_dim=prompt_embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(image_size, image_size),
        mask_in_chans=16,
    )

    mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformerEncoder(
            depth=2,
            embedding_dim=prompt_embed_dim,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=prompt_embed_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )

    model = model_class(
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
    )

    return model
