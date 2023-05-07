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
from deepvision.models.load_weights import load_weights
from deepvision.models.object_detection.vision_transformer_detector.vit_det import (
    ViTDetB,
)
from deepvision.models.object_detection.vision_transformer_detector.vit_det import (
    ViTDetH,
)
from deepvision.models.object_detection.vision_transformer_detector.vit_det import (
    ViTDetL,
)
from deepvision.models.segmentation.sam.sam_pt import SAM_PT

# All SAM models differ only in the ViTDet backbone
MODEL_CONFIGS = {
    "SAM": {
        "prompt_embed_dim": 256,
        "image_size": 1024,
        "vit_patch_size": 16,
        "mask_in_chans": 16,
        "num_multimask_outputs": 3,
        "transformer_depth": 2,
        "transformer_mlp_dim": 2048,
        "transformer_num_heads": 8,
        "iou_head_depth": 3,
        "iou_head_hidden_dim": 256,
    },
}

MODEL_BACKBONES = {"tensorflow": None, "pytorch": SAM_PT}


def SAM_B(
    backend,
    weights="SA-1B",
    **kwargs,
):
    if weights == "SA-1B":
        weights = load_weights("SAM_B", True, backend)

    model_class = MODEL_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )

    image_embedding_size = (
        MODEL_CONFIGS["SAM"]["image_size"] // MODEL_CONFIGS["SAM"]["vit_patch_size"]
    )

    image_encoder = ViTDetB(backend="pytorch")

    prompt_encoder = PromptEncoder(
        embed_dim=MODEL_CONFIGS["SAM"]["prompt_embed_dim"],
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(
            MODEL_CONFIGS["SAM"]["image_size"],
            MODEL_CONFIGS["SAM"]["image_size"],
        ),
        mask_in_chans=MODEL_CONFIGS["SAM"]["mask_in_chans"],
    )

    mask_decoder = MaskDecoder(
        num_multimask_outputs=MODEL_CONFIGS["SAM"]["num_multimask_outputs"],
        transformer=TwoWayTransformerEncoder(
            depth=MODEL_CONFIGS["SAM"]["transformer_depth"],
            embedding_dim=MODEL_CONFIGS["SAM"]["prompt_embed_dim"],
            mlp_dim=MODEL_CONFIGS["SAM"]["transformer_mlp_dim"],
            num_heads=MODEL_CONFIGS["SAM"]["transformer_num_heads"],
            backend=backend,
        ),
        transformer_dim=MODEL_CONFIGS["SAM"]["prompt_embed_dim"],
        iou_head_depth=MODEL_CONFIGS["SAM"]["iou_head_depth"],
        iou_head_hidden_dim=MODEL_CONFIGS["SAM"]["iou_head_hidden_dim"],
    )

    model = model_class(
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
        weights=weights,
    )

    return model


def SAM_L(
    backend,
    weights="SA-1B",
    **kwargs,
):
    if weights == "SA-1B":
        weights = load_weights("SAM_L", True, backend)

    model_class = MODEL_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )

    image_embedding_size = (
        MODEL_CONFIGS["SAM"]["image_size"] // MODEL_CONFIGS["SAM"]["vit_patch_size"]
    )

    image_encoder = ViTDetL(backend="pytorch")

    prompt_encoder = PromptEncoder(
        embed_dim=MODEL_CONFIGS["SAM"]["prompt_embed_dim"],
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(
            MODEL_CONFIGS["SAM"]["image_size"],
            MODEL_CONFIGS["SAM"]["image_size"],
        ),
        mask_in_chans=MODEL_CONFIGS["SAM"]["mask_in_chans"],
    )

    mask_decoder = MaskDecoder(
        num_multimask_outputs=MODEL_CONFIGS["SAM"]["num_multimask_outputs"],
        transformer=TwoWayTransformerEncoder(
            depth=MODEL_CONFIGS["SAM"]["transformer_depth"],
            embedding_dim=MODEL_CONFIGS["SAM"]["prompt_embed_dim"],
            mlp_dim=MODEL_CONFIGS["SAM"]["transformer_mlp_dim"],
            num_heads=MODEL_CONFIGS["SAM"]["transformer_num_heads"],
            backend=backend,
        ),
        transformer_dim=MODEL_CONFIGS["SAM"]["prompt_embed_dim"],
        iou_head_depth=MODEL_CONFIGS["SAM"]["iou_head_depth"],
        iou_head_hidden_dim=MODEL_CONFIGS["SAM"]["iou_head_hidden_dim"],
    )

    model = model_class(
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
        weights=weights,
    )

    return model


def SAM_H(
    backend,
    weights="SA-1B",
    **kwargs,
):
    if weights == "SA-1B":
        weights = load_weights("SAM_H", True, backend)

    model_class = MODEL_BACKBONES.get(backend)

    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )

    image_embedding_size = (
        MODEL_CONFIGS["SAM"]["image_size"] // MODEL_CONFIGS["SAM"]["vit_patch_size"]
    )

    image_encoder = ViTDetH(backend="pytorch")

    prompt_encoder = PromptEncoder(
        embed_dim=MODEL_CONFIGS["SAM"]["prompt_embed_dim"],
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(
            MODEL_CONFIGS["SAM"]["image_size"],
            MODEL_CONFIGS["SAM"]["image_size"],
        ),
        mask_in_chans=MODEL_CONFIGS["SAM"]["mask_in_chans"],
    )

    mask_decoder = MaskDecoder(
        num_multimask_outputs=MODEL_CONFIGS["SAM"]["num_multimask_outputs"],
        transformer=TwoWayTransformerEncoder(
            depth=MODEL_CONFIGS["SAM"]["transformer_depth"],
            embedding_dim=MODEL_CONFIGS["SAM"]["prompt_embed_dim"],
            mlp_dim=MODEL_CONFIGS["SAM"]["transformer_mlp_dim"],
            num_heads=MODEL_CONFIGS["SAM"]["transformer_num_heads"],
            backend=backend,
        ),
        transformer_dim=MODEL_CONFIGS["SAM"]["prompt_embed_dim"],
        iou_head_depth=MODEL_CONFIGS["SAM"]["iou_head_depth"],
        iou_head_hidden_dim=MODEL_CONFIGS["SAM"]["iou_head_hidden_dim"],
    )

    model = model_class(
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
        weights=weights,
    )

    return model
