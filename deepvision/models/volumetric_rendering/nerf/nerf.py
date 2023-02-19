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

from deepvision.models.volumetric_rendering.nerf.nerf_pt import NeRFPT
from deepvision.models.volumetric_rendering.nerf.nerf_tf import NeRFTF

MODEL_CONFIGS = {
    "NeRF": {
        "": 0,
    },
}

MODEL_BACKBONES = {"tensorflow": NeRFTF, "pytorch": NeRFPT}


def NeRF(
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