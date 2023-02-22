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

from deepvision.models.volumetric.nerf.nerf_pt import NeRFPT
from deepvision.models.volumetric.nerf.nerf_tf import NeRFTF

"""
A few different setups for a NeRF-style model. The parameters used to instantiate a NeRF model equivalent to the
official implementation is `NeRF`, with smaller and larger options available to fit hardware limitations of some machines.
"""

MODEL_CONFIGS = {
    "NeRFTiny": {
        "depth": 2,
        "width": 64,
    },
    "NeRFSmall": {
        "depth": 4,
        "width": 128,
    },
    "NeRFMedium": {
        "depth": 6,
        "width": 128,
    },
    "NeRF": {
        "depth": 8,
        "width": 256,
    },
    "NeRFLarge": {
        "depth": 12,
        "width": 256,
    },
}

MODEL_BACKBONES = {"tensorflow": NeRFTF, "pytorch": NeRFPT}


def NeRFTiny(
    backend,
    input_shape=(None, None, None),
    **kwargs,
):
    model_class = MODEL_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )
    model = model_class(
        input_shape=input_shape,
        depth=MODEL_CONFIGS["NeRFTiny"]["depth"],
        width=MODEL_CONFIGS["NeRFTiny"]["width"],
        **kwargs,
    )

    return model


def NeRFSmall(
    backend,
    input_shape=(None, None, None),
    **kwargs,
):
    model_class = MODEL_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )
    model = model_class(
        input_shape=input_shape,
        depth=MODEL_CONFIGS["NeRFSmall"]["depth"],
        width=MODEL_CONFIGS["NeRFSmall"]["width"],
        **kwargs,
    )

    return model


def NeRFMedium(
    backend,
    input_shape=(None, None, None),
    **kwargs,
):
    model_class = MODEL_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )
    model = model_class(
        input_shape=input_shape,
        depth=MODEL_CONFIGS["NeRFMedium"]["depth"],
        width=MODEL_CONFIGS["NeRFMedium"]["width"],
        **kwargs,
    )

    return model


def NeRF(
    backend,
    input_shape=(None, None, None),
    **kwargs,
):
    model_class = MODEL_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )
    model = model_class(
        input_shape=input_shape,
        depth=MODEL_CONFIGS["NeRF"]["depth"],
        width=MODEL_CONFIGS["NeRF"]["width"],
        **kwargs,
    )

    return model


def NeRFLarge(
    backend,
    input_shape=(None, None, None),
    **kwargs,
):
    model_class = MODEL_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )
    model = model_class(
        input_shape=input_shape,
        depth=MODEL_CONFIGS["NeRFLarge"]["depth"],
        width=MODEL_CONFIGS["NeRFLarge"]["width"],
        **kwargs,
    )

    return model
