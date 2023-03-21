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

from deepvision.models.segmentation.backbones.mit.mit_pt import __MiTPT
from deepvision.models.segmentation.backbones.mit.mit_tf import __MiTTF

MODEL_CONFIGS = {
    "B0": {"embedding_dims": [32, 64, 160, 256], "depths": [2, 2, 2, 2]},
    "B1": {"embedding_dims": [64, 128, 320, 512], "depths": [2, 2, 2, 2]},
    "B2": {"embedding_dims": [64, 128, 320, 512], "depths": [3, 4, 6, 3]},
    "B3": {"embedding_dims": [64, 128, 320, 512], "depths": [3, 4, 18, 3]},
    "B4": {"embedding_dims": [64, 128, 320, 512], "depths": [3, 8, 27, 3]},
    "B5": {"embedding_dims": [64, 128, 320, 512], "depths": [3, 6, 40, 3]},
}

MODEL_BACKBONES = {"tensorflow": __MiTTF, "pytorch": __MiTPT}


def MiTB0(input_shape, backend):
    model_class = MODEL_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )
    return model_class(
        input_shape=input_shape,
        embed_dims=MODEL_CONFIGS["B0"]["embedding_dims"],
        depths=MODEL_CONFIGS["B0"]["depths"],
    )


def MiTB1(input_shape, backend):
    model_class = MODEL_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )
    return model_class(
        input_shape=input_shape,
        embed_dims=MODEL_CONFIGS["B1"]["embedding_dims"],
        depths=MODEL_CONFIGS["B1"]["depths"],
    )


def MiTB2(input_shape, backend):
    model_class = MODEL_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )
    return model_class(
        input_shape=input_shape,
        embed_dims=MODEL_CONFIGS["B2"]["embedding_dims"],
        depths=MODEL_CONFIGS["B2"]["depths"],
    )


def MiTB3(input_shape, backend):
    model_class = MODEL_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )
    return model_class(
        input_shape=input_shape,
        embed_dims=MODEL_CONFIGS["B3"]["embedding_dims"],
        depths=MODEL_CONFIGS["B3"]["depths"],
    )


def MiTB4(input_shape, backend):
    model_class = MODEL_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )
    return model_class(
        input_shape=input_shape,
        embed_dims=MODEL_CONFIGS["B4"]["embedding_dims"],
        depths=MODEL_CONFIGS["B4"]["depths"],
    )


def MiTB5(input_shape, backend):
    model_class = MODEL_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )
    return model_class(
        input_shape=input_shape,
        embed_dims=MODEL_CONFIGS["B5"]["embedding_dims"],
        depths=MODEL_CONFIGS["B5"]["depths"],
    )
