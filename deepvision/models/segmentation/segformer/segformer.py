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

from deepvision.models.classification.mix_transformer import mit
from deepvision.models.segmentation.segformer.segformer_pt import __SegFormerPT
from deepvision.models.segmentation.segformer.segformer_tf import __SegFormerTF

MODEL_BACKBONES = {"tensorflow": __SegFormerTF, "pytorch": __SegFormerPT}


def SegFormerB0(
    input_shape,
    num_classes,
    backend,
    softmax_output,
    input_tensor=None,
):
    model_class = MODEL_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )

    backbone = mit.MiTB0(
        input_shape=input_shape,
        include_top=False,
        backend=backend,
        name="mit_b0_backbone",
        as_backbone=True,
    )
    return model_class(
        embed_dim=256,
        num_classes=num_classes,
        softmax_output=softmax_output,
        input_shape=input_shape,
        input_tensor=input_tensor,
        backbone=backbone,
    )


def SegFormerB1(
    input_shape,
    num_classes,
    backend,
    softmax_output,
    input_tensor=None,
):
    model_class = MODEL_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )

    backbone = mit.MiTB1(
        input_shape=input_shape,
        include_top=False,
        backend=backend,
        name="mit_b1_backbone",
        as_backbone=True,
    )
    return model_class(
        embed_dim=256,
        num_classes=num_classes,
        softmax_output=softmax_output,
        input_shape=input_shape,
        input_tensor=input_tensor,
        backbone=backbone,
    )


def SegFormerB2(
    input_shape,
    num_classes,
    backend,
    softmax_output,
    input_tensor=None,
):
    model_class = MODEL_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )

    backbone = mit.MiTB2(
        input_shape=input_shape,
        include_top=False,
        backend=backend,
        name="mit_b2_backbone",
        as_backbone=True,
    )
    return model_class(
        embed_dim=768,
        num_classes=num_classes,
        softmax_output=softmax_output,
        input_shape=input_shape,
        input_tensor=input_tensor,
        backbone=backbone,
    )


def SegFormerB3(
    input_shape,
    num_classes,
    backend,
    softmax_output,
    input_tensor=None,
):
    model_class = MODEL_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )

    backbone = mit.MiTB3(
        input_shape=input_shape,
        include_top=False,
        backend=backend,
        name="mit_b3_backbone",
        as_backbone=True,
    )
    return model_class(
        embed_dim=768,
        num_classes=num_classes,
        softmax_output=softmax_output,
        input_shape=input_shape,
        input_tensor=input_tensor,
        backbone=backbone,
    )


def SegFormerB4(
    input_shape,
    num_classes,
    backend,
    softmax_output,
    input_tensor=None,
):
    model_class = MODEL_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )

    backbone = mit.MiTB4(
        input_shape=input_shape,
        include_top=False,
        backend=backend,
        name="mit_b4_backbone",
        as_backbone=True,
    )
    return model_class(
        embed_dim=768,
        num_classes=num_classes,
        softmax_output=softmax_output,
        input_shape=input_shape,
        input_tensor=input_tensor,
        backbone=backbone,
    )


def SegFormerB5(
    input_shape,
    num_classes,
    backend,
    softmax_output,
    input_tensor=None,
):
    model_class = MODEL_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )

    backbone = mit.MiTB5(
        input_shape=input_shape,
        include_top=False,
        backend=backend,
        name="mit_b5_backbone",
        as_backbone=True,
    )
    return model_class(
        embed_dim=768,
        num_classes=num_classes,
        softmax_output=softmax_output,
        input_shape=input_shape,
        input_tensor=input_tensor,
        backbone=backbone,
    )
