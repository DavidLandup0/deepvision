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

import math

import tensorflow as tf
import torch
import torch.nn as nn
from tensorflow.keras import layers


class __PatchingAndEmbeddingTF(layers.Layer):
    def __init__(self, project_dim, patch_size, padding="valid", **kwargs):
        super().__init__(**kwargs)
        self.project_dim = project_dim
        self.patch_size = patch_size
        self.padding = padding
        if patch_size < 0:
            raise ValueError(
                f"The patch_size cannot be a negative number. Received {patch_size}"
            )
        if padding not in ["valid", "same"]:
            raise ValueError(
                f"Padding must be either 'same' or 'valid', but {padding} was passed."
            )
        self.projection = layers.Conv2D(
            filters=self.project_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding=self.padding,
        )

    def build(self, input_shape):
        self.class_token = self.add_weight(
            shape=[1, 1, self.project_dim], name="class_token", trainable=True
        )
        self.num_patches = (
            input_shape[1] // self.patch_size * input_shape[2] // self.patch_size
        )
        self.position_embedding = layers.Embedding(
            input_dim=self.num_patches + 1, output_dim=self.project_dim
        )

    def call(
        self,
        images,
        interpolate=False,
        interpolate_width=None,
        interpolate_height=None,
        patch_size=None,
    ):

        """Calls the PatchingAndEmbedding layer on a batch of images.
        Args:
            images: A `tf.Tensor` of shape [batch, width, height, depth]
            interpolate: A `bool` to enable or disable interpolation
            interpolate_height: An `int` representing interpolated height
            interpolate_width: An `int` representing interpolated width
            patch_size: An `int` representing the new patch size if interpolation is used

        Returns:
            `A tf.Tensor` of shape [batch, patch_num+1, embedding_dim]
        """
        # Turn images into patches and project them onto `project_dim`
        patches = self.projection(images)
        patch_shapes = tf.shape(patches)
        patches_flattened = tf.reshape(
            patches,
            shape=(
                patch_shapes[0],
                patch_shapes[-2] * patch_shapes[-2],
                patch_shapes[-1],
            ),
        )

        # Add learnable class token before linear projection and positional embedding
        flattened_shapes = tf.shape(patches_flattened)
        class_token_broadcast = tf.cast(
            tf.broadcast_to(
                self.class_token,
                [flattened_shapes[0], 1, flattened_shapes[-1]],
            ),
            dtype=patches_flattened.dtype,
        )
        patches_flattened = tf.concat([class_token_broadcast, patches_flattened], 1)
        positions = tf.range(start=0, limit=self.num_patches + 1, delta=1)

        if interpolate and None not in (
            interpolate_width,
            interpolate_height,
            patch_size,
        ):
            (
                interpolated_embeddings,
                class_token,
            ) = self.__interpolate_positional_embeddings(
                self.position_embedding(positions),
                interpolate_width,
                interpolate_height,
                patch_size,
            )
            addition = patches_flattened + interpolated_embeddings
            encoded = tf.concat([class_token, addition], 1)
        elif interpolate and None in (
            interpolate_width,
            interpolate_height,
            patch_size,
        ):
            raise ValueError(
                "`None of `interpolate_width`, `interpolate_height` and `patch_size` cannot be None if `interpolate` is True"
            )
        else:
            encoded = patches_flattened + self.position_embedding(positions)
        return encoded

    def __interpolate_positional_embeddings(self, embedding, height, width, patch_size):
        """
        Allows for pre-trained position embedding interpolation. This trick allows you to fine-tune a ViT
        on higher resolution images than it was trained on.

        Based on:
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_tf_vit.py
        """

        dimensionality = embedding.shape[-1]

        class_token = tf.expand_dims(embedding[:1, :], 0)
        patch_positional_embeddings = embedding[1:, :]

        h0 = height // patch_size
        w0 = width // patch_size

        new_shape = tf.constant(int(math.sqrt(self.num_patches)))

        interpolated_embeddings = tf.image.resize(
            images=tf.reshape(
                patch_positional_embeddings,
                shape=(
                    1,
                    new_shape,
                    new_shape,
                    dimensionality,
                ),
            ),
            size=(h0, w0),
            method="bicubic",
        )

        reshaped_embeddings = tf.reshape(
            tensor=interpolated_embeddings, shape=(1, -1, dimensionality)
        )

        return reshaped_embeddings, class_token

    def get_config(self):
        config = {
            "project_dim": self.project_dim,
            "patch_size": self.patch_size,
            "padding": self.padding,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class __PatchingAndEmbeddingPT(torch.nn.Module):
    def __init__(self, project_dim, input_shape, patch_size, padding="valid", **kwargs):
        super().__init__(**kwargs)
        self.project_dim = project_dim
        self.patch_size = patch_size
        self.input_shape = input_shape
        self.padding = padding

        if patch_size < 0:
            raise ValueError(
                f"The patch_size cannot be a negative number. Received {patch_size}"
            )
        if padding not in ["valid", "same"]:
            raise ValueError(
                f"Padding must be either 'same' or 'valid', but {padding} was passed."
            )
        self.projection = nn.Conv2d(
            input_shape[0],
            self.project_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=self.padding,
        )

        self.class_token = nn.Parameter(torch.rand(1, 1, self.project_dim))
        self.num_patches = (
            input_shape[1] // self.patch_size * input_shape[2] // self.patch_size
        )
        self.position_embedding = torch.nn.Embedding(
            num_embeddings=self.num_patches + 1, embedding_dim=self.project_dim
        )

    def forward(
        self,
        images,
        interpolate=False,
        interpolate_width=None,
        interpolate_height=None,
        patch_size=None,
    ):

        """Calls the PatchingAndEmbedding layer on a batch of images.
        Args:
            images: A `torch.Tensor` of shape [batch, channels, width, height]
            interpolate: A `bool` to enable or disable interpolation
            interpolate_height: An `int` representing interpolated height
            interpolate_width: An `int` representing interpolated width
            patch_size: An `int` representing the new patch size if interpolation is used

        Returns:
            `A torch.Tensor` of shape [batch, patch_num+1, embedding_dim]
        """
        # Turn images into patches and project them onto `project_dim`
        patches = self.projection(images)
        patch_shapes = patches.shape

        patches_flattened = torch.reshape(
            patches,
            shape=(
                patch_shapes[0],
                patch_shapes[-1] * patch_shapes[-1],
                patch_shapes[1],
            ),
        )

        # Add learnable class token before linear projection and positional embedding
        flattened_shapes = patches_flattened.shape
        # [1, 1, project_dim] -> [B, 1, project_dim]
        class_token_broadcast = self.class_token.expand(flattened_shapes[0], -1, -1)
        patches_flattened = torch.cat([class_token_broadcast, patches_flattened], 1)
        positions = torch.arange(start=0, end=self.num_patches + 1, step=1).to(
            patches_flattened.device
        )

        if interpolate and None not in (
            interpolate_width,
            interpolate_height,
            patch_size,
        ):
            (
                interpolated_embeddings,
                class_token,
            ) = self.__interpolate_positional_embeddings(
                self.position_embedding(positions),
                interpolate_width,
                interpolate_height,
                patch_size,
            )
            addition = patches_flattened + interpolated_embeddings
            encoded = torch.cat([class_token, addition], 1)
        elif interpolate and None in (
            interpolate_width,
            interpolate_height,
            patch_size,
        ):
            raise ValueError(
                "`None of `interpolate_width`, `interpolate_height` and `patch_size` cannot be None if `interpolate` is True"
            )
        else:
            encoded = patches_flattened + self.position_embedding(positions)
        return encoded

    def __interpolate_positional_embeddings(self, embedding, height, width, patch_size):
        """
        Allows for pre-trained position embedding interpolation. This trick allows you to fine-tune a ViT
        on higher resolution images than it was trained on.

        Based on:
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py
        """

        dimensionality = embedding.shape[-1]

        class_token = embedding[:1, :].unsqueeze(0)
        patch_positional_embeddings = embedding[1:, :]

        h0 = height // patch_size
        w0 = width // patch_size

        new_shape = int(math.sqrt(self.num_patches))

        interpolated_embeddings = torch.nn.functional.interpolate(
            patch_positional_embeddings.reshape(
                1, new_shape, new_shape, dimensionality
            ),
            size=(h0, w0),
            mode="bicubic",
        )

        reshaped_embeddings = interpolated_embeddings.reshape(1, -1, dimensionality)

        return reshaped_embeddings, class_token


LAYER_BACKBONES = {
    "tensorflow": __PatchingAndEmbeddingTF,
    "pytorch": __PatchingAndEmbeddingPT,
}


def PatchingAndEmbedding(
    project_dim, patch_size, backend, input_shape=None, padding="valid"
):
    """
    Layer to patchify images, prepend a class token, positionally embed and
    create a projection of patches for Vision Transformers

    The layer expects a batch of input images and returns batches of patches, flattened as a sequence
    and projected onto `project_dims`. If the height and width of the images
    aren't divisible by the patch size, the supplied padding type is used (or 'valid' by default).

    Reference:
        An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
        by Alexey Dosovitskiy et al. (https://arxiv.org/abs/2010.11929)

        Acknowledgements and other implementations:
        - The TensorFlow layer was originally implemented by David Landup for KerasCV.
        - HuggingFace's implementations: https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py
            and https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_tf_vit.py
        - Ross Wightman's implementations: https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/vision_transformer.py

    Args:
        project_dim: the dimensionality of the project_dim
        input_shape: the input height and width (ignored for TensorFlow version)
        patch_size: the patch size
        padding: default 'valid', the padding to apply for patchifying images

    Returns:
        Patchified and linearly projected input images, including a prepended learnable class token
        with shape (batch, num_patches+1, project_dim)

    Basic usage:

    ```
    images = torch.rand(1, 3, 224, 224)
    encoded_patches = deepvision.layers.PatchingAndEmbedding(project_dim=1024,
                                                             patch_size=16,
                                                             input_shape=(3, 224, 224),
                                                             backend='pytorch')(images)
    print(encoded_patches.shape) # torch.Size([1, 197, 1024])

    images = tf.random.normal([1, 224, 224, 3])
    encoded_patches = deepvision.layers.PatchingAndEmbedding(project_dim=1024,
                                                             patch_size=16,
                                                             input_shape=(224, 224, 3),
                                                             backend='tensorflow')(images)
    print(encoded_patches.shape) # (1, 197, 1024)
    ```
    """

    layer_class = LAYER_BACKBONES.get(backend)
    if layer_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {LAYER_BACKBONES.keys()}"
        )

    layer = layer_class(
        project_dim=project_dim,
        patch_size=patch_size,
        input_shape=input_shape,
        padding=padding,
    )

    return layer
