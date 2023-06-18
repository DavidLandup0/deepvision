# encode images
from collections import OrderedDict
from typing import Tuple
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import tensorflow as tf

from deepvision.layers.residual_transformer_encoder import ResidualTransformerEncoder

from deepvision.utils.utils import parse_model_inputs


class __CLIPImageEncoderTF(tf.keras.Model):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        input_tensor=None,
        **kwargs,
    ):
        
        inputs = parse_model_inputs("tensorflow", None, input_tensor)
        x = inputs
    

        x = tf.keras.layers.Conv2D(
            filters=width,
            kernel_size=patch_size,
            strides=patch_size,
            use_bias=False,
        )(x)

        x = tf.reshape(x, (x.shape[0], x.shape[1], -1))
        x = tf.transpose(x, perm=(0, 2, 1))
        
        scale = width ** -0.5
        class_embedding = tf.Variable(
            scale * tf.random.normal((width,))
        )

        class_embedding = tf.expand_dims(self.class_embedding, axis=0)
        class_embedding = tf.tile(class_embedding, (x.shape[0], 1, 1))
        x = tf.concat([class_embedding, x], axis=1)  # shape = [*, grid ** 2 + 1, width]
        positional_embedding = tf.Variable(
            scale * tf.random.normal(((input_resolution // patch_size) ** 2 + 1, width))
        )
        x = x + positional_embedding
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

        x = tf.transpose(x, perm=(1, 0, 2))  # NLD -> LND
        x = ResidualTransformerEncoder(
            width, layers, heads, backend="tensorflow"
        )(x)
        x = tf.transpose(x, perm=(1, 0, 2))  # LND -> NLD

        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x[:, 0, :])

        proj = tf.Variable(
            scale * tf.random.normal((width, output_dim))
        )

        if proj is not None:
            x = tf.matmul(x, self.proj)

        output = x
        
        super().__init__(
            inputs=inputs,
            outputs=output,
            **kwargs,
        )

        """self.channels = embed_dims
        self.num_stages = num_stages
        self.output_channels = embed_dims
        self.classes = classes
        self.include_top = include_top
        self.as_backbone = as_backbone
        self.pooling = pooling

        self.patch_embedding_layers = []
        self.transformer_blocks = []"""

    """
    def call(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = tf.reshape(x, (x.shape[0], x.shape[1], -1))  # shape = [*, width, grid ** 2]
        x = tf.transpose(x, perm=(0, 2, 1))  # shape = [*, grid ** 2, width]
        class_embedding = tf.expand_dims(self.class_embedding, axis=0)
        class_embedding = tf.tile(class_embedding, (x.shape[0], 1, 1))
        x = tf.concat([class_embedding, x], axis=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding
        x = self.ln_pre(x)

        x = tf.transpose(x, perm=(1, 0, 2))  # NLD -> LND
        x = self.transformer(x)
        x = tf.transpose(x, perm=(1, 0, 2))  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = tf.matmul(x, self.proj)

        return x
    """


class __CLIPImageEncoderPT(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        """
        The `conv1`, `class_embedding` and `positional_embedding`s are effectively the
        `PatchingAndEmbedding` layer, but with a -0.5 scale factor on the class/positional embeddings.
        TODO: @davidlandup0
            Find a way to either update the existing PatchingAndEmbedding layer to allow for this generally,
            or subclass a new layer specifically for this. Weight porting will be made more difficult, but a total
            remapping is needed anyway due to porting the weights to TensorFlow, so changing state dict names
            is trivial, as long as we can guarantee the same operations across both implementations.
        """
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = nn.LayerNorm(width)

        self.transformer = ResidualTransformerEncoder(
            width, layers, heads, backend="pytorch"
        )

        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


MODEL_BACKBONES = {
    "tensorflow": __CLIPImageEncoderTF,
    "pytorch": __CLIPImageEncoderPT,
}


def CLIPImageEncoder(
    input_resolution, patch_size, width, layers, heads, output_dim, backend
):
    model_class = MODEL_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )

    model = model_class(input_resolution, patch_size, width, layers, heads, output_dim)

    return model
