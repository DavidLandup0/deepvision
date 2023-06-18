import tensorflow as tf
import torch
import torch.nn as nn


class __CLIPPatchingAndEmbeddingPT(nn.Module):
    def __init__(self, width, patch_size, input_resolution):
        super().__init__()

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

    def forward(self, x):
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

        return x


class __CLIPPatchingAndEmbeddingTF(tf.keras.layers.Layer):
    def __init__(self, width, patch_size, input_resolution):
        super().__init__()

        self.conv1 = tf.keras.layers.Conv2D(
            filters=width,
            kernel_size=patch_size,
            strides=patch_size,
            use_bias=False,
        )

        scale = width**-0.5
        self.class_embedding = self.add_weight(
            shape=(
                scale
                * tf.random.normal(
                    (
                        1,
                        1,
                        width,
                    )
                )
            ).shape,
            trainable=True,
        )

        self.positional_embedding = self.add_weight(
            shape=(
                scale
                * tf.random.normal(((input_resolution // patch_size) ** 2 + 1, width))
            ).shape,
            trainable=True,
        )

    def call(self, x):
        x = self.conv1(x)  # shape = [*, grid, grid, width]
        x = tf.transpose(x, perm=[0, 3, 1, 2])  # shape = [*, width, grid, grid]
        shape = tf.shape(x)
        x = tf.reshape(
            x, [shape[0], shape[1], shape[2] * shape[3]]
        )  # shape = [*, width, grid ** 2]
        x = tf.transpose(x, perm=(0, 2, 1))  # shape = [*, grid ** 2, width]

        scale = self.class_embedding.shape[2] ** -0.5
        class_embedding = self.class_embedding * scale

        shape = tf.shape(x)
        x = tf.concat([class_embedding, x], axis=1)  # shape = [*, grid ** 2 + 1, width]
        positional_embedding = self.positional_embedding * scale
        x = x + positional_embedding

        return x


LAYER_BACKBONES = {
    "tensorflow": __CLIPPatchingAndEmbeddingTF,
    "pytorch": __CLIPPatchingAndEmbeddingPT,
}


def CLIPPatchingAndEmbedding(width, patch_size, input_resolution, backend):
    layer_class = LAYER_BACKBONES.get(backend)
    if layer_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {LAYER_BACKBONES.keys()}"
        )

    layer = layer_class(
        width=width, patch_size=patch_size, input_resolution=input_resolution
    )

    return layer
