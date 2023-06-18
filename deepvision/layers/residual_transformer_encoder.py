import tensorflow as tf
import torch

from deepvision.layers.residual_attention import ResidualAttention


class __ResidualTransformerEncoderPT(torch.nn.Module):
    def __init__(
        self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = torch.nn.Sequential(
            *[
                ResidualAttention(width, heads, attn_mask, backend="pytorch")
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class __ResidualTransformerEncoderTF(tf.keras.layers.Layer):
    def __init__(
        self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = tf.keras.Sequential(
            [
                ResidualAttention(width, heads, attn_mask, backend="tensorflow")
                for _ in range(layers)
            ]
        )

    def call(self, x: torch.Tensor):
        return self.resblocks(x)


LAYER_BACKBONES = {
    "tensorflow": __ResidualTransformerEncoderTF,
    "pytorch": __ResidualTransformerEncoderPT,
}


def ResidualTransformerEncoder(
    width,
    layers,
    heads,
    backend,
    attn_mask=None,
):
    model_class = LAYER_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {LAYER_BACKBONES.keys()}"
        )

    model = model_class(width, layers, heads, attn_mask)

    return model
