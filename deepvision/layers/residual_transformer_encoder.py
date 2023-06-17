import torch
from torch import nn

from deepvision.layers.residual_attention import ResidualAttention


class __ResidualTransformerEncoderPT(nn.Module):
    def __init__(
        self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[
                ResidualAttention(width, heads, attn_mask, backend="pytorch")
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class __ResidualTransformerEncoderTF(nn.Module):
    def __init__(
        self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[
                ResidualAttention(width, heads, attn_mask, backend="tensorflow")
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor):
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
