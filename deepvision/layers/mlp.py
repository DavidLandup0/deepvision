import torch
from torch import nn
import tensorflow as tf
import torch.functional as F


class MLP_PT(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        output_dim: int,
        num_layers: int,
        input_dim: int = None,
        activation=nn.GELU,
        output_act=False,
    ) -> None:
        super().__init__()
        h = [embed_dim] * (num_layers - 1)
        self.num_layers = num_layers
        if input_dim is None:
            input_dim = output_dim

        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.activation = activation()
        self.output_act = output_act

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.activation(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.output_act:
            x = self.activation(x)
        return x


class MLP_TF(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dim: int,
        output_dim: int,
        num_layers: int,
        input_dim: int = None,
        act=nn.GELU,
    ) -> None:
        super().__init__()
        pass

    def call(self, x):
        pass


def MLP(
    embed_dim: int, output_dim: int, num_layers: int, input_dim: int = None, act=None
):
    pass
