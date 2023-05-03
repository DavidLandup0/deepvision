from torch import nn
import tensorflow as tf


class __MLP_PT(nn.Module):
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


class __MLP_TF(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dim: int,
        output_dim: int,
        num_layers: int,
        input_dim: int = None,
        activation=None,
        output_act=False,
    ) -> None:
        super().__init__()
        pass

    def call(self, x):
        pass


LAYER_BACKBONES = {
    "tensorflow": __MLP_TF,
    "pytorch": __MLP_PT,
}


def MLP(
    embed_dim: int,
    output_dim: int,
    backend,
    input_dim: int = None,
    num_layers: int = 2,
    activation=None,
    output_act=False,
):
    """
    Generic helper MLP. Allows for creating sequential MLP networks, typically used as prediction heads for semantic segmentation, depth estimation,
    or within Transformer-style blocks.

    This layer is made available as part of the public API because it's reused within multiple architectures,
    to solidify terminology. For example:
        - In some, implementations, the "embedding dimensionality" refers to the output, projected dimensionality.
        - In others, the "embedding dimensionality" is the hidden/latent dimensionality, not the projected output.

    To avoid having many helper MLP classes/functions, with differing terminology, one main helper MLP can be used
    and customized to niche down to each individual implementation.

    Args:
        embed_dim:
        output_dim:
        num_layers:
        input_dim:
        activation:
        output_act:

    Returns:

    """

    layer_class = LAYER_BACKBONES.get(backend)
    if layer_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {LAYER_BACKBONES.keys()}"
        )

    layer = layer_class(
        embed_dim=embed_dim,
        output_dim=output_dim,
        input_dim=input_dim,
        num_layers=num_layers,
        activation=activation,
        output_act=output_act,
    )

    return layer
