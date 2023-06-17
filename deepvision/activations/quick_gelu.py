import tensorflow as tf
import torch


class __QuickGELUPT(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class __QuickGELUTF(tf.keras.layers.Layer):
    def call(self, x: tf.Tensor):
        return x * tf.math.sigmoid(1.702 * x)


ACTIVATION_BACKBONES = {"tensorflow": __QuickGELUTF, "pytorch": __QuickGELUPT}


def QuickGELU(backend):
    activation_class = ACTIVATION_BACKBONES.get(backend)
    if activation_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {ACTIVATION_BACKBONES.keys()}"
        )

    activation = activation_class()

    return activation
