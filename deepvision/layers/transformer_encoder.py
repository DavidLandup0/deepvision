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

import tensorflow as tf
import torch
import torch.nn as nn
from tensorflow.keras import layers


class __TransformerEncoderPT(torch.nn.Module):
    """
    Transformer encoder block implementation as a `torch.nn.Module`.
    A custom TransformerEncoderPT implementation is used to maintain identical implementations between the
    PyTorch and TensorFlow counterparts, instead of utilizing the built-in `torch.nn.TransformerEncoderLayer`.

    Note that the input and output dimensionality of the layer must stay the same,
    due to the residual addition in the layer.

    Args:
        project_dim: the dimensionality of the projection of the encoder, and output of the `MultiheadAttention`
        mlp_dim: the intermediate dimensionality of the MLP head before projecting to `project_dim`
        num_heads: the number of heads for the `MultiheadAttention` layer
        mlp_dropout: default 0.1, the dropout rate to apply between the layers of the MLP head of the encoder
        attention_dropout: default 0.1, the dropout rate to apply in the MultiHeadAttention layer
        activation: default 'torch.nn.GELU', the activation function to apply in the MLP head - should be a function
        layer_norm_epsilon: default 1e-06, the epsilon for `LayerNorm` layers
    """

    def __init__(
        self,
        project_dim,
        num_heads,
        mlp_dim,
        mlp_dropout=0.1,
        attention_dropout=0.1,
        activation=torch.nn.GELU,
        layer_norm_epsilon=1e-06,
        name=None,  # Ignored but added for generalizability between backends
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.project_dim = project_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.mlp_dropout = mlp_dropout
        self.attention_dropout = attention_dropout
        self.activation = activation
        self.layer_norm_epsilon = layer_norm_epsilon
        self.mlp_units = [mlp_dim, project_dim]

        self.layer_norm1 = nn.LayerNorm(self.project_dim, eps=self.layer_norm_epsilon)
        self.layer_norm2 = nn.LayerNorm(self.project_dim, eps=self.layer_norm_epsilon)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.project_dim,
            num_heads=self.num_heads,
            dropout=self.attention_dropout,
        )
        self.linear1 = nn.Linear(project_dim, self.mlp_units[0])
        self.linear2 = nn.Linear(self.mlp_units[0], self.mlp_units[1])

    def forward(self, inputs):
        """Calls the Transformer Encoder on an input sequence.
        Args:
            inputs: A `torch.Tensor` of shape [batch, sequence, projection]

        Returns:
            `A torch.Tensor` of shape [batch, sequence+1, project_dim]
        """

        if inputs.shape[-1] != self.project_dim:
            raise ValueError(
                f"The input and output dimensionality must be the same, but the TransformerEncoder was provided with {inputs.shape[-1]} and {self.project_dim}"
            )

        x = self.layer_norm1(inputs)
        attn, attn_weights = self.attn(x, x, x)
        x = nn.Dropout(self.mlp_dropout)(attn)
        x = x + inputs

        y = self.layer_norm2(x)

        y = self.linear1(y)
        if self.activation == torch.nn.GELU:
            y = self.activation(approximate="tanh")(y)
        else:
            y = self.activation()(y)
        y = nn.Dropout(self.mlp_dropout)(y)
        y = self.linear2(y)
        y = nn.Dropout(self.mlp_dropout)(y)

        output = x + y

        return output


class __TransformerEncoderTF(layers.Layer):
    """
    Transformer encoder block implementation as a Keras Layer.
    Originally implemented by David Landup for KerasCV.

    Note that the input and output dimensionality of the layer must stay the same,
    due to the residual addition in the layer.

    Args:
        project_dim: the dimensionality of the projection of the encoder, and output of the `MultiHeadAttention`
        mlp_dim: the intermediate dimensionality of the MLP head before projecting to `project_dim`
        num_heads: the number of heads for the `MultiHeadAttention` layer
        mlp_dropout: default 0.1, the dropout rate to apply between the layers of the MLP head of the encoder
        attention_dropout: default 0.1, the dropout rate to apply in the MultiHeadAttention layer
        activation: default 'tf.activations.gelu', the activation function to apply in the MLP head - should be a function
        layer_norm_epsilon: default 1e-06, the epsilon for `LayerNormalization` layers
    """

    def __init__(
        self,
        project_dim,
        num_heads,
        mlp_dim,
        mlp_dropout=0.1,
        attention_dropout=0.1,
        activation=tf.keras.activations.gelu,
        layer_norm_epsilon=1e-06,
        **kwargs,
    ):

        super().__init__(**kwargs)
        self.project_dim = project_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.mlp_dropout = mlp_dropout
        self.attention_dropout = attention_dropout
        self.activation = activation
        self.layer_norm_epsilon = layer_norm_epsilon
        self.mlp_units = [mlp_dim, project_dim]

        self.layer_norm1 = layers.LayerNormalization(epsilon=self.layer_norm_epsilon)
        self.layer_norm2 = layers.LayerNormalization(epsilon=self.layer_norm_epsilon)
        self.attn = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.project_dim // self.num_heads,
            dropout=self.attention_dropout,
        )
        self.dense1 = layers.Dense(self.mlp_units[0])
        self.dense2 = layers.Dense(self.mlp_units[1])

    def call(self, inputs):
        """Calls the Transformer Encoder on an input sequence.
        Args:
            inputs: A `tf.Tensor` of shape [batch, sequence, projection]

        Returns:
            `A tf.Tensor` of shape [batch, sequence+1, project_dim]
        """

        if inputs.shape[-1] != self.project_dim:
            raise ValueError(
                f"The input and output dimensionality must be the same, but the TransformerEncoder was provided with {inputs.shape[-1]} and {self.project_dim}"
            )

        x = self.layer_norm1(inputs)
        x = self.attn(x, x)
        x = layers.Dropout(self.mlp_dropout)(x)
        x = layers.Add()([x, inputs])

        y = self.layer_norm2(x)

        y = self.dense1(y)
        if self.activation == tf.keras.activations.gelu:
            y = self.activation(y, approximate=True)
        else:
            y = self.activation(y)
        y = layers.Dropout(self.mlp_dropout)(y)
        y = self.dense2(y)
        y = layers.Dropout(self.mlp_dropout)(y)

        output = layers.Add()([x, y])

        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "project_dim": self.project_dim,
                "mlp_dim": self.mlp_dim,
                "num_heads": self.num_heads,
                "attention_dropout": self.attention_dropout,
                "mlp_dropout": self.mlp_dropout,
                "activation": self.activation,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        activation = config.pop("activation")
        activation = tf.keras.activations.deserialize(activation)
        return cls(activation=activation, **config)


LAYER_BACKBONES = {
    "tensorflow": __TransformerEncoderTF,
    "pytorch": __TransformerEncoderPT,
}


def TransformerEncoder(
    project_dim,
    mlp_dim,
    num_heads,
    backend,
    mlp_dropout=0.1,
    attention_dropout=0.1,
    activation=None,
    layer_norm_epsilon=1e-06,
    name=None,
):
    """
    Transformer encoder block implementation as a `torch.nn.Module` or `tf.keras.Layer`.
    A custom TransformerEncoder layer is exposed to maintain identical implementations between the
    PyTorch and TensorFlow counterparts, instead of utilizing the built-in `torch.nn.TransformerEncoderLayer`.

    Note that the input and output dimensionality of the layer must stay the same,
    due to the residual addition in the layer.

    Args:
        project_dim: the dimensionality of the projection of the encoder, and output of the Multi Head Attention
        mlp_dim: the intermediate dimensionality of the MLP head before projecting to `project_dim`
        num_heads: the number of heads for the Multi Head Attention layer
        mlp_dropout: default 0.1, the dropout rate to apply between the layers of the MLP head of the encoder
        attention_dropout: default 0.1, the dropout rate to apply in the Multi Head Attention layer
        activation: default None, the activation function to apply in the MLP head - should be a function
            if not supplied, torch.nn.GELU and tf.keras.activations.gelu() are used respectively.
        layer_norm_epsilon: default 1e-06, the epsilon for layer norm layers

    Basic usage:

    ```
    project_dim = 1024
    mlp_dim = 3072
    num_heads = 4

    tensor = torch.rand(1, 197, 1024)
    trans_encoded = deepvision.layers.TransformerEncoder(project_dim=1024,
                                                         mlp_dim=3072,
                                                         num_heads=8,
                                                         backend='pytorch')(tensor)
    print(trans_encoded.shape) # torch.Size([1, 197, 1024])

    tensor = tf.random.normal([1, 197, 1024])
    trans_encoded = deepvision.layers.TransformerEncoder(project_dim=1024,
                                                         mlp_dim=3072,
                                                         num_heads=8,
                                                         backend='tensorflow')(tensor)
    print(trans_encoded.shape) # TensorShape([1, 197, 1024])
    ```
    """
    layer_class = LAYER_BACKBONES.get(backend)
    if layer_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {LAYER_BACKBONES.keys()}"
        )
    if activation is None:
        activation = (
            tf.keras.activations.gelu if backend == "tensorflow" else torch.nn.GELU
        )
    layer = layer_class(
        project_dim,
        num_heads,
        mlp_dim,
        mlp_dropout,
        attention_dropout,
        activation,
        layer_norm_epsilon,
        name=name,
    )

    return layer
