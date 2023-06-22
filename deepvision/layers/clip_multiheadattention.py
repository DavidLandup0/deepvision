import math
from typing import Tuple
from typing import Type

import tensorflow as tf
import torch
from torch import Tensor
from torch import nn


class __FusedMultiheadAttentionTF(tf.keras.layers.Layer):
    """
    The "short path" implementation of `tf.keras.layers.MultiHeadAttention`, to mimic the behavior of
    PyTorch's `torch.nn.MultiheadAttention` class, when the embedding dimensionality of the key, query and value are equal.

    When the embedding dimensionality of the key, query and value are equal, they can be fused into a single operation with 3*
    the embedding dimensionality. PyTorch's `torch.nn.MultiheadAttention` does this by default, producing an `in_proj` and `out_proj` field,
    instead of the `q_proj`, `k_proj` and `v_proj` fields. This makes weight porting between the short path of the PyTorch implementation
    and the default TensorFlow implementation impossible.

    Thus, this layer is implemented *only* for TensorFlow, as it mimics the default existing behavior of the PyTorch layer, and will be used
    whenever the embedding dimensionality of the key, query and value is equal.
    """

    def __init__(self, project_dim, num_heads, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.project_dim = project_dim
        
        self.head_dim = project_dim // num_heads
        
        self.wqkv = tf.keras.layers.Dense(project_dim * 3, use_bias=False)
        self.dense = tf.keras.layers.Dense(project_dim)

        if self.project_dim % num_heads != 0:
            raise ValueError("project_dim must be divisible by num_heads")

    def _separate_heads(self, x: tf.Tensor, num_heads: int) -> tf.Tensor:
        input_shape = tf.shape(x)
        b, n, c = input_shape
        x = tf.reshape(x, [b, n, num_heads, c // num_heads])
        return tf.transpose(x, [0, 2, 1, 3])  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: tf.Tensor) -> tf.Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, [b, n_tokens, n_heads * c_per_head])  # B x N_tokens x C

    def call(self, inputs):
        qkv = self.wqkv(inputs)        
        qkv = self._separate_heads(qkv, self.num_heads)
        
        query, key, value = tf.split(qkv, 3, axis=2)
        
        scaled_attention_logits = tf.matmul(query, key, transpose_b=True)
        scaled_attention_logits = scaled_attention_logits / tf.math.sqrt(tf.cast(self.head_dim, dtype=tf.float32))
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        attention_output = tf.matmul(attention_weights, value)
        
        attention_output = self._recombine_heads(attention_output)
        
        output = self.dense(attention_output)
        
        return output


def FusedMultiheadAttention(project_dim, num_heads):
    layer = __FusedMultiheadAttentionTF(
        project_dim=project_dim,
        num_heads=num_heads,
    )

    return layer
