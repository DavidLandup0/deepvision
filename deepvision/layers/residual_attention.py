from collections import OrderedDict

import tensorflow as tf
import torch
from torch import nn

from deepvision.activations.quick_gelu import QuickGELU


class __ResidualAttentionPT(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU(backend="pytorch")),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class __ResidualAttentionTF(tf.keras.layers.Layer):
    def __init__(self, d_model, n_head, attn_mask=None):
        super().__init__()

        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads=n_head, key_dim=d_model
        )
        self.ln_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(d_model * 4, name="c_fc"),
                QuickGELU(backend="tensorflow", name="gelu"),
                tf.keras.layers.Dense(d_model, name="c_proj"),
            ]
        )
        self.ln_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.attn_mask = attn_mask

    def attention(self, x):
        self.attn_mask = (
            tf.cast(self.attn_mask, dtype=x.dtype)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, attention_mask=self.attn_mask)

    def call(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


LAYER_BACKBONES = {
    "tensorflow": __ResidualAttentionTF,
    "pytorch": __ResidualAttentionPT,
}


def ResidualAttention(d_model, n_head, attn_mask, backend):
    model_class = LAYER_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {LAYER_BACKBONES.keys()}"
        )

    model = model_class(d_model, n_head, attn_mask)

    return model
