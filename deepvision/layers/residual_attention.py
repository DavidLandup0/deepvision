from collections import OrderedDict

import tensorflow as tf
import torch
from torch import nn

from deepvision.activations.quick_gelu import QuickGELU
from deepvision.layers.multiheadattention import MultiHeadAttention


class __ResidualAttentionPT(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, mha='custom'):
        super().__init__()

        self.mha = mha
        if mha == 'custom':
            self.attn = MultiHeadAttention(d_model, n_head, backend='pytorch')
        else:
            self.attn = torch.nn.MultiheadAttention(d_model, n_head)
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
        attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        
        if self.attn_mask is not None and self.mha == 'custom':
            if self.attn_mask.shape != (x.shape[1], 1, x.shape[0], x.shape[0]):
                attn_mask = self.attn_mask[None, None, :, :].expand(x.shape[1], 1, x.shape[0], x.shape[0])
            
        elif self.mha=='default':
            x =  self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]
            return x
        
        x =  self.attn(x, output_attentions=False, causal_attention_mask=attn_mask)
        return x
        

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class __ResidualAttentionTF(tf.keras.layers.Layer):
    def __init__(self, d_model, n_head, attn_mask=None):
        super().__init__()

        self.attn = MultiHeadAttention(d_model, n_head, backend='tensorflow')
        self.ln_1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(d_model * 4, name="c_fc"),
                QuickGELU(backend="tensorflow", name="gelu"),
                tf.keras.layers.Dense(d_model, name="c_proj"),
            ]
        )
        self.ln_2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn_mask = attn_mask

    def attention(self, x):
        self.attn_mask = (
            tf.cast(self.attn_mask, dtype=x.dtype)
            if self.attn_mask is not None
            else None
        )

        return self.attn(x, attention_mask=self.attn_mask)

    def call(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


LAYER_BACKBONES = {
    "tensorflow": __ResidualAttentionTF,
    "pytorch": __ResidualAttentionPT,
}


def ResidualAttention(d_model, n_head, attn_mask, backend, mha='custom'):
    model_class = LAYER_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {LAYER_BACKBONES.keys()}"
        )

    model = model_class(d_model, n_head, attn_mask, mha=mha)

    return model
