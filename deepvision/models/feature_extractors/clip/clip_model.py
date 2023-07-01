from collections import OrderedDict
from typing import Tuple
from typing import Union

import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
from torch import nn

from deepvision.layers.residual_transformer_encoder import ResidualTransformerEncoder
from deepvision.models.feature_extractors.clip.clip_image_encoder import (
    CLIPImageEncoder,
)
from deepvision.models.load_weights import load_weights

MODEL_CONFIGS = {
    "CLIP_B16": {
        "embed_dim": 512,
        "context_length": 77,
        "vocab_size": 49408,
        "transformer_width": 512,
        "transformer_heads": 8,
        "transformer_layers": 12,
        "vision_layers": 12,
        "vision_width": 768,
        "image_resolution": 224,
        "vision_patch_size": 16,
    },
    "CLIP_B32": {
        "embed_dim": 512,
        "context_length": 77,
        "vocab_size": 49408,
        "transformer_width": 512,
        "transformer_heads": 8,
        "transformer_layers": 12,
        "vision_layers": 12,
        "vision_width": 768,
        "image_resolution": 224,
        "vision_patch_size": 32,
    },
}


class __CLIPPT(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        # vision
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,
        # text
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        mha='custom',
    ):
        super().__init__()

        self.context_length = context_length

        vision_heads = vision_width // 64
        self.visual = CLIPImageEncoder(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            backend="pytorch",
            mha=mha,
        )

        self.transformer = ResidualTransformerEncoder(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            backend="pytorch",
            mha=mha,
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width)
        )
        self.ln_final = nn.LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        #self.initialize_parameters()

    def build_attention_mask(self):
        # Lazily create a causal attention mask, with full attention between the vision tokens.
        # Pytorch uses an additive attention mask, so it's filled with `-inf`
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.patch_embed.conv1.weight.dtype

    """    
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width**-0.5) * (
            (2 * self.transformer.layers) ** -0.5
        )
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width**-0.5)
            """

    def encode_images(self, image):
        x = self.visual(image.type(self.dtype))
        print('VISUAL OUTPUT', x)
        return x

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        print('TOKEN EMBEDDING', x)

        x = x + self.positional_embedding.type(self.dtype)
        print('POSITIONAL EMBEDDING', x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        print('TRANSFORMER EMBEDDING', x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        print('LN FINAL', x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        print('ENCODE TEXT OUTPUT', x)

        return x

    def forward(self, image, text):
        image_features = self.encode_images(image)
        text_features = self.encode_text(text)

        # Normalize the features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # Compute cosine similarity, and return as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


class __CLIPTF(tf.keras.Model):
    def __init__(
        self,
        embed_dim,
        image_resolution,
        vision_layers,
        vision_width,
        vision_patch_size,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
    ):
        super().__init__()

        self.context_length = context_length

        vision_heads = vision_width // 64
        self.visual = CLIPImageEncoder(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            backend="tensorflow",
        )

        self.transformer = ResidualTransformerEncoder(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            backend="tensorflow",
        )

        self.vocab_size = vocab_size
        self.token_embedding = tf.keras.layers.Embedding(vocab_size, transformer_width)
        self.positional_embedding = self.add_weight(
            shape=[self.context_length, transformer_width], name="positional_embedding"
        )
        self.ln_final = tf.keras.layers.LayerNormalization()

        self.text_projection = self.add_weight(
            shape=(transformer_width, embed_dim), name="text_projection"
        )
        self.logit_scale = tf.Variable(
            tf.ones([]) * tf.math.log(1 / 0.07), name="logit_scale"
        )

    def build_attention_mask(self):
        mask = tf.ones((self.context_length, self.context_length))  # * float("-inf")
        mask = tf.linalg.band_part(mask, 0, -1)  # zero out the lower diagonal
        # mask = tf.where(tf.equal(mask, 0), mask, tf.fill(mask.shape, float("-inf")))
        return tf.cast(mask, tf.float32)

    def encode_images(self, image):
        return self.visual(image)

    def encode_text(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = self.transformer(x)
        x = self.ln_final(x)

        batch_size = x.shape[0]
        indices_stack = tf.stack(
            [
                tf.range(batch_size, dtype=tf.int32),
                tf.cast(tf.argmax(text, axis=1), tf.int32),
            ],
            axis=-1,
        )
        selected_features = tf.gather_nd(x, indices_stack)
        x = tf.matmul(selected_features, self.text_projection)

        return x

    def call(self, image, text):
        image_features = self.encode_images(image)
        text_features = self.encode_text(text)

        image_features = image_features / tf.norm(image_features, axis=1, keepdims=True)
        text_features = text_features / tf.norm(text_features, axis=1, keepdims=True)

        logit_scale = tf.exp(self.logit_scale)
        logits_per_image = logit_scale * tf.matmul(
            image_features, text_features, transpose_b=True
        )
        logits_per_text = tf.transpose(logits_per_image)

        return logits_per_image, logits_per_text


MODEL_BACKBONES = {
    "tensorflow": __CLIPTF,
    "pytorch": __CLIPPT,
}


def CLIP_B16(pretrained=True, backend=None, mha='custom'):
    embed_dim = MODEL_CONFIGS["CLIP_B16"]["embed_dim"]
    context_length = MODEL_CONFIGS["CLIP_B16"]["context_length"]
    vocab_size = MODEL_CONFIGS["CLIP_B16"]["vocab_size"]
    transformer_width = MODEL_CONFIGS["CLIP_B16"]["transformer_width"]
    transformer_heads = MODEL_CONFIGS["CLIP_B16"]["transformer_heads"]
    transformer_layers = MODEL_CONFIGS["CLIP_B16"]["transformer_layers"]
    vision_layers = MODEL_CONFIGS["CLIP_B16"]["vision_layers"]
    vision_width = MODEL_CONFIGS["CLIP_B16"]["vision_width"]
    vision_patch_size = MODEL_CONFIGS["CLIP_B16"]["vision_patch_size"]
    image_resolution = MODEL_CONFIGS["CLIP_B16"]["image_resolution"]

    model_class = MODEL_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )

    model = model_class(
        embed_dim,
        image_resolution,
        vision_layers,
        vision_width,
        vision_patch_size,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
        mha,
    )

    # PyTorch only
    if pretrained:
        weights = load_weights("CLIP_B16", True, backend)
        model.load_state_dict(weights)

    return model


def CLIP_B32():
    return None


def CLIP_L14():
    return None
