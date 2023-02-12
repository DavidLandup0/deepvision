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
from tensorflow.keras import layers

from deepvision.layers import PatchingAndEmbedding
from deepvision.layers import TransformerEncoder
from deepvision.utils.utils import parse_model_inputs


class ViTTF(tf.keras.Model):
    def __init__(
        self,
        include_top,
        input_shape=(None, None, 3),
        input_tensor=None,
        pooling=None,
        classes=None,
        patch_size=None,
        transformer_layer_num=None,
        project_dim=None,
        num_heads=None,
        mlp_dim=None,
        mlp_dropout=None,
        attention_dropout=None,
        activation=None,
        **kwargs,
    ):

        if include_top and not classes:
            raise ValueError(
                "If `include_top` is True, you should specify `classes`. "
                f"Received: classes={classes}"
            )

        if include_top and pooling:
            raise ValueError(
                f"`pooling` must be `None` when `include_top=True`."
                f"Received pooling={pooling} and include_top={include_top}. "
            )

        if not include_top and pooling is None:
            raise ValueError(f"`pooling` must be specified when `include_top=False`.")

        inputs = parse_model_inputs("tensorflow", input_shape, input_tensor)
        x = inputs

        encoded_patches = PatchingAndEmbedding(
            project_dim=project_dim,
            patch_size=patch_size,
            input_shape=input_shape,
            backend="tensorflow",
        )(x)
        encoded_patches = layers.Dropout(mlp_dropout)(encoded_patches)

        for i in range(transformer_layer_num):
            encoded_patches = TransformerEncoder(
                project_dim=project_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                mlp_dropout=mlp_dropout,
                attention_dropout=attention_dropout,
                activation=activation,
                backend="tensorflow",
                name=f"transformer_encoder_{i}",
            )(encoded_patches)

        layer_norm = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        if include_top:
            x = layers.GlobalAveragePooling1D(name="avg_pool")(layer_norm)
            output = layers.Dense(classes, activation="softmax", name="predictions")(x)
        else:
            if pooling == "token":
                output = layers.Lambda(lambda rep: rep[:, 0], name="token_pool")(
                    layer_norm
                )
            elif pooling == "avg":
                output = layers.GlobalAveragePooling1D(name="avg_pool")(layer_norm)
            elif pooling == "max":
                output = layers.GlobalMaxPooling1D(name="max_pool")(layer_norm)

        super().__init__(
            inputs={
                "inputs": inputs,
            },
            outputs={
                "output": output,
            },
            **kwargs,
        )
        self.include_top = include_top
        self.pooling = pooling
        self.classes = classes
        self.transformer_layer_num = transformer_layer_num
        self.project_dim = project_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.mlp_dropout = mlp_dropout
        self.attention_dropout = attention_dropout
        self.activation = activation
