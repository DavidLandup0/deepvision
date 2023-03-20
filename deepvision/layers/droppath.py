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


@tf.keras.utils.register_keras_serializable(package="deepvision")
class __DropPathTF(tf.keras.layers.Layer):
    def __init__(self, drop_prob=None, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x):
        input_shape = tf.shape(x)
        batch_size = input_shape[0]
        rank = x.shape.rank
        shape = (batch_size,) + (1,) * (rank - 1)
        random_tensor = (1 - self.drop_prob) + tf.random.uniform(shape, dtype=x.dtype)
        path_mask = tf.floor(random_tensor)
        output = tf.math.divide(x, 1 - self.drop_prob) * path_mask
        return output

    def get_config(self):
        base_config = super().get_config()
        config = {"drop_prob": self.drop_prob}
        return {**base_config, **config}


class __DropPathPT(torch.nn.Module):
    def __init__(self, drop_prob=None, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def forward(self, x):
        input_shape = x.shape
        batch_size = input_shape[0]
        rank = len(x.shape)
        shape = (batch_size,) + (1,) * (rank - 1)
        random_tensor = (1 - self.drop_prob) + torch.rand(*shape).to(x.dtype)
        path_mask = torch.floor(random_tensor)
        output = x / (1 - self.drop_prob) * path_mask
        return output


LAYER_BACKBONES = {
    "tensorflow": __DropPathTF,
    "pytorch": __DropPathPT,
}


def DropPath(
    drop_prob,
    backend,
    **kwargs,
):
    layer_class = LAYER_BACKBONES.get(backend)
    if layer_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {LAYER_BACKBONES.keys()}"
        )

    layer = layer_class(
        drop_prob,
        **kwargs,
    )

    return layer
