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
class __StochaticDepthTF(tf.keras.layers.Layer):
    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob

    def call(self, x):
        if self.drop_prob == 0.0 or not self.trainable:
            return x
        keep_mask = 1 - self.drop_prob
        shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
        random_tensor = keep_mask + tf.random.uniform(shape)
        random_tensor = tf.floor(random_tensor)
        random_tensor = tf.divide(x, keep_mask) * random_tensor
        return random_tensor

    def get_config(self):
        base_config = super().get_config()
        config = {"drop_prob": self.drop_prob}
        return {**base_config, **config}


class __StochaticDepthPT(torch.nn.Module):
    """
    Based on: https://github.com/sithu31296/semantic-segmentation/blob/main/semseg/models/layers/common.py
    """

    def __init__(self, drop_prob=None, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        kp = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = kp + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(kp) * random_tensor


LAYER_BACKBONES = {
    "tensorflow": __StochaticDepthTF,
    "pytorch": __StochaticDepthPT,
}


def StochasticDepth(
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
