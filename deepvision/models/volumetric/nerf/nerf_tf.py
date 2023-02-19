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

from deepvision.utils.utils import parse_model_inputs


class NeRFTF(tf.keras.Model):
    def __init__(
        self,
        input_shape=(None, None, 3),
        input_tensor=None,
        depth=None,
        width=None,
        **kwargs,
    ):

        inputs = parse_model_inputs("tensorflow", input_shape, input_tensor)
        x = inputs

        for i in range(depth):
            x = layers.Dense(units=width, activation="relu")(x)
            if i % 4 == 0 and i > 0:
                x = layers.concatenate([x, inputs], axis=-1)
        output = layers.Dense(4)(x)

        super().__init__(
            inputs={
                "inputs": inputs,
            },
            outputs={
                "output": output,
            },
            **kwargs,
        )
