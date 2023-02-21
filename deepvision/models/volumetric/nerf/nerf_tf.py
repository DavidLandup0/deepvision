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

from deepvision.models.volumetric.volumetric_utils import render_rgb_depth_tf


class NeRFTF(tf.keras.Model):
    def __init__(
        self,
        input_shape=(None, None, 3),
        input_tensor=None,
        depth=None,
        width=None,
        **kwargs,
    ):

        inputs = tf.keras.Input(shape=input_shape)
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

        self.depth = depth
        self.width = width
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.psnr_metric = tf.keras.metrics.Mean(name="psnr")

    def train_step(self, inputs):
        # Get the images and the rays.
        (images, rays) = inputs
        (rays_flat, t_vals) = rays

        with tf.GradientTape() as tape:
            # Get the predictions from the model.
            rgb, _ = render_rgb_depth_tf(
                model=self,
                rays_flat=rays_flat,
                t_vals=t_vals,
                img_height=images.shape[1],
                img_width=images.shape[2],
                num_ray_samples=tf.shape(t_vals)[-1],
            )
            loss = self.loss(images, rgb)

        # Get the trainable variables.
        trainable_variables = self.trainable_variables

        # Get the gradeints of the trainiable variables with respect to the loss.
        gradients = tape.gradient(loss, trainable_variables)

        # Apply the grads and optimize the model.
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        # Get the PSNR of the reconstructed images and the source images.
        psnr = tf.image.psnr(images, rgb, max_val=1.0)

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.psnr_metric.update_state(psnr)
        return {"loss": self.loss_tracker.result(), "psnr": self.psnr_metric.result()}
