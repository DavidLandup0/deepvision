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

from deepvision.models.volumetric.volumetric_utils import nerf_render_image_and_depth_tf


class NeRFTF(tf.keras.Model):
    def __init__(
        self,
        input_shape=(None, None, None),
        depth=None,
        width=None,
        **kwargs,
    ):
        """
        Neural Radiance Field (NeRF) model, implemented in TensorFlow.

        Args:
            input_shape: the shape of the input tensor
            depth: the depth of the model (i.e. the number of layers to stack)
            width: the 'channels' of each stacked layer (i.e. the number of dense units in each layer)
        """

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
        (images, rays) = inputs
        (rays_flat, t_vals) = rays

        with tf.GradientTape() as tape:
            rgb, _, _ = nerf_render_image_and_depth_tf(
                model=self,
                rays_flat=rays_flat,
                t_vals=t_vals,
                img_height=images.shape[1],
                img_width=images.shape[2],
                num_ray_samples=tf.shape(t_vals)[-1],
            )
            loss = self.loss(images, rgb)

        # Compute gradients and apply optimizer step
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Compute Peak Signal-to-Noise Ratio (PSNR) between the predicted images and actual images
        psnr = tf.image.psnr(images, rgb, max_val=1.0)

        self.loss_tracker.update_state(loss)
        self.psnr_metric.update_state(psnr)
        return {"loss": self.loss_tracker.result(), "psnr": self.psnr_metric.result()}

    def test_step(self, inputs):
        (images, rays) = inputs
        (rays_flat, t_vals) = rays

        rgb, _, _ = nerf_render_image_and_depth_tf(
            model=self,
            rays_flat=rays_flat,
            t_vals=t_vals,
            img_height=images.shape[1],
            img_width=images.shape[2],
            num_ray_samples=tf.shape(t_vals)[-1],
        )
        val_loss = self.loss(images, rgb)

        # Compute Peak Signal-to-Noise Ratio (PSNR) between the predicted images and actual images
        val_psnr = tf.image.psnr(images, rgb, max_val=1.0)

        self.loss_tracker.update_state(val_loss)
        self.psnr_metric.update_state(val_psnr)
        return {
            "loss": self.loss_tracker.result(),
            "psnr": self.psnr_metric.result(),
        }
