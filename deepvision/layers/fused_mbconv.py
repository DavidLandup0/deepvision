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
import torch.nn as nn
from tensorflow.keras import layers

from deepvision.utils.utils import same_padding


@tf.keras.utils.register_keras_serializable(package="deepvision")
class __FusedMBConvTF(layers.Layer):
    def __init__(
        self,
        input_filters: int,
        output_filters: int,
        expand_ratio=1,
        kernel_size=3,
        strides=1,
        se_ratio=0.0,
        bn_momentum=0.9,
        activation=tf.keras.activations.swish,
        dropout: float = 0.8,
        **kwargs,
    ):

        super().__init__(**kwargs)
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.expand_ratio = expand_ratio
        self.kernel_size = kernel_size
        self.strides = strides
        self.se_ratio = se_ratio
        self.bn_momentum = bn_momentum
        self.activation = activation
        self.dropout = dropout
        self.filters = self.input_filters * self.expand_ratio
        self.filters_se = max(1, int(input_filters * se_ratio))

        self.conv1 = layers.Conv2D(
            filters=self.filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            use_bias=False,
        )
        self.bn1 = layers.BatchNormalization(
            momentum=self.bn_momentum,
        )

        self.se_conv1 = layers.Conv2D(
            self.filters_se,
            1,
            padding="same",
            activation=self.activation,
        )

        self.se_conv2 = layers.Conv2D(
            self.filters,
            1,
            padding="same",
            activation="sigmoid",
        )

        self.output_conv = layers.Conv2D(
            filters=self.output_filters,
            kernel_size=1 if expand_ratio != 1 else kernel_size,
            strides=1,
            padding="same",
            use_bias=False,
        )

        self.bn_out = layers.BatchNormalization(momentum=self.bn_momentum)

    def call(self, inputs):
        # Expansion
        if self.expand_ratio != 1:
            x = self.conv1(inputs)
            x = self.bn1(x)
            x = self.activation(x)
        else:
            x = inputs

        # Squeeze-and-Excite
        if 0 < self.se_ratio <= 1:
            se = layers.GlobalAveragePooling2D()(x)
            se = layers.Reshape((1, 1, self.filters))(se)

            se = self.se_conv1(se)
            se = self.se_conv2(se)

            x = layers.multiply([x, se])

        # Output projection
        x = self.output_conv(x)
        x = self.bn_out(x)
        if self.expand_ratio == 1:
            x = self.activation(x)

        # Residual addition with dropout
        if self.strides == 1 and self.input_filters == self.output_filters:
            if self.dropout:
                x = layers.Dropout(
                    self.dropout,
                )(x)
            x = layers.add([x, inputs])
        return x

    def get_config(self):
        config = {
            "input_filters": self.input_filters,
            "output_filters": self.output_filters,
            "expand_ratio": self.expand_ratio,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "se_ratio": self.se_ratio,
            "bn_momentum": self.bn_momentum,
            "activation": self.activation,
            "dropout": self.dropout,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        activation = config.pop("activation")
        activation = tf.keras.activations.deserialize(activation)
        return cls(activation=activation, **config)


class __FusedMBConvPT(nn.Module):
    def __init__(
        self,
        input_filters: int,
        output_filters: int,
        expand_ratio=1,
        kernel_size=3,
        strides=1,
        se_ratio=0.0,
        bn_momentum=0.9,
        activation=nn.SiLU(),
        dropout: float = 0.8,
        name=None,  # Ignored but added for generalizability between backends
        **kwargs,
    ):

        super().__init__(**kwargs)
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.expand_ratio = expand_ratio
        self.kernel_size = kernel_size
        self.stride = strides
        self.se_ratio = se_ratio
        self.bn_momentum = bn_momentum
        self.activation = activation
        self.dropout = dropout
        self.filters = self.input_filters * self.expand_ratio
        self.filters_se = max(1, int(input_filters * se_ratio))

        if self.expand_ratio != 1:
            self.conv1 = nn.Conv2d(
                in_channels=self.input_filters,
                out_channels=self.filters,
                kernel_size=kernel_size,
                stride=strides,
                padding=same_padding(kernel_size, strides),
                bias=False,
            )
            self.bn1 = nn.BatchNorm2d(self.filters, momentum=self.bn_momentum)

        if 0 < self.se_ratio <= 1:
            self.se_conv1 = nn.Conv2d(self.filters, self.filters_se, 1, padding="same")
            self.se_conv2 = nn.Conv2d(self.filters_se, self.filters, 1, padding="same")

        self.output_conv = nn.Conv2d(
            in_channels=self.filters,
            out_channels=self.output_filters,
            kernel_size=1 if expand_ratio != 1 else kernel_size,
            stride=1,
            padding="same",
            bias=False,
        )

        self.bn_out = nn.BatchNorm2d(self.output_filters, momentum=self.bn_momentum)

    def forward(self, inputs):
        if self.expand_ratio != 1:
            x = self.conv1(inputs)
            x = self.bn1(x)
            x = self.activation()(x)
        else:
            x = inputs

        # Squeeze-and-Excite
        if 0 < self.se_ratio <= 1:
            se = nn.AvgPool2d(x.shape[2])(x)
            # No need to reshape, output is already [B, C, 1, 1]
            # se = se.reshape(x.shape[0], self.filters, 1, 1)

            se = self.se_conv1(se)
            se = self.activation()(se)
            se = self.se_conv2(se)
            se = nn.Sigmoid()(se)
            x = x * se

        # Output projection
        x = self.output_conv(x)
        x = self.bn_out(x)
        if self.expand_ratio == 1:
            x = self.activation()(x)

        # Residual addition with dropout
        if self.stride == 1 and self.input_filters == self.output_filters:
            if self.dropout:
                x = nn.Dropout(self.dropout)(x)
            x = x + inputs
        return x


LAYER_BACKBONES = {
    "tensorflow": __FusedMBConvTF,
    "pytorch": __FusedMBConvPT,
}


def FusedMBConv(
    input_filters: int,
    output_filters: int,
    backend,
    expand_ratio=1,
    kernel_size=3,
    strides=1,
    se_ratio=0.0,
    bn_momentum=0.9,
    activation=None,
    dropout: float = 0.8,
    **kwargs,
):
    """
    Implementation of the FusedMBConv (Fused Mobile Inverted Residual Bottleneck) block.


    FusedMBConv blocks have been popularized, and primarily used in EfficientNetV2s. They're fundamentally
    similar to MBConv blocks, but replace the depthwise and 1x1 output convolutions with a single, fused 3x3
    convolution block, begetting the name. FusedMBConv blocks and MBConv blocks are the basis of the EfficientNetV2
    family, and can be used in mobile-friendly, edge-friendly or low-latency efficient networks, which use few
    computational resources, but provide competitive performance (accuracy-wise).

    The efficiency comes from a narrow-wide-narrow structure, regular convolutions, as opposed
    to the conventional wide-narrow/bottleneck-wide structure in blocks throughout many architectures.

    Given their usefulness and general applicability in production, FusedMBConv blocks are made as a public API.

    Acknowledgements and other implementations:
        - The TensorFlow layer was originally implemented by @AdityaKane2001 for KerasCV.

    References:
        EfficientNet-EdgeTPU: Creating Accelerator-Optimized Neural Networks with AutoML - https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html.
        EfficientNetV2: Smaller Models and Faster Training - https://arxiv.org/abs/2104.00298v3.

    Basic usage:

    ```
    inputs = tf.random.normal(shape=(1, 64, 64, 32))
    layer = deepvision.layers.FusedMBConv(input_filters=32, output_filters=32, backend='tensorflow')
    output = layer(inputs)
    print(output.shape) # (1, 64, 64, 32)

    inputs = torch.rand(1, 32, 64, 64)
    layer = deepvision.layers.FusedMBConv(input_filters=32, output_filters=32, backend='pytorch')
    output = layer(inputs)
    print(output.shape) # torch.Size([1, 32, 64, 64])
    ```
    """
    layer_class = LAYER_BACKBONES.get(backend)
    if layer_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {LAYER_BACKBONES.keys()}"
        )

    if activation is None:
        activation = tf.keras.activations.swish if backend == "tensorflow" else nn.SiLU

    layer = layer_class(
        input_filters=input_filters,
        output_filters=output_filters,
        expand_ratio=expand_ratio,
        kernel_size=kernel_size,
        strides=strides,
        se_ratio=se_ratio,
        bn_momentum=bn_momentum,
        activation=activation,
        dropout=dropout,
        **kwargs,
    )

    return layer


def tf_to_pt(layer, dummy_input=None):
    """
    Params:
    layer: TensorFlow layer to convert weights from.
    dummy_input: Dummy input, mimicking the expected input for the translated PyTorch layer.

    Returns:
        PyTorch MBConv block with weights transferred from the provided TensorFlow layer.
    """
    if not isinstance(layer, __FusedMBConvTF):
        raise ValueError(f"Layer type not supported, received: {type(layer)}")

    # Pass dummy input through to
    # get variables under `layer.variables`
    if dummy_input is None:
        dummy_input = tf.random.normal([1, 32, 32, layer.input_filters])
    layer(dummy_input)

    pytorch_mbconv = __FusedMBConvPT(
        input_filters=layer.input_filters,
        output_filters=layer.output_filters,
        expand_ratio=layer.expand_ratio,
        kernel_size=layer.kernel_size,
        strides=layer.strides,
        se_ratio=layer.se_ratio,
        bn_momentum=layer.bn_momentum,
        activation=torch.nn.SiLU,
        dropout=layer.dropout,
    )

    if layer.expand_ratio != 1:
        # conv1 and bn1
        pytorch_mbconv.conv1.weight.data = torch.nn.Parameter(
            torch.from_numpy(tf.transpose(layer.conv1.kernel, (3, 2, 0, 1)).numpy())
        )
        pytorch_mbconv.bn1.weight.data = torch.nn.Parameter(
            torch.from_numpy(layer.bn1.gamma.numpy())
        )
        pytorch_mbconv.bn1.bias.data = torch.nn.Parameter(
            torch.from_numpy(layer.bn1.beta.numpy())
        )
        pytorch_mbconv.bn1.running_mean.data = torch.nn.Parameter(
            torch.from_numpy(layer.bn1.moving_mean.numpy())
        )
        pytorch_mbconv.bn1.running_var.data = torch.nn.Parameter(
            torch.from_numpy(layer.bn1.moving_variance.numpy())
        )

    if 0 < layer.se_ratio <= 1:
        pytorch_mbconv.se_conv1.weight.data = torch.nn.Parameter(
            torch.from_numpy(tf.transpose(layer.se_conv1.kernel, (3, 2, 0, 1)).numpy())
        )
        pytorch_mbconv.se_conv1.bias.data = torch.nn.Parameter(
            torch.from_numpy(layer.se_conv1.bias.numpy())
        )
        pytorch_mbconv.se_conv2.weight.data = torch.nn.Parameter(
            torch.from_numpy(tf.transpose(layer.se_conv2.kernel, (3, 2, 0, 1)).numpy())
        )
        pytorch_mbconv.se_conv2.bias.data = torch.nn.Parameter(
            torch.from_numpy(layer.se_conv2.bias.numpy())
        )
    pytorch_mbconv.output_conv.weight.data = torch.nn.Parameter(
        torch.from_numpy(tf.transpose(layer.output_conv.kernel, (3, 2, 0, 1)).numpy())
    )

    pytorch_mbconv.bn_out.weight.data = torch.nn.Parameter(
        torch.from_numpy(layer.bn_out.gamma.numpy())
    )
    pytorch_mbconv.bn_out.bias.data = torch.nn.Parameter(
        torch.from_numpy(layer.bn_out.beta.numpy())
    )
    pytorch_mbconv.bn_out.running_mean.data = torch.nn.Parameter(
        torch.from_numpy(layer.bn_out.moving_mean.numpy())
    )
    pytorch_mbconv.bn_out.running_var.data = torch.nn.Parameter(
        torch.from_numpy(layer.bn_out.moving_variance.numpy())
    )

    return pytorch_mbconv


def pt_to_tf(layer, dummy_input=None):
    """
    Params:
    layer: PyTorch layer to convert weights from.
    dummy_input: Dummy input, mimicking the expected input for the translated TensorFlow layer.

    Returns:
        TensorFlow MBConv block with weights transferred from the provided PyTorch layer.
    """
    if not isinstance(layer, __FusedMBConvPT):
        raise ValueError(f"Layer type not supported, received: {type(layer)}")

    tensorflow_mbconv = __FusedMBConvTF(
        input_filters=layer.input_filters,
        output_filters=layer.output_filters,
        expand_ratio=layer.expand_ratio,
        kernel_size=layer.kernel_size,
        strides=layer.stride,
        se_ratio=layer.se_ratio,
        bn_momentum=layer.bn_momentum,
        activation=tf.keras.activations.swish,
        dropout=layer.dropout,
    )

    # Pass dummy input through to
    # get variables under `layer.variables`
    if dummy_input is None:
        dummy_input = torch.rand(1, layer.input_filters, 224, 224)

    tf_dummy_input = tf.convert_to_tensor(
        dummy_input.detach().cpu().numpy().transpose(0, 2, 3, 1)
    )
    tensorflow_mbconv(tf_dummy_input)

    if layer.expand_ratio != 1:
        # conv1 and bn1
        tensorflow_mbconv.conv1.kernel.assign(
            tf.convert_to_tensor(
                layer.conv1.weight.data.permute(2, 3, 1, 0).detach().cpu().numpy()
            )
        )
        tensorflow_mbconv.bn1.gamma.assign(
            tf.convert_to_tensor(layer.bn1.weight.data.detach().cpu().numpy())
        )

        tensorflow_mbconv.bn1.beta.assign(
            tf.convert_to_tensor(layer.bn1.bias.data.detach().cpu().numpy())
        )

        tensorflow_mbconv.bn1.moving_mean.assign(
            tf.convert_to_tensor(layer.bn1.running_mean.data.detach().cpu().numpy())
        )

        tensorflow_mbconv.bn1.moving_variance.assign(
            tf.convert_to_tensor(layer.bn1.running_var.data.detach().cpu().numpy())
        )

    if 0 < layer.se_ratio <= 1:
        tensorflow_mbconv.se_conv1.kernel.assign(
            tf.convert_to_tensor(
                layer.se_conv1.weight.data.permute(2, 3, 1, 0).detach().cpu().numpy()
            )
        )
        tensorflow_mbconv.se_conv1.bias.assign(
            tf.convert_to_tensor(layer.se_conv1.bias.data.detach().cpu().numpy())
        )
        tensorflow_mbconv.se_conv2.kernel.assign(
            tf.convert_to_tensor(
                layer.se_conv2.weight.data.permute(2, 3, 1, 0).detach().cpu().numpy()
            )
        )
        tensorflow_mbconv.se_conv2.bias.assign(
            tf.convert_to_tensor(layer.se_conv2.bias.data.detach().cpu().numpy())
        )

    tensorflow_mbconv.output_conv.kernel.assign(
        tf.convert_to_tensor(
            layer.output_conv.weight.data.permute(2, 3, 1, 0).detach().cpu().numpy()
        )
    )

    tensorflow_mbconv.bn_out.gamma.assign(
        tf.convert_to_tensor(layer.bn_out.weight.data.detach().cpu().numpy())
    )

    tensorflow_mbconv.bn_out.beta.assign(
        tf.convert_to_tensor(layer.bn_out.bias.data.detach().cpu().numpy())
    )

    tensorflow_mbconv.bn_out.moving_mean.assign(
        tf.convert_to_tensor(layer.bn_out.running_mean.data.detach().cpu().numpy())
    )

    tensorflow_mbconv.bn_out.moving_variance.assign(
        tf.convert_to_tensor(layer.bn_out.running_var.data.detach().cpu().numpy())
    )

    return tensorflow_mbconv
