import tensorflow as tf
import torch.nn as nn
from keras import backend
from tensorflow.keras import layers

from deepvision.utils.utils import same_padding


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
            data_format="channels_last",
            use_bias=False,
            name=self.name + "expand_conv",
        )
        self.bn1 = layers.BatchNormalization(
            momentum=self.bn_momentum,
            name=self.name + "expand_bn",
        )

        self.bn2 = layers.BatchNormalization(
            momentum=self.bn_momentum, name=self.name + "bn"
        )

        self.se_conv1 = layers.Conv2D(
            self.filters_se,
            1,
            padding="same",
            activation=self.activation,
            name=self.name + "se_reduce",
        )

        self.se_conv2 = layers.Conv2D(
            self.filters,
            1,
            padding="same",
            activation="sigmoid",
            name=self.name + "se_expand",
        )

        self.output_conv = layers.Conv2D(
            filters=self.output_filters,
            kernel_size=1 if expand_ratio != 1 else kernel_size,
            strides=1,
            padding="same",
            data_format="channels_last",
            use_bias=False,
            name=self.name + "project_conv",
        )

        self.bn3 = layers.BatchNormalization(
            momentum=self.bn_momentum, name=self.name + "project_bn"
        )

    def build(self, input_shape):
        if self.name is None:
            self.name = backend.get_uid("block0")

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
            se = layers.GlobalAveragePooling2D(name=self.name + "se_squeeze")(x)
            se = layers.Reshape((1, 1, self.filters), name=self.name + "se_reshape")(se)

            se = self.se_conv1(se)
            se = self.se_conv2(se)

            x = layers.multiply([x, se], name=self.name + "se_excite")

        # Output projection
        x = self.output_conv(x)
        x = self.bn3(x)
        if self.expand_ratio == 1:
            x = self.activation(x)

        # Residual addition with dropout
        if self.strides == 1 and self.input_filters == self.output_filters:
            if self.dropout:
                x = layers.Dropout(
                    self.dropout,
                    noise_shape=(None, 1, 1, 1),
                    name=self.name + "drop",
                )(x)
            x = layers.add([x, inputs], name=self.name + "add")
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

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=self.filters,
            kernel_size=kernel_size,
            stride=strides,
            padding=same_padding(kernel_size, strides),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.filters, momentum=self.bn_momentum)
        self.bn2 = nn.BatchNorm2d(self.filters, momentum=self.bn_momentum)

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

        self.bn3 = nn.BatchNorm2d(self.output_filters, momentum=self.bn_momentum)

    def forward(self, inputs):
        if self.expand_ratio != 1:
            x = self.conv1(inputs)
            x = self.bn1(x)
            x = self.activation()(x)
        else:
            x = inputs

        # Squeeze-and-Excite
        if 0 < self.se_ratio <= 1:
            se = x.mean(dim=2)
            se = se.reshape(1, 1, self.filters)

            se = self.se_conv1(se)
            se = self.activation()(se)
            se = self.se_conv2(se)
            se = nn.Sigmoid()(se)
            x = x * se

        # Output projection
        x = self.output_conv(x)
        x = self.bn3(x)
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
