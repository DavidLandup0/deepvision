import torch
from deepvision import utils

MODEL_CONFIGS = {
    "ResNet18V2": {
        "stackwise_filters": [64, 128, 256, 512],
        "stackwise_blocks": [2, 2, 2, 2],
        "stackwise_strides": [1, 2, 2, 2],
    },

}


def BasicBlock(
        filters, kernel_size=3, stride=1, dilation=1, conv_shortcut=False, name=None
):
    def apply(x):
        use_preactivation = torch.nn.BatchNorm()(x)
        use_preactivation = torch.nn.ReLU()(use_preactivation)

        s = stride if dilation == 1 else 1
        if conv_shortcut:
            shortcut = torch.nn.Conv2d(filters, 1, strides=s)(use_preactivation)
        else:
            shortcut = (
                torch.nn.MaxPool2D(1, strides=stride)(x)
                if s > 1
                else x
            )

        x = torch.nn.Conv2d(
            filters,
            kernel_size,
            padding="SAME",
            strides=1,
            use_bias=False
        )(use_preactivation)
        x = torch.nn.BatchNorm(epsilon=1.001e-5, name=name + "_1_bn"
                               )(x)
        x = torch.nn.ReLU()(x)

        x = torch.nn.Conv2d(
            filters,
            kernel_size,
            strides=s,
            padding="same",
            dilation_rate=dilation,
            use_bias=False,
        )(x)

        x = shortcut + x
        return x

    return apply


def Block(filters, kernel_size=3, stride=1, dilation=1, conv_shortcut=False, name=None):
    def apply(x):
        use_preactivation = torch.nn.BatchNorm(epsilon=1.001e-5, name=name + "_use_preactivation_bn"
                                               )(x)

        use_preactivation = torch.nn.ReLU()(use_preactivation)

        s = stride if dilation == 1 else 1
        if conv_shortcut:
            shortcut = torch.nn.Conv2d(
                4 * filters,
                1,
                strides=s
            )(use_preactivation)
        else:
            shortcut = (
                torch.nn.MaxPool2D(1, strides=stride)(x)
                if s > 1
                else x
            )

        x = torch.nn.Conv2d(filters, 1, strides=1, use_bias=False, name=name + "_1_conv")(
            use_preactivation
        )
        x = torch.nn.BatchNorm()(x)
        x = torch.nn.ReLU()(x)

        x = torch.nn.Conv2d(
            filters,
            kernel_size,
            strides=s,
            use_bias=False,
            padding="same",
            dilation_rate=dilation,
        )(x)
        x = torch.nn.BatchNorm()(x)
        x = torch.nn.ReLU()(x)

        x = torch.nn.Conv2d(4 * filters, 1)(x)
        x = shortcut + x
        return x

    return apply


def Stack(
        filters,
        blocks,
        stride=2,
        dilations=1,
        name=None,
        block_fn=Block,
        first_shortcut=True,
        stack_index=1,
):
    def apply(x):
        x = block_fn(filters, conv_shortcut=first_shortcut, name=name + "_block1")(x)
        for i in range(2, blocks):
            x = block_fn(filters, dilation=dilations, name=name + "_block" + str(i))(x)
        x = block_fn(
            filters,
            stride=stride,
            dilation=dilations,
            name=name + "_block" + str(blocks),
        )(x)
        return x

    return apply


class ResNetV2(torch.nn.Module):
    def __init__(self,
                 stackwise_filters,
                 stackwise_blocks,
                 stackwise_strides,
                 include_top=True,
                 stackwise_dilations=None,
                 name="ResNetV2",
                 weights=None,
                 input_shape=(None, None, 3),
                 input_tensor=None,
                 pooling=None,
                 classes=None,
                 classifier_activation="softmax",
                 block_fn=Block,
                 **kwargs, ):
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
        self.stackwise_dilations = stackwise_dilations
        self.stackwise_filters = stackwise_filters
        self.stackwise_blocks = stackwise_blocks
        self.stackwise_strides = stackwise_strides
        self.include_top = include_top
        self.classes = classes

        self.conv1 = torch.nn.Conv2d(input_shape[1], 64, 7,
            strides=2,
            use_bias=True,
            padding="same",
        )
        self.maxpool1 = torch.nn.MaxPool2D(3, strides=2, padding="same")


    def forward(self, input_shape, input_tensor):
        inputs = utils.parse_model_inputs('pytorch', input_shape, input_tensor)
        x = inputs

        x = self.conv1(x)
        x = self.maxpool1(x)

        num_stacks = len(self.stackwise_filters)
        if self.stackwise_dilations is None:
            stackwise_dilations = [1] * num_stacks

        stack_level_outputs = {}
        for stack_index in range(num_stacks):
            x = Stack(
                filters=self.stackwise_filters[stack_index],
                blocks=self.tackwise_blocks[stack_index],
                stride=self.stackwise_strides[stack_index],
                dilations=stackwise_dilations[stack_index],
                block_fn=self.block_fn,
                first_shortcut=self.block_fn == Block or stack_index > 0,
                stack_index=stack_index,
            )(x)
            stack_level_outputs[stack_index + 2] = x

        x = torch.nn.BatchNorm()(x)
        x = torch.nn.ReLU()(x)
        if self.include_top:
            # [B, C, F, F] -> [B, avg C]
            x = torch.nn.AvgPool2D(x.shape[2])(x).flatten(1)
            x = torch.nn.Linear(x.shape[1], self.classes)(x)
            x = torch.nn.Softmax()(x)
        else:
            if self.pooling == "avg":
                x = torch.nn.AvgPool2D(x.shape[2])(x).flatten(1)
            elif self.pooling == "max":
                x = torch.nn.MaxPool2D(x.shape[2])(x).flatten(1)
        return x


def ResNet18V2(**kwargs):
    model = ResNetV2()
    return model
