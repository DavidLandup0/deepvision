import torch

from deepvision.utils.utils import parse_model_inputs
from deepvision.utils.utils import same_padding


class ResNetV2Block(torch.nn.Module):
    def __init__(
        self,
        in_filters,
        out_filters,
        kernel_size=3,
        stride=1,
        dilation=1,
        conv_shortcut=False,
        type="basic",
    ):
        super().__init__()
        s = stride if dilation == 1 else 1

        self.type = type
        self.preact_bn = torch.nn.BatchNorm2d(in_filters)

        if conv_shortcut:
            self.shortcut = torch.nn.Conv2d(
                in_filters,
                4 * out_filters if type == "bottleneck" else out_filters,
                kernel_size=1,
                stride=s,
            )
        else:
            self.shortcut = (
                torch.nn.MaxPool2d(kernel_size=1, stride=stride)
                if s > 1
                else torch.nn.Identity()
            )

        self.conv2 = torch.nn.Conv2d(
            in_filters,
            out_filters,
            kernel_size=1 if type == "bottleneck" else kernel_size,
            stride=1,
            bias=False,
            padding=same_padding(
                kernel_size=1 if type == "bottleneck" else kernel_size, stride=1
            ),
        )

        self.bn2 = torch.nn.BatchNorm2d(out_filters)
        self.conv3 = torch.nn.Conv2d(
            out_filters,
            out_filters,
            kernel_size,
            stride=s,
            bias=False,
            padding=same_padding(kernel_size=kernel_size, stride=s),
            dilation=dilation,
        )
        if self.type == "bottleneck":
            self.bn3 = torch.nn.BatchNorm2d(out_filters)
            self.conv4 = torch.nn.Conv2d(out_filters, 4 * out_filters, 1)

    def forward(self, input):
        x = self.preact_bn(input)
        x = torch.nn.ReLU()(x)
        shortcut = self.shortcut(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.nn.ReLU()(x)
        x = self.conv3(x)

        if self.type == "bottleneck":
            x = self.bn3(x)
            x = torch.nn.ReLU()(x)
            x = self.conv4(x)

        x = x + shortcut
        return x


class Stack(torch.nn.Module):
    def __init__(
        self,
        in_filters,
        out_filters,
        blocks,
        stride=2,
        dilations=1,
        block_type=None,
        first_shortcut=True,
        stack_index=1,
    ):
        super().__init__()

        self.block_1 = ResNetV2Block(
            in_filters,
            out_filters,
            conv_shortcut=first_shortcut,
            type=block_type,
        )
        self.middle_blocks = torch.nn.ModuleList()
        for i in range(2, blocks):
            self.middle_blocks.append(
                ResNetV2Block(
                    in_filters,
                    out_filters,
                    dilation=dilations,
                    type=block_type,
                )
            )
        self.final_block = ResNetV2Block(
            out_filters,
            out_filters,
            stride=stride,
            dilation=dilations,
            type=block_type,
        )

    def forward(self, inputs):
        x = self.block_1(inputs)
        for block in self.middle_blocks:
            x = block(x)
        x = self.final_block(x)
        return x


class ResNetV2PT(torch.nn.Module):
    def __init__(
        self,
        stackwise_filters,
        stackwise_blocks,
        stackwise_strides,
        include_top,
        stackwise_dilations=None,
        input_shape=(3, None, None),
        input_tensor=None,
        pooling=None,
        classes=None,
        block_type=None,
        **kwargs,
    ):
        super().__init__()

        self.include_top = include_top
        self.pooling = pooling
        self.classes = classes
        self.stackwise_dilations = stackwise_dilations
        self.stackwise_filters = stackwise_filters
        self.stackwise_blocks = stackwise_blocks
        self.stackwise_strides = stackwise_strides

        if self.include_top and not self.classes:
            raise ValueError(
                "If `include_top` is True, you should specify `classes`. "
                f"Received: classes={self.classes}"
            )

        if self.include_top and self.pooling:
            raise ValueError(
                f"`pooling` must be `None` when `include_top=True`."
                f"Received pooling={self.pooling} and include_top={self.include_top}. "
            )

        self.conv1 = torch.nn.Conv2d(
            input_shape[0],
            64,
            kernel_size=7,
            stride=2,
            bias=True,
            padding=same_padding(kernel_size=7, stride=2),
        )
        self.maxpool1 = torch.nn.MaxPool2d(
            kernel_size=3, stride=2, padding=same_padding(kernel_size=3, stride=2)
        )

        self.stacks = torch.nn.ModuleList()

        num_stacks = len(self.stackwise_filters)
        if self.stackwise_dilations is None:
            stackwise_dilations = [1] * num_stacks

        prev_filters = self.stackwise_filters[0]

        for stack_index in range(num_stacks):
            self.stacks.append(
                Stack(
                    in_filters=prev_filters,
                    out_filters=self.stackwise_filters[stack_index],
                    blocks=self.stackwise_blocks[stack_index],
                    stride=self.stackwise_strides[stack_index],
                    dilations=stackwise_dilations[stack_index],
                    block_type=block_type,
                    first_shortcut=block_type == "bottleneck" or stack_index > 0,
                    stack_index=stack_index,
                )
            )
            prev_filters = self.stackwise_filters[stack_index]

        self.batchnorm = torch.nn.BatchNorm2d(self.stackwise_filters[-1])
        self.pool = (
            torch.nn.AvgPool2d(7)
            if self.pooling == "avg" or self.pooling is None
            else torch.nn.MaxPool2d(7)
        )
        self.top_dense = torch.nn.Linear(self.stackwise_filters[-1], classes)

    def forward(self, input_tensor):
        inputs = parse_model_inputs("pytorch", input_tensor.shape, input_tensor)
        x = inputs

        x = self.conv1(x)
        x = self.maxpool1(x)

        for stack in self.stacks:
            x = stack(x)

        x = self.batchnorm(x)
        x = torch.nn.ReLU()(x)

        # [B, C, F, F] -> [B, avg C]
        x = self.pool(x).flatten(1)
        if self.include_top:
            x = self.top_dense(x)
            x = torch.nn.Softmax(dim=1)(x)

        return x
