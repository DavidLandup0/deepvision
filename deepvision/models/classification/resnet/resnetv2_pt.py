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


import pytorch_lightning as pl
import torchmetrics
from torch import nn

from deepvision.utils.utils import parse_model_inputs
from deepvision.utils.utils import same_padding


class ResNetV2Block(nn.Module):
    def __init__(
        self,
        in_filters,
        out_filters,
        kernel_size=3,
        stride=1,
        dilation=1,
        conv_shortcut=False,
        block_type="basic",
    ):
        super().__init__()
        s = stride if dilation == 1 else 1

        self.block_type = block_type
        self.preact_bn = nn.BatchNorm2d(in_filters)

        if conv_shortcut:
            self.shortcut = nn.Conv2d(
                in_filters,
                4 * out_filters if block_type == "bottleneck" else out_filters,
                kernel_size=1,
                stride=s,
            )
        else:
            self.shortcut = (
                nn.MaxPool2d(kernel_size=1, stride=stride) if s > 1 else nn.Identity()
            )

        self.conv2 = nn.Conv2d(
            in_filters,
            out_filters,
            kernel_size=1 if block_type == "bottleneck" else kernel_size,
            stride=1,
            bias=False,
            padding=same_padding(
                kernel_size=1 if block_type == "bottleneck" else kernel_size, stride=1
            ),
        )

        self.bn2 = nn.BatchNorm2d(out_filters)
        self.conv3 = nn.Conv2d(
            out_filters,
            out_filters,
            kernel_size,
            stride=s,
            bias=False,
            padding=same_padding(kernel_size=kernel_size, stride=s),
            dilation=dilation,
        )
        if self.block_type == "bottleneck":
            self.bn3 = nn.BatchNorm2d(out_filters)
            self.conv4 = nn.Conv2d(out_filters, 4 * out_filters, 1)

    def forward(self, inputs):
        x = self.preact_bn(inputs)
        x = nn.ReLU()(x)
        shortcut = self.shortcut(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)

        if self.block_type == "bottleneck":
            x = self.bn3(x)
            x = nn.ReLU()(x)
            x = self.conv4(x)

        x = x + shortcut
        return x


class Stack(nn.Module):
    def __init__(
        self,
        in_filters,
        out_filters,
        blocks,
        stride=2,
        dilations=1,
        block_type=None,
        first_shortcut=True,
    ):
        super().__init__()

        self.block_1 = ResNetV2Block(
            in_filters,
            out_filters,
            conv_shortcut=first_shortcut,
            block_type=block_type,
        )
        self.middle_blocks = nn.ModuleList()
        for i in range(2, blocks):
            self.middle_blocks.append(
                ResNetV2Block(
                    4 * out_filters,
                    out_filters,
                    dilation=dilations,
                    block_type=block_type,
                )
            )
        self.final_block = ResNetV2Block(
            4 * out_filters if block_type == "bottleneck" else out_filters,
            out_filters,
            stride=stride,
            dilation=dilations,
            block_type=block_type,
        )

    def forward(self, inputs):
        x = self.block_1(inputs)
        for block in self.middle_blocks:
            x = block(x)
        x = self.final_block(x)
        return x


class ResNetV2PT(pl.LightningModule):
    def __init__(
        self,
        stackwise_filters,
        stackwise_blocks,
        stackwise_strides,
        include_top,
        stackwise_dilations=None,
        input_shape=(3, None, None),
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

        if self.include_top and self.classes:
            self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=classes)

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

        if not self.include_top and self.pooling is None:
            raise ValueError(f"`pooling` must be specified when `include_top=False`.")

        self.conv1 = nn.Conv2d(
            input_shape[0],
            64,
            kernel_size=7,
            stride=2,
            bias=True,
            padding=same_padding(kernel_size=7, stride=2),
        )
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=same_padding(kernel_size=3, stride=2)
        )

        self.stacks = nn.ModuleList()

        num_stacks = len(self.stackwise_filters)
        if self.stackwise_dilations is None:
            stackwise_dilations = [1] * num_stacks

        prev_filters = self.stackwise_filters[0]

        for stack_index in range(num_stacks):
            self.stacks.append(
                Stack(
                    in_filters=prev_filters
                    if block_type == "basic" or stack_index == 0
                    else prev_filters * 4,
                    out_filters=self.stackwise_filters[stack_index],
                    blocks=self.stackwise_blocks[stack_index],
                    stride=self.stackwise_strides[stack_index],
                    dilations=stackwise_dilations[stack_index],
                    block_type=block_type,
                    first_shortcut=block_type == "bottleneck" or stack_index > 0,
                )
            )
            prev_filters = self.stackwise_filters[stack_index]
        final_dim = (
            self.stackwise_filters[-1] * 4
            if block_type == "bottleneck"
            else self.stackwise_filters[-1]
        )
        self.batchnorm = nn.BatchNorm2d(final_dim)
        self.pool = (
            nn.AvgPool2d(7)
            if self.pooling == "avg" or self.pooling is None
            else nn.MaxPool2d(7)
        )
        if self.include_top:
            self.top_dense = nn.Linear(final_dim, classes)

    def forward(self, input_tensor):
        inputs = parse_model_inputs("pytorch", input_tensor.shape, input_tensor)
        x = inputs

        x = self.conv1(x)
        x = self.maxpool1(x)

        for stack in self.stacks:
            x = stack(x)

        x = self.batchnorm(x)
        x = nn.ReLU()(x)

        if self.include_top:
            # [B, C, F, F] -> [B, avg C]
            x = self.pool(x).flatten(1)
            x = self.top_dense(x)
            x = nn.Softmax(dim=1)(x)
        else:
            if self.pooling == "avg":
                x = nn.AvgPool2d(x.shape[2])(x).flatten(1)
            elif self.pooling == "max":
                x = nn.MaxPool2d(x.shape[2])(x).flatten(1)

        return x

    def compile(self, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer

    def configure_optimizers(self):
        optimizer = self.optimizer
        return optimizer

    def compute_loss(self, outputs, targets):
        return self.loss(outputs, targets)

    def training_step(self, train_batch, batch_idx):
        inputs, targets = train_batch
        outputs = self.forward(inputs)
        loss = self.compute_loss(outputs, targets)
        self.log(
            "loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        if self.include_top:
            acc = self.acc(outputs, targets)
            self.log(
                "acc",
                acc,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
        return loss

    def validation_step(self, val_batch, batch_idx):
        inputs, targets = val_batch
        outputs = self.forward(inputs)
        loss = self.compute_loss(outputs, targets)
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        if self.include_top:
            val_acc = self.acc(outputs, targets)
            self.log(
                "val_acc",
                val_acc,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
        return loss
