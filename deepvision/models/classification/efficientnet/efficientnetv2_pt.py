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

import math

import pytorch_lightning as pl
import torchmetrics
from torch import nn

from deepvision.layers import FusedMBConv
from deepvision.layers import MBConv
from deepvision.utils.utils import parse_model_inputs
from deepvision.utils.utils import same_padding


def _make_divisible(filter_num, width_coefficient, depth_divisor, min_depth):
    """
    Adapted from the official MobileNetV2 implementation to accommodate for the width coefficient:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    filter_num = filter_num * width_coefficient
    if min_depth is None:
        min_depth = depth_divisor
    new_v = max(
        min_depth, int(filter_num + depth_divisor / 2) // depth_divisor * depth_divisor
    )
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * filter_num:
        new_v += depth_divisor
    return int(new_v)


class EfficientNetV2PT(pl.LightningModule):
    def __init__(
        self,
        include_top,
        width_coefficient,
        depth_coefficient,
        input_shape=(3, None, None),
        input_tensor=None,
        pooling=None,
        classes=None,
        dropout_rate=0.2,
        drop_connect_rate=0.2,
        depth_divisor=8,
        min_depth=8,
        bn_momentum=0.9,
        activation=nn.SiLU,
        blockwise_kernel_sizes=None,
        blockwise_num_repeat=None,
        blockwise_input_filters=None,
        blockwise_output_filters=None,
        blockwise_expand_ratios=None,
        blockwise_se_ratios=None,
        blockwise_strides=None,
        blockwise_conv_type=None,
        **kwargs,
    ):
        super().__init__()

        self.include_top = include_top
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.model_input_shape = input_shape
        self.pooling = pooling
        self.classes = classes
        self.dropout_rate = dropout_rate
        self.drop_connect_rate = drop_connect_rate
        self.depth_divisor = depth_divisor
        self.min_depth = min_depth
        self.bn_momentum = bn_momentum
        self.activation = activation
        self.blockwise_kernel_sizes = blockwise_kernel_sizes
        self.blockwise_num_repeat = blockwise_num_repeat
        self.blockwise_input_filters = blockwise_input_filters
        self.blockwise_output_filters = blockwise_output_filters
        self.blockwise_expand_ratios = blockwise_expand_ratios
        self.blockwise_se_ratios = blockwise_se_ratios
        self.blockwise_strides = blockwise_strides
        self.blockwise_conv_type = blockwise_conv_type

        if include_top and classes:
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

        stem_out_channels = _make_divisible(
            filter_num=self.blockwise_input_filters[0],
            width_coefficient=self.width_coefficient,
            min_depth=self.min_depth,
            depth_divisor=self.depth_divisor,
        )
        self.stem_conv = nn.Conv2d(
            in_channels=input_shape[0],
            out_channels=stem_out_channels,
            kernel_size=3,
            stride=2,
            padding=same_padding(3, 2),
            bias=False,
        )
        self.stem_bn = nn.BatchNorm2d(stem_out_channels, momentum=bn_momentum)
        self.blocks = nn.ModuleList()
        block_num = sum(self.blockwise_num_repeat)
        for block_index in range(len(blockwise_num_repeat)):
            # Scale the input/output filters by the
            # width coefficient, and make them divisible again
            # as there's no guarantee that they'll be integers after scaling
            input_channels = _make_divisible(
                blockwise_input_filters[block_index],
                width_coefficient,
                min_depth,
                depth_divisor,
            )

            output_channels = _make_divisible(
                blockwise_output_filters[block_index],
                width_coefficient,
                min_depth,
                depth_divisor,
            )
            # Num repeats * depth_coefficient -> then round them up to an integer
            repeats = int(
                math.ceil(depth_coefficient * blockwise_num_repeat[block_index])
            )
            # For each repeat in the list of repeats, add a block (MBConv or FusedMBConv)
            for repeat in range(repeats):
                if blockwise_conv_type[block_index] == "mbconv":
                    conv_type = MBConv
                else:
                    conv_type = FusedMBConv

                conv_block = conv_type(
                    input_filters=output_channels if repeat > 0 else input_channels,
                    output_filters=output_channels,
                    expand_ratio=blockwise_expand_ratios[block_index],
                    kernel_size=blockwise_kernel_sizes[block_index],
                    strides=1 if repeat > 0 else blockwise_strides[block_index],
                    se_ratio=blockwise_se_ratios[block_index],
                    activation=activation,
                    bn_momentum=bn_momentum,
                    dropout=drop_connect_rate * block_index / block_num,
                    backend="pytorch",
                    name=f"block{block_index+1}_{repeat+1}",
                )
                self.blocks.append(conv_block)

        top_channels = _make_divisible(
            filter_num=1280,
            width_coefficient=width_coefficient,
            min_depth=min_depth,
            depth_divisor=depth_divisor,
        )

        self.top_conv = nn.Conv2d(
            in_channels=blockwise_output_filters[-1],
            out_channels=top_channels,
            kernel_size=1,
            stride=1,
            padding="same",
            bias=False,
        )
        self.top_bn = nn.BatchNorm2d(
            top_channels,
            momentum=bn_momentum,
        )
        if self.include_top:
            self.top_dense = nn.Linear(top_channels, classes)

    def forward(self, input_tensor):
        inputs = parse_model_inputs("pytorch", input_tensor.shape, input_tensor)
        x = inputs

        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.activation()(x)

        for block in self.blocks:
            x = block(x)
        x = self.top_conv(x)
        x = self.top_bn(x)
        x = self.activation()(x)

        if self.include_top:
            # [B, C, F, F] -> [B, avg C]
            x = nn.AvgPool2d(x.shape[2])(x).flatten(1)
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
            loss.item(),
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
            loss.item(),
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

    def get_config(self):
        config = {
            "include_top": self.include_top,
            "width_coefficient": self.width_coefficient,
            "depth_coefficient": self.depth_coefficient,
            "pooling": self.pooling,
            "classes": self.classes,
            "dropout_rate": self.dropout_rate,
            "drop_connect_rate": self.drop_connect_rate,
            "depth_divisor": self.depth_divisor,
            "min_depth": self.min_depth,
            "bn_momentum": self.bn_momentum,
            "activation": self.activation,
            "blockwise_kernel_sizes": self.blockwise_kernel_sizes,
            "blockwise_num_repeat": self.blockwise_num_repeat,
            "blockwise_input_filters": self.blockwise_input_filters,
            "blockwise_output_filters": self.blockwise_output_filters,
            "blockwise_expand_ratios": self.blockwise_expand_ratios,
            "blockwise_se_ratios": self.blockwise_se_ratios,
            "blockwise_strides": self.blockwise_strides,
            "blockwise_conv_type": self.blockwise_conv_type,
        }

        return config
