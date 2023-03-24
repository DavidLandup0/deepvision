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
import torch
import torchmetrics
from torch.nn import functional as F

from deepvision.layers.segformer_segmentation_head import SegFormerHead


class __SegFormerPT(pl.LightningModule):
    def __init__(
        self,
        num_classes=None,
        backbone=None,
        embed_dim=None,
        softmax_output=None,
        name=None,
        input_shape=None,
        input_tensor=None,
    ):
        super().__init__()
        self.backbone = backbone
        self.decode_head = SegFormerHead(
            in_dims=backbone.output_channels,
            embed_dim=embed_dim,
            num_classes=num_classes,
            backend="pytorch",
        )
        self.softmax_output = softmax_output
        self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        y = self.backbone(x)
        y = self.decode_head(y)
        y = F.interpolate(
            y, size=x.shape[2:], mode="bilinear", align_corners=False
        )  # to original image shape
        if self.softmax_output:
            y = torch.nn.Softmax(1)(y)
        return y

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
        val_acc = self.acc(outputs, targets)
        self.log(
            "val_acc",
            val_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss
