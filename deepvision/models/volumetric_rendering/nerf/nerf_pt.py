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


class NeRFPT(pl.LightningModule):
    def __init__(
        self,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        super().__init__()

    def forward(self, input_tensor):
        inputs = parse_model_inputs("pytorch", input_tensor.shape, input_tensor)
        x = inputs

        output = x

        return output

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
        return loss
