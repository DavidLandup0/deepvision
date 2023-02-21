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

from deepvision.utils.utils import parse_model_inputs


class NeRFPT(pl.LightningModule):
    def __init__(
        self,
        input_shape=(None, None, 3),
        depth=None,
        width=None,
        **kwargs,
    ):
        """
        Neural Radiance Field (NeRF) model, implemented in PyTorch.

        Args:
            input_shape: the shape of the input tensor
            depth: the depth of the model (i.e. the number of layers to stack)
            width: the 'channels' of each stacked layer (i.e. the number of dense units in each layer)
        """
        super().__init__()

        self.layers = torch.nn.ModuleList()

        for i in range(depth):
            if i % 4 == 0 and i > 0:
                self.layers.append(torch.nn.Linear(width + input_shape[-1], width))
            else:
                self.layers.append(
                    torch.nn.Linear(width if i > 0 else input_shape[-1], width)
                )

        self.output = torch.nn.Linear(width, 4)

    def forward(self, input_tensor):
        x = input_tensor

        for index, layer in enumerate(self.layers):
            x = layer(x)
            x = torch.nn.ReLU()(x)
            if index % 4 == 0 and index > 0:
                x = torch.concat(x, input_tensor)
        output = self.output(x)

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
