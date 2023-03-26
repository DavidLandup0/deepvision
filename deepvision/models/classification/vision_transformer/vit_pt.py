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

from deepvision.layers import PatchingAndEmbedding, TransformerEncoder
from deepvision.utils.utils import parse_model_inputs


class ViTPT(pl.LightningModule):
    def __init__(
        self,
        include_top,
        input_shape=(None, None, 3),
        input_tensor=None,
        pooling=None,
        as_backbone=None,
        classes=None,
        patch_size=None,
        transformer_layer_num=None,
        project_dim=None,
        num_heads=None,
        mlp_dim=None,
        mlp_dropout=None,
        attention_dropout=None,
        activation=None,
        weights=None,
        **kwargs,
    ):
        super().__init__()

        self.include_top = include_top
        self.pooling = pooling
        self.classes = classes
        self.transformer_layer_num = transformer_layer_num
        self.project_dim = project_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.mlp_dropout = mlp_dropout
        self.attention_dropout = attention_dropout
        self.activation = activation
        self.patch_size = patch_size
        self.as_backbone = as_backbone

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

        if self.include_top and self.as_backbone:
            raise ValueError(
                f"`as_backbone` must be `False` when `include_top=True`."
                f"Received as_backbone={self.as_backbone} and include_top={self.include_top}. "
            )

        if self.as_backbone and self.classes:
            raise ValueError(
                f"`as_backbone` must be `False` when `classes` are set."
                f"Received as_backbone={self.as_backbone} and classes={self.classes}. "
            )

        # Enforce (3, 224, 224) for  default weights and then
        # interpolate for higher resolutions
        self.patching_and_embedding = PatchingAndEmbedding(
            project_dim=project_dim,
            patch_size=patch_size,
            input_shape=(3, 224, 224),
            backend="pytorch",
        )

        self.transformer_layers = nn.ModuleList()
        for _ in range(self.transformer_layer_num):
            self.transformer_layers.append(
                TransformerEncoder(
                    project_dim=self.project_dim,
                    num_heads=self.num_heads,
                    mlp_dim=self.mlp_dim,
                    mlp_dropout=self.mlp_dropout,
                    attention_dropout=self.attention_dropout,
                    activation=self.activation,
                    backend="pytorch",
                )
            )

        self.layer_norm = nn.LayerNorm(project_dim, eps=1e-6)
        if self.include_top:
            self.linear = nn.Linear(project_dim, classes)
        if self.include_top and self.classes:
            self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=classes)

    def forward(self, input_tensor):
        inputs = parse_model_inputs("pytorch", input_tensor.shape, input_tensor)
        x = inputs
        outputs = []

        if x.shape != (3, 224, 224):
            encoded_patches = self.patching_and_embedding(
                x,
                interpolate=True,
                patch_size=self.patch_size,
                interpolate_height=x.shape[2],
                interpolate_width=x.shape[3],
            )
        else:
            encoded_patches = self.patching_and_embedding(x)
        x = nn.Dropout(self.mlp_dropout)(encoded_patches)

        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)
            outputs.append(x)

        layer_norm = self.layer_norm(x)

        if self.include_top:
            output = layer_norm[:, 0]
            output = self.linear(output)
            output = nn.Softmax(dim=1)(output)
        elif self.as_backbone:
            return outputs
        else:
            if self.pooling == "avg":
                output = layer_norm.mean(dim=1)
            elif self.pooling == "max":
                output = layer_norm.max(dim=1)
            elif self.pooling == "token":
                output = layer_norm[:, 0]

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
