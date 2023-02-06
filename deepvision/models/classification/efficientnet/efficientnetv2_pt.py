import pytorch_lightning as pl
import torchmetrics
from torch import nn

from deepvision.utils.utils import parse_model_inputs
from deepvision.utils.utils import same_padding


class EfficientNetV2PT(pl.LightningModule):
    def __init__(
        self,
        include_top,
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

    def forward(self, input_tensor):
        inputs = parse_model_inputs("pytorch", input_tensor.shape, input_tensor)
        x = inputs

        # [B, C, F, F] -> [B, avg C]
        x = self.pool(x).flatten(1)
        if self.include_top:
            x = self.top_dense(x)
            x = nn.Softmax(dim=1)(x)

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
        self.log("loss", loss, on_epoch=True, prog_bar=True)
        acc = self.acc(outputs, targets)
        self.log("acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        inputs, targets = val_batch
        outputs = self.forward(inputs)
        loss = self.compute_loss(outputs, targets)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        val_acc = self.acc(outputs, targets)
        self.log("val_acc", val_acc, on_epoch=True, prog_bar=True)
        return loss
