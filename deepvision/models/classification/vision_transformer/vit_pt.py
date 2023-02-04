import pytorch_lightning as pl
import torchmetrics
from torch import nn

from deepvision.layers import PatchingAndEmbedding
from deepvision.layers import TransformerEncoder
from deepvision.utils.utils import parse_model_inputs


class ViTPT(pl.LightningModule):
    def __init__(
        self,
        include_top,
        input_shape=(None, None, 3),
        input_tensor=None,
        pooling=None,
        classes=None,
        patch_size=None,
        transformer_layer_num=None,
        project_dim=None,
        num_heads=None,
        mlp_dim=None,
        mlp_dropout=None,
        attention_dropout=None,
        activation=None,
        **kwargs,
    ):
        super().__init__()

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

        self.patching_and_embedding = PatchingAndEmbedding(
            project_dim=project_dim,
            patch_size=patch_size,
            input_shape=input_shape,
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
        self.linear = nn.Linear(project_dim, classes)

        self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=classes)

    def forward(self, input_tensor):
        inputs = parse_model_inputs("pytorch", input_tensor.shape, input_tensor)
        x = inputs

        encoded_patches = self.patching_and_embedding(x)
        x = nn.Dropout(self.mlp_dropout)(encoded_patches)

        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)

        layer_norm = self.layer_norm(x)

        if self.include_top:
            output = layer_norm.mean(dim=1)
            output = self.linear(output)
            output = nn.Softmax(dim=1)(output)
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
        self.log("loss", loss, on_epoch=True, prog_bar=True)
        self.acc(outputs, targets)
        self.log("acc", self.accuracy, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        inputs, targets = val_batch
        outputs = self.forward(inputs)
        loss = self.compute_loss(outputs, targets)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.val_acc.update(outputs, targets)
        return loss

    def training_epoch_end(self, outs):
        self.acc.reset()

    def validation_epoch_end(self, outputs):
        self.log("val_acc_epoch", self.val_acc.compute())
        self.val_acc.reset()
