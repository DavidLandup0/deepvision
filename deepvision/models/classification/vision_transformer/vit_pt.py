from torch import nn

from deepvision.layers import PatchingAndEmbedding
from deepvision.layers import TransformerEncoder
from deepvision.utils.utils import parse_model_inputs


class ViTPT(nn.Module):
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

        self.include_top = include_top
        self.pooling = pooling
        self.classes = classes

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

        self.patching_and_embedding = PatchingAndEmbedding(
            project_dim=project_dim,
            patch_size=patch_size,
            input_shape=input_shape,
            backend="pytorch",
        )
        self.transformer_layer_num = transformer_layer_num
        self.project_dim = project_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.mlp_dropout = mlp_dropout
        self.attention_dropout = attention_dropout
        self.activation = activation
        self.layer_norm = nn.LayerNorm(project_dim, eps=1e-6)
        self.linear = nn.Linear(project_dim, classes)

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

    def forward(self, input_tensor):
        inputs = parse_model_inputs("pytorch", input_tensor.shape, input_tensor)
        x = inputs

        encoded_patches = self.patching_and_embedding(x)
        x = nn.Dropout(self.mlp_dropout)(encoded_patches)

        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)

        output = self.layer_norm(encoded_patches)

        if self.include_top:
            x = nn.AvgPool1d(x.shape[1])(x)
            output = self.linear(x)
            output = nn.Softmax(dim=1)(output)
        else:
            if self.pooling == "token":
                output = x[:, 0]
            elif self.pooling == "avg":
                output = nn.AvgPool1d(x.shape[1])(x)
            elif self.pooling == "max":
                output = nn.MaxPool1d(x.shape[1])(x)

        return output
