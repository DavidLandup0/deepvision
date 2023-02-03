import tensorflow as tf
from tensorflow.keras import layers

from deepvision.layers import PatchingAndEmbedding
from deepvision.layers import TransformerEncoder
from deepvision.utils.utils import parse_model_inputs


class ViTTF(tf.keras.Model):
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

        inputs = parse_model_inputs("tensorflow", input_shape, input_tensor)
        x = inputs

        encoded_patches = PatchingAndEmbedding(project_dim, patch_size)(x)
        encoded_patches = layers.Dropout(mlp_dropout)(encoded_patches)

        for _ in range(transformer_layer_num):
            encoded_patches = TransformerEncoder(
                project_dim=project_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                mlp_dropout=mlp_dropout,
                attention_dropout=attention_dropout,
                activation=activation,
            )(encoded_patches)

        output = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        if include_top:
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            output = layers.Dense(classes, activation="softmax", name="predictions")(x)
        else:
            if pooling == "avg":
                output = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            elif pooling == "max":
                output = layers.GlobalMaxPooling2D(name="max_pool")(x)

        super().__init__(
            inputs={
                "inputs": inputs,
            },
            outputs={
                "output": output,
            },
            **kwargs,
        )
        self.include_top = include_top
        self.pooling = pooling
        self.classes = classes
