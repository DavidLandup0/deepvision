import tensorflow as tf
from tensorflow.keras import layers

from deepvision.layers import Identity
from deepvision.utils.utils import parse_model_inputs

class ViTTF(tf.keras.Model):
    def __init__(
        self,
        include_top,
        input_shape=(None, None, 3),
        input_tensor=None,
        pooling=None,
        classes=None,
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

        # ...

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
