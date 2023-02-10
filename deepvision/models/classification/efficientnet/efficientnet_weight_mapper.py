import tensorflow as tf
import torch

from deepvision.models.classification.efficientnet.efficientnetv2_pt import (
    EfficientNetV2PT,
)
from deepvision.models.classification.efficientnet.efficientnetv2_tf import (
    EfficientNetV2TF,
)

MODEL_BACKBONES = {"tensorflow": EfficientNetV2TF, "pytorch": EfficientNetV2PT}


def load(filepath, origin, target, kwargs=None):
    if origin == "tensorflow":
        model = tf.keras.models.load_model(filepath)
        for var in model.variables:
            print(var.shape)
    elif origin == "pytorch":
        if kwargs is None:
            raise ValueError(
                f"kwargs are required to be passed for PyTorch model instantiation."
            )
        model = MODEL_BACKBONES.get(origin)
        model = model(**kwargs)
        model.load_state_dict(torch.load(filepath))
        for var in model.parameters():
            print(var.shape)
    else:
        raise ValueError(
            f"Backend not supported: {origin}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )


