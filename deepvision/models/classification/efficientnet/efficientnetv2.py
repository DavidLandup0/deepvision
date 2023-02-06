from deepvision.models.classification.efficientnet.efficientnetv2_pt import (
    EfficientNetV2PT,
)
from deepvision.models.classification.efficientnet.efficientnetv2_tf import (
    EfficientNetV2TF,
)

MODEL_CONFIGS = {
    "EfficientNetV2B0": {},
}

MODEL_BACKBONES = {"tensorflow": EfficientNetV2TF, "pytorch": EfficientNetV2PT}


def EfficientNetV2B0(
    backend,
    include_top,
    classes=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    **kwargs,
):
    model_class = MODEL_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )
    model = model_class(
        #
    )
    return model
