from deepvision.models.classification.vision_transformer.vit_tf import ViTTF
from deepvision.models.classification.vision_transformer.vit_pt import ViTPT

MODEL_CONFIGS = {
    "ViTTiny16": {
        "patch_size": 16,
        "transformer_layer_num": 12,
        "project_dim": 192,
        "mlp_dim": 768,
        "num_heads": 3,
        "mlp_dropout": 0.0,
        "attention_dropout": 0.0,
    },
}

MODEL_BACKBONES = {"tensorflow": ViTTF, "pytorch": ViTPT}


def ViTTiny16(
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
        include_top,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        patch_size=MODEL_CONFIGS["ViTTiny16"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViTTiny16"]["transformer_layer_num"],
        project_dim=MODEL_CONFIGS["ViTTiny16"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViTTiny16"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViTTiny16"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViTTiny16"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViTTiny16"]["attention_dropout"],
        **kwargs,
    )

    return model