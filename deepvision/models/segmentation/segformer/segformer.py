from deepvision.models.segmentation.backbones.mit import mit
from deepvision.models.segmentation.segformer.segformer_pt import __SegFormerPT

MODEL_BACKBONES = {"tensorflow": None, "pytorch": __SegFormerPT}


def SegFormerB0(input_shape, num_classes, backend, softmax_output=True):
    model_class = MODEL_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )

    backbone = mit.MiTB0(input_shape=input_shape, backend="pytorch")
    return model_class(
        embed_dims=256,
        num_classes=num_classes,
        softmax_output=softmax_output,
        backbone=backbone,
    )


def SegFormerB1(input_shape, num_classes, backend, softmax_output=True):
    model_class = MODEL_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )

    backbone = mit.MiTB1(input_shape=input_shape, backend="pytorch")
    return model_class(
        embed_dims=256,
        num_classes=num_classes,
        softmax_output=softmax_output,
        backbone=backbone,
    )


def SegFormerB2(input_shape, num_classes, backend, softmax_output=True):
    model_class = MODEL_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )

    backbone = mit.MiTB2(input_shape=input_shape, backend="pytorch")
    return model_class(
        embed_dims=768,
        num_classes=num_classes,
        softmax_output=softmax_output,
        backbone=backbone,
    )


def SegFormerB3(input_shape, num_classes, backend, softmax_output=True):
    model_class = MODEL_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )

    backbone = mit.MiTB3(input_shape=input_shape, backend="pytorch")
    return model_class(
        embed_dims=768,
        num_classes=num_classes,
        softmax_output=softmax_output,
        backbone=backbone,
    )


def SegFormerB4(input_shape, num_classes, backend, softmax_output=True):
    model_class = MODEL_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )

    backbone = mit.MiTB4(input_shape=input_shape, backend="pytorch")
    return model_class(
        embed_dims=768,
        num_classes=num_classes,
        softmax_output=softmax_output,
        backbone=backbone,
    )


def SegFormerB5(input_shape, num_classes, backend, softmax_output=True):
    model_class = MODEL_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )

    backbone = mit.MiTB5(input_shape=input_shape, backend="pytorch")
    return model_class(
        embed_dims=768,
        num_classes=num_classes,
        softmax_output=softmax_output,
        backbone=backbone,
    )
