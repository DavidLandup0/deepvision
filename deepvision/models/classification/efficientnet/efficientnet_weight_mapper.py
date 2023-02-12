import math

import tensorflow as tf
import torch

from deepvision.layers import fused_mbconv
from deepvision.layers import mbconv
from deepvision.layers.fused_mbconv import __FusedMBConvPT
from deepvision.layers.fused_mbconv import __FusedMBConvTF
from deepvision.layers.mbconv import __MBConvPT
from deepvision.layers.mbconv import __MBConvTF
from deepvision.models.classification.efficientnet.efficientnetv2 import (
    EfficientNetV2B0,
)
from deepvision.models.classification.efficientnet.efficientnetv2 import (
    EfficientNetV2B1,
)
from deepvision.models.classification.efficientnet.efficientnetv2 import (
    EfficientNetV2B2,
)
from deepvision.models.classification.efficientnet.efficientnetv2 import (
    EfficientNetV2B3,
)
from deepvision.models.classification.efficientnet.efficientnetv2 import EfficientNetV2L
from deepvision.models.classification.efficientnet.efficientnetv2 import EfficientNetV2M
from deepvision.models.classification.efficientnet.efficientnetv2 import EfficientNetV2S
from deepvision.models.classification.efficientnet.efficientnetv2_pt import (
    EfficientNetV2PT,
)
from deepvision.models.classification.efficientnet.efficientnetv2_tf import (
    EfficientNetV2TF,
)

MODEL_ARCHITECTURES = {
    "EfficientNetV2B0": EfficientNetV2B0,
    "EfficientNetV2B1": EfficientNetV2B1,
    "EfficientNetV2B2": EfficientNetV2B2,
    "EfficientNetV2B3": EfficientNetV2B3,
    "EfficientNetV2S": EfficientNetV2S,
    "EfficientNetV2M": EfficientNetV2M,
    "EfficientNetV2L": EfficientNetV2L,
}

MODEL_BACKBONES = {"tensorflow": EfficientNetV2TF, "pytorch": EfficientNetV2PT}


def load(
    filepath,
    origin,
    target,
    dummy_input,
    kwargs=None,
    architecture=None,
    freeze_bn=True,
):
    """
    Basic usage:

    ```
    ### TensorFlow to PyTorch

    dummy_input_tf = tf.ones([1, 224, 224, 3])
    dummy_input_torch = torch.ones(1, 3, 224, 224)

    tf_model = deepvision.models.EfficientNetV2B0(include_top=False,
                                              pooling='avg',
                                              input_shape=(224, 224, 3),
                                              backend='tensorflow')

    tf_model.save('effnet.h5')

    from deepvision.models.classification.efficientnet import efficientnet_weight_mapper
    pt_model = efficientnet_weight_mapper.load(filepath='effnet.h5',
                                    origin='tensorflow',
                                    target='pytorch',
                                    dummy_input=dummy_input_tf)

    print(tf_model(dummy_input_tf)['output'].numpy())
    print(pt_model(dummy_input_torch).detach().cpu().numpy())
    # True
    np.allclose(tf_model(dummy_input_tf)['output'].numpy(), pt_model(dummy_input_torch).detach().cpu().numpy())

    ### PyTorch to TensorFlow
    dummy_input_tf = tf.ones([1, 224, 224, 3])
    dummy_input_torch = torch.ones(1, 3, 224, 224)

    pt_model = deepvision.models.EfficientNetV2B0(include_top=False,
                                              pooling='avg',
                                              input_shape=(3, 224, 224),
                                              backend='pytorch')
    torch.save(pt_model.state_dict(), 'effnet.pt')

    from deepvision.models.classification.efficientnet import efficientnet_weight_mapper

    kwargs = {'include_top': False, 'pooling':'avg', 'input_shape':(3, 224, 224)}
    tf_model = efficientnet_weight_mapper.load(filepath='effnet.pt',
                                    origin='pytorch',
                                    target='tensorflow',
                                    architecture='EfficientNetV2B0',
                                    kwargs=kwargs,
                                    dummy_input=dummy_input_torch)


    pt_model.eval()
    print(pt_model(dummy_input_torch).detach().cpu().numpy())
    print(tf_model(dummy_input_tf)['output'].numpy())
    # True
    np.allclose(tf_model(dummy_input_tf)['output'].numpy(), pt_model(dummy_input_torch).detach().cpu().numpy())
    ```
    """
    if origin == "tensorflow":
        # Temporarily need to supply this as custom_objects() due to a bug while
        # saving Functional Subclassing models
        model = tf.keras.models.load_model(
            filepath, custom_objects={"EfficientNetV2TF": EfficientNetV2TF}
        )
        model_config = model.get_config()
        target_model = EfficientNetV2PT(
            include_top=model_config["include_top"],
            classes=model_config["classes"],
            input_shape=tf.transpose(tf.squeeze(dummy_input), (2, 0, 1)).shape,
            pooling=model_config["pooling"],
            width_coefficient=model_config["width_coefficient"],
            depth_coefficient=model_config["depth_coefficient"],
            blockwise_kernel_sizes=model_config["blockwise_kernel_sizes"],
            blockwise_num_repeat=model_config["blockwise_num_repeat"],
            blockwise_input_filters=model_config["blockwise_input_filters"],
            blockwise_output_filters=model_config["blockwise_output_filters"],
            blockwise_expand_ratios=model_config["blockwise_expand_ratios"],
            blockwise_se_ratios=model_config["blockwise_se_ratios"],
            blockwise_strides=model_config["blockwise_strides"],
            blockwise_conv_type=model_config["blockwise_conv_type"],
        )
        # Copy stem
        target_model.stem_conv.weight.data = torch.nn.Parameter(
            torch.from_numpy(tf.transpose(model.layers[1].kernel, (3, 2, 0, 1)).numpy())
        )
        # Copy BatchNorm
        target_model.stem_bn.weight.data = torch.nn.Parameter(
            torch.from_numpy(model.layers[2].gamma.numpy())
        )
        target_model.stem_bn.bias.data = torch.nn.Parameter(
            torch.from_numpy(model.layers[2].beta.numpy())
        )
        target_model.stem_bn.running_mean.data = torch.nn.Parameter(
            torch.from_numpy(model.layers[2].moving_mean.numpy())
        )
        target_model.stem_bn.running_var.data = torch.nn.Parameter(
            torch.from_numpy(model.layers[2].moving_variance.numpy())
        )

        tf_blocks = [
            block
            for block in model.layers
            if isinstance(block, __FusedMBConvTF) or isinstance(block, __MBConvTF)
        ]

        for pt_block, tf_block in zip(target_model.blocks, tf_blocks):
            if isinstance(tf_block, __FusedMBConvTF):
                converted_block = fused_mbconv.tf_to_pt(tf_block)
                pt_block.load_state_dict(converted_block.state_dict())
            if isinstance(tf_block, __MBConvTF):
                converted_block = mbconv.tf_to_pt(tf_block)
                pt_block.load_state_dict(converted_block.state_dict())

        target_model.top_conv.weight.data = torch.nn.Parameter(
            torch.from_numpy(
                tf.transpose(
                    model.layers[-5 if model_config["include_top"] else -4].kernel,
                    (3, 2, 0, 1),
                ).numpy()
            )
        )
        if model_config["include_top"]:
            # Copy top BatchNorm
            target_model.top_bn.weight.data = torch.nn.Parameter(
                torch.from_numpy(model.layers[-4].gamma.numpy())
            )
            target_model.top_bn.bias.data = torch.nn.Parameter(
                torch.from_numpy(model.layers[-4].beta.numpy())
            )
            target_model.top_bn.running_mean.data = torch.nn.Parameter(
                torch.from_numpy(model.layers[-4].moving_mean.numpy())
            )
            target_model.top_bn.running_var.data = torch.nn.Parameter(
                torch.from_numpy(model.layers[-4].moving_variance.numpy())
            )

            # Copy head
            target_model.top_dense.weight.data = torch.nn.Parameter(
                torch.from_numpy(model.layers[-1].kernel.numpy().transpose(1, 0))
            )
            target_model.top_dense.bias.data = torch.nn.Parameter(
                torch.from_numpy(model.layers[-1].bias.numpy())
            )
        if freeze_bn:
            # Freeze all BatchNorm2d layers
            for module in target_model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.eval()
                    module.weight.requires_grad = False
                    module.bias.requires_grad = False

        return target_model

    elif origin == "pytorch":

        if kwargs is None:
            raise ValueError(
                f"'kwargs' cannot be None, and are required for PyTorch model construction."
            )
        if kwargs is None:
            raise ValueError(
                f"'architecture' cannot be None, and is required for PyTorch model construction."
            )

        model = MODEL_ARCHITECTURES.get(architecture)
        model = model(backend="pytorch", **kwargs)
        model.load_state_dict(torch.load(filepath))

        model_config = model.get_config()
        target_model = EfficientNetV2TF(
            include_top=model_config["include_top"],
            classes=model_config["classes"],
            input_shape=dummy_input.squeeze(0).permute(1, 2, 0).shape,
            pooling=model_config["pooling"],
            width_coefficient=model_config["width_coefficient"],
            depth_coefficient=model_config["depth_coefficient"],
            blockwise_kernel_sizes=model_config["blockwise_kernel_sizes"],
            blockwise_num_repeat=model_config["blockwise_num_repeat"],
            blockwise_input_filters=model_config["blockwise_input_filters"],
            blockwise_output_filters=model_config["blockwise_output_filters"],
            blockwise_expand_ratios=model_config["blockwise_expand_ratios"],
            blockwise_se_ratios=model_config["blockwise_se_ratios"],
            blockwise_strides=model_config["blockwise_strides"],
            blockwise_conv_type=model_config["blockwise_conv_type"],
        )

        # Copy stem
        target_model.layers[1].kernel.assign(
            tf.convert_to_tensor(
                model.stem_conv.weight.data.permute(2, 3, 1, 0).detach().cpu().numpy()
            )
        )

        # Copy BatchNorm
        target_model.layers[2].gamma.assign(
            tf.convert_to_tensor(model.stem_bn.weight.data.detach().cpu().numpy())
        )

        target_model.layers[2].beta.assign(
            tf.convert_to_tensor(model.stem_bn.bias.data.detach().cpu().numpy())
        )

        target_model.layers[2].moving_mean.assign(
            tf.convert_to_tensor(model.stem_bn.running_mean.data.detach().cpu().numpy())
        )

        target_model.layers[2].moving_variance.assign(
            tf.convert_to_tensor(model.stem_bn.running_var.data.detach().cpu().numpy())
        )

        tf_blocks = [
            block
            for block in target_model.layers
            if isinstance(block, __FusedMBConvTF) or isinstance(block, __MBConvTF)
        ]

        for tf_block, pt_block in zip(tf_blocks, model.blocks):
            if isinstance(pt_block, __FusedMBConvPT):
                converted_block = fused_mbconv.pt_to_tf(pt_block)
                tf_block.set_weights(converted_block.weights)
            if isinstance(pt_block, __MBConvPT):
                converted_block = mbconv.pt_to_tf(pt_block)
                tf_block.set_weights(converted_block.weights)

        target_model.layers[-5 if model_config["include_top"] else -4].kernel.assign(
            tf.convert_to_tensor(
                model.top_conv.weight.data.permute(2, 3, 1, 0).detach().cpu().numpy()
            )
        )

        if model_config["include_top"]:
            # Copy top BatchNorm
            target_model.layers[-4].gamma.assign(
                tf.convert_to_tensor(model.top_bn.weight.data.detach().cpu().numpy())
            )

            target_model.layers[-4].beta.assign(
                tf.convert_to_tensor(model.top_bn.bias.data.detach().cpu().numpy())
            )

            target_model.layers[-4].moving_mean.assign(
                tf.convert_to_tensor(
                    model.top_bn.running_mean.data.detach().cpu().numpy()
                )
            )

            target_model.layers[-4].moving_variance.assign(
                tf.convert_to_tensor(
                    model.top_bn.running_var.data.detach().cpu().numpy()
                )
            )

            # Copy head
            target_model.layers[-1].kernel.assign(
                tf.convert_to_tensor(
                    model.top_dense.weight.data.permute(1, 0).detach().cpu().numpy()
                )
            )

            target_model.layers[-1].bias.assign(
                tf.convert_to_tensor(model.top_dense.bias.data.detach().cpu().numpy())
            )

        if freeze_bn:
            # Freeze all BatchNorm2d layers
            for layer in target_model.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False

        return target_model

    else:
        raise ValueError(
            f"Backend not supported: {origin}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )
