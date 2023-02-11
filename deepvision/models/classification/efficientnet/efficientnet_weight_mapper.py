import math

import tensorflow as tf
import torch

from deepvision.layers import fused_mbconv
from deepvision.layers import mbconv
from deepvision.layers.fused_mbconv import __FusedMBConvPT
from deepvision.layers.fused_mbconv import __FusedMBConvTF
from deepvision.layers.mbconv import __MBConvPT
from deepvision.layers.mbconv import __MBConvTF
from deepvision.models.classification.efficientnet.efficientnetv2_pt import (
    EfficientNetV2PT,
)
from deepvision.models.classification.efficientnet.efficientnetv2_tf import (
    EfficientNetV2TF,
)

MODEL_BACKBONES = {"tensorflow": EfficientNetV2TF, "pytorch": EfficientNetV2PT}


def load(filepath, origin, target, freeze_bn=True):
    """
    Basic usage:

    ```
    dummy_input = np.random.rand(1, 224, 224, 3)
    dummy_input_tf = tf.convert_to_tensor(dummy_input)
    dummy_input_torch = torch.from_numpy(dummy_input).permute(0, 3, 1, 2).float()

    tf_model = deepvision.models.EfficientNetV2B0(include_top=True,
                                     classes=10,
                                     input_shape=(224, 224, 3),
                                     backend='tensorflow')

    tf_model(dummy_input_tf) # {'output': <tf.Tensor: shape=(1, 10), dtype=float32, numpy=array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]], dtype=float32)>}

    from deepvision.models.classification.efficientnet import efficientnet_weight_mapper
    pt_model = efficientnet_weight_mapper.load(filepath='effnet.h5',
                                    origin='tensorflow',
                                    target='pytorch',
                                    dummy_input=dummy_input_tf)

    pt_model(dummy_input_torch) # tensor([[0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000]], grad_fn=<SoftmaxBackward0>)
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
            input_shape=model_config["model_input_shape"],
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
        model = MODEL_BACKBONES.get(origin)
        model = model(**kwargs)
        model.load_state_dict(torch.load(filepath))
        for layer in model.children():
            if isinstance(layer, torch.nn.modules.container.ModuleList):
                for element in layer:
                    print(type(element))
            else:
                print(type(layer))
    else:
        raise ValueError(
            f"Backend not supported: {origin}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )
