# Copyright 2023 David Landup
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

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


def load_tf_to_pt(
    filepath,
    dummy_input,
    kwargs=None,
    freeze_bn=True,
):
    """
    Basic usage:

    ```
    dummy_input_tf = tf.ones([1, 224, 224, 3])
    dummy_input_torch = torch.ones(1, 3, 224, 224)

    tf_model = deepvision.models.EfficientNetV2B0(include_top=False,
                                              pooling='avg',
                                              input_shape=(224, 224, 3),
                                              backend='tensorflow')

    tf_model.save('effnet.h5')

    from deepvision.models.classification.efficientnet import efficientnet_weight_mapper
    pt_model = efficientnet_weight_mapper.load_tf_to_pt(filepath='effnet.h5', dummy_input=dummy_input_tf)

    print(tf_model(dummy_input_tf)['output'].numpy())
    print(pt_model(dummy_input_torch).detach().cpu().numpy())
    # True
    np.allclose(tf_model(dummy_input_tf)['output'].numpy(), pt_model(dummy_input_torch).detach().cpu().numpy())
    """
    with torch.no_grad():
        # Temporarily need to supply this as custom_objects() due to a bug while
        # saving Functional Subclassing models
        model = tf.keras.models.load_model(
            filepath, custom_objects={"EfficientNetV2TF": EfficientNetV2TF}
        )
        # Run dummy_input through the model to initialize
        # model.variables
        model(dummy_input)

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

        """
        As noted in: https://discuss.pytorch.org/t/out-of-memory-error-when-resume-training-even-though-my-gpu-is-empty/30757/5
        Sometimes, on some devices, PyTorch-based networks throw a CUDA OOM when loaded directly on the GPU. To avoid this,
        we now *save* the model and load it back, mapping to the CPU and then pushing back to the original model device.
        """
        device = target_model.device
        original_filepath = os.path.splitext(filepath)[0]
        target_model.to("cpu")
        torch.save(target_model.state_dict(), f"converted_{original_filepath}.pt")

        target_model.load_state_dict(
            torch.load(f"converted_{original_filepath}.pt", map_location="cpu"),
        )
        target_model.to(device)
        target_model.zero_grad()

        if freeze_bn:
            # Freeze all BatchNorm2d layers
            for module in target_model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.eval()
                    module.weight.requires_grad = False
                    module.bias.requires_grad = False

    return target_model


def load_pt_to_tf(
    filepath,
    dummy_input,
    kwargs=None,
    architecture=None,
    freeze_bn=True,
):

    """
    Basic usage:

    ```
    dummy_input_tf = tf.ones([1, 224, 224, 3])
    dummy_input_torch = torch.ones(1, 3, 224, 224)

    pt_model = deepvision.models.EfficientNetV2B0(include_top=False,
                                              pooling='avg',
                                              input_shape=(3, 224, 224),
                                              backend='pytorch')
    torch.save(pt_model.state_dict(), 'effnet.pt')

    from deepvision.models.classification.efficientnet import efficientnet_weight_mapper

    kwargs = {'include_top': False, 'pooling':'avg', 'input_shape':(3, 224, 224)}
    tf_model = efficientnet_weight_mapper.load_pt_to_tf(filepath='effnet.pt',
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

    if kwargs is None:
        raise ValueError(
            f"'kwargs' cannot be None, and are required for PyTorch model construction."
        )
    if architecture is None:
        raise ValueError(
            f"'architecture' cannot be None, and is required for PyTorch model construction."
        )
    with torch.no_grad():
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
        dummy_input = tf.convert_to_tensor(
            dummy_input.permute(0, 2, 3, 1).detach().cpu().numpy()
        )
        # Run dummy_input through the model to initialize
        # model.variables
        target_model(dummy_input)

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
