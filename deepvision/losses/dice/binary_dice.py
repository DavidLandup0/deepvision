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

import tensorflow as tf
import torch

from deepvision.losses.dice import dice_utils


@tf.keras.utils.register_keras_serializable(package="deepvision")
class __BinaryDiceLossTF(tf.keras.losses.Loss):
    def __init__(
        self,
        beta=1,
        from_logits=False,
        class_ids=None,
        axis=[1, 2],
        loss_type=None,
        label_smoothing=0.0,
        epsilon=1e-07,
        per_image=False,
        name="binary_dice",
        **kwargs,
    ):
        """
        Basic TensorFlow usage:

        ```
        y_true = tf.random.uniform([5, 10, 10, 1], 0, maxval=2, dtype=tf.int32)
        y_pred = tf.random.uniform([5, 10, 10, 1], 0, maxval=1)

        # Regular dice loss
        dice = deepvision.losses.BinaryDiceLoss(backend='tensorflow')

        # Generalized Dice Loss
        dice = deepvision.losses.BinaryDiceLoss(backend='tensorflow', loss_type='generalized')

        # Adaptive Dice Loss
        dice = deepvision.losses.BinaryDiceLoss(backend='tensorflow', loss_type='adaptive')

        # Compute Dice Loss between y_true and y_pred
        dice(y_true, y_pred).numpy() # 0.16937321

        # Supplying sample weights
        dice(y_true, y_pred, sample_weight=tf.constant([[0.5, 0.5]])).numpy() # 0.08619368
        ```

        Usage with the TensorFlow `compile()` API:

        ```
        model.compile(optimizer='adam', loss=deepvision.losses.BinaryDice(backend='tensorflow'))
        ```
        """
        super().__init__(name=name, **kwargs)

        self.beta = beta
        self.from_logits = from_logits
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        self.epsilon = epsilon
        self.class_ids = class_ids
        self.per_image = per_image
        self.axis = axis

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        label_smoothing = tf.convert_to_tensor(self.label_smoothing, dtype=y_pred.dtype)

        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)

        if tf.cast(label_smoothing, dtype=tf.bool):
            y_true = self._smooth_labels(y_true, y_pred, label_smoothing)

        if self.class_ids is not None:
            y_true = tf.gather(y_true, self.class_ids, axis=-1)
            y_pred = tf.gather(y_pred, self.class_ids, axis=-1)

        if self.axis not in [[1, 2], [1, 2, 3], [2, 3], [2, 3, 4]]:
            raise ValueError(
                f"`axis` value should be [1, 2] or [1, 2, 3] for 2D and 3D channels_last input respectively, and [2, 3] or [2, 3, 4] for 2D and 3D channels_first input respectively. Got {self.axis}"
            )

        numerator, denominator = dice_utils._calculate_dice_numerator_denominator_tf(
            y_true, y_pred, self.beta, self.axis, self.epsilon
        )

        if self.loss_type == "generalized":
            dice_score = dice_utils._generalized_dice_score_tf(
                y_true, numerator, denominator, self.axis
            )
        elif self.loss_type == "adaptive":
            dice_score = dice_utils._adaptive_dice_score_tf(numerator, denominator)
        else:
            dice_score = numerator / denominator

        if self.per_image and self.axis in [[1, 2], [1, 2, 3]]:
            dice_score = tf.reduce_mean(dice_score, axis=0)
        elif self.per_image and self.axis in [[2, 3], [2, 3, 4]]:
            dice_score = tf.reduce_mean(dice_score, axis=1)
        else:
            dice_score = tf.reduce_mean(dice_score)

        return 1 - dice_score

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta": self.beta,
                "from_logits": self.from_logits,
                "class_ids": self.class_ids,
                "loss_type": self.loss_type,
                "label_smoothing": self.label_smoothing,
                "epsilon": self.epsilon,
                "per_image": self.per_image,
                "axis": self.axis,
            }
        )
        return config


class __BinaryDiceLossPT(torch.nn.Module):
    def __init__(
        self,
        beta=1,
        from_logits=False,
        class_ids=None,
        axis=[1, 2],
        loss_type=None,
        label_smoothing=0.0,
        epsilon=1e-07,
        per_image=False,
        name=None,  # No usage, kept for API compatibility
    ):
        """
        Basic PyTorch usage:

        ```
        y_true = torch.randint(0, 2, (5, 10, 10, 1), dtype=torch.int32)
        y_pred = torch.rand((5, 10, 10, 1))

        # Regular dice loss
        dice = deepvision.losses.BinaryDiceLoss(backend='pytorch')

        # Generalized Dice Loss
        dice = deepvision.losses.BinaryDiceLoss(backend='pytorch', loss_type='generalized')

        # Adaptive Dice Loss
        dice = deepvision.losses.BinaryDiceLoss(backend='pytorch', loss_type='adaptive')

        # Compute Dice Loss between y_true and y_pred
        print(dice(y_true, y_pred).numpy()) # 0.16937321
        ```
        """
        super().__init__()

        self.beta = beta
        self.from_logits = from_logits
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        self.epsilon = epsilon
        self.class_ids = class_ids
        self.per_image = per_image
        self.axis = axis

    def forward(self, y_true, y_pred):
        y_true = y_true.float()
        y_pred = y_pred.float()

        label_smoothing = torch.tensor(self.label_smoothing, dtype=y_pred.dtype)

        if self.from_logits:
            y_pred = torch.sigmoid(y_pred)

        if label_smoothing != 0:
            y_true = self._smooth_labels(y_true, y_pred, label_smoothing)

        if self.class_ids is not None:
            y_true = y_true[:, :, :, self.class_ids]
            y_pred = y_pred[:, :, :, self.class_ids]

        if self.axis not in [[1, 2], [1, 2, 3], [2, 3], [2, 3, 4]]:
            raise ValueError(
                f"`axis` value should be [1, 2] or [1, 2, 3] for 2D and 3D channels_last input respectively, and [2, 3] or [2, 3, 4] for 2D and 3D channels_first input respectively. Got {self.axis}"
            )

        numerator, denominator = dice_utils._calculate_dice_numerator_denominator_pt(
            y_true, y_pred, self.beta, self.axis, self.epsilon
        )

        if self.loss_type == "generalized":
            dice_score = dice_utils._generalized_dice_score_pt(
                y_true, numerator, denominator, self.axis
            )
        elif self.loss_type == "adaptive":
            dice_score = dice_utils._adaptive_dice_score_pt(numerator, denominator)
        else:
            dice_score = numerator / denominator

        if self.per_image and self.axis in [[1, 2], [1, 2, 3]]:
            dice_score = dice_score.mean(dim=0)
        elif self.per_image and self.axis in [[2, 3], [2, 3, 4]]:
            dice_score = dice_score.mean(dim=1)
        else:
            dice_score = dice_score.mean()

        return 1 - dice_score


LAYER_BACKBONES = {
    "tensorflow": __BinaryDiceLossTF,
    "pytorch": __BinaryDiceLossPT,
}


def BinaryDiceLoss(
    backend,
    beta=1,
    from_logits=False,
    class_ids=None,
    axis=[1, 2],
    loss_type=None,
    label_smoothing=0.0,
    epsilon=1e-07,
    per_image=False,
    name="binary_dice",
):
    """Compute the dice loss between the binary labels and predictions.

    The loss function is applicable to 2D and 3D semantic segmentation, and labels
    should be binary.

    The binary dice loss is a commonly used loss function for evaluating the similarity between
    binary segmentation masks. It measures the overlap between the predicted mask and the ground truth mask.

    Acknowledgements:
        - TensorFlow version was originally written by (innat)[https://github.com/innat] and expanded/ported to PyTorch by David Landup.

    Three loss types are supported:
        - Regular Binary Dice loss
        - Generalized Binary Dice loss
        - Adaptive Binary Dice loss

    Regular Binary Dice Loss:
    -------------------------
    The regular binary dice loss computes the dice score as the ratio of twice the intersection
    between the predicted mask and the ground truth mask to the sum of the number of pixels in
    both masks. It ranges from 0 to 1, where 0 indicates no overlap and 1 indicates perfect overlap.

    Generalized Binary Dice Loss:
    -----------------------------
    The generalized binary dice loss extends the regular binary dice loss to handle imbalanced datasets
    by assigning weights to the classes based on the inverse square of the class frequencies. It aims to
    provide a more robust loss function that accounts for the class imbalance and encourages better
    performance on minority classes.

    Adaptive Binary Dice Loss:
    --------------------------
    The adaptive binary dice loss further addresses the class imbalance issue by adaptively adjusting
    the weights based on the individual dice scores of each class. It assigns higher weights to classes
    with lower dice scores, which helps to focus the training process on the classes that are harder to
    segment accurately. This adaptive weighting scheme can improve the performance on challenging classes
    and achieve better overall segmentation accuracy.

    Args:
        beta: default 1, A coefficient for balancing precision and recall in the dice score.
              Values greater than 1 favor recall, while values less than 1 favor precision.
        from_logits: default False, A boolean indicating whether the y_pred vector is a probability distribution or logits.
                     If True, logits are converted to probabilities via a Sigmoid function.
        class_ids: default None, Which classes to use for computing the dice loss. If None, all  classes are used.
        axis: default [1, 2], the axes for which to calculate the dice loss for. For 2D segmentation, [1, 2] and [2, 3] are used
              for computing the channels_last and channels_first formats. For 3D segmentation, [1, 2, 3] and [2, 3, 4] are used for
              computing the channels_last and channels_first formats.
        loss_type: default None, The type of dice score to compute. Supported types are None for regular binary dice loss,
                    'generalized' for generalized binary dice loss and 'adaptive' for adaptive binary dice loss.
        label_smoothing: default 0.0, The extent to which label smoothing is applied on the ground truth labels.
        epsilon: default 1e-07, The value to add when dividing by small values to avoid division by 0.
        per_image: default False, whether to calculate the loss as mean of all scores for each image in the batch or not.


    Basic TensorFlow usage:

    ```
    y_true = tf.random.uniform([5, 10, 10, 1], 0, maxval=2, dtype=tf.int32)
    y_pred = tf.random.uniform([5, 10, 10, 1], 0, maxval=1)

    # Regular dice loss
    dice = deepvision.losses.BinaryDiceLoss(backend='tensorflow')

    # Generalized Dice Loss
    dice = deepvision.losses.BinaryDiceLoss(backend='tensorflow', loss_type='generalized')

    # Adaptive Dice Loss
    dice = deepvision.losses.BinaryDiceLoss(backend='tensorflow', loss_type='adaptive')

    # Compute Dice Loss between y_true and y_pred
    dice(y_true, y_pred).numpy() # 0.16937321

    # Supplying sample weights
    dice(y_true, y_pred, sample_weight=tf.constant([[0.5, 0.5]])).numpy() # 0.08619368
    ```

    Usage with the TensorFlow `compile()` API:

    ```
    model.compile(optimizer='adam', loss=deepvision.losses.BinaryDice(backend='tensorflow'))
    ```


    Basic PyTorch usage:

    ```
    y_true = torch.randint(0, 2, (5, 10, 10, 1), dtype=torch.int32)
    y_pred = torch.rand((5, 10, 10, 1))

    # Regular dice loss
    dice = deepvision.losses.BinaryDiceLoss(backend='pytorch')

    # Generalized Dice Loss
    dice = deepvision.losses.BinaryDiceLoss(backend='pytorch', loss_type='generalized')

    # Adaptive Dice Loss
    dice = deepvision.losses.BinaryDiceLoss(backend='pytorch', loss_type='adaptive')

    # Compute Dice Loss between y_true and y_pred
    print(dice(y_true, y_pred).numpy()) # 0.16937321
    ```
    """

    layer_class = LAYER_BACKBONES.get(backend)
    if layer_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {LAYER_BACKBONES.keys()}"
        )

    layer = layer_class(
        beta=beta,
        from_logits=from_logits,
        class_ids=class_ids,
        axis=axis,
        loss_type=loss_type,
        label_smoothing=label_smoothing,
        epsilon=epsilon,
        per_image=per_image,
        name="binary_dice",
    )

    return layer
