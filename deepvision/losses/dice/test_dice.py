import numpy as np
import tensorflow as tf
import torch

from deepvision.losses.dice.binary_dice import BinaryDiceLoss


def test_binary_dice_loss_numeric():
    y_true = np.array(([[[[1], [1]], [[1], [0]]], [[[0], [1]], [[1], [1]]]]))

    y_pred = np.array(
        (
            [
                [[[0.09264243], [0.8862786]], [[0.45386922], [0.76872504]]],
                [[[0.21945655], [0.8341843]], [[0.26319265], [0.74839675]]],
            ]
        )
    )

    y_true_tf = tf.convert_to_tensor(y_true, dtype=tf.int32)
    y_pred_tf = tf.convert_to_tensor(y_pred)

    y_true_pt = torch.from_numpy(y_true)
    y_pred_pt = torch.from_numpy(y_pred)

    tf_result = BinaryDiceLoss(backend="tensorflow")(y_true_tf, y_pred_tf).numpy()
    pt_result = BinaryDiceLoss(backend="pytorch")(y_true_pt, y_pred_pt).numpy()

    assert np.isclose(tf_result, pt_result)

    tf_result = BinaryDiceLoss(backend="tensorflow", loss_type="generalized")(
        y_true_tf, y_pred_tf
    ).numpy()
    pt_result = BinaryDiceLoss(backend="pytorch", loss_type="generalized")(
        y_true_pt, y_pred_pt
    ).numpy()

    assert np.isclose(tf_result, pt_result)

    tf_result = BinaryDiceLoss(backend="tensorflow", loss_type="adaptive")(
        y_true_tf, y_pred_tf
    ).numpy()
    pt_result = BinaryDiceLoss(backend="pytorch", loss_type="adaptive")(
        y_true_pt, y_pred_pt
    ).numpy()

    assert np.isclose(tf_result, pt_result)
