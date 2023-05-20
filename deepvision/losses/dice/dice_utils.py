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


def _calculate_dice_numerator_denominator_tf(y_true, y_pred, beta, axis, epsilon):
    true_positive = tf.reduce_sum(y_true * y_pred, axis=axis)
    false_positive = tf.reduce_sum(y_pred, axis=axis) - true_positive
    false_negative = tf.reduce_sum(y_true, axis=axis) - true_positive

    power_beta = 1 + beta**2
    numerator = power_beta * true_positive + epsilon
    denominator = (
        (power_beta * true_positive)
        + (beta**2 * false_negative)
        + false_positive
        + epsilon
    )
    return (numerator, denominator)


def _calculate_dice_numerator_denominator_pt(y_true, y_pred, beta, axis, epsilon):
    true_positive = torch.sum(y_true * y_pred, axis=axis)
    false_positive = torch.sum(y_pred, axis=axis) - true_positive
    false_negative = torch.sum(y_true, axis=axis) - true_positive

    power_beta = 1 + beta**2
    numerator = power_beta * true_positive + epsilon
    denominator = (
        (power_beta * true_positive)
        + (beta**2 * false_negative)
        + false_positive
        + epsilon
    )
    return (numerator, denominator)


def _generalized_dice_score_tf(y_true, numerator, denominator, axis):
    weight = tf.math.reciprocal(tf.square(tf.reduce_sum(y_true, axis=axis)))

    weighted_numerator = tf.reduce_sum(weight * numerator)
    weighted_denominator = tf.reduce_sum(weight * denominator)
    general_dice_score = weighted_numerator / weighted_denominator

    return general_dice_score


def _generalized_dice_score_pt(y_true, numerator, denominator, axis):
    weight = 1 / torch.square(torch.sum(y_true, dim=axis))

    weighted_numerator = torch.sum(weight * numerator)
    weighted_denominator = torch.sum(weight * denominator)
    general_dice_score = weighted_numerator / weighted_denominator

    return general_dice_score


def _adaptive_dice_score_tf(numerator, denominator):
    # Calculate the dice scores
    dice_score = numerator / denominator
    weights = tf.exp(-1.0 * dice_score)
    weighted_dice = tf.reduce_sum(weights * dice_score)
    normalizer = tf.cast(tf.size(input=dice_score), dtype=tf.float32) * tf.exp(-1.0)
    norm_dice_score = weighted_dice / normalizer

    return norm_dice_score


def _adaptive_dice_score_pt(numerator, denominator):
    dice_score = numerator / denominator
    weights = torch.exp(-1.0 * dice_score)
    weighted_dice = torch.sum(weights * dice_score)
    normalizer = dice_score.shape[0] * torch.exp(torch.tensor(-1.0))
    norm_dice_score = weighted_dice / normalizer
    return norm_dice_score


def _smooth_labels_tf(self, y_true, y_pred, label_smoothing):
    num_classes = tf.cast(tf.shape(y_true)[-1], y_pred.dtype)
    return y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)


def _smooth_labels_pt(self, y_true, y_pred, label_smoothing):
    num_classes = y_true.size(-1).to(y_pred.dtype)
    return y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)
