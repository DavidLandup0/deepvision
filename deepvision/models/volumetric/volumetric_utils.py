import tensorflow as tf
import torch


def nerf_render_image_and_depth_tf(
    model, rays_flat, t_vals, img_height, img_width, num_ray_samples
):
    """Generates the RGB image and depth map from model prediction.

    The code was adapted from the official implementation at [NeRF: Neural Radiance Fields](https://github.com/bmild/nerf)
    And the useful comments for readability were written by Aritra Roy Gosthipaty and Ritwik Raha at [3D volumetric rendering with NeRF](https://keras.io/examples/vision/nerf/)

    Args:
        model: The MLP model that is trained to predict the rgb and
            volume density of the volumetric scene.
        rays_flat: The flattened rays that serve as the input to
            the NeRF model.
        t_vals: The sample points for the rays.

    Returns:
        Tuple of rgb image and depth map.
    """

    predictions = model(rays_flat, 0)["output"]
    predictions = tf.reshape(
        predictions,
        shape=(tf.shape(predictions)[0], img_height, img_width, num_ray_samples, 4),
    )

    # Slice the predictions into rgb and sigma.
    rgb = tf.sigmoid(predictions[..., :-1])
    sigma_a = tf.nn.relu(predictions[..., -1])

    # Get the distance of adjacent intervals.
    delta = t_vals[..., 1:] - t_vals[..., :-1]
    # delta shape = (num_samples)
    delta = tf.concat(
        [
            delta,
            tf.broadcast_to(
                [1e10], shape=(tf.shape(predictions)[0], img_height, img_width, 1)
            ),
        ],
        axis=-1,
    )
    alpha = 1.0 - tf.exp(-sigma_a * delta[:, None, None, :])

    # Get transmittance.
    exp_term = 1.0 - alpha
    epsilon = 1e-10
    transmittance = tf.math.cumprod(exp_term + epsilon, axis=-1, exclusive=True)
    weights = alpha * transmittance
    rgb = tf.reduce_sum(weights[..., None] * rgb, axis=-2)
    depth_map = tf.reduce_sum(weights * t_vals[:, None, None], axis=-1)
    return rgb, depth_map


def nerf_render_image_and_depth_pt(
    model, rays_flat, t_vals, img_height, img_width, num_ray_samples
):
    """Generates the RGB image and depth map from model prediction.

    The code was adapted from the official implementation at [NeRF: Neural Radiance Fields](https://github.com/bmild/nerf)
    And the useful comments for readability were written by Aritra Roy Gosthipaty and Ritwik Raha at [3D volumetric rendering with NeRF](https://keras.io/examples/vision/nerf/)

    Args:
        model: The MLP model that is trained to predict the rgb and
            volume density of the volumetric scene.
        rays_flat: The flattened rays that serve as the input to
            the NeRF model.
        t_vals: The sample points for the rays.

    Returns:
        Tuple of rgb image and depth map.
    """
    predictions = model(rays_flat)
    predictions = predictions.reshape(
        predictions.shape[0], img_height, img_width, num_ray_samples, 4
    )

    # Slice the predictions into rgb and sigma.
    rgb = torch.nn.Sigmoid()(predictions[..., :-1])
    sigma_a = torch.nn.ReLU()(predictions[..., -1])

    # Get the distance of adjacent intervals.
    delta = t_vals[..., 1:] - t_vals[..., :-1]
    delta = torch.cat(
        [
            delta,
            torch.broadcast_to(
                input=torch.tensor([1e10], device=delta.device),
                size=(predictions.shape[0], img_height, img_width, 1),
            ),
        ],
        dim=-1,
    )
    alpha = 1.0 - torch.exp(-sigma_a * delta[:, None, None, :])

    # Get transmittance.
    exp_term = 1.0 - alpha
    epsilon = 1e-10
    transmittance = torch.cumprod(exp_term + epsilon, dim=-1)
    weights = alpha * transmittance
    rgb = torch.sum(weights[..., None] * rgb, dim=-2)
    depth_map = torch.sum(weights * t_vals[:, None, None], dim=-1)
    return rgb, depth_map
