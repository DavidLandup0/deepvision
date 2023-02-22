import tensorflow as tf
import torch


def nerf_render_image_and_depth_tf(
    model, rays_flat, t_vals, img_height, img_width, num_ray_samples
):
    """Generates the RGB image and depth map from model prediction.

    The code was adapted from the official implementation at [NeRF: Neural Radiance Fields](https://github.com/bmild/nerf) and
        Aritra Roy Gosthipaty and Ritwik Raha's at [3D volumetric rendering with NeRF](https://keras.io/examples/vision/nerf/)

        Args:
        model: the NeRF model to predict the rgb and volume density of the volumetric scene
        rays_flat: the flattened rays that to input into the NeRF model
        t_vals: ray sample points
        img_width: the image width
        img_height: the image height
        num_ray_samples: the number of samples in each ray

    Returns:
        The rgb_map (3-channel output image), depth_map and acc_map
    """
    # Run model predictions and reshape the output
    predictions = model(rays_flat, 0)["output"]
    predictions = tf.reshape(
        predictions,
        shape=(tf.shape(predictions)[0], img_height, img_width, num_ray_samples, 4),
    )

    # Compute opacity (sigma) and colors (rgb)
    rgb = tf.sigmoid(predictions[..., :-1])
    sigma_a = tf.nn.relu(predictions[..., -1])

    # Distances to adjacent intervals
    delta = tf.concat(
        [
            t_vals[..., 1:] - t_vals[..., :-1],
            tf.broadcast_to(
                [1e10], shape=(tf.shape(predictions)[0], img_height, img_width, 1)
            ),
        ],
        axis=-1,
    )
    alpha = 1.0 - tf.exp(-sigma_a * delta[:, None, None, :])

    exp_term = 1.0 - alpha
    transmittance = tf.math.cumprod(exp_term + 1e-10, axis=-1, exclusive=True)
    weights = alpha * transmittance

    rgb_map = tf.reduce_sum(weights[..., None] * rgb, axis=-2)
    depth_map = tf.reduce_sum(weights * t_vals[:, None, None], axis=-1)
    acc_map = tf.reduce_sum(weights, axis=-1)
    return rgb_map, depth_map, acc_map


def nerf_render_image_and_depth_pt(
    model, rays_flat, t_vals, img_height, img_width, num_ray_samples
):
    """Generates the RGB image and depth map from model prediction.

    The code was adapted from the official implementation at [NeRF: Neural Radiance Fields](https://github.com/bmild/nerf) and
        Aritra Roy Gosthipaty and Ritwik Raha's at [3D volumetric rendering with NeRF](https://keras.io/examples/vision/nerf/)

    Args:
        model: the NeRF model to predict the rgb and volume density of the volumetric scene
        rays_flat: the flattened rays that to input into the NeRF model
        t_vals: ray sample points
        img_width: the image width
        img_height: the image height
        num_ray_samples: the number of samples in each ray

    Returns:
        The rgb_map (3-channel output image), depth_map and acc_map
    """

    # Run model predictions and reshape the output
    predictions = model(rays_flat)
    predictions = predictions.reshape(
        predictions.shape[0], img_height, img_width, num_ray_samples, 4
    )

    # Compute opacity (sigma) and colors (rgb)
    rgb = torch.nn.Sigmoid()(predictions[..., :-1])
    sigma_a = torch.nn.ReLU()(predictions[..., -1])

    # Distances to adjacent intervals
    delta = torch.cat(
        [
            t_vals[..., 1:] - t_vals[..., :-1],
            torch.broadcast_to(
                input=torch.tensor([1e10], device=rgb.device),
                size=(predictions.shape[0], img_height, img_width, 1),
            ),
        ],
        dim=-1,
    )
    alpha = 1.0 - torch.exp(-sigma_a * delta[:, None, None, :])

    exp_term = 1.0 - alpha
    transmittance = torch.cumprod(exp_term + 1e-10, dim=-1)
    weights = alpha * transmittance

    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    depth_map = torch.sum(weights * t_vals[:, None, None], dim=-1)
    acc_map = torch.sum(weights, -1)
    return rgb_map, depth_map, acc_map
