import tensorflow as tf


def load_tiny_nerf(images, poses, focal, pos_embed=16, num_ray_samples=32):
    """
    Loads and returns a `tf.data.Dataset`, containing the "tiny_nerf" dataset, as per
        [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934)

    The code and original docstrings were adapted from the official implementation at
        [NeRF: Neural Radiance Fields](https://github.com/bmild/nerf)
        And the useful comments for readability were written by Aritra Roy Gosthipaty and Ritwik Raha at
        [3D volumetric rendering with NeRF](https://keras.io/examples/vision/nerf/)

    Args:
        images: np.ndarray, images in the tiny_nerf dataset
        poses: np.ndarray, camera poses in the tiny_nerf dataset
        focal: np.ndarray, camera focal lengths in the tiny_nerf dataset

    Returns:
        tf.data.Dataset of (img_batch, ray_batch)

    """
    (num_images, height, width, _) = images.shape
    img_ds = tf.data.Dataset.from_tensor_slices(images)
    pose_ds = tf.data.Dataset.from_tensor_slices(poses)
    ray_ds = pose_ds.map(
        lambda pose: map_fn(
            pose=pose,
            height=height,
            width=width,
            focal=focal,
            pos_embed=pos_embed,
            num_ray_samples=num_ray_samples,
        )
    )
    ds = tf.data.Dataset.zip((img_ds, ray_ds))
    return ds


def encode_position(x, pos_embed):
    """Encodes the position into its corresponding Fourier feature.

    Args:
        x: The input coordinate.

    Returns:
        Fourier features tensors of the position.
    """
    positions = [x]
    for i in range(pos_embed):
        for fn in [tf.sin, tf.cos]:
            positions.append(fn(2.0**i * x))
    return tf.concat(positions, axis=-1)


def get_rays(height, width, focal, pose):
    """Computes origin point and direction vector of rays.

    Args:
        height: Height of the image.
        width: Width of the image.
        focal: The focal length between the images and the camera.
        pose: The pose matrix of the camera.

    Returns:
        Tuple of origin point and direction vector for rays.
    """
    # Build a meshgrid for the rays.
    i, j = tf.meshgrid(
        tf.range(width, dtype=tf.float32),
        tf.range(height, dtype=tf.float32),
        indexing="xy",
    )

    # Normalize the x axis coordinates.
    transformed_i = (i - width * 0.5) / focal

    # Normalize the y axis coordinates.
    transformed_j = (j - height * 0.5) / focal

    # Create the direction unit vectors.
    directions = tf.stack([transformed_i, -transformed_j, -tf.ones_like(i)], axis=-1)

    # Get the camera matrix.
    camera_matrix = pose[:3, :3]
    height_width_focal = pose[:3, -1]

    # Get origins and directions for the rays.
    transformed_dirs = directions[..., None, :]
    camera_dirs = transformed_dirs * camera_matrix
    ray_directions = tf.reduce_sum(camera_dirs, axis=-1)
    ray_origins = tf.broadcast_to(height_width_focal, tf.shape(ray_directions))

    # Return the origins and directions.
    return (ray_origins, ray_directions)


def render_flat_rays(
    ray_origins, ray_directions, near, far, pos_embed, num_ray_samples, rand=False
):
    """Renders the rays and flattens it.

    Args:
        ray_origins: The origin points for rays.
        ray_directions: The direction unit vectors for the rays.
        near: The near bound of the volumetric scene.
        far: The far bound of the volumetric scene.
        num_samples: Number of sample points in a ray.
        rand: Choice for randomising the sampling strategy.

    Returns:
       Tuple of flattened rays and sample points on each rays.
    """
    # Compute 3D query points.
    # Equation: r(t) = o+td -> Building the "t" here.
    t_vals = tf.linspace(near, far, num_ray_samples)
    if rand:
        # Inject uniform noise into sample space to make the sampling continuous.
        shape = list(ray_origins.shape[:-1]) + [num_ray_samples]
        noise = tf.random.uniform(shape=shape) * (far - near) / num_ray_samples
        t_vals = t_vals + noise

    # Equation: r(t) = o + td -> Building the "r" here.
    rays = ray_origins[..., None, :] + (
        ray_directions[..., None, :] * t_vals[..., None]
    )
    rays_flat = tf.reshape(rays, [-1, 3])
    rays_flat = encode_position(rays_flat, pos_embed)
    return rays_flat, t_vals


def map_fn(pose, height, width, focal, pos_embed, num_ray_samples):
    """Maps individual pose to flattened rays and sample points.

    Args:
        pose: The pose matrix of the camera.

    Returns:
        Tuple of flattened rays and sample points corresponding to the
        camera pose.
    """
    (ray_origins, ray_directions) = get_rays(
        height=height, width=width, focal=focal, pose=pose
    )
    (rays_flat, t_vals) = render_flat_rays(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        near=2.0,
        far=6.0,
        pos_embed=pos_embed,
        num_ray_samples=num_ray_samples,
        rand=True,
    )
    return rays_flat, t_vals
