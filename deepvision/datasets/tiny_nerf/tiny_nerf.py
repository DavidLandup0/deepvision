import os

import numpy as np
import requests

from deepvision.datasets.tiny_nerf import tiny_nerf_pt
from deepvision.datasets.tiny_nerf import tiny_nerf_tf

file_name = "tiny_nerf_data.npz"
url = "https://people.eecs.berkeley.edu/~bmild/nerf/tiny_nerf_data.npz"

DATASET_BACKENDS = {
    "tensorflow": tiny_nerf_tf,
    "pytorch": tiny_nerf_pt,
}


def load_tiny_nerf(
    pos_embed=16,
    num_ray_samples=32,
    save_path=None,
    download=False,
    validation_split=None,
    backend=None,
):
    dataset_class = DATASET_BACKENDS.get(backend)
    if dataset_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {DATASET_BACKENDS.keys()}"
        )
    if validation_split > 1.0:
        raise ValueError(
            f"The `validation_split` cannot be set to a value higher than `1.0` as it represents a [0..1] bound percentage value of the dataset. "
            f"Received {validation_split}"
        )

    deepvision_dataset_location = os.path.expanduser(
        os.path.join("~", "deepvision_datasets", "tiny_nerf")
    )
    if not os.path.exists(deepvision_dataset_location):
        os.makedirs(deepvision_dataset_location)

    save_path = os.path.join(deepvision_dataset_location, save_path)

    if not os.path.exists(save_path) or download:
        print(f"Downloading dataset and storing under {save_path}")
        file_data = requests.get(url).content
        with open(save_path, "wb") as file:
            file.write(file_data)
        print(f"Dataset download finished.")
    else:
        print(
            f"Dataset already exists under {save_path}. Skipping download and re-using existing data. "
            f"If you want to download again, set `download=True`."
        )

    data = np.load(save_path)
    images = data["images"]
    (poses, focal) = (data["poses"], data["focal"])
    if validation_split:
        split = int(len(images) * (1 - validation_split))
        train_images, train_poses = images[:split], poses[:split]
        valid_images, valid_poses = images[split:], poses[split:]

        train_dataset = dataset_class.load_tiny_nerf(
            train_images, train_poses, focal, pos_embed, num_ray_samples
        )
        valid_dataset = dataset_class.load_tiny_nerf(
            valid_images, valid_poses, focal, pos_embed, num_ray_samples
        )

        return train_dataset, valid_dataset
    else:
        dataset = dataset_class.load_tiny_nerf(
            images, poses, focal, pos_embed, num_ray_samples
        )
        return dataset
