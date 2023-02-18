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


def load_tiny_nerf(save_path=None, download=False, backend=None):
    dataset_class = DATASET_BACKENDS.get(backend)
    if dataset_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {DATASET_BACKENDS.keys()}"
        )

    deepvision_dataset_location = os.path.expanduser(
        os.path.join("~", "deepvision_datasets")
    )
    if not os.path.exists(deepvision_dataset_location):
        os.mkdir(deepvision_dataset_location)

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

    dataset = dataset_class.load_tiny_nerf(images, poses, focal)

    return dataset
