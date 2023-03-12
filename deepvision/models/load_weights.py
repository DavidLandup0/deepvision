import os

import requests
from tqdm import tqdm


def load_weights(model_name, include_top, backend):
    if not include_top:
        model_name += "-notop"

    if backend == "tensorflow":
        weight_path = MODELS_TF.get(model_name, None)
    else:
        weight_path = MODELS_PT.get(model_name, None)

    if weight_path is None:
        raise ValueError(f"Weights do not exist for {model_name}")

    weight_location = os.path.expanduser(
        os.path.join("~", "deepvision_weights", model_name)
    )
    if not os.path.isdir(weight_location):
        os.makedirs(weight_location)

    save_path = os.path.join(weight_location, model_name + ".h5")

    if not os.path.exists(save_path):
        print(f"Downloading weights and storing under {save_path}")
        file_data = requests.get(weight_path, stream=True)
        total = int(file_data.headers.get("content-length", 0))
        with open(save_path, "wb") as file, tqdm(
            desc=save_path,
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in file_data.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
        print(f"Weight download finished.")

    return save_path


MODELS_TF = {
    "ViTTiny16": "https://github.com/DavidLandup0/deepvision/releases/download/0.1.5-weights/ViTTiny16.h5",
    "ViTTiny16-notop": "https://github.com/DavidLandup0/deepvision/releases/download/0.1.5-weights/ViTTiny16-notop.h5",
    "ViTS16": "https://github.com/DavidLandup0/deepvision/releases/download/0.1.5-weights/ViTS16.h5",
    "ViTS16-notop": "https://github.com/DavidLandup0/deepvision/releases/download/0.1.5-weights/ViTS16-notop.h5",
    "ViTB16": "https://github.com/DavidLandup0/deepvision/releases/download/0.1.5-weights/ViTB16.h5",
    "ViTB16-notop": "https://github.com/DavidLandup0/deepvision/releases/download/0.1.5-weights/ViTB16-notop.h5",
    "ViTL16": "https://github.com/DavidLandup0/deepvision/releases/download/0.1.5-weights/ViTL16.h5",
    "ViTL16-notop": "https://github.com/DavidLandup0/deepvision/releases/download/0.1.5-weights/ViTL16-notop.h5",
    "ViTS32": "https://github.com/DavidLandup0/deepvision/releases/download/0.1.5-weights/ViTS32.h5",
    "ViTS32-notop": "https://github.com/DavidLandup0/deepvision/releases/download/0.1.5-weights/ViTS32-notop.h5",
    "ViTB32": "https://github.com/DavidLandup0/deepvision/releases/download/0.1.5-weights/ViTB32.h5",
    "ViTB32-notop": "https://github.com/DavidLandup0/deepvision/releases/download/0.1.5-weights/ViTB32-notop.h5",
}

MODELS_PT = {}
