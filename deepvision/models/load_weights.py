import os
import requests

def load_weights(model_name, include_top):
    if include_top:
        model_name += '-notop'

    weight_path = MODELS[model_name]
    if weight_path is None:
        raise ValueError(f'Weights do not exist for {model_name}')

    weight_location = os.path.expanduser(
        os.path.join("~", "deepvision_weights", model_name)
    )
    if not os.path.exists(weight_location):
        os.mkdir(weight_location)

    save_path = os.path.join(weight_location, model_name)

    if not os.path.exists(save_path):
        print(f"Downloading weights and storing under {save_path}")
        file_data = requests.get(weight_path).content
        with open(save_path, "wb") as file:
            file.write(file_data)
        print(f"Weight download finished.")

    return save_path



MODELS = {
    "ViTTiny16" : "1",
    "ViTTiny16-notop" : "2",
    "ViTS16" : "3",
    "ViTS16-notop" : "4",
    "ViTB16" : "5",
    "ViTB16-notop": "6",
    "ViTL16" : "7",
    "ViTL16-notop" : "8",
    "ViTS32" : "9",
    "ViTS32-notop" : "10",
    "ViTB32" : "11",
    "ViTB32-notop" : "12",

}