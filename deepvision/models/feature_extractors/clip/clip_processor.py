from typing import List
from typing import Union

import torch
from PIL import Image
from pkg_resources import packaging
from torchvision.transforms import CenterCrop
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor

from deepvision.models.feature_extractors.clip.clip_tokenizer import CLIPTokenizer

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


class __CLIPProcessorPT:
    def __init__(self, input_resolution):
        self.image_transform = Compose(
            [
                Resize(input_resolution, interpolation=BICUBIC),
                CenterCrop(input_resolution),
                _convert_image_to_rgb,
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        self.tokenizer = CLIPTokenizer()

    def process_image(self, image):
        if isinstance(image, str):
            image = Image.open(image)
        return self.image_transform(image)

    def process_texts(self, texts, context_length: int = 77, truncate: bool = False):
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        all_tokens = [
            [sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts
        ]

        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate:
                    tokens = tokens[:context_length]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(
                        f"Input {texts[i]} is too long for context length {context_length}"
                    )
            result[i, : len(tokens)] = torch.tensor(tokens)

        return result

    def process_pair(self, image, texts):
        image = self.process_image(image)
        texts = self.process_texts(texts)
        return (image, texts)


LAYER_BACKBONES = {
    "tensorflow": None,
    "pytorch": __CLIPProcessorPT,
}


def CLIPProcessor(resolution, backend):
    layer_class = LAYER_BACKBONES.get(backend)
    if layer_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {LAYER_BACKBONES.keys()}"
        )

    layer = layer_class(input_resolution=resolution)

    return layer
