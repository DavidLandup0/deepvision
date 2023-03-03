from setuptools import find_packages
from setuptools import setup

setup(
    name="deepvision-toolkit",
    version="0.1.5",
    description="PyTorch and TensorFlow/Keras image models with automatic weight conversions and equal API/implementations - Vision Transformer (ViT), ResNetV2, EfficientNetV2, NeRFs, (planned...) DeepLabV3+, ConvNeXtV2, YOLO, etc.",
    url="https://github.com/DavidLandup0/deepvision",
    author="David Landup",
    author_email="david.landup.0@gmail.com",
    license="Apache License 2.0",
    packages=find_packages(),
    readme="README.md",
    install_requires=[
        "matplotlib",
        "pytorch_lightning",
        "scikit_learn",
        "tensorflow",
        "torch",
        "torchmetrics",
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
)
