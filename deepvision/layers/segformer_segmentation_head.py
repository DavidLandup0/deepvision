import torch
from torch import nn, Tensor
from typing import Tuple
from torch.nn import functional as F
import tensorflow as tf


class MLP(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.projection = nn.Linear(dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(2).transpose(1, 2)
        x = self.projection(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.activate = nn.ReLU(True)

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))


class SegFormerHead(nn.Module):
    def __init__(self, dims, embed_dim=256, num_classes=19):
        super().__init__()
        for i, dim in enumerate(dims):
            self.add_module(f"linear_layer_{i+1}", MLP(dim, embed_dim))

        # To fuse multiple layer outputs into a single feature map using a Conv2d
        self.linear_fuse = ConvModule(embed_dim * 4, embed_dim)
        # Final segmentation output
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, features: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        B, _, H, W = features[0].shape
        outs = [
            self.linear_layer_1(features[0])
            .permute(0, 2, 1)
            .reshape(B, -1, *features[0].shape[-2:])
        ]

        for i, feature in enumerate(features[1:]):
            cf = (
                eval(f"self.linear_layer_{i+2}")(feature)
                .permute(0, 2, 1)
                .reshape(B, -1, *feature.shape[-2:])
            )
            outs.append(
                F.interpolate(cf, size=(H, W), mode="bilinear", align_corners=False)
            )

        seg = self.linear_fuse(torch.cat(outs[::-1], dim=1))
        seg = self.linear_pred(self.dropout(seg))
        return seg
