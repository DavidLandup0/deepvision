from typing import Tuple

import tensorflow as tf
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.projection = nn.Linear(dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.projection(x)
        return x


class SegFormerHead(nn.Module):
    def __init__(self, dims, embed_dim=256, num_classes=19):
        super().__init__()
        self.linear_layers = torch.nn.ModuleList()
        for i, dim in enumerate(dims):
            self.linear_layers.append(MLP(dim, embed_dim))

        # To fuse multiple layer outputs into a single feature map using a Conv2d
        self.linear_fuse = torch.nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
        )
        # Final segmentation output
        self.seg_out = nn.Conv2d(embed_dim, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, features: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        B, _, H, W = features[0].shape
        outs = []
        for feature, layer in zip(features, self.linear_layers):
            projected_features = (
                layer(feature)
                .permute(0, 2, 1)
                .reshape(B, -1, feature.shape[-2], feature.shape[-1])
            )
            outs.append(
                F.interpolate(
                    projected_features,
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                )
            )

        seg = self.seg_out(self.dropout(self.linear_fuse(torch.cat(outs[::-1], dim=1))))

        return seg
