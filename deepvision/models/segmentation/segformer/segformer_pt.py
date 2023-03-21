import torch
from torch import Tensor
from torch.nn import functional as F

from deepvision.layers.segformer_segmentation_head import SegFormerHead


class __SegFormerPT(torch.nn.Module):
    def __init__(self, num_classes, backbone, embed_dim, softmax_output=None):
        super().__init__()
        self.backbone = backbone
        self.decode_head = SegFormerHead(
            in_dims=backbone.output_channels,
            embed_dim=embed_dim,
            num_classes=num_classes,
            backend="pytorch",
        )
        self.softmax_output = softmax_output

    def forward(self, x):
        y = self.backbone(x)
        y = self.decode_head(y)
        y = F.interpolate(
            y, size=x.shape[2:], mode="bilinear", align_corners=False
        )  # to original image shape
        if self.softmax_output:
            y = torch.nn.Softmax(1)(y)
        return y
