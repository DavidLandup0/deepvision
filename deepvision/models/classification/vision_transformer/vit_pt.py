from torch import nn

from deepvision.utils.utils import parse_model_inputs
from deepvision.utils.utils import same_padding

class ViTPT(nn.Module):
    def __init__(
        self,
        include_top,
        input_shape=(3, None, None),
        pooling=None,
        classes=None,
        block_type=None,
        **kwargs,
    ):
        super().__init__()

        self.include_top = include_top
        self.pooling = pooling
        self.classes = classes

        if self.include_top and not self.classes:
            raise ValueError(
                "If `include_top` is True, you should specify `classes`. "
                f"Received: classes={self.classes}"
            )

        if self.include_top and self.pooling:
            raise ValueError(
                f"`pooling` must be `None` when `include_top=True`."
                f"Received pooling={self.pooling} and include_top={self.include_top}. "
            )



    def forward(self, input_tensor):
        inputs = parse_model_inputs("pytorch", input_tensor.shape, input_tensor)
        x = inputs


        if self.include_top:
            x = self.top_dense(x)
            x = nn.Softmax(dim=1)(x)

        return x
