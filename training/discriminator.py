import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function


class GradReverse(Function):
    """
    https://zhuanlan.zhihu.com/p/263827804
    """

    @staticmethod
    def forward(ctx, x, lambd, **kwargs: None):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, *grad_output):
        # can't stop in backward call in pycharm because it's called on a different thread:
        # https://discuss.pytorch.org/t/custom-backward-breakpoint-doesnt-get-hit/6473/6
        # print("static backward called!")
        return grad_output[0] * -ctx.lambd, None

class AugmentationDiscriminator(nn.Module):
    """
    Discriminator mlp, takes in both sentence embeddings, predict augmentation type
    """

    def __init__(
        self,
        input_size: int,
        num_layers: int,
        num_labels: int,
        gradient_reverse_multiplier: float = 1.0,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, input_size * 2))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(input_size * 2, input_size * 2))
        self.layers.append(nn.Linear(input_size * 2, num_labels))
        self.lambd = gradient_reverse_multiplier

    def forward(self, x):
        x = GradReverse.apply(x, self.lambd)  # reverse gradient in backwards pass
        for dense in self.layers:
            x = self.dropout(x)
            x = dense(x)
            x = self.activation(x)
        return x
