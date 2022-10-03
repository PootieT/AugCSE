import torch
import torch.nn as nn


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, hidden_size: int, num_layers: int, num_heads: int = 1):
        super().__init__()
        self.num_heads = num_heads
        self.activation = nn.Tanh()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size * num_heads))

    def forward(self, x, eval=False, **kwargs):
        bs, hdim = x.shape
        if eval and len(self.layers) == 1:
            return x

        for dense in self.layers if not eval else self.layers[:-1]:
            x = dense(x)
            x = self.activation(x)
            if self.num_heads > 1:
                # max pool across multi-head
                x = torch.max(x.view(bs, hdim, self.num_heads), dim=2)[0]
        return x


class ProjectionMLP(nn.Module):
    """
    https://github.com/voidism/DiffCSE/blob/master/diffcse/models.py
    """

    def __init__(self, hidden_size):
        super().__init__()
        in_dim = hidden_size
        hidden_dim = hidden_size * 2
        out_dim = hidden_size
        affine = False
        list_layers = [nn.Linear(in_dim, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)]
        list_layers += [nn.Linear(hidden_dim, out_dim, bias=False), nn.BatchNorm1d(out_dim, affine=affine)]
        self.net = nn.Sequential(*list_layers)

    def forward(self, x, eval=False, **kwargs):
        return x if eval else self.net(x)
