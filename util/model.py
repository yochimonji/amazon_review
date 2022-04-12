from typing import Tuple, cast

import torch
from torch import Tensor, nn
from torchtext.legacy.data import Field

from .activation import Tanh


class MyClassifier(nn.Module):
    def __init__(self, emb_dim: int, v_size: int, max_length: int, class_num: int, text_field: Field = None):
        super().__init__()
        self.embed = nn.Embedding(v_size, emb_dim)
        if text_field:
            self.embed.weight.data.copy_(text_field.vocab.vectors)
        self.linear = nn.Linear(emb_dim * max_length, 508)
        self.mlp = nn.Sequential(
            nn.Linear(508, 256),
            nn.BatchNorm1d(256),
            Tanh(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            Tanh(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            Tanh(),
            nn.Dropout(0.25),
            nn.Linear(64, class_num),
        )

    def forward(self, sentence: Tensor) -> Tuple[Tensor, Tensor]:
        embedding = cast(Tensor, self.embed(sentence))
        embedding = embedding.view(embedding.shape[0], -1)
        embedding = cast(Tensor, self.linear(embedding))
        output = cast(Tensor, self.mlp(embedding))
        return embedding, output


class MMD(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, y: Tensor, kernel: str) -> Tensor:
        """Emprical maximum mean discrepancy. The lower the result
        the more evidence that distributions are the same.

        Args:
            x: first sample, distribution P
            y: second sample, distribution Q
            kernel: kernel type such as "multiscale" or "rbf"
        """
        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = xx.diag().unsqueeze(0).expand_as(xx)
        ry = yy.diag().unsqueeze(0).expand_as(yy)

        dxx = rx.t() + rx - 2.0 * xx  # Used for A in (1)
        dyy = ry.t() + ry - 2.0 * yy  # Used for B in (1)
        dxy = rx.t() + ry - 2.0 * zz  # Used for C in (1)

        device = x.device
        XX, YY, XY = (
            torch.zeros(xx.shape).to(device),
            torch.zeros(xx.shape).to(device),
            torch.zeros(xx.shape).to(device),
        )

        if kernel == "multiscale":

            bandwidth_range = [0.2, 0.5, 0.9, 1.3]
            for a in bandwidth_range:
                XX += a**2 * (a**2 + dxx) ** -1
                YY += a**2 * (a**2 + dyy) ** -1
                XY += a**2 * (a**2 + dxy) ** -1

        if kernel == "rbf":

            bandwidth_range = [10, 15, 20, 50]
            for a in bandwidth_range:
                XX += torch.exp(-0.5 * dxx / a)
                YY += torch.exp(-0.5 * dyy / a)
                XY += torch.exp(-0.5 * dxy / a)

        return torch.mean(XX + YY - 2.0 * XY)
