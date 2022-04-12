import torch
from torch import Tensor, nn


class Tanh(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return torch.tanh(input)


class Relu(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        zeros = torch.zeros(input.size()).to(input.device)
        return torch.maximum(input, zeros)
