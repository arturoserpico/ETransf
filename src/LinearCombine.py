import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearCombine(nn.Module):
    def __init__(self, d_model: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(2 * d_model, d_model, bias=bias)

    def forward(self, a, b):
        x = torch.cat([a, b], dim=-1)
        return self.proj(x)