import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.attention = nn.Linear(d_model, 1)

    def forward(self, x):
        scores = self.attention(x)

        weights = F.softmax(scores, dim=1)

        pooled = (weights * x).sum(dim=1)

        return pooled