import torch
import torch.nn as nn
import torch.nn.functional as F
import TransformerBlock
import LinearCombine
import TransformerBlock
import AttentionPooling

class EClassTransformer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers):
        super().__init__()

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock.TransformerBlock(d_model, n_heads)
                for _ in range(n_layers)
            ]
        )

        self.norm = nn.RMSNorm(d_model)

        self.pool = AttentionPooling.AttentionPooling(d_model)

        self.linear = nn.Linear(d_model, d_model)

    def forward(self, nodes):
        return self.linear(self.pool(self.norm(self.transformer_blocks(nodes))))
