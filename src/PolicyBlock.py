import torch
import torch.nn as nn
import torch.nn.functional as F
import TransformerBlock
import LinearCombine
import TransformerBlock
import AttentionPooling
import ENodeEmbedding
import EClassEmbedding

class PolicyBlock(nn.Module):
    def __init__(self, action_count, d_model, n_heads, n_layers):
        super().__init__()

        self.enode_eclass_combine = LinearCombine.LinearCombine(d_model)

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock.TransformerBlock(d_model, n_heads)
                for _ in range(n_layers)
            ]
        )

        self.norm = nn.RMSNorm(d_model)

        self.linear = nn.Linear(d_model, action_count)
    def forward(self, nodes, classes):
        tokens = self.enode_eclass_combine(nodes, classes)
        
        tokens = self.transformer_blocks(tokens)

        tokens = self.norm(tokens)

        return self.linear(tokens)