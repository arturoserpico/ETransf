import torch
import torch.nn as nn
import torch.nn.functional as F
import TransformerBlock
import LinearCombine
import TransformerBlock
import AttentionPooling

class ENodeTransformer(nn.Module):
    def __init__(self, n_operators, max_arity, d_model, n_heads, n_layers):
        super().__init__()

        self.d_model = d_model

        self.op_embedding = nn.Embedding(n_operators, d_model)

        self.op_param_embedding = nn.Linear(1, d_model)

        self.op_param_combine = LinearCombine.LinearCombine(d_model)

        self.op_args_combine = LinearCombine.LinearCombine(d_model)

        self.pos_embedding = nn.Embedding(max_arity, d_model)

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock.TransformerBlock(d_model, n_heads)
                for _ in range(n_layers)
            ]
        )

        self.norm = nn.RMSNorm(d_model)

        self.pool = AttentionPooling.AttentionPooling(d_model)

        self.linear = nn.Linear(d_model, d_model)

    def forward(self, ops, ops_parameters, args):
        B = ops.shape[0]
        T = args.shape[1] if args.dim() > 1 else 0
        D = self.d_model

        ops_embeddings = self.op_embedding(ops)  # (B, D)

        op_args_embeddings = self.op_param_embedding(ops_parameters)

        ops_embeddings = self.op_param_combine(ops_embeddings, op_args_embeddings)

        if T == 0:
            return self.linear(ops_embeddings)

        ops_embeddings = ops_embeddings.unsqueeze(1).expand(B, T, D)

        tokens = self.op_args_combine(args, ops_embeddings)

        pos = torch.arange(T, device=args.device)
        pos_embeddings = self.pos_embedding(pos)

        tokens = tokens + pos_embeddings

        tokens = self.transformer_blocks(tokens)

        tokens = self.norm(tokens)

        return self.linear(self.pool(tokens))