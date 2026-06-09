from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import TransformerBlock
import LinearCombine
import TransformerBlock
import AttentionPooling
from EGraph import *

class OperatorNetwork(nn.Module):
    def __init__(self, d_model: int, arity: int, n_hidden: int):
        super().__init__()

        self.d_model = d_model
        self.arity = arity

        self.net = nn.Sequential(
            nn.Linear(arity * d_model + 1, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, d_model)
        )
    
    def forward(self, param: int, children: torch.Tensor = torch.Tensor([[]])):
        T, D = children.shape

        assert D == self.d_model
        assert T == self.arity

        input = children.flatten()

        input = torch.cat([input, torch.tensor([param], dtype=torch.float)]).to(children.device)

        return self.net(input)



class ENodeNetworks(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 OP: OpTable, 
                 n_hidden_per_arg: int, 
                 n_hidden_per_param: int):
        super().__init__()

        self.d_model = d_model

        self.networks: list[None | OperatorNetwork] = [None] * len(OP)

        for id, _ in OP._id_to_name.items():
            arity = OP.arity(id)
            self.networks[id] = OperatorNetwork(d_model, arity, n_hidden_per_arg * arity + n_hidden_per_param)

    def forward(self, op: int, param: int, children: torch.Tensor = torch.Tensor([[]])):
        return self.networks[op](param, children)