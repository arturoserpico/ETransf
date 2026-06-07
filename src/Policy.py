from typing import Any

import egglog
import torch
import torch.nn as nn
import torch.nn.functional as F

import ENodeEmbedding, EClassEmbedding, PolicyBlock

class Policy(nn.Module):
    def __init__(self, d_model, n_layers, n_layers_node, n_layers_class, n_heads, n_heads_node, n_heads_class, max_arity, op_count, action_count):
        self.enode_embedding = ENodeEmbedding.ENodeTransformer(op_count, max_arity, d_model, n_heads_node, n_layers_node)
        self.eclass_emdedding = EClassEmbedding.EClassTransformer(d_model, n_heads_class, n_layers_class)
        self.core = PolicyBlock.PolicyBlock(action_count, d_model, n_heads, n_layers)

