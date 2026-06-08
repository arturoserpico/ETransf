from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import ENodeEmbedding, EClassEmbedding, PolicyBlock, EGraph

class Policy(nn.Module):
    def __init__(self, 
                 rules: list[EGraph.RewriteRule], 
                 d_model: int, 
                 n_layers: int, 
                 n_layers_node: int, 
                 n_layers_class: int, 
                 n_heads: int, 
                 n_heads_node: int,
                 n_heads_class: int, 
                 max_arity: int, 
                 op_count: int):
        super().__init__()
        
        self.rules = rules
        self.d_model = d_model
        self.enode_embedding = ENodeEmbedding.ENodeTransformer(op_count, max_arity, d_model, n_heads_node, n_layers_node)
        self.eclass_emdedding = EClassEmbedding.EClassTransformer(d_model, n_heads_class, n_layers_class)
        self.target_eclass_linear = nn.Linear(d_model, d_model)
        self.core = PolicyBlock.PolicyBlock(len(rules) + 3, d_model, n_heads, n_layers)
        self.loop_resulution_vector = nn.Parameter(torch.randn(d_model))
        self.softmax = nn.Softmax(dim=-1)

    def make_class_embedding(self, eg: EGraph.EGraph, target_eclass: int, eclass: EGraph.EClass, 
                             visited_nodes: dict[EGraph.ENode, torch.Tensor], 
                             visited_classes: dict[EGraph.EClass, torch.Tensor]) -> torch.Tensor:
        if eclass in visited_classes.keys():
            return self.loop_resulution_vector
        
        visited_classes[eclass] = torch.empty(self.d_model)

        collect_embeddings = []

        for enode in eclass.nodes:
            collect_embeddings.append(self.make_node_embedding(eg, target_eclass, enode, visited_nodes, visited_classes))

        tensor = torch.stack(collect_embeddings).unsqueeze(0)

        device = next(self.parameters()).device

        tensor = tensor.to(device)

        embedding = self.eclass_emdedding(tensor).squeeze(0)

        if eclass.id == target_eclass:
            embedding = self.target_eclass_linear(embedding)

        visited_classes[eclass] = embedding 

        return embedding

        


    def make_node_embedding(self, eg: EGraph.EGraph, target_eclass: int, enode: EGraph.ENode, 
                             visited_nodes: dict[EGraph.ENode, torch.Tensor], 
                             visited_classes: dict[EGraph.EClass, torch.Tensor]) -> torch.Tensor:
        if enode in visited_nodes.keys():
            return self.loop_resulution_vector
        
        visited_nodes[enode] = torch.empty(self.d_model)
        
        collect_embeddings = []

        for eclass in enode.child_classes(eg):
            collect_embeddings.append(self.make_class_embedding(eg, target_eclass, eclass, visited_nodes, visited_classes))

        tensor = torch.empty(1, 0, self.d_model) if len(collect_embeddings) == 0 else torch.stack(collect_embeddings).unsqueeze(0)

        device = next(self.parameters()).device

        tensor = tensor.to(device)

        embedding = self.enode_embedding(torch.tensor([enode.op], device=device), torch.tensor([enode.param], dtype=torch.float32, device=device), tensor).squeeze(0)
        
        visited_nodes[enode] = embedding

        return embedding

    def make_all_embedding(self, eg: EGraph.EGraph, target_eclass: int) -> tuple[list[EGraph.ENode], list[torch.Tensor], list[torch.Tensor]]:
        visited_nodes: dict[EGraph.ENode, torch.Tensor] = {}
        visited_classes: dict[EGraph.EClass, torch.Tensor] = {}

        for eclass in eg.eclasses.values():
            if len(visited_nodes) == eg.enode_count and len(visited_classes) == eg.eclass_count:
                break

            self.make_class_embedding(eg, target_eclass, eclass, visited_nodes, visited_classes)

        nodes = []
        nodes_embeddings = []
        class_embeddings = []

        for enode, node_embedding in visited_nodes.items():
            nodes.append(enode)
            nodes_embeddings.append(node_embedding)
            for eclass, class_embedding in visited_classes.items():
                if enode in eclass.nodes:
                    class_embeddings.append(class_embedding)

        return nodes, nodes_embeddings, class_embeddings
        

    def forward(self, eg: EGraph.EGraph, target_eclass: int):
        nodes, nodes_embeddings, class_embeddings = self.make_all_embedding(eg, target_eclass)

        nodes_tensor = torch.stack(nodes_embeddings).unsqueeze(0)
        classes_tensor = torch.stack(class_embeddings).unsqueeze(0)

        result = self.core(nodes_tensor, classes_tensor).squeeze(0)

        for i in range(len(nodes)):
            for j in range(len(self.rules)):
                if not eg.is_rule_applicable_at(self.rules[j], nodes[i]):
                    result[i, j + 3] = float('-inf')

        return nodes, self.softmax(result)


