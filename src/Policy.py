from typing import Any

import egglog
import torch
import torch.nn as nn
import torch.nn.functional as F

import ENodeEmbedding, EClassEmbedding, PolicyBlock, EGraph

class Policy(nn.Module):
    def __init__(self, d_model, n_layers, n_layers_node, n_layers_class, n_heads, n_heads_node, n_heads_class, max_arity, op_count, action_count):
        super().__init__()
        
        self.d_model = d_model
        self.enode_embedding = ENodeEmbedding.ENodeTransformer(op_count, max_arity, d_model, n_heads_node, n_layers_node)
        self.eclass_emdedding = EClassEmbedding.EClassTransformer(d_model, n_heads_class, n_layers_class)
        self.core = PolicyBlock.PolicyBlock(action_count, d_model, n_heads, n_layers)
        self.loop_resulution_vector = nn.Parameter(torch.randn(d_model))

    def make_class_embedding(self, eg: EGraph.EGraph, eclass: EGraph.EClass, 
                             visited_nodes: dict[EGraph.ENode, torch.Tensor], 
                             visited_classes: dict[EGraph.EClass, torch.Tensor]) -> torch.Tensor:
        if eclass in visited_classes.keys():
            return self.loop_resulution_vector

        collect_embeddings = []

        for enode in eclass.nodes:
            collect_embeddings.append(self.make_node_embedding(eg, enode, visited_nodes, visited_classes))

        tensor = torch.stack(collect_embeddings).unsqueeze(0)

        embedding = self.eclass_emdedding(tensor).squeeze(0)

        visited_classes[eclass] = embedding 

        return embedding

        


    def make_node_embedding(self, eg: EGraph.EGraph, enode: EGraph.ENode, 
                             visited_nodes: dict[EGraph.ENode, torch.Tensor], 
                             visited_classes: dict[EGraph.EClass, torch.Tensor]) -> torch.Tensor:
        if enode in visited_nodes.keys():
            return self.loop_resulution_vector
        
        collect_embeddings = []

        for eclass in enode.child_classes(eg):
            collect_embeddings.append(self.make_class_embedding(eg, eclass, visited_nodes, visited_classes))

        tensor = torch.empty(1, 0, self.d_model) if len(collect_embeddings) == 0 else torch.stack(collect_embeddings).unsqueeze(0)

        embedding = self.enode_embedding(torch.tensor([enode.op]), torch.tensor([enode.param], dtype=torch.float32), tensor).squeeze(0)
        
        visited_nodes[enode] = embedding

        return embedding

    def make_all_embedding(self, eg: EGraph.EGraph) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        visited_nodes: dict[EGraph.ENode, torch.Tensor] = {}
        visited_classes: dict[EGraph.EClass, torch.Tensor] = {}

        for eclass in eg.eclasses.values():
            if len(visited_nodes) == eg.enode_count and len(visited_classes) == eg.eclass_count:
                break

            self.make_class_embedding(eg, eclass, visited_nodes, visited_classes)

        nodes_embeddings = []
        class_embeddings = []

        for enode, node_embedding in visited_nodes.items():
            nodes_embeddings.append(node_embedding)
            for eclass, class_embedding in visited_classes.items():
                if enode in eclass.nodes:
                    class_embeddings.append(class_embedding)

        return nodes_embeddings, class_embeddings
        

    def forward(self, eg: EGraph.EGraph):
        nodes_embeddings, class_embeddings = self.make_all_embedding(eg)

        nodes_tensor = torch.stack(nodes_embeddings).unsqueeze(0)
        classes_tensor = torch.stack(class_embeddings).unsqueeze(0)

        return self.core(nodes_tensor, classes_tensor).squeeze(0)
