import torch
import torch.nn as nn
from torch_scatter import scatter
from library.config import (
    HIDDEN_DIM,
    NUM_INTERACTIONS,
    ATOM_TYPES,
    NUM_RBF,
    NUM_SPHERICAL,
    NUM_RADIAL,
    NUM_COUPLING_TYPES,
    OUTPUT_DIM,
)


class EmbeddingBlock(nn.Module):
    """
    Initializes atom and edge embeddings.
    Edge embeddings are derived from the concatenation of source atom embedding,
    target atom embedding, and the RBF expansion of the edge length.
    """

    def __init__(self, hidden_dim, num_rbf, num_atom_types):
        super(EmbeddingBlock, self).__init__()
        self.atom_embedding = nn.Embedding(num_atom_types, hidden_dim)
        self.rbf_lin = nn.Linear(num_rbf, hidden_dim)

        # Initialize edge embedding: Atom_src || Atom_dst || RBF -> Hidden
        self.edge_init = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_attr, edge_index):
        # x: (NumNodes,) - Atom type indices
        # edge_attr: (NumEdges, NumRBF) - RBF features
        # edge_index: (2, NumEdges) - Connectivity

        # 1. Embed atoms
        h_nodes = self.atom_embedding(x)  # (NumNodes, Hidden)

        # 2. Create initial edge embeddings
        src, dst = edge_index
        h_src = h_nodes[src]
        h_dst = h_nodes[dst]
        h_rbf = self.rbf_lin(edge_attr)

        # Concatenate and project
        edge_input = torch.cat([h_src, h_dst, h_rbf], dim=-1)
        h_edges = self.edge_init(edge_input)  # (NumEdges, Hidden)

        return h_edges, h_nodes


class InteractionBlock(nn.Module):
    """
    Performs directional message passing.
    Updates edge features based on interactions with preceding edges (triplets),
    weighted by spherical basis functions (angles + distances).
    """

    def __init__(self, hidden_dim, sbf_dim):
        super(InteractionBlock, self).__init__()

        # Linear transformation for incoming edge features (k->j)
        self.lin_kj = nn.Linear(hidden_dim, hidden_dim)

        # Linear transformation for geometric features (SBF of triplet k-j-i)
        self.lin_sbf = nn.Linear(sbf_dim, hidden_dim, bias=False)

        # MLP for updating the edge state (j->i)
        self.mlp_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h_edges, triplet_attr, triplet_index, num_edges):
        # h_edges: (NumEdges, Hidden)
        # triplet_attr: (NumTriplets, SBF_Dim)
        # triplet_index: (2, NumTriplets) -> [edge_idx_kj, edge_idx_ji]

        idx_kj, idx_ji = triplet_index

        # 1. Transform incoming edges (k->j)
        m_kj = self.lin_kj(h_edges)  # (NumEdges, Hidden)
        m_kj = m_kj[idx_kj]  # Gather for triplets -> (NumTriplets, Hidden)

        # 2. Transform geometric features
        w_sbf = self.lin_sbf(triplet_attr)  # (NumTriplets, Hidden)

        # 3. Interaction (Hadamard product)
        # Modulate the message by the geometric weight
        m_kji = m_kj * w_sbf

        # 4. Aggregate messages to target edges (j->i)
        # Sum all messages from k->j directed towards j->i
        m_ji_agg = scatter(m_kji, idx_ji, dim=0, dim_size=num_edges, reduce="add")

        # 5. Update state with Residual Connection
        h_edges_new = h_edges + self.mlp_update(m_ji_agg)

        return h_edges_new


class OutputBlock(nn.Module):
    """
    Predicts the scalar coupling constant for specific atom pairs.
    Concatenates the final edge embedding, atom embeddings, and coupling type embedding.
    """

    def __init__(self, hidden_dim, num_coupling_types, output_dim):
        super(OutputBlock, self).__init__()
        self.type_embedding = nn.Embedding(num_coupling_types, hidden_dim)

        # Input: EdgeState + AtomSrc + AtomDst + CouplingType
        input_dim = 4 * hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, h_edges, h_nodes, target_edge_index, target_type, edge_index):
        # h_edges: (NumEdges, Hidden) - Final edge representations
        # h_nodes: (NumNodes, Hidden) - Initial atom embeddings
        # target_edge_index: (NumTargets,) - Indices of edges corresponding to coupling pairs
        # target_type: (NumTargets,) - Indices of coupling types
        # edge_index: (2, NumEdges) - Graph connectivity

        # 1. Gather features for the specific target edges
        h_target_edges = h_edges[target_edge_index]  # (NumTargets, Hidden)

        # 2. Identify the atoms connected by these edges
        src_idx = edge_index[0, target_edge_index]
        dst_idx = edge_index[1, target_edge_index]

        h_src = h_nodes[src_idx]
        h_dst = h_nodes[dst_idx]

        # 3. Embed the coupling type
        h_type = self.type_embedding(target_type)

        # 4. Concatenate all information
        out_input = torch.cat([h_target_edges, h_src, h_dst, h_type], dim=-1)

        # 5. Predict
        out = self.mlp(out_input)

        return out


class DMPNN(nn.Module):
    """
    Directional Message Passing Neural Network with Spherical Basis Functions.
    """

    def __init__(self):
        super(DMPNN, self).__init__()

        self.hidden_dim = HIDDEN_DIM
        self.num_interactions = NUM_INTERACTIONS
        self.num_rbf = NUM_RBF
        self.sbf_dim = NUM_SPHERICAL * NUM_RADIAL
        self.num_atom_types = len(ATOM_TYPES)
        self.num_coupling_types = NUM_COUPLING_TYPES

        # 1. Embedding Block
        self.embedding = EmbeddingBlock(
            self.hidden_dim, self.num_rbf, self.num_atom_types
        )

        # 2. Interaction Blocks (Message Passing)
        self.interactions = nn.ModuleList(
            [
                InteractionBlock(self.hidden_dim, self.sbf_dim)
                for _ in range(self.num_interactions)
            ]
        )

        # 3. Output Block (Single Readout)
        # Using a single readout after the final layer is preferred for specific pairwise interactions
        # Cite solution_lesson_node_00013
        self.output_block = OutputBlock(
            self.hidden_dim, self.num_coupling_types, OUTPUT_DIM
        )

    def forward(self, data):
        # Unpack data dictionary
        x = data["x"]
        edge_index = data["edge_index"]
        edge_attr = data["edge_attr"]
        triplet_index = data["triplet_index"]
        triplet_attr = data["triplet_attr"]
        target_edge_index = data["target_edge_index"]
        target_type = data["target_type"]

        num_edges = edge_index.shape[1]

        # 1. Initial Embedding
        h_edges, h_nodes = self.embedding(x, edge_attr, edge_index)

        # 2. Directional Message Passing
        for interaction in self.interactions:
            h_edges = interaction(h_edges, triplet_attr, triplet_index, num_edges)

        # 3. Final Readout
        pred = self.output_block(
            h_edges, h_nodes, target_edge_index, target_type, edge_index
        )

        return pred
