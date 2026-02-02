import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Batch
from library.config import Config
from library.utils import GaussianRBF, Standardizer, compute_rmsle


class GatedGCNLayer(MessagePassing):
    """
    Gated Graph Convolution Layer.
    Updates node features based on neighbor features and edge gating.
    This layer is used for both the Atom Graph (updating atoms via bonds)
    and the Line Graph (updating bonds via angles).
    """

    def __init__(self, node_dim, edge_dim, dropout=0.0):
        super().__init__(aggr="add")  # Aggregation method: Sum

        # Gating mechanism: Computes a soft gate (0-1) for each message
        # gate_ij = sigmoid(W_src * x_i + W_dst * x_j + W_edge * e_ij)
        self.gate_linear_src = nn.Linear(node_dim, edge_dim)
        self.gate_linear_dst = nn.Linear(node_dim, edge_dim)
        self.gate_linear_edge = nn.Linear(edge_dim, edge_dim)

        # Message transformation: W_msg * x_j
        self.msg_linear = nn.Linear(node_dim, node_dim)

        # Update mechanism for the node itself after aggregation
        self.update_linear = nn.Linear(node_dim, node_dim)
        self.norm = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)

        # Activation function
        self.act = nn.SiLU()

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: Node features (N, node_dim)
            edge_index: Graph connectivity (2, E)
            edge_attr: Edge features (E, edge_dim)
        Returns:
            Updated node features (N, node_dim)
        """
        # Propagate messages
        # x and edge_attr are passed to the message function
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Residual connection + Transformation + Norm
        out = x + self.dropout(self.update_linear(out))
        out = self.norm(out)
        return out

    def message(self, x_i, x_j, edge_attr):
        """
        Computes the message for each edge (j -> i).
        x_i: Target node features (E, node_dim)
        x_j: Source node features (E, node_dim)
        edge_attr: Edge features (E, edge_dim)
        """
        # Compute Gate
        # We project node features to edge_dim to match edge_attr dimension for the gate
        gate_input = (
            self.gate_linear_src(x_i)
            + self.gate_linear_dst(x_j)
            + self.gate_linear_edge(edge_attr)
        )
        gate = torch.sigmoid(gate_input)

        # Compute raw message from source node
        msg = self.msg_linear(x_j)

        # Apply gate
        return gate * msg


class CrystalGNN(nn.Module):
    """
    Crystal Graph Neural Network.
    Simplified architecture focusing on pairwise distance-based interactions.
    (Cite solution_lesson_node_00016: Simpler models generalize better on small datasets)
    """

    def __init__(self):
        super().__init__()

        dim = Config.EMBEDDING_DIM

        # ---------------------------------------------------------------------
        # 1. Initial Embeddings
        # ---------------------------------------------------------------------
        # Atom embedding
        self.atom_embedding = nn.Embedding(100, dim)

        # Distance RBF expansion and projection
        # (Cite solution_lesson_node_00002: RBF encoding of distances)
        self.distance_rbf = GaussianRBF(0.0, Config.CUTOFF_RADIUS, Config.RBF_NUM_BINS)
        self.distance_proj = nn.Linear(Config.RBF_NUM_BINS, dim)

        # ---------------------------------------------------------------------
        # 2. Interaction Blocks
        # ---------------------------------------------------------------------
        self.blocks = nn.ModuleList()
        for _ in range(Config.NUM_BLOCKS):
            # Atom Graph Layer: Updates atom features using bond features.
            atom_layer = GatedGCNLayer(node_dim=dim, edge_dim=dim)
            self.blocks.append(atom_layer)

        # ---------------------------------------------------------------------
        # 3. Output Heads
        # ---------------------------------------------------------------------
        self.pool = global_mean_pool

        # Head for Formation Energy
        self.head_formation = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1)
        )

        # Head for Bandgap Energy
        self.head_bandgap = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1)
        )

    def forward(self, data):
        """
        Forward pass of the model.
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr_rbf = data.edge_attr  # Raw distances

        # ---------------------------------------------------------------------
        # Initialization
        # ---------------------------------------------------------------------
        # Embed Atoms
        h_atoms = self.atom_embedding(x)  # (N_atoms, dim)

        # Embed Bonds (Static edge features)
        h_bonds = self.distance_rbf(edge_attr_rbf)  # (N_bonds, bins)
        h_bonds = self.distance_proj(h_bonds)  # (N_bonds, dim)

        # ---------------------------------------------------------------------
        # Message Passing Blocks
        # ---------------------------------------------------------------------
        for block in self.blocks:
            # Update atom representations by passing messages across bonds.
            # We use the static distance embeddings as edge attributes for gating.
            h_atoms = block(h_atoms, edge_index, h_bonds)

        # ---------------------------------------------------------------------
        # Readout and Prediction
        # ---------------------------------------------------------------------
        # Global mean pooling of atom features
        h_graph = self.pool(h_atoms, data.batch)

        # Predict targets
        pred_formation = self.head_formation(h_graph)
        pred_bandgap = self.head_bandgap(h_graph)

        # Concatenate predictions
        return torch.cat([pred_formation, pred_bandgap], dim=1)
