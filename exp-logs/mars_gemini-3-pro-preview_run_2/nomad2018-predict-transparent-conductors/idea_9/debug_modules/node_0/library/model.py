import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GlobalAttention

from library.config import Config
from library.utils import GaussianSmearing


class CGCNNConv(MessagePassing):
    """
    The Crystal Graph Convolutional Neural Network (CGCNN) layer.
    Implements the gated convolution:
    v_i^(t+1) = v_i^t + sum_{j in N(i)} sigma(z_{ij}W_f + b_f) * g(z_{ij}W_s + b_s)
    where z_{ij} = v_i^t || v_j^t || u_{ij}
    """

    def __init__(self, node_dim, edge_dim, aggr="add", **kwargs):
        super(CGCNNConv, self).__init__(aggr=aggr, **kwargs)
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        # The input to the linear layers is the concatenation of:
        # node_i (node_dim) + node_j (node_dim) + edge_attr (edge_dim)
        in_dim = 2 * node_dim + edge_dim

        # Filter (sigmoid activation) and Core (softplus activation) linear transformations
        self.lin_f = nn.Linear(in_dim, node_dim)
        self.lin_s = nn.Linear(in_dim, node_dim)

        # Batch normalization for training stability
        self.bn = nn.BatchNorm1d(node_dim)

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: Node features (num_nodes, node_dim)
            edge_index: Graph connectivity (2, num_edges)
            edge_attr: Edge features (num_edges, edge_dim)
        """
        # Calculate messages and aggregate
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Residual connection and Batch Norm
        # Note: x is the node features from the previous layer
        out = self.bn(out) + x
        return out

    def message(self, x_i, x_j, edge_attr):
        """
        Construct message for each edge.
        x_i: Features of target nodes (num_edges, node_dim)
        x_j: Features of source nodes (num_edges, node_dim)
        edge_attr: Edge features (num_edges, edge_dim)
        """
        # Concatenate source node, target node, and edge features
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)

        # Filter gate (sigmoid)
        gate = torch.sigmoid(self.lin_f(z))

        # Core signal (softplus)
        core = F.softplus(self.lin_s(z))

        # Element-wise product
        return gate * core


class DualStreamCGCNN(nn.Module):
    """
    Dual-Stream Late-Fusion CGCNN Architecture.

    Stream 1 (Local): CGCNN backbone processes the crystal graph to learn local atomic interactions.
                      Aggregated via Global Attention Pooling.
    Stream 2 (Global): MLP processes global features (lattice parameters, composition).

    Fusion: The outputs of both streams are concatenated.
    Readout: Separate MLP heads predict formation energy and bandgap energy.
    """

    def __init__(self):
        super(DualStreamCGCNN, self).__init__()

        # ---------------------------------------------------------------------
        # Stream 1: Local Graph
        # ---------------------------------------------------------------------

        # 1. Atom Embedding: Map atomic indices to dense vectors
        self.embedding = nn.Embedding(Config.NUM_ATOM_TYPES, Config.ATOM_EMBEDDING_DIM)

        # 2. Edge Features: Expand distances using Gaussian RBF
        self.rbf = GaussianSmearing(
            start=0.0, stop=Config.CUTOFF_RADIUS, n_gaussians=Config.N_RBF
        )

        # 3. Interaction Layers: Stack of CGCNNConv layers
        self.conv_layers = nn.ModuleList(
            [
                CGCNNConv(node_dim=Config.HIDDEN_DIM, edge_dim=Config.N_RBF)
                for _ in range(Config.N_INTERACTION_LAYERS)
            ]
        )

        # 4. Global Attention Pooling
        # Learn to weight nodes based on their features before aggregation
        # Gate NN maps node features to a scalar attention score
        gate_nn = nn.Sequential(
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM // 2, 1),
        )
        self.pooling = GlobalAttention(gate_nn=gate_nn)

        # ---------------------------------------------------------------------
        # Stream 2: Global Features
        # ---------------------------------------------------------------------
        # MLP to process lattice parameters and composition
        self.global_mlp = nn.Sequential(
            nn.Linear(Config.N_GLOBAL_FEATURES, Config.HIDDEN_DIM),
            nn.BatchNorm1d(Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM),
            nn.BatchNorm1d(Config.HIDDEN_DIM),
            nn.ReLU(),
        )

        # ---------------------------------------------------------------------
        # Fusion & Readout
        # ---------------------------------------------------------------------
        # Input dim = Graph Stream Output (HIDDEN_DIM) + Global Stream Output (HIDDEN_DIM)
        fusion_dim = Config.HIDDEN_DIM * 2

        # Head for Target 1: Formation Energy
        self.head_formation = nn.Sequential(
            nn.Linear(fusion_dim, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(Config.HIDDEN_DIM, 1),
        )

        # Head for Target 2: Bandgap Energy
        self.head_bandgap = nn.Sequential(
            nn.Linear(fusion_dim, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(Config.HIDDEN_DIM, 1),
        )

    def forward(self, data):
        """
        Forward pass of the model.

        Args:
            data: A torch_geometric.data.Data batch object containing:
                - x: Atom types (Batch_Nodes,)
                - edge_index: Graph connectivity (2, Batch_Edges)
                - edge_attr: Edge distances (Batch_Edges,)
                - global_feat: Global features (Batch_Size, N_Global_Features)
                - batch: Batch index for each node (Batch_Nodes,)

        Returns:
            Tensor of shape (Batch_Size, 2) containing [formation_energy, bandgap]
        """
        # ---------------------------------------------------------------------
        # Stream 1: Local Graph Processing
        # ---------------------------------------------------------------------
        # Node Embedding
        x = self.embedding(data.x)  # (Num_Nodes, Hidden_Dim)

        # Edge Feature Expansion (Distance -> RBF)
        edge_feat = self.rbf(data.edge_attr)  # (Num_Edges, N_RBF)

        # Message Passing Layers
        for conv in self.conv_layers:
            x = conv(x, data.edge_index, edge_feat)

        # Global Pooling (Attention-based aggregation)
        # Aggregates node features into a single vector per graph
        graph_embedding = self.pooling(x, data.batch)  # (Batch_Size, Hidden_Dim)

        # ---------------------------------------------------------------------
        # Stream 2: Global Feature Processing
        # ---------------------------------------------------------------------
        # Process global features (Lattice + Composition)
        global_embedding = self.global_mlp(data.global_feat)  # (Batch_Size, Hidden_Dim)

        # ---------------------------------------------------------------------
        # Fusion
        # ---------------------------------------------------------------------
        # Concatenate local and global representations
        fused = torch.cat(
            [graph_embedding, global_embedding], dim=1
        )  # (Batch_Size, 2*Hidden_Dim)

        # ---------------------------------------------------------------------
        # Readout
        # ---------------------------------------------------------------------
        # Predict targets using specialized heads
        out_formation = self.head_formation(fused)
        out_bandgap = self.head_bandgap(fused)

        # Concatenate predictions: (Batch_Size, 2)
        return torch.cat([out_formation, out_bandgap], dim=1)
