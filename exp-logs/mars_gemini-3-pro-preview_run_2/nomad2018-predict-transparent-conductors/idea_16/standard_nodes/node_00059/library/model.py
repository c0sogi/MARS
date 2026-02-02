import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add
from library.config import Config
from library.utils import set_seed

# Ensure reproducibility upon module import
set_seed(Config.SEED)


class RBFExpansion(nn.Module):
    """
    Expands scalar distances into a Gaussian basis set.
    This allows the network to learn non-linear relationships with distance.
    """

    def __init__(self, dmin, dmax, n_bins):
        super().__init__()
        # Register centers as a buffer so it moves to device with the model
        # but is not treated as a trainable parameter.
        self.register_buffer("centers", torch.linspace(dmin, dmax, n_bins))
        self.sigma = (dmax - dmin) / n_bins

    def forward(self, distances):
        """
        Args:
            distances: Tensor of shape [num_edges]
        Returns:
            Tensor of shape [num_edges, n_bins]
        """
        return torch.exp(
            -((distances.unsqueeze(1) - self.centers) ** 2) / self.sigma**2
        )


class CGCNNLayer(nn.Module):
    """
    Crystal Graph Convolutional Neural Network Layer.
    Performs gated message passing where edge features modulate the node interactions.
    """

    def __init__(self, atom_dim, edge_dim):
        super().__init__()
        # The linear layer processes the concatenation of source node, target node, and edge features.
        # It outputs 2 * atom_dim to be split into filter and core gates.
        self.linear = nn.Linear(2 * atom_dim + edge_dim, 2 * atom_dim)
        self.bn = nn.BatchNorm1d(2 * atom_dim)

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: Node features [num_nodes, atom_dim]
            edge_index: Edge indices [2, num_edges] (source, target)
            edge_attr: Edge features [num_edges, edge_dim]
        Returns:
            Aggregated messages [num_nodes, atom_dim]
        """
        src, dest = edge_index

        # Concatenate source node features, target node features, and edge features
        # Shape: [num_edges, 2 * atom_dim + edge_dim]
        z = torch.cat([x[src], x[dest], edge_attr], dim=1)

        # Transform via linear layer and batch norm
        z = self.linear(z)
        z = self.bn(z)

        # Split into filter (gate) and core (signal) parts
        z_filter, z_core = z.chunk(2, dim=1)

        # Apply Gated Activation: Sigmoid(filter) * Softplus(core)
        # This allows the network to learn which interactions are important.
        message = torch.sigmoid(z_filter) * F.softplus(z_core)

        # Aggregate messages to target nodes (dest) using summation
        # Shape: [num_nodes, atom_dim]
        aggr_message = scatter_add(message, dest, dim=0, dim_size=x.size(0))

        return aggr_message


class IR_CGCNN(nn.Module):
    """
    Initial-Residual Crystal Graph Convolutional Network.

    Implements the strategy of anchoring updates to the initial node embedding
    to preserve stoichiometric information throughout the network depth.

    Update Rule: h_{l+1} = Activation(Norm(Conv(h_l) + h_l + h_0))
    """

    def __init__(self, config):
        super().__init__()

        self.atom_dim = config.ATOM_EMBEDDING_DIM
        self.edge_dim = config.EDGE_EMBEDDING_DIM

        # 1. Embedding Layer for Atomic Numbers
        # Covers atomic numbers up to 100
        self.embedding = nn.Embedding(101, self.atom_dim)

        # 2. Edge Distance Expansion and Projection
        self.rbf = RBFExpansion(
            dmin=config.RBF_MIN, dmax=config.RBF_MAX, n_bins=config.RBF_BINS
        )
        self.edge_project = nn.Linear(config.RBF_BINS, self.edge_dim)

        # 3. Interaction Layers (CGCNN Layers)
        self.layers = nn.ModuleList(
            [CGCNNLayer(self.atom_dim, self.edge_dim) for _ in range(config.NUM_LAYERS)]
        )

        # 4. Normalization Layers for the Residual Connection
        # One BatchNorm per layer to normalize the sum (Conv + h + h0)
        self.layer_norms = nn.ModuleList(
            [nn.BatchNorm1d(self.atom_dim) for _ in range(config.NUM_LAYERS)]
        )

        # 5. Dropout for regularization
        self.dropout = nn.Dropout(config.DROPOUT)

        # 6. Decoupled Prediction Heads
        # Independent MLPs for Formation Energy and Bandgap Energy
        self.head_formation = nn.Sequential(
            nn.Linear(self.atom_dim, 64), nn.Softplus(), nn.Linear(64, 1)
        )

        self.head_bandgap = nn.Sequential(
            nn.Linear(self.atom_dim, 64), nn.Softplus(), nn.Linear(64, 1)
        )

    def forward(self, atom_fea, edge_index, edge_fea, batch_index):
        """
        Forward pass of the model.

        Args:
            atom_fea: Atomic numbers [num_atoms]
            edge_index: Graph connectivity [2, num_edges]
            edge_fea: Edge distances [num_edges]
            batch_index: Batch index mapping atoms to graphs [num_atoms]

        Returns:
            predictions: Tensor of shape [batch_size, 2] containing
                         [formation_energy, bandgap_energy]
        """
        # --- Initial Embedding ---
        # h0 serves as the permanent memory of chemical identity
        h0 = self.embedding(atom_fea)  # [num_atoms, atom_dim]
        h = h0.clone()

        # --- Edge Feature Processing ---
        # Expand distances to RBF basis
        edge_rbf = self.rbf(edge_fea)  # [num_edges, n_bins]
        # Project RBF features to embedding dimension
        edge_attr = self.edge_project(edge_rbf)  # [num_edges, edge_dim]
        # Apply non-linearity to edge features
        edge_attr = F.softplus(edge_attr)

        # --- Interaction Blocks ---
        for i, layer in enumerate(self.layers):
            # 1. Convolution (Message Passing)
            aggr_message = layer(h, edge_index, edge_attr)

            # 2. Amplified Residual Connection (Cite solution_lesson_node_00054)
            # Weight self-loop more heavily to preserve stoichiometry.
            # Removed Initial Residual (+ h0) as it restricts shallow GNNs (Cite solution_lesson_node_00057).
            h_next = aggr_message + 2 * h

            # 3. Normalization
            h_next = self.layer_norms[i](h_next)

            # 4. Activation
            h_next = F.softplus(h_next)

            # 5. Dropout
            h_next = self.dropout(h_next)

            # Update node state
            h = h_next

        # --- Global Pooling ---
        # Average node features for each graph in the batch
        # pooled shape: [batch_size, atom_dim]
        pooled = scatter_mean(h, batch_index, dim=0)

        # --- Prediction Heads ---
        # Predict targets independently
        pred_formation = self.head_formation(pooled)
        pred_bandgap = self.head_bandgap(pooled)

        # Concatenate predictions
        return torch.cat([pred_formation, pred_bandgap], dim=1)
