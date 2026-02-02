import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from library.config import Config


class RBFExpansion(nn.Module):
    """
    Expands scalar distances into a vector of Radial Basis Functions (Gaussian).
    """

    def __init__(self, min_dist=0.0, max_dist=5.0, num_rbf=60):
        super(RBFExpansion, self).__init__()
        self.num_rbf = num_rbf
        # Create centers linearly spaced
        centers = torch.linspace(min_dist, max_dist, num_rbf)
        self.register_buffer("centers", centers)
        # Sigma determined by the spacing
        width = (max_dist - min_dist) / num_rbf
        self.sigma = width

    def forward(self, distance):
        """
        Args:
            distance: Tensor of shape (E,), (E, 1), or (E, 3)
        Returns:
            Tensor of shape (E, num_rbf)
        """
        # Cite debug_lesson_16: Request Scalar Distances Directly to Prevent Empty Array Shape Errors
        # Handle vector distances (E, 3) by computing norm to ensure robustness
        if distance.dim() == 2 and distance.size(1) == 3:
            distance = torch.norm(distance, dim=1)

        if distance.dim() == 1:
            distance = distance.unsqueeze(1)

        return torch.exp(-((distance - self.centers) ** 2) / (self.sigma**2))


class CGCNNLayer(MessagePassing):
    """
    Crystal Graph Convolutional Neural Network Layer.
    Uses static edge features and gated activation.
    """

    def __init__(self, node_dim, edge_dim):
        super(CGCNNLayer, self).__init__(
            aggr="add"
        )  # 'add' aggregation as per CGCNN paper

        # Input dimension for message function: node_i + node_j + edge_attr
        msg_input_dim = 2 * node_dim + edge_dim

        # Linear transformations for filter and core parts of the message
        self.linear_filter = nn.Linear(msg_input_dim, node_dim)
        self.linear_core = nn.Linear(msg_input_dim, node_dim)

        # Batch Normalization
        self.bn_filter = nn.BatchNorm1d(node_dim)
        self.bn_core = nn.BatchNorm1d(node_dim)

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: Node features (N, node_dim)
            edge_index: Graph connectivity (2, E)
            edge_attr: Edge features (E, edge_dim)
        """
        # Propagate messages
        # flow='source_to_target' is default
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        """
        Constructs messages:
        z_ij = Concat(x_i, x_j, e_ij)
        m_ij = Sigmoid(Linear_f(z_ij)) * Softplus(Linear_c(z_ij))
        """
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)

        # Filter part (gate)
        gated = self.linear_filter(z)
        gated = self.bn_filter(gated)
        gated = torch.sigmoid(gated)

        # Core part
        core = self.linear_core(z)
        core = self.bn_core(core)
        core = F.softplus(core)

        return gated * core

    def update(self, aggr_out, x):
        """
        Update node features: x' = x + sum(messages)
        """
        # Residual connection
        return x + aggr_out


class MSR_CGCNN(nn.Module):
    """
    Multi-Scale Readout Crystal Graph Convolutional Network.

    Features:
    - RBF Edge Expansion
    - Stacked CGCNN Layers
    - Jumping Knowledge (JK) via Multi-Scale Pooling
    - Decoupled Global Feature Stream
    - Late Fusion
    """

    def __init__(self, config=Config):
        super(MSR_CGCNN, self).__init__()

        # Hyperparameters
        self.node_dim = config.HIDDEN_CHANNELS
        self.edge_dim = config.NUM_RBF
        self.num_layers = config.NUM_INTERACTION_LAYERS
        self.global_dim = len(config.GLOBAL_FEATURES)
        self.dropout = config.DROPOUT_RATE

        # 1. Node Embedding
        # Assuming atomic numbers up to ~100. 118 is safe upper bound.
        self.embedding = nn.Embedding(118, self.node_dim)

        # 2. Edge Featurization
        self.rbf = RBFExpansion(
            min_dist=0.0, max_dist=config.NEIGHBOR_CUTOFF, num_rbf=self.edge_dim
        )

        # 3. Graph Backbone (CGCNN Layers)
        self.conv_layers = nn.ModuleList(
            [CGCNNLayer(self.node_dim, self.edge_dim) for _ in range(self.num_layers)]
        )

        # 4. Global Feature Stream
        # Simple MLP to process lattice/composition info
        self.global_mlp = nn.Sequential(
            nn.Linear(self.global_dim, self.node_dim),
            nn.BatchNorm1d(self.node_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.node_dim, self.node_dim),
            nn.ReLU(),
        )

        # 5. Fusion & Readout
        # Input to final MLP is: (Num_Layers * Node_Dim) + Global_Emb_Dim
        # Because we pool every layer (JK) and concat global features
        fusion_dim = (self.num_layers * self.node_dim) + self.node_dim

        self.readout_mlp = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.Softplus(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.Softplus(),
            nn.Dropout(self.dropout),
            nn.Linear(128, len(config.TARGET_COLS)),
        )

    def forward(self, data):
        """
        Forward pass.

        Args:
            data: torch_geometric Data object containing:
                - x: Atomic numbers (N,)
                - edge_index: (2, E)
                - edge_attr: Distances (E,)
                - global_x: Global features (B, G)
                - batch: Batch indices (N,)
        """
        # 1. Embed Nodes
        # x shape: (N, node_dim)
        x = self.embedding(data.x)

        # 2. Expand Edges
        # edge_attr shape: (E, edge_dim)
        edge_attr = self.rbf(data.edge_attr)

        # 3. Process Global Features
        # global_emb shape: (B, node_dim)
        global_emb = self.global_mlp(data.global_x)

        # 4. Message Passing with Jumping Knowledge
        layer_outputs = []

        for conv in self.conv_layers:
            x = conv(x, data.edge_index, edge_attr)
            # Apply Softplus activation between layers as per standard CGCNN implementations
            # Note: The layer update itself is residual (x + m), here we activate the result
            x = F.softplus(x)

            # Store pooled representation of this scale
            # pooled shape: (B, node_dim)
            pooled = global_mean_pool(x, data.batch)
            layer_outputs.append(pooled)

        # 5. Multi-Scale Fusion
        # Concatenate all layer outputs: (B, num_layers * node_dim)
        graph_emb = torch.cat(layer_outputs, dim=1)

        # Late Fusion with Global Features
        # total_emb shape: (B, fusion_dim)
        total_emb = torch.cat([graph_emb, global_emb], dim=1)

        # 6. Final Prediction
        out = self.readout_mlp(total_emb)

        return out
