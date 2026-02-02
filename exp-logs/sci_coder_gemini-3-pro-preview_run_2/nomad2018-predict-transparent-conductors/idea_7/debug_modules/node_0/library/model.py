import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from library.config import (
    NUM_ATOM_TYPES,
    HIDDEN_DIM,
    N_GCN_LAYERS,
    RBF_BINS,
    RBF_GAMMA,
    DROPOUT,
    NUM_GLOBAL_FEATURES,
    NUM_TARGETS,
    CUTOFF_RADIUS,
)


class GaussianSmearing(nn.Module):
    """
    Expands distances into a vector of Radial Basis Functions (RBFs).
    """

    def __init__(
        self, start=0.0, stop=CUTOFF_RADIUS, n_gaussians=RBF_BINS, gamma=RBF_GAMMA
    ):
        super().__init__()
        offset = torch.linspace(start, stop, n_gaussians)
        self.coeff = -0.5 / ((stop - start) / (n_gaussians - 1)) ** 2
        self.register_buffer("offset", offset)
        # Gamma can be learned or fixed. Here we use the config fixed value logic
        # but adapted to the width formulation often used in CGCNN.
        # However, to strictly follow the config RBF_GAMMA if it implies a specific width:
        # We will use the provided gamma directly if it's meant to be the coefficient.
        # Assuming RBF_GAMMA is the width parameter (sigma) or similar, but standard
        # CGCNN implementation calculates coeff based on bin spacing.
        # We will stick to the standard CGCNN formulation using the spacing.

    def forward(self, dist):
        # dist: [num_edges, 1]
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class CGCNNConv(MessagePassing):
    """
    Gated Crystal Graph Convolutional Layer.
    """

    def __init__(self, node_dim, edge_dim):
        super().__init__(aggr="add")  # "Add" aggregation is standard for CGCNN

        # Input dimension: node_i + node_j + edge_attr
        self.input_dim = 2 * node_dim + edge_dim

        # Linear transformation for the message
        # We output 2 * node_dim to split into filter and core/gate parts
        self.lin = nn.Linear(self.input_dim, 2 * node_dim)
        self.bn = nn.BatchNorm1d(2 * node_dim)

    def forward(self, x, edge_index, edge_attr):
        # x: [num_nodes, node_dim]
        # edge_index: [2, num_edges]
        # edge_attr: [num_edges, edge_dim]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # Concatenate features
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)

        # Transform
        z = self.lin(z)
        z = self.bn(z)

        # Split into filter and core/gate
        filter_part, core_part = z.chunk(2, dim=-1)

        # Gated activation: Sigmoid(filter) * Softplus(core)
        # Note: Original CGCNN uses Sigmoid * Softplus (or Tanh)
        return torch.sigmoid(filter_part) * F.softplus(core_part)


class DSGCN(nn.Module):
    """
    Dual-Stream Geometric-Compositional Network.

    Stream 1 (Local): CGCNN on the crystal graph.
    Stream 2 (Global): MLP on lattice parameters and composition.
    Fusion: Concatenation -> MLP -> Targets.
    """

    def __init__(self):
        super().__init__()

        # --- Local Geometric Stream ---
        # Node embedding
        self.node_embedding = nn.Embedding(NUM_ATOM_TYPES, HIDDEN_DIM)

        # Edge expansion
        self.rbf = GaussianSmearing(start=0.0, stop=CUTOFF_RADIUS, n_gaussians=RBF_BINS)

        # GCN Layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(N_GCN_LAYERS):
            self.convs.append(CGCNNConv(node_dim=HIDDEN_DIM, edge_dim=RBF_BINS))
            self.bns.append(nn.BatchNorm1d(HIDDEN_DIM))

        # --- Global Compositional Stream ---
        # Input: NUM_GLOBAL_FEATURES (10)
        # We project this to HIDDEN_DIM to match the graph embedding size roughly
        self.global_mlp = nn.Sequential(
            nn.Linear(NUM_GLOBAL_FEATURES, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.ReLU(),
        )

        # --- Fusion & Readout ---
        # Input: Graph Embedding (HIDDEN_DIM) + Global Embedding (HIDDEN_DIM)
        fusion_dim = HIDDEN_DIM * 2

        self.readout_mlp = nn.Sequential(
            nn.Linear(fusion_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM // 2, NUM_TARGETS),
        )

    def forward(self, data):
        """
        Args:
            data (torch_geometric.data.Data): Batch of crystal graphs.
                - x: Node features (atomic indices)
                - edge_index: Graph connectivity
                - edge_attr: Edge distances
                - global_feat: Global features [batch_size, 1, num_global_features]
                - batch: Batch vector mapping nodes to graphs
        """
        # --- Local Stream ---
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # 1. Embed nodes
        x = self.node_embedding(x)  # [num_nodes, HIDDEN_DIM]

        # 2. Expand edge distances
        edge_feat = self.rbf(edge_attr)  # [num_edges, RBF_BINS]

        # 3. Message Passing
        for conv, bn in zip(self.convs, self.bns):
            # Residual connection
            x_update = conv(x, edge_index, edge_feat)
            x_update = bn(x_update)
            x = x + F.dropout(x_update, p=DROPOUT, training=self.training)

        # 4. Global Pooling
        graph_embedding = global_mean_pool(x, batch)  # [batch_size, HIDDEN_DIM]

        # --- Global Stream ---
        # data.global_feat is [batch_size, 1, 10], squeeze to [batch_size, 10]
        global_input = data.global_feat.squeeze(1)
        global_embedding = self.global_mlp(global_input)  # [batch_size, HIDDEN_DIM]

        # --- Fusion ---
        # Concatenate graph and global representations
        fused = torch.cat(
            [graph_embedding, global_embedding], dim=1
        )  # [batch_size, 2*HIDDEN_DIM]

        # --- Readout ---
        out = self.readout_mlp(fused)  # [batch_size, NUM_TARGETS]

        return out
