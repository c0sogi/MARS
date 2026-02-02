import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, global_mean_pool, global_max_pool, knn_graph
from library.config import Config


class DFCGN(nn.Module):
    """
    Dual-Frame Causal Graph Network (DF-CGN).

    A Graph Neural Network that fuses raw and canonical coordinate frames
    and utilizes a spatiotemporal k-NN topology to predict neutrino direction.
    """

    def __init__(self):
        super(DFCGN, self).__init__()

        # Hyperparameters
        self.k = Config.K_NEIGHBORS
        self.alpha = Config.TIME_SCALE_ALPHA
        self.hidden_dim = Config.HIDDEN_DIM
        self.dropout = Config.DROPOUT

        # ---------------------------------------------------------
        # 1. Input Embedding
        # ---------------------------------------------------------
        # Fuses Raw (x,y,z,t), Canonical (x',y',z'), and Features (q, aux)
        # Input shape: (Batch * Pulses, 9)
        self.embedding = nn.Sequential(
            nn.Linear(Config.IN_CHANNELS, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
        )

        # ---------------------------------------------------------
        # 2. Backbone: Dynamic Edge Convolution Layers
        # ---------------------------------------------------------
        self.conv_layers = nn.ModuleList()

        for _ in range(Config.NUM_LAYERS):
            # The neural network used within EdgeConv.
            # EdgeConv inputs are [h_i, h_j - h_i], so input dim is 2 * hidden_dim.
            # This implicitly captures the "Global Context Feature" (z'_j - z'_i)
            # because z' (canonical z) is part of the embedded features.
            nn_conv = nn.Sequential(
                nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(),
            )

            # aggr='max' is standard for PointNet/DGCNN architectures
            self.conv_layers.append(EdgeConv(nn=nn_conv, aggr="max"))

        # ---------------------------------------------------------
        # 3. Global Pooling & Predictor Head
        # ---------------------------------------------------------
        # Concatenate Max and Mean pooling for robust global representation
        self.pool_dim = self.hidden_dim * 2

        self.head = nn.Sequential(
            nn.Linear(self.pool_dim, Config.LATENT_DIM),
            nn.BatchNorm1d(Config.LATENT_DIM),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(Config.LATENT_DIM, 3),  # Outputs (nx, ny, nz) vector
        )

    def forward(self, data):
        """
        Forward pass of the DF-CGN.

        Args:
            data (torch_geometric.data.Data): Batch containing:
                - x (Tensor): Node features of shape (Total_Pulses, 9).
                              Indices: 0-2 (raw pos), 3 (time), 4-6 (canonical pos), 7 (charge), 8 (aux).
                - batch (Tensor): Batch indices of shape (Total_Pulses,).

        Returns:
            torch.Tensor: Predicted direction vectors of shape (Batch_Size, 3).
        """
        x, batch = data.x, data.batch

        # ---------------------------------------------------------
        # A. Causal Graph Construction (Spatiotemporal k-NN)
        # ---------------------------------------------------------
        # We construct the graph topology based on physical spacetime proximity.
        # Distance metric: D^2 = |dx|^2 + alpha * |dt|^2

        # Extract Raw Position (x, y, z) -> Indices 0, 1, 2
        raw_pos = x[:, 0:3]

        # Extract Time (t) -> Index 3
        time = x[:, 3:4]

        # Scale time component
        # We scale t by sqrt(alpha) so that Euclidean distance on the concatenated
        # vector corresponds to the spatiotemporal metric.
        t_scaled = time * (self.alpha**0.5)

        # Concatenate to form the metric space coordinates
        st_coords = torch.cat([raw_pos, t_scaled], dim=1)

        # Compute k-NN graph dynamically
        # This connects pulses that are close in space AND time (causally linked)
        edge_index = knn_graph(st_coords, k=self.k, batch=batch)

        # ---------------------------------------------------------
        # B. Feature Encoding & Message Passing
        # ---------------------------------------------------------
        # Embed input features
        h = self.embedding(x)

        # Pass through DynEdge layers
        # We use the same causal topology for all layers
        for conv in self.conv_layers:
            # Apply convolution
            h_new = conv(h, edge_index)

            # Residual Connection
            h = h + h_new

        # ---------------------------------------------------------
        # C. Global Aggregation & Prediction
        # ---------------------------------------------------------
        # Global Pooling
        h_max = global_max_pool(h, batch)
        h_mean = global_mean_pool(h, batch)

        # Concatenate representations
        h_pool = torch.cat([h_max, h_mean], dim=1)

        # Regress direction vector
        out = self.head(h_pool)

        return out
