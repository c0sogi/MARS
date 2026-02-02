import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv, global_max_pool, global_mean_pool
from library.config import Config


class IceCubeDGCN(nn.Module):
    """
    Dynamic Graph Convolutional Network for Neutrino Direction Prediction.

    Treats the neutrino event as a point cloud of pulses. Uses Dynamic Edge Convolution
    to learn local geometric features that evolve with network depth, followed by
    global pooling and an MLP head to regress the 3D direction vector.
    """

    def __init__(self):
        super(IceCubeDGCN, self).__init__()

        # Hyperparameters from Config
        self.k = Config.K_NEIGHBORS
        self.input_dim = Config.INPUT_DIM
        self.hidden_dim = Config.HIDDEN_DIM
        self.output_dim = Config.OUTPUT_DIM
        self.num_layers = Config.NUM_LAYERS
        self.dropout_rate = Config.DROPOUT

        self.convs = nn.ModuleList()

        # --- Layer 1: Input Projection & EdgeConv ---
        # DynamicEdgeConv expects an MLP h(cat(x_i, x_j - x_i))
        # Input to MLP is 2 * in_channels (concatenation of node feature and edge feature)
        mlp1 = nn.Sequential(
            nn.Linear(2 * self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.convs.append(DynamicEdgeConv(mlp1, k=self.k, aggr="max"))

        # --- Layers 2 to N: Deep Feature Extraction ---
        for _ in range(self.num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.LeakyReLU(0.2),
            )
            self.convs.append(DynamicEdgeConv(mlp, k=self.k, aggr="max"))

        # --- Global Pooling & Prediction Head ---
        # We concatenate Global Max and Global Mean pooling
        self.pool_dim = self.hidden_dim * 2

        self.head = nn.Sequential(
            nn.Linear(self.pool_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.BatchNorm1d(self.hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim // 2, self.output_dim),
        )

    def forward(self, data):
        """
        Forward pass of the DGCN.

        Args:
            data: PyG Data or Batch object containing:
                - x (Tensor): Node features [num_nodes, input_dim]
                - batch (Tensor): Batch indices [num_nodes] mapping nodes to graphs

        Returns:
            Tensor: Predicted 3D direction vectors [batch_size, 3]
        """
        x, batch = data.x, data.batch

        # 1. Dynamic Graph Convolutions
        # First layer (Input -> Hidden)
        x = self.convs[0](x, batch)

        # Subsequent layers (Hidden -> Hidden) with Residual Connections
        for i in range(1, self.num_layers):
            identity = x
            x = self.convs[i](x, batch)
            x = x + identity  # Residual connection

        # 2. Global Pooling
        # Aggregate node features to graph-level features
        x_max = global_max_pool(x, batch)
        x_mean = global_mean_pool(x, batch)

        # Concatenate max and mean features
        x_pool = torch.cat([x_max, x_mean], dim=1)

        # 3. Prediction Head
        out = self.head(x_pool)

        return out


def cosine_similarity_loss(pred, target):
    """
    Computes the Cosine Similarity Loss: 1 - mean(cosine_similarity(pred, target)).
    Optimizes the angular alignment between predicted and true vectors.

    Args:
        pred (Tensor): Predicted vectors of shape [batch_size, 3].
        target (Tensor): Ground truth unit vectors of shape [batch_size, 3].

    Returns:
        Tensor: Scalar loss value.
    """
    # Normalize predictions to unit vectors
    pred_norm = F.normalize(pred, p=2, dim=1)

    # Normalize targets (ensure they are unit vectors)
    target_norm = F.normalize(target, p=2, dim=1)

    # Compute cosine similarity: (a . b) / (|a| |b|)
    # Since vectors are normalized, this is just the dot product
    cos_sim = torch.sum(pred_norm * target_norm, dim=1)

    # Loss = 1 - Average Cosine Similarity
    # Range: [0, 2]. 0 means perfect alignment, 2 means opposite direction.
    loss = 1.0 - cos_sim.mean()

    return loss
