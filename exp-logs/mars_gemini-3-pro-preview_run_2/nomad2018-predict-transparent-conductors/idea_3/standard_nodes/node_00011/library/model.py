import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import softmax
from library.config import Config


class GaussianRBF(nn.Module):
    """
    Expands scalar distances into a vector of Gaussian Radial Basis Functions.
    This provides a continuous, high-resolution representation of interatomic distances.
    """

    def __init__(self, num_bins, cutoff):
        super().__init__()
        self.num_bins = num_bins
        self.cutoff = cutoff
        # Register centers as buffers so they are part of the state dict but not trainable parameters
        # Linearly spaced centers from 0 to cutoff
        centers = torch.linspace(0, cutoff, num_bins)
        self.register_buffer("centers", centers)

        # Width (sigma) determined by the spacing between centers
        if num_bins > 1:
            width = centers[1] - centers[0]
        else:
            width = 1.0
        # gamma = 1 / (2 * sigma^2), approximating sigma with width
        self.gamma = 1.0 / (width**2)

    def forward(self, distances):
        """
        Args:
            distances: Tensor of shape (num_edges,) representing edge lengths.
        Returns:
            Tensor of shape (num_edges, num_bins)
        """
        # Expand dimensions for broadcasting: (E, 1) - (1, Bins) -> (E, Bins)
        diff = distances.unsqueeze(1) - self.centers.unsqueeze(0)
        return torch.exp(-self.gamma * (diff**2))


class DistanceBiasedAttention(MessagePassing):
    """
    Multi-Head Self-Attention mechanism with Geometric Bias.
    The attention scores are biased by a learnable projection of the edge distance RBFs.
    Score_ij = (Q_i . K_j) / sqrt(d) + Bias(e_ij)
    """

    def __init__(self, embed_dim, num_heads, rbf_bins):
        super().__init__(aggr="add", node_dim=0)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dim must be divisible by num_heads"

        # Projections for Query, Key, Value
        self.lin_q = nn.Linear(embed_dim, embed_dim)
        self.lin_k = nn.Linear(embed_dim, embed_dim)
        self.lin_v = nn.Linear(embed_dim, embed_dim)

        # Projection for edge geometric bias
        # Projects RBF features to one bias value per head
        self.lin_edge_bias = nn.Linear(rbf_bins, num_heads)

        # Output projection
        self.lin_out = nn.Linear(embed_dim, embed_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_q.weight)
        nn.init.xavier_uniform_(self.lin_k.weight)
        nn.init.xavier_uniform_(self.lin_v.weight)
        nn.init.xavier_uniform_(self.lin_edge_bias.weight)
        nn.init.xavier_uniform_(self.lin_out.weight)
        # Initialize biases to zero
        if self.lin_q.bias is not None:
            nn.init.zeros_(self.lin_q.bias)
        if self.lin_k.bias is not None:
            nn.init.zeros_(self.lin_k.bias)
        if self.lin_v.bias is not None:
            nn.init.zeros_(self.lin_v.bias)
        if self.lin_edge_bias.bias is not None:
            nn.init.zeros_(self.lin_edge_bias.bias)
        if self.lin_out.bias is not None:
            nn.init.zeros_(self.lin_out.bias)

    def forward(self, x, edge_index, edge_rbf):
        """
        Args:
            x: Node features (N, embed_dim)
            edge_index: Graph connectivity (2, E)
            edge_rbf: Edge RBF features (E, rbf_bins)
        """
        # Propagate messages.
        # x is passed to message() where it can be indexed as x_i (target) and x_j (source)
        out = self.propagate(edge_index, x=x, edge_rbf=edge_rbf)

        # Final linear projection
        out = self.lin_out(out)
        return out

    def message(self, x_i, x_j, edge_rbf, index, ptr, size_i):
        """
        Computes the attention scores and values for each edge.

        Args:
            x_i: Target node features (E, embed_dim)
            x_j: Source node features (E, embed_dim)
            edge_rbf: (E, rbf_bins)
            index: Target node indices (for softmax normalization)
            ptr, size_i: Helper arguments for softmax
        """
        # Calculate Q, K, V
        # Reshape to (E, num_heads, head_dim)
        q_i = self.lin_q(x_i).view(-1, self.num_heads, self.head_dim)
        k_j = self.lin_k(x_j).view(-1, self.num_heads, self.head_dim)
        v_j = self.lin_v(x_j).view(-1, self.num_heads, self.head_dim)

        # Calculate geometric bias: (E, num_heads)
        # This corresponds to Bias(e_ij) in the formula
        bias_ij = self.lin_edge_bias(edge_rbf)

        # Attention score: (Q . K) / sqrt(d)
        # (E, heads, head_dim) * (E, heads, head_dim) -> sum over last dim -> (E, heads)
        score = (q_i * k_j).sum(dim=-1) / (self.head_dim**0.5)

        # Add geometric bias to the score
        score = score + bias_ij

        # Softmax over neighbors (for each head)
        # index is the index of the target node (i) for aggregation
        alpha = softmax(score, index, ptr, size_i)

        # Weighted sum of values
        # alpha: (E, heads) -> (E, heads, 1) for broadcasting
        # v_j: (E, heads, head_dim)
        out = alpha.unsqueeze(-1) * v_j

        # Flatten heads: (E, heads * head_dim) = (E, embed_dim)
        out = out.view(-1, self.embed_dim)

        return out


class GraphTransformerLayer(nn.Module):
    """
    A single layer of the Distance-Biased Graph Transformer.
    Consists of Distance-Biased Attention and a Feed-Forward Network (FFN),
    wrapped with Residual Connections and Layer Normalization.
    """

    def __init__(self, embed_dim, num_heads, rbf_bins, dropout=0.1):
        super().__init__()

        # Attention Block
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = DistanceBiasedAttention(embed_dim, num_heads, rbf_bins)
        self.dropout1 = nn.Dropout(dropout)

        # Feed Forward Block
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, edge_index, edge_rbf):
        # Pre-Norm Architecture

        # 1. Attention Block
        x_norm = self.norm1(x)
        attn_out = self.attention(x_norm, edge_index, edge_rbf)
        x = x + self.dropout1(attn_out)

        # 2. FFN Block
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out

        return x


class DBGT(nn.Module):
    """
    Distance-Biased Graph Transformer (DB-GT) Model.
    """

    def __init__(self, config=Config):
        super().__init__()

        self.embed_dim = config.EMBEDDING_DIM
        self.rbf_bins = config.RBF_BINS
        self.cutoff = config.CUTOFF_RADIUS

        # Node Embedding
        # +1 for safety/padding
        self.node_embedding = nn.Embedding(config.MAX_ATOMIC_NUMBER + 1, self.embed_dim)

        # Edge RBF Featurizer
        self.rbf = GaussianRBF(self.rbf_bins, self.cutoff)

        # Stack of Transformer Layers
        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    embed_dim=self.embed_dim,
                    num_heads=config.NUM_HEADS,
                    rbf_bins=self.rbf_bins,
                    dropout=config.DROPOUT,
                )
                for _ in range(config.N_LAYERS)
            ]
        )

        # Output Head
        # Predicts 2 targets: formation_energy and bandgap
        self.output_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.SiLU(),  # Swish activation
            nn.Linear(self.embed_dim, len(config.TARGET_COLS)),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, data):
        """
        Args:
            data: A dictionary or PyG Batch object containing:
                - x: Atomic numbers (N,)
                - edge_index: Graph connectivity (2, E)
                - edge_attr: Edge distances (E,)
                - batch: Batch indices for nodes (N,)
        """
        # Unpack data
        if isinstance(data, dict):
            x = data["x"]
            edge_index = data["edge_index"]
            edge_dists = data["edge_attr"]
            batch = data["batch"]
        else:
            x = data.x
            edge_index = data.edge_index
            edge_dists = data.edge_attr
            batch = data.batch

        # 1. Node Embedding
        h = self.node_embedding(x)  # (N, embed_dim)

        # 2. Edge Featurization (RBF)
        edge_rbf = self.rbf(edge_dists)  # (E, rbf_bins)

        # 3. Transformer Layers
        for layer in self.layers:
            h = layer(h, edge_index, edge_rbf)

        # 4. Global Pooling
        # Aggregate node features to graph features using mean pooling
        h_graph = global_mean_pool(h, batch)  # (batch_size, embed_dim)

        # 5. Output Prediction
        out = self.output_head(h_graph)  # (batch_size, 2)

        return out
