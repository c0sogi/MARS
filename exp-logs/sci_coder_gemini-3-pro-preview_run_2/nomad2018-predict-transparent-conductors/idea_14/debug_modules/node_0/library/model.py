import torch
import torch.nn as nn
from torch_geometric.nn import CGConv, global_mean_pool
from library.config import Config


class InteractionBlock(nn.Module):
    """
    Interaction Block consisting of:
    1. Gated Graph Convolution (CGConv) with Batch Normalization
    2. Inverted Bottleneck Feed-Forward Network (FFN)
    3. Residual Connection
    """

    def __init__(self, hidden_dim, ffn_hidden_dim, dropout_rate):
        super(InteractionBlock, self).__init__()

        # Gated Graph Convolution
        # channels: Node feature dimension
        # dim: Edge feature dimension (projected RBF)
        # batch_norm: Whether to apply BN inside CGConv (True per description)
        self.conv = CGConv(channels=hidden_dim, dim=hidden_dim, batch_norm=True)

        # Inverted Bottleneck FFN: Linear -> Act -> Linear
        # Projects features to a higher dimension and back
        self.ffn_linear1 = nn.Linear(hidden_dim, ffn_hidden_dim)
        self.ffn_act = nn.Softplus()
        self.ffn_linear2 = nn.Linear(ffn_hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # Final activation after residual addition
        self.final_act = nn.Softplus()

    def forward(self, x, edge_index, edge_attr):
        # 1. Gated Convolution + BN
        # x: [num_nodes, hidden_dim]
        x_conv = self.conv(x, edge_index, edge_attr)

        # 2. Inverted Bottleneck FFN
        out = self.ffn_linear1(x_conv)
        out = self.ffn_act(out)
        out = self.ffn_linear2(out)
        out = self.dropout(out)

        # 3. Residual Connection
        # The input to the block (x) is added to the output of the FFN
        x_out = self.final_act(x + out)

        return x_out


class CGCNN_IB(nn.Module):
    """
    Crystal Graph Convolutional Network with Inverted Bottleneck FFN (CGCNN-IB).

    Architecture:
    - Node Embedding (Atomic Numbers)
    - Edge Embedding (Linear projection of RBF features)
    - Stack of Interaction Blocks (CGConv + FFN + Residual)
    - Global Mean Pooling
    - Decoupled Prediction Heads for Formation and Bandgap Energy
    """

    def __init__(self, config=Config):
        super(CGCNN_IB, self).__init__()

        # Hyperparameters from Config
        self.atom_embedding_dim = config.ATOM_EMBEDDING_DIM
        self.edge_embedding_dim = config.EDGE_EMBEDDING_DIM
        self.num_rbf = config.NUM_RBF_BINS
        self.num_blocks = config.NUM_INTERACTION_BLOCKS
        self.ffn_hidden = config.FFN_HIDDEN_DIM
        self.dropout_rate = config.DROPOUT_RATE

        # 1. Node Embedding
        # Maps atomic numbers to learnable vectors
        # 100 is a safe upper bound for atomic numbers in standard datasets
        self.node_embedding = nn.Embedding(100, self.atom_embedding_dim)

        # 2. Edge Embedding
        # Projects RBF-expanded edge features to the embedding dimension
        self.edge_embedding = nn.Linear(self.num_rbf, self.edge_embedding_dim)

        # 3. Interaction Backbone
        self.blocks = nn.ModuleList(
            [
                InteractionBlock(
                    hidden_dim=self.atom_embedding_dim,
                    ffn_hidden_dim=self.ffn_hidden,
                    dropout_rate=self.dropout_rate,
                )
                for _ in range(self.num_blocks)
            ]
        )

        # 4. Readout Heads
        # Decoupled Multi-Layer Perceptrons for each target

        # Head for Formation Energy
        self.head_formation = nn.Sequential(
            nn.Linear(self.atom_embedding_dim, 128),
            nn.Softplus(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 64),
            nn.Softplus(),
            nn.Linear(64, 1),
        )

        # Head for Bandgap Energy
        self.head_bandgap = nn.Sequential(
            nn.Linear(self.atom_embedding_dim, 128),
            nn.Softplus(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 64),
            nn.Softplus(),
            nn.Linear(64, 1),
        )

    def forward(self, data):
        """
        Forward pass of the model.

        Args:
            data: PyG Data object containing:
                - x: Atomic numbers [num_nodes]
                - edge_index: Graph connectivity [2, num_edges]
                - edge_attr: RBF expanded edge distances [num_edges, num_rbf]
                - batch: Batch vector mapping nodes to graphs [num_nodes]

        Returns:
            torch.Tensor: Predictions of shape [batch_size, 2]
        """
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Embed Nodes
        x = self.node_embedding(x)

        # Embed Edges (Project RBF features)
        edge_attr = self.edge_embedding(edge_attr)

        # Apply Interaction Blocks
        for block in self.blocks:
            x = block(x, edge_index, edge_attr)

        # Global Pooling (Mean)
        x_pool = global_mean_pool(x, batch)

        # Predict Targets independently
        out_formation = self.head_formation(x_pool)
        out_bandgap = self.head_bandgap(x_pool)

        # Concatenate results
        return torch.cat([out_formation, out_bandgap], dim=1)
