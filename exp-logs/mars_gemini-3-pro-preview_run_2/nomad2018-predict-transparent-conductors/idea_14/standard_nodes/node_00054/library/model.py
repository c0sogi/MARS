import torch
import torch.nn as nn
from torch_geometric.nn import CGConv, global_mean_pool
from library.config import Config


class InteractionBlock(nn.Module):
    """
    Interaction Block consisting of:
    1. Gated Graph Convolution (CGConv) with Batch Normalization
    2. Activation and Dropout

    Note: CGConv internally adds the residual connection x + message.
    Cite solution_lesson_node_00050: Avoid Dense FFN Expansions in GNN Blocks for Small Datasets.
    """

    def __init__(self, hidden_dim, dropout_rate):
        super(InteractionBlock, self).__init__()

        # Gated Graph Convolution
        # channels: Node feature dimension
        # dim: Edge feature dimension (projected RBF)
        # batch_norm: Whether to apply BN inside CGConv
        self.conv = CGConv(channels=hidden_dim, dim=hidden_dim, batch_norm=True)

        # Activation and Dropout
        self.activation = nn.Softplus()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, edge_attr):
        # CGConv returns x_i + aggregated_messages
        x = self.conv(x, edge_index, edge_attr)

        # Apply activation and dropout
        x = self.activation(x)
        x = self.dropout(x)
        return x


class CGCNN(nn.Module):
    """
    Crystal Graph Convolutional Network (CGCNN).

    Architecture:
    - Node Embedding (Atomic Numbers)
    - Edge Embedding (Linear projection of RBF features) - Cite solution_lesson_node_00042
    - Stack of Interaction Blocks (CGConv)
    - Global Mean Pooling
    - Decoupled Prediction Heads for Formation and Bandgap Energy
    """

    def __init__(self, config=Config):
        super(CGCNN, self).__init__()

        # Hyperparameters from Config
        self.atom_embedding_dim = config.ATOM_EMBEDDING_DIM
        self.edge_embedding_dim = config.EDGE_EMBEDDING_DIM
        self.num_rbf = config.NUM_RBF_BINS
        self.num_blocks = config.NUM_INTERACTION_BLOCKS
        self.dropout_rate = config.DROPOUT_RATE

        # 1. Node Embedding
        self.node_embedding = nn.Embedding(100, self.atom_embedding_dim)

        # 2. Edge Embedding
        # Projects RBF-expanded edge features to the embedding dimension
        self.edge_embedding = nn.Linear(self.num_rbf, self.edge_embedding_dim)

        # 3. Interaction Backbone
        self.blocks = nn.ModuleList(
            [
                InteractionBlock(
                    hidden_dim=self.atom_embedding_dim,
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
