import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import softmax
from library.config import Config
from library.utils import GaussianRBF


class EdgeUpdateGATLayer(MessagePassing):
    """
    A Graph Attention Layer that performs dynamic edge feature updates.

    The layer operates in two steps:
    1. Edge Update: e'_ij = MLP(e_ij || x_i || x_j)
    2. Message Passing: x'_i = sum(alpha_ij * W * x_j), where alpha_ij is derived from e'_ij
    """

    def __init__(self, hidden_dim, num_heads, dropout):
        # Set aggr='add' because we perform the weighted sum manually in message()
        # via attention, but standard GAT implementation often uses 'add' aggregation
        # of weighted messages.
        super().__init__(aggr="add", node_dim=0)

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout

        # MLP for updating edge features based on incident nodes and current edge features
        # Input: Node_i (H) + Node_j (H) + Edge_ij (H) = 3*H
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Attention mechanism: Projects updated edge features to attention logits
        # We produce one score per head
        self.att_linear = nn.Linear(hidden_dim, num_heads)

        # Linear transformation for the node features (Value in attention)
        self.msg_linear = nn.Linear(hidden_dim, hidden_dim)

        # Normalization for edge update residual
        self.norm_edge = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: Node features [N, hidden_dim]
            edge_index: Graph connectivity [2, E]
            edge_attr: Edge features [E, hidden_dim]

        Returns:
            x_new: Updated node features [N, hidden_dim]
            edge_attr_new: Updated edge features [E, hidden_dim]
        """
        row, col = edge_index

        # 1. Dynamic Edge Update
        # Concatenate source node, target node, and existing edge features
        edge_input = torch.cat([x[row], x[col], edge_attr], dim=-1)

        # Compute new edge features
        edge_update = self.edge_mlp(edge_input)

        # Residual connection and Normalization for edges
        edge_attr_new = self.norm_edge(edge_attr + edge_update)

        # 2. Message Propagation
        # Propagate uses the *updated* edge attributes to compute attention
        x_new = self.propagate(edge_index, x=x, edge_attr=edge_attr_new)

        return x_new, edge_attr_new

    def message(self, x_j, edge_attr, index):
        """
        Computes the message for each edge.

        Args:
            x_j: Features of neighbor nodes [E, hidden_dim]
            edge_attr: Updated edge features [E, hidden_dim]
            index: Target node indices [E] (used for softmax normalization)
        """
        # Compute attention logits from edge features: [E, num_heads]
        alpha_logits = self.att_linear(edge_attr)

        # Apply LeakyReLU (standard GAT practice, though optional here since input is MLP)
        alpha_logits = F.leaky_relu(alpha_logits, negative_slope=0.2)

        # Normalize attention scores using softmax over neighbors
        # index corresponds to the target node 'i'
        alpha = softmax(alpha_logits, index)

        # Apply dropout to attention weights
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Transform neighbor node features: [E, hidden_dim]
        msg = self.msg_linear(x_j)

        # Reshape for multi-head attention: [E, num_heads, head_dim]
        msg = msg.view(-1, self.num_heads, self.head_dim)

        # Reshape alpha for broadcasting: [E, num_heads, 1]
        alpha = alpha.unsqueeze(-1)

        # Weighted message: [E, num_heads, head_dim]
        weighted_msg = msg * alpha

        # Flatten heads back to hidden_dim: [E, hidden_dim]
        return weighted_msg.view(-1, self.hidden_dim)


class EUGAT(nn.Module):
    """
    Edge-Updated Graph Attention Network.
    """

    def __init__(self):
        super().__init__()

        self.hidden_dim = Config.HIDDEN_DIM
        self.num_layers = Config.NUM_LAYERS
        self.num_heads = Config.NUM_HEADS
        self.dropout = Config.DROPOUT

        # 1. Node Embedding
        # Maps atomic numbers to continuous vectors
        # Assuming max atomic number < 100 (In is 49)
        self.node_emb = nn.Embedding(100, self.hidden_dim)

        # 2. Edge Embedding
        # Expands scalar distances using RBF
        self.rbf = GaussianRBF(
            start=0.0, stop=Config.CUTOFF_RADIUS, n_centers=Config.NUM_RBF
        )
        # Projects RBF features to hidden dimension
        self.edge_emb = nn.Linear(Config.NUM_RBF, self.hidden_dim)

        # 3. Interaction Layers
        self.layers = nn.ModuleList(
            [
                EdgeUpdateGATLayer(self.hidden_dim, self.num_heads, self.dropout)
                for _ in range(self.num_layers)
            ]
        )

        # Layer Norms for node residual connections
        self.norms = nn.ModuleList(
            [nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)]
        )

        # 4. Output Head
        self.out_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, len(Config.TARGET_COLS)),
        )

    def forward(self, data):
        """
        Forward pass of the model.

        Args:
            data: PyG Data object containing x, edge_index, edge_attr, batch
        """
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Initial Node Embeddings
        x = self.node_emb(x)  # [N, hidden_dim]

        # Initial Edge Embeddings
        # edge_attr is [E, 1] (distances), squeeze to [E]
        edge_attr = self.rbf(edge_attr.squeeze(-1))  # [E, num_rbf]
        edge_attr = self.edge_emb(edge_attr)  # [E, hidden_dim]

        # Message Passing Layers
        for layer, norm in zip(self.layers, self.norms):
            x_res = x

            # Apply layer: updates both nodes and edges
            x_updated, edge_attr_updated = layer(x, edge_index, edge_attr)

            # Update edge features for the next layer
            edge_attr = edge_attr_updated

            # Residual connection for nodes
            x = x_res + F.dropout(x_updated, p=self.dropout, training=self.training)
            x = norm(x)

        # Global Pooling
        # Aggregate node features to graph representation
        x_graph = global_mean_pool(x, batch)  # [batch_size, hidden_dim]

        # Prediction Head
        out = self.out_mlp(x_graph)  # [batch_size, num_targets]

        return out
