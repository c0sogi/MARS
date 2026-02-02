import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DilatedConvBackbone(nn.Module):
    """
    Dilated Residual Convolutional Network.
    Captures local and medium-range sequence motifs using dilated convolutions
    while preserving sequence length.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilations, dropout):
        super().__init__()
        # Initial projection to target channel dimension
        self.entry_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        self.layers = nn.ModuleList()
        for d in dilations:
            # Calculate padding to maintain sequence length
            # For kernel_size=3, padding = dilation
            padding = d * (kernel_size - 1) // 2

            conv = nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=d,
            )
            self.layers.append(conv)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        # Input: (Batch, Seq_Len, Channels)
        # Conv1d expects: (Batch, Channels, Seq_Len)
        x = x.permute(0, 2, 1)
        x = self.entry_conv(x)

        for conv in self.layers:
            res = x
            out = self.act(conv(x))
            out = self.dropout(out)
            x = out + res  # Residual connection

        # Return to (Batch, Seq_Len, Channels)
        return x.permute(0, 2, 1)


class GraphRefinementModule(nn.Module):
    """
    Graph Neural Network Module using Multi-Head Attention (GAT-style).
    Uses the structural adjacency matrix as an attention mask to restrict
    information flow to physically connected bases.
    """

    def __init__(
        self, in_channels, hidden_channels, out_channels, heads, layers, dropout
    ):
        super().__init__()

        # Project input features to GNN hidden dimension
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        self.layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for _ in range(layers):
            # MultiheadAttention handles the "Graph Attention" logic
            # when provided with a structural mask
            self.layers.append(
                nn.MultiheadAttention(
                    embed_dim=hidden_channels,
                    num_heads=heads,
                    dropout=dropout,
                    batch_first=True,
                )
            )
            self.norm_layers.append(nn.LayerNorm(hidden_channels))

        self.output_proj = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        """
        Args:
            x: Node features (Batch, Seq_Len, Channels)
            adj: Adjacency matrix (Batch, Seq_Len, Seq_Len) with 1.0 for edges
        """
        B, L, _ = x.shape

        # 1. Prepare Attention Mask from Adjacency Matrix
        # We need to ensure self-loops so a node can attend to itself
        eye = torch.eye(L, device=adj.device).unsqueeze(0).expand(B, -1, -1)
        adj_with_self = torch.clamp(adj + eye, max=1.0)

        # Convert binary adjacency to additive mask:
        # Edge (1.0) -> 0.0 (Attention allowed)
        # No Edge (0.0) -> -inf (Attention blocked)
        # Shape: (Batch, Seq_Len, Seq_Len)
        attn_mask = (1.0 - adj_with_self) * -1e9

        # 2. GNN Layers
        h = self.input_proj(x)

        for i, layer in enumerate(self.layers):
            res = h
            # attn_mask supports (Batch*NumHeads, L, L) or (Batch, L, L)
            # We pass (Batch, L, L). PyTorch broadcasts this across heads.
            out, _ = layer(h, h, h, attn_mask=attn_mask, need_weights=False)

            # Residual + Norm
            h = self.norm_layers[i](res + self.dropout(out))

        return self.output_proj(h)


class BiGRUHead(nn.Module):
    """
    Bidirectional GRU to capture global sequence dependencies.
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

    def forward(self, x):
        # Output: (Batch, Seq_Len, Hidden_Size * 2)
        out, _ = self.gru(x)
        return out


class GEHN(nn.Module):
    """
    Graph-Enhanced Hybrid Network (GEHN).
    Combines Dilated CNNs, Structure-aware Graph Attention, and BiGRUs.
    """

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = Config()

        # 1. Dilated CNN Backbone
        self.cnn = DilatedConvBackbone(
            in_channels=config.input_channels,
            out_channels=config.cnn_channels,
            kernel_size=config.cnn_kernel_size,
            dilations=config.cnn_dilations,
            dropout=config.dropout,
        )

        # 2. Graph Refinement (GNN)
        self.gnn = GraphRefinementModule(
            in_channels=config.cnn_channels,
            hidden_channels=config.gnn_hidden_channels,
            out_channels=config.gnn_out_channels,
            heads=config.gnn_heads,
            layers=config.gnn_layers,
            dropout=config.gnn_dropout,
        )

        # 3. BiGRU Head
        self.rnn = BiGRUHead(
            input_size=config.rnn_input_size,
            hidden_size=config.rnn_hidden_size,
            num_layers=config.rnn_layers,
            dropout=config.dropout,
        )

        # 4. Final Projection
        # BiGRU output is hidden_size * 2
        rnn_out_dim = config.rnn_hidden_size * 2
        self.final_proj = nn.Linear(rnn_out_dim, config.num_targets)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs, adj):
        """
        Args:
            inputs: (Batch, Seq_Len, Channels)
            adj: (Batch, Seq_Len, Seq_Len)
        Returns:
            logits: (Batch, Seq_Len, Num_Targets)
        """
        # 1. Extract local motifs
        x = self.cnn(inputs)

        # 2. Refine using structural connectivity
        x = self.gnn(x, adj)

        # 3. Aggregate global context
        x = self.rnn(x)

        # 4. Predict
        x = self.dropout(x)
        out = self.final_proj(x)

        return out
