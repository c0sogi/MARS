import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
import math
from library.config import Config


class SinusoidalPositionalEncoding(nn.Module):
    """
    Injects sinusoidal positional encodings into the input embeddings.
    Standard implementation as used in Transformers.
    """

    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer so it's part of state_dict but not a parameter
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Total_Nodes, d_model)
        Returns:
            Tensor of shape (Total_Nodes, d_model) with PE added.
        """
        # In the batched graph, nodes are stacked.
        # We assume each graph in the batch has exactly Config.SEQ_LEN nodes.
        # We generate indices [0, 1, ..., 106, 0, 1, ..., 106, ...]
        seq_len = Config.SEQ_LEN
        num_nodes = x.size(0)

        # Calculate position indices for the entire batch
        pos_indices = torch.arange(num_nodes, device=x.device) % seq_len

        # Retrieve encodings and add to input
        return x + self.pe[pos_indices]


class RNAGNN(nn.Module):
    """
    Relational Graph Neural Network for RNA Degradation Prediction.

    Architecture:
    1. Embeddings (Seq, Struct, Loop)
    2. Positional Encoding
    3. Stack of RGCNConv Layers (with Residuals, BN, ReLU, Dropout)
    4. MLP Decoder
    """

    def __init__(self):
        super().__init__()

        # 1. Embeddings
        self.seq_embed = nn.Embedding(Config.VOCAB_SIZE_SEQ, Config.EMBED_DIM)
        self.struct_embed = nn.Embedding(Config.VOCAB_SIZE_STRUCT, Config.EMBED_DIM)
        self.loop_embed = nn.Embedding(Config.VOCAB_SIZE_LOOP, Config.EMBED_DIM)

        # Input projection: 3 * EMBED_DIM -> HIDDEN_DIM
        input_dim = 3 * Config.EMBED_DIM
        self.input_proj = nn.Linear(input_dim, Config.HIDDEN_DIM)

        # 2. Positional Encoding
        self.pos_encoder = SinusoidalPositionalEncoding(
            Config.HIDDEN_DIM, max_len=Config.SEQ_LEN + 10
        )

        # 3. GNN Layers (RGCN)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(Config.NUM_LAYERS):
            # RGCNConv handles heterogeneous edges (Backbone vs BasePair)
            conv = RGCNConv(
                in_channels=Config.HIDDEN_DIM,
                out_channels=Config.HIDDEN_DIM,
                num_relations=Config.NUM_EDGE_TYPES,
            )
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(Config.HIDDEN_DIM))

        self.dropout = nn.Dropout(Config.DROPOUT)

        # 4. Decoder
        self.decoder = nn.Sequential(
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.HIDDEN_DIM, Config.NUM_TARGETS),
        )

    def forward(self, data):
        """
        Forward pass of the model.

        Args:
            data: PyG Data object containing:
                - x: (N, 3) LongTensor of indices [seq, struct, loop]
                - edge_index: (2, E) LongTensor
                - edge_attr: (E,) LongTensor of relation types

        Returns:
            out: (N, NUM_TARGETS) Tensor of predictions
        """
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_attr

        # --- Embedding Stage ---
        # x[:, 0] is sequence, x[:, 1] is structure, x[:, 2] is loop_type
        emb_seq = self.seq_embed(x[:, 0])
        emb_struct = self.struct_embed(x[:, 1])
        emb_loop = self.loop_embed(x[:, 2])

        # Concatenate and project
        h = torch.cat([emb_seq, emb_struct, emb_loop], dim=-1)
        h = self.input_proj(h)

        # Add Positional Encoding
        h = self.pos_encoder(h)

        # --- GNN Stage ---
        for conv, bn in zip(self.convs, self.bns):
            h_in = h

            # Message Passing
            h = conv(h, edge_index, edge_type)
            h = bn(h)
            h = F.relu(h)
            h = self.dropout(h)

            # Residual Connection
            h = h + h_in

        # --- Decoding Stage ---
        out = self.decoder(h)

        return out
