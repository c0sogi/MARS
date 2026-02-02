import torch
import torch.nn as nn
from library.config import Config


class RNAGRUNet(nn.Module):
    """
    Bidirectional GRU Network for RNA degradation prediction.

    Architecture:
    - Embeddings for Sequence, Structure, and Loop Type
    - Bidirectional GRU Encoder
    - Linear Decoder
    """

    def __init__(self):
        super(RNAGRUNet, self).__init__()

        # 1. Embeddings
        self.seq_embedding = nn.Embedding(Config.VOCAB_SIZE_SEQ, Config.EMBED_DIM)
        self.struct_embedding = nn.Embedding(Config.VOCAB_SIZE_STRUCT, Config.EMBED_DIM)
        self.loop_embedding = nn.Embedding(Config.VOCAB_SIZE_LOOP, Config.EMBED_DIM)

        # Input dimension for GRU: 3 * Embed_Dim
        self.input_dim = 3 * Config.EMBED_DIM

        # 2. Encoder: Bidirectional GRU
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=Config.HIDDEN_DIM,
            num_layers=Config.LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.DROPOUT if Config.LAYERS > 1 else 0,
        )

        # 3. Decoder: Linear projection
        # Input: Hidden_Dim * 2 (bidirectional)
        # Output: 5 targets
        self.decoder = nn.Linear(Config.HIDDEN_DIM * 2, len(Config.TARGET_COLS))

    def forward(self, sequence, structure, loop_type):
        """
        Forward pass of the network.

        Args:
            sequence (torch.Tensor): Shape (Batch, Seq_Len)
            structure (torch.Tensor): Shape (Batch, Seq_Len)
            loop_type (torch.Tensor): Shape (Batch, Seq_Len)

        Returns:
            torch.Tensor: Predictions of shape (Batch, Seq_Len, 5)
        """
        # Embed inputs
        emb_seq = self.seq_embedding(sequence)
        emb_struct = self.struct_embedding(structure)
        emb_loop = self.loop_embedding(loop_type)

        # Concatenate features: (Batch, Seq_Len, 3 * Embed_Dim)
        x = torch.cat([emb_seq, emb_struct, emb_loop], dim=2)

        # Pass through GRU
        # Output: (Batch, Seq_Len, Hidden_Dim * 2)
        x, _ = self.gru(x)

        # Pass through Decoder
        # Output: (Batch, Seq_Len, 5)
        x = self.decoder(x)

        return x
