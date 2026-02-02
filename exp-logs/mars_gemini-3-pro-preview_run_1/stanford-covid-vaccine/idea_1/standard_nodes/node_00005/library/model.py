import torch
import torch.nn as nn
from library.config import Config


class RNAGRUNet(nn.Module):
    """
    Bidirectional GRU Network for RNA degradation prediction.
    Cite solution_lesson_node_00004
    """

    def __init__(self):
        super(RNAGRUNet, self).__init__()

        # 1. Embeddings
        self.seq_embedding = nn.Embedding(Config.VOCAB_SIZE_SEQ, Config.EMBED_DIM)
        self.struct_embedding = nn.Embedding(Config.VOCAB_SIZE_STRUCT, Config.EMBED_DIM)
        self.loop_embedding = nn.Embedding(Config.VOCAB_SIZE_LOOP, Config.EMBED_DIM)

        self.input_channels = 3 * Config.EMBED_DIM

        # 2. Encoder: Bidirectional GRU
        self.encoder = nn.GRU(
            input_size=self.input_channels,
            hidden_size=Config.HIDDEN_DIM,
            num_layers=Config.LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.DROPOUT if Config.LAYERS > 1 else 0,
        )

        # 3. Decoder: Linear projection
        # Input: (Batch, Seq_Len, 2 * Hidden_Dim)
        self.decoder = nn.Linear(Config.HIDDEN_DIM * 2, len(Config.TARGET_COLS))

    def forward(self, sequence, structure, loop_type):
        # Embed inputs: (Batch, Seq_Len, Embed_Dim)
        emb_seq = self.seq_embedding(sequence)
        emb_struct = self.struct_embedding(structure)
        emb_loop = self.loop_embedding(loop_type)

        # Concatenate: (Batch, Seq_Len, 3 * Embed_Dim)
        x = torch.cat([emb_seq, emb_struct, emb_loop], dim=2)

        # Encoder (GRU): (Batch, Seq_Len, 2 * Hidden_Dim)
        x, _ = self.encoder(x)

        # Decoder: (Batch, Seq_Len, 5)
        x = self.decoder(x)

        return x
