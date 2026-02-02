import torch
import torch.nn as nn
import math
from library.config import (
    VOCAB_SIZE_SEQ,
    VOCAB_SIZE_LOOP,
    VOCAB_SIZE_STRUCT,
    EMBED_DIM_SEQ,
    EMBED_DIM_LOOP,
    EMBED_DIM_STRUCT,
    HIDDEN_DIM_GRU,
    NUM_LAYERS_GRU,
    DROPOUT_GRU,
    NUM_LAYERS_TRANSFORMER,
    NHEAD_TRANSFORMER,
    DIM_FEEDFORWARD,
    DROPOUT_TRANSFORMER,
    NUM_TARGETS,
)


class HybridRNNTransformer(nn.Module):
    """
    Hybrid architecture combining BiGRU and Transformer Encoder.
    Inputs: Sequence, Structure (Dot-Bracket), Predicted Loop Type.
    """

    def __init__(self):
        super(HybridRNNTransformer, self).__init__()

        # 1. Embeddings
        self.embed_seq = nn.Embedding(VOCAB_SIZE_SEQ, EMBED_DIM_SEQ)
        self.embed_struct = nn.Embedding(VOCAB_SIZE_STRUCT, EMBED_DIM_STRUCT)
        self.embed_loop = nn.Embedding(VOCAB_SIZE_LOOP, EMBED_DIM_LOOP)

        # Input dimension (sum of embeddings)
        self.input_dim = EMBED_DIM_SEQ + EMBED_DIM_STRUCT + EMBED_DIM_LOOP

        # 2. BiGRU Encoder
        # Captures local sequential dependencies
        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=HIDDEN_DIM_GRU,
            num_layers=NUM_LAYERS_GRU,
            dropout=DROPOUT_GRU if NUM_LAYERS_GRU > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )

        # BiGRU output dimension is 2 * HIDDEN_DIM_GRU
        self.d_model = 2 * HIDDEN_DIM_GRU

        # 3. Transformer Encoder
        # Captures long-range dependencies
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=NHEAD_TRANSFORMER,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT_TRANSFORMER,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=NUM_LAYERS_TRANSFORMER
        )

        # 4. Output Head
        self.fc_out = nn.Linear(self.d_model, NUM_TARGETS)

    def forward(self, seq, struct, loop):
        """
        Args:
            seq: (Batch, Seq_Len) - Tokenized sequence
            struct: (Batch, Seq_Len) - Tokenized structure
            loop: (Batch, Seq_Len) - Tokenized loop type
        """
        # Embed
        emb_s = self.embed_seq(seq)
        emb_st = self.embed_struct(struct)
        emb_l = self.embed_loop(loop)

        # Concatenate features
        x = torch.cat([emb_s, emb_st, emb_l], dim=-1)

        # BiGRU Pass
        x, _ = self.gru(x)

        # Transformer Pass
        x = self.transformer(x)

        # Output Projection
        out = self.fc_out(x)
        return out


def weighted_masked_mse_loss(preds, targets, masks, weights):
    """
    Calculates the Mean Squared Error, masked to valid positions.
    Ignores sample weights to avoid overfitting to clean data (Cite Lesson 00011).
    """
    # Squared Error: (B, L, 5)
    squared_error = (preds - targets) ** 2

    # Apply Mask (broadcast over last dim): (B, L, 5)
    masked_squared_error = squared_error * masks.unsqueeze(-1)

    # Total error sum
    total_loss = masked_squared_error.sum()

    # Normalization: Count of valid positions
    normalization = masks.sum() * targets.shape[-1]

    # Avoid division by zero
    epsilon = 1e-8
    loss = total_loss / (normalization + epsilon)

    return loss
