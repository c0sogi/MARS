import torch
import torch.nn as nn
from library.config import (
    VOCAB_SIZE_SEQ,
    VOCAB_SIZE_STRUCT,
    VOCAB_SIZE_LOOP,
    EMBED_DIM_SEQ,
    EMBED_DIM_STRUCT,
    EMBED_DIM_LOOP,
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
    A Hybrid Neural Network combining Bidirectional GRU for sequential inductive bias
    and Transformer Encoder for global attention mechanisms.
    """

    def __init__(self):
        super(HybridRNNTransformer, self).__init__()

        # 1. Embeddings
        self.embed_seq = nn.Embedding(VOCAB_SIZE_SEQ, EMBED_DIM_SEQ)
        self.embed_struct = nn.Embedding(VOCAB_SIZE_STRUCT, EMBED_DIM_STRUCT)
        self.embed_loop = nn.Embedding(VOCAB_SIZE_LOOP, EMBED_DIM_LOOP)

        # Calculate total input dimension for the GRU
        input_dim = EMBED_DIM_SEQ + EMBED_DIM_STRUCT + EMBED_DIM_LOOP

        # 2. Bidirectional GRU Encoder
        # Captures sequential context and local dependencies
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=HIDDEN_DIM_GRU,
            num_layers=NUM_LAYERS_GRU,
            dropout=DROPOUT_GRU if NUM_LAYERS_GRU > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

        # The output dimension of Bi-GRU is 2 * hidden_size
        gru_output_dim = 2 * HIDDEN_DIM_GRU

        # 3. Transformer Encoder
        # Refines features with global self-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=gru_output_dim,
            nhead=NHEAD_TRANSFORMER,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT_TRANSFORMER,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=NUM_LAYERS_TRANSFORMER
        )

        # 4. Output Head
        self.fc_out = nn.Linear(gru_output_dim, NUM_TARGETS)

    def forward(self, x_seq, x_struct, x_loop):
        """
        Args:
            x_seq: (Batch, Seq_Len) - Sequence indices
            x_struct: (Batch, Seq_Len) - Structure indices
            x_loop: (Batch, Seq_Len) - Loop type indices
        Returns:
            out: (Batch, Seq_Len, Num_Targets)
        """
        # Embed and Concatenate
        emb_s = self.embed_seq(x_seq)  # (B, L, E_seq)
        emb_st = self.embed_struct(x_struct)  # (B, L, E_struct)
        emb_l = self.embed_loop(x_loop)  # (B, L, E_loop)

        # (B, L, Total_Embed_Dim)
        x = torch.cat([emb_s, emb_st, emb_l], dim=-1)

        # Pass through Bi-GRU
        # gru_out: (B, L, 2*Hidden)
        gru_out, _ = self.gru(x)

        # Pass through Transformer Encoder
        # transformer_out: (B, L, 2*Hidden)
        transformer_out = self.transformer_encoder(gru_out)

        # Project to targets
        # out: (B, L, 5)
        out = self.fc_out(transformer_out)

        return out


def weighted_masked_mse_loss(preds, targets, masks, weights):
    """
    Calculates the Mean Squared Error, masked to valid positions and weighted by signal-to-noise ratio.

    Args:
        preds: (Batch, Seq_Len, 5) - Predicted values
        targets: (Batch, Seq_Len, 5) - Ground truth values
        masks: (Batch, Seq_Len) - 1.0 for scored positions, 0.0 otherwise
        weights: (Batch,) - Signal-to-noise weights for each sample

    Returns:
        loss: Scalar tensor
    """
    # Squared Error: (B, L, 5)
    squared_error = (preds - targets) ** 2

    # Apply Mask (broadcast over last dim): (B, L, 5)
    # Zeros out errors for unscored positions (indices 68+)
    masked_squared_error = squared_error * masks.unsqueeze(-1)

    # Apply Sample Weights (broadcast over sequence and target dims): (B, L, 5)
    # weights is (B,), view as (B, 1, 1)
    weighted_error = masked_squared_error * weights.view(-1, 1, 1)

    # Normalize
    # We sum the weighted errors and divide by the sum of effective weights (mask * weight)
    # This prevents the loss from scaling with batch size or being biased by zero-padding

    # Total weighted error sum
    total_loss = weighted_error.sum()

    # Calculate normalization factor
    # Sum of weights for all valid positions across all targets
    # masks shape (B, L), weights shape (B,)
    # effective_weights = masks * weights.unsqueeze(-1) -> (B, L)
    # We have 5 targets, so multiply by 5
    normalization = (masks * weights.unsqueeze(-1)).sum() * targets.shape[-1]

    # Avoid division by zero
    epsilon = 1e-8
    loss = total_loss / (normalization + epsilon)

    return loss
