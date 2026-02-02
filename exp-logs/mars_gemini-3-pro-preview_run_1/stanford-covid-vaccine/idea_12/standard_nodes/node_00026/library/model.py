import torch
import torch.nn as nn
from library.config import Config


class ResidualBiLSTMBlock(nn.Module):
    """
    A Pre-LayerNorm Residual Bidirectional LSTM Block.
    Architecture: x = x + Dropout(LSTM(LayerNorm(x)))

    This configuration (Pre-LN) is generally more stable for training deep
    residual networks compared to Post-LN.
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        # The residual stream dimension is 2 * hidden_dim because the LSTM is bidirectional.
        # We project the input to this size before the blocks, so input_size == output_size.
        self.d_model = 2 * hidden_dim

        self.ln = nn.LayerNorm(self.d_model)
        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, d_model)
        residual = x

        # Pre-LayerNorm
        out = self.ln(x)

        # LSTM
        # out shape: (Batch, Seq_Len, 2 * hidden_dim)
        out, _ = self.lstm(out)

        # Dropout
        out = self.dropout(out)

        # Residual Connection
        return residual + out


class InteractionAwareModel(nn.Module):
    """
    Interaction-Aware Deep Residual BiLSTM Model.

    Integrates explicit physicochemical semantics (Bond Types) and geometric
    structure (Signed Distances) with a deep recurrent backbone.
    """

    def __init__(self):
        super().__init__()

        # =====================================================================
        # 1. Feature Embeddings
        # =====================================================================
        self.seq_emb = nn.Embedding(Config.VOCAB_SIZE_SEQ, Config.EMBED_DIM_SEQ)
        self.loop_emb = nn.Embedding(Config.VOCAB_SIZE_LOOP, Config.EMBED_DIM_LOOP)
        self.bond_emb = nn.Embedding(Config.VOCAB_SIZE_BOND, Config.EMBED_DIM_BOND)

        # Note: The distance feature is passed as a pre-computed float tensor
        # of shape (B, L, EMBED_DIM_DISTANCE), so no embedding layer is needed.

        # =====================================================================
        # 2. Input Projection
        # =====================================================================
        # Calculate total input dimension from concatenated features
        # 32 (Seq) + 32 (Loop) + 32 (Bond) + 32 (Dist) = 128
        input_dim = Config.INPUT_DIM

        # The backbone operates on 2 * HIDDEN_DIM (768)
        self.backbone_dim = 2 * Config.HIDDEN_DIM

        self.input_proj = nn.Linear(input_dim, self.backbone_dim)
        self.dropout_in = nn.Dropout(Config.DROPOUT)

        # =====================================================================
        # 3. Deep Residual Backbone
        # =====================================================================
        self.blocks = nn.ModuleList(
            [
                ResidualBiLSTMBlock(Config.HIDDEN_DIM, Config.DROPOUT)
                for _ in range(Config.NUM_LAYERS)
            ]
        )

        # =====================================================================
        # 4. Output Head
        # =====================================================================
        # Projects to the 3 scored targets: reactivity, deg_Mg_pH10, deg_Mg_50C
        self.head = nn.Linear(self.backbone_dim, Config.NUM_TARGETS)

    def forward(self, inputs):
        """
        Forward pass of the model.

        Args:
            inputs (dict): Dictionary containing:
                - 'seq': LongTensor (Batch, Seq_Len)
                - 'loop': LongTensor (Batch, Seq_Len)
                - 'bond': LongTensor (Batch, Seq_Len)
                - 'dist': FloatTensor (Batch, Seq_Len, Embed_Dim_Dist)

        Returns:
            torch.Tensor: Predictions of shape (Batch, Seq_Len, 3)
        """
        # Unpack inputs
        seq = inputs["seq"]
        loop = inputs["loop"]
        bond = inputs["bond"]
        dist = inputs["dist"]

        # Embed categorical features
        e_seq = self.seq_emb(seq)  # (B, L, 32)
        e_loop = self.loop_emb(loop)  # (B, L, 32)
        e_bond = self.bond_emb(bond)  # (B, L, 32)

        # Concatenate all features
        # Result shape: (B, L, 128)
        x = torch.cat([e_seq, e_loop, e_bond, dist], dim=-1)

        # Project to backbone dimension
        # (B, L, 128) -> (B, L, 768)
        x = self.input_proj(x)
        x = self.dropout_in(x)

        # Pass through Deep Residual BiLSTM blocks
        for block in self.blocks:
            x = block(x)

        # Output projection
        # (B, L, 768) -> (B, L, 3)
        logits = self.head(x)

        return logits
