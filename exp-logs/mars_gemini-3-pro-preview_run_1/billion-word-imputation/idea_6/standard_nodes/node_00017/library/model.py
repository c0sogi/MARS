import torch
import torch.nn as nn
import math
from library.config import Config


class BifurcatedTransformer(nn.Module):
    """
    Bifurcated Interleaved Transformer (Idea 6).

    A Split-Stream Transformer architecture that shares low-level contextual
    processing but bifurcates into two specialized streams:
    1. Localization Stream: Detects structural breaks (missing words).
    2. Identification Stream: Predicts the semantic identity of missing words.
    """

    def __init__(self):
        super().__init__()

        self.d_model = Config.EMBED_DIM
        self.vocab_size = Config.VOCAB_SIZE
        self.dropout_p = Config.DROPOUT
        self.max_len = Config.MAX_SEQ_LEN

        # ----------------------------------------------------------------------
        # 1. Embeddings
        # ----------------------------------------------------------------------
        self.token_embedding = nn.Embedding(
            self.vocab_size, self.d_model, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_len, self.d_model)
        self.dropout = nn.Dropout(self.dropout_p)

        # ----------------------------------------------------------------------
        # 2. Shared Context Encoder
        # ----------------------------------------------------------------------
        # Base definition for a transformer layer
        # norm_first=True (Pre-LN) is generally more stable for deeper networks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=Config.NHEAD,
            dim_feedforward=Config.DIM_FEEDFORWARD,
            dropout=self.dropout_p,
            batch_first=True,
            norm_first=True,
        )

        # Bottom K layers shared between tasks
        self.shared_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.SHARED_LAYERS
        )

        # ----------------------------------------------------------------------
        # 3. Bifurcated Streams
        # ----------------------------------------------------------------------

        # Stream A: Localization (Gap Detection)
        # Focuses on syntactic integrity
        self.loc_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.BRANCH_LAYERS
        )
        self.loc_head = nn.Linear(self.d_model, 1)

        # Stream B: Identification (Word Prediction)
        # Focuses on semantic context
        self.id_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.BRANCH_LAYERS
        )
        self.id_head = nn.Linear(self.d_model, self.vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize parameters for better convergence."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids (torch.Tensor): Shape (Batch, SeqLen). Token indices.
            attention_mask (torch.Tensor): Shape (Batch, SeqLen).
                                           1 for valid tokens, 0 for padding.

        Returns:
            loc_logits (torch.Tensor): Shape (Batch, SeqLen, 1).
                                       Logits for gap detection.
            id_logits (torch.Tensor): Shape (Batch, SeqLen, VocabSize).
                                      Logits for word identification.
        """
        batch_size, seq_len = input_ids.size()

        # Generate position indices
        positions = (
            torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        # Embed and add position info
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        # Create padding mask for Transformer
        # PyTorch expects True for positions to be IGNORED (padded)
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None

        # ----------------------------------------------------------------------
        # Shared Processing
        # ----------------------------------------------------------------------
        shared_features = self.shared_encoder(
            x, src_key_padding_mask=src_key_padding_mask
        )

        # ----------------------------------------------------------------------
        # Stream A: Localization
        # ----------------------------------------------------------------------
        loc_features = self.loc_encoder(
            shared_features, src_key_padding_mask=src_key_padding_mask
        )
        loc_logits = self.loc_head(loc_features)

        # ----------------------------------------------------------------------
        # Stream B: Identification
        # ----------------------------------------------------------------------
        id_features = self.id_encoder(
            shared_features, src_key_padding_mask=src_key_padding_mask
        )
        id_logits = self.id_head(id_features)

        return loc_logits, id_logits
