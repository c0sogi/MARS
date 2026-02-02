import torch
import torch.nn as nn
import math
from library.config import Config
from library.utils import setup_logger

logger = setup_logger("model")


class InterleavedTransformer(nn.Module):
    """
    Interleaved Gap-Token Transformer.

    This model processes sequences where 'Gap' tokens are interleaved between 'Word' tokens.
    It uses distinct token type embeddings to differentiate structure from content.
    Two decoupled heads predict:
      1. Localization: Whether a word is missing at a specific Gap.
      2. Identification: Which word is missing (if any).
    """

    def __init__(self):
        super(InterleavedTransformer, self).__init__()

        self.vocab_size = Config.VOCAB_SIZE
        self.embed_dim = Config.EMBED_DIM
        self.hidden_dim = Config.HIDDEN_DIM
        self.num_layers = Config.NUM_LAYERS
        self.num_heads = Config.NUM_HEADS
        self.dropout_prob = Config.DROPOUT
        self.max_seq_len = Config.MAX_SEQ_LEN

        # 1. Embeddings
        # Word Embedding: Maps token IDs to vectors
        self.word_embedding = nn.Embedding(
            num_embeddings=self.vocab_size + 3,  # +3 for PAD, UNK, GAP
            embedding_dim=self.embed_dim,
            padding_idx=Config.PAD_IDX,
        )

        # Positional Embedding: Learnable vectors for positions
        self.position_embedding = nn.Embedding(
            num_embeddings=self.max_seq_len, embedding_dim=self.embed_dim
        )

        # Token Type Embedding: 0 for Words, 1 for Gaps
        self.token_type_embedding = nn.Embedding(
            num_embeddings=2, embedding_dim=self.embed_dim
        )

        self.emb_layer_norm = nn.LayerNorm(self.embed_dim)
        self.emb_dropout = nn.Dropout(self.dropout_prob)

        # 2. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim,
            dropout=self.dropout_prob,
            activation="gelu",
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=self.num_layers
        )

        # 3. Decoupled Heads
        # Localization Head: Binary classification (Missing vs Not Missing)
        # Output shape: (Batch, SeqLen, 1)
        self.localization_head = nn.Linear(self.embed_dim, 1)

        # Identification Head: Multi-class classification (Vocabulary)
        # Output shape: (Batch, SeqLen, VocabSize)
        # Note: We predict over the full vocab size (including special tokens,
        # though we only care about real words)
        self.identification_head = nn.Linear(self.embed_dim, self.vocab_size + 3)

        self._init_weights()

        logger.info(f"Model initialized. Parameters: {self.count_parameters():,}")

    def _init_weights(self):
        """Initialize weights for better convergence."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids, token_type_ids, attention_mask=None):
        """
        Args:
            input_ids: (Batch, SeqLen) - Token indices
            token_type_ids: (Batch, SeqLen) - 0 for words, 1 for gaps
            attention_mask: (Batch, SeqLen) - 1 for valid tokens, 0 for padding

        Returns:
            loc_logits: (Batch, SeqLen, 1)
            id_logits: (Batch, SeqLen, VocabSize)
        """
        batch_size, seq_len = input_ids.size()

        # --- Embedding Layer ---

        # Create position IDs: [0, 1, ..., seq_len-1]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Sum embeddings
        word_emb = self.word_embedding(input_ids)
        pos_emb = self.position_embedding(position_ids)
        type_emb = self.token_type_embedding(token_type_ids)

        embeddings = word_emb + pos_emb + type_emb
        embeddings = self.emb_layer_norm(embeddings)
        embeddings = self.emb_dropout(embeddings)

        # --- Transformer Encoder ---

        # Create padding mask for Transformer
        # PyTorch Transformer expects `src_key_padding_mask` where True indicates padding (to be ignored)
        # attention_mask is 1 for keep, 0 for ignore. So we invert it.
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None

        encoder_output = self.transformer_encoder(
            src=embeddings, src_key_padding_mask=src_key_padding_mask
        )

        # --- Output Heads ---

        # Apply heads to all tokens.
        # We will mask out non-Gap tokens or padding tokens in the loss function.

        loc_logits = self.localization_head(encoder_output)
        id_logits = self.identification_head(encoder_output)

        return loc_logits, id_logits
