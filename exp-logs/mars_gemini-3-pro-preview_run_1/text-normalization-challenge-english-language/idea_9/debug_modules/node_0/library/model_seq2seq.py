import math
import torch
import torch.nn as nn
from library.config import Config


class PositionalEncoding(nn.Module):
    """
    Standard Sinusoidal Positional Encoding.
    Injects information about the relative or absolute position of the tokens in the sequence.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        # Add positional encoding to embedding (slicing to current seq_len)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerFallback(nn.Module):
    """
    Character-level Transformer Seq2Seq model for text normalization.
    Conditioned on the token class.
    """

    def __init__(
        self,
        char_vocab_size,
        class_vocab_size,
        d_model=Config.SEQ2SEQ_EMBEDDING_DIM,
        nhead=Config.SEQ2SEQ_NUM_HEADS,
        num_encoder_layers=Config.SEQ2SEQ_NUM_LAYERS,
        num_decoder_layers=Config.SEQ2SEQ_NUM_LAYERS,
        dim_feedforward=Config.SEQ2SEQ_PF_DIM,
        dropout=Config.SEQ2SEQ_DROPOUT,
        pad_idx=Config.PAD_IDX,
    ):
        super(TransformerFallback, self).__init__()

        self.d_model = d_model
        self.pad_idx = pad_idx

        # 1. Embeddings
        self.char_embedding = nn.Embedding(
            char_vocab_size, d_model, padding_idx=pad_idx
        )
        self.class_embedding = nn.Embedding(
            class_vocab_size, d_model, padding_idx=pad_idx
        )

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # 3. Transformer
        # batch_first=True ensures input/output tensors are (batch, seq, feature)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # 4. Output Head
        self.fc_out = nn.Linear(d_model, char_vocab_size)

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz, device):
        """
        Generates a causal mask for the decoder to prevent attending to future tokens.
        Returns: (sz, sz) boolean mask where True indicates position should be masked.
        """
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def create_src_mask(self, src, pad_idx):
        """
        Creates padding mask for source sequence.
        Args:
            src: (batch, seq_len)
        Returns:
            (batch, seq_len) boolean mask (True where padding exists)
        """
        return src == pad_idx

    def create_tgt_mask(self, tgt, pad_idx):
        """
        Creates padding mask for target sequence.
        Args:
            tgt: (batch, seq_len)
        Returns:
            (batch, seq_len) boolean mask
        """
        return tgt == pad_idx

    def forward(self, src, tgt, class_id):
        """
        Args:
            src: Source char indices (batch, src_len)
            tgt: Target char indices (batch, tgt_len)
            class_id: Class indices (batch) or (batch, 1)

        Returns:
            logits: (batch, tgt_len, char_vocab_size)
        """
        device = src.device

        # -------------------------------------------------------
        # 1. Prepare Source Sequence with Class Conditioning
        # -------------------------------------------------------
        # Embed source chars: (batch, src_len, d_model)
        src_emb = self.char_embedding(src) * math.sqrt(self.d_model)

        # Embed class: (batch, 1, d_model)
        if class_id.dim() == 1:
            class_id = class_id.unsqueeze(1)
        class_emb = self.class_embedding(class_id) * math.sqrt(self.d_model)

        # Prepend class embedding to source sequence
        # New shape: (batch, src_len + 1, d_model)
        src_input = torch.cat([class_emb, src_emb], dim=1)

        # Add positional encoding
        src_input = self.pos_encoder(src_input)

        # -------------------------------------------------------
        # 2. Prepare Target Sequence
        # -------------------------------------------------------
        # Embed target chars: (batch, tgt_len, d_model)
        tgt_emb = self.char_embedding(tgt) * math.sqrt(self.d_model)
        tgt_input = self.pos_encoder(tgt_emb)

        # -------------------------------------------------------
        # 3. Create Masks
        # -------------------------------------------------------
        # Source Padding Mask
        # We must account for the prepended class token which is never padding.
        # Original src mask: (batch, src_len)
        src_pad_mask = self.create_src_mask(src, self.pad_idx)
        # Prepend 'False' (not padding) for the class token column
        batch_size = src.size(0)
        class_token_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
        # Combined mask: (batch, src_len + 1)
        src_key_padding_mask = torch.cat([class_token_mask, src_pad_mask], dim=1)

        # Target Padding Mask: (batch, tgt_len)
        tgt_key_padding_mask = self.create_tgt_mask(tgt, self.pad_idx)

        # Target Causal Mask: (tgt_len, tgt_len)
        tgt_seq_len = tgt.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len, device)

        # -------------------------------------------------------
        # 4. Transformer Forward Pass
        # -------------------------------------------------------
        # Output shape: (batch, tgt_len, d_model)
        output = self.transformer(
            src=src_input,
            tgt=tgt_input,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        # -------------------------------------------------------
        # 5. Projection
        # -------------------------------------------------------
        # Shape: (batch, tgt_len, char_vocab_size)
        logits = self.fc_out(output)

        return logits
