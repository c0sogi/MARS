import torch
import torch.nn as nn
import math
from library import config


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

        pe = pe.unsqueeze(0).transpose(0, 1)  # Shape: [max_len, 1, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class SemioticTransformer(nn.Module):
    """
    Encoder-Decoder Transformer for Text Normalization.

    Encoder: Processes character-level input to understand the structure of symbols/numbers.
    Decoder: Generates BPE-level output tokens for the normalized text.
    """

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        num_decoder_layers=config.NUM_DECODER_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT,
        max_len=512,
    ):
        super(SemioticTransformer, self).__init__()

        self.d_model = d_model

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_len)
        self.pos_decoder = PositionalEncoding(d_model, dropout, max_len=max_len)

        # Transformer Backbone
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # We will handle [seq_len, batch, dim] internally
        )

        # Output Head
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        # Initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize parameters with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src,
        tgt,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
        tgt_mask=None,
        memory_key_padding_mask=None,
    ):
        """
        Forward pass of the Transformer.

        Args:
            src: [batch_size, src_len]
            tgt: [batch_size, tgt_len]
            src_key_padding_mask: [batch_size, src_len] (True for pad tokens)
            tgt_key_padding_mask: [batch_size, tgt_len] (True for pad tokens)
            tgt_mask: [tgt_len, tgt_len] (Causal mask)

        Returns:
            output: [batch_size, tgt_len, tgt_vocab_size]
        """
        # Transpose to [seq_len, batch_size] for PyTorch Transformer (batch_first=False)
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        # Embed and Add Position Info
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_decoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))

        # Pass through Transformer
        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        # Project to Vocab Size
        output = self.generator(output)

        # Transpose back to [batch_size, seq_len, vocab_size]
        return output.transpose(0, 1)

    def generate_square_subsequent_mask(self, sz):
        """
        Generates a causal mask for the decoder to prevent attending to future tokens.
        Returns: [sz, sz] float tensor with -inf in upper triangle.
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask
