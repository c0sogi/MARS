import torch
import torch.nn as nn
import math
from library.config import Config


class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as
    the embeddings, so that the two can be summed.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer to be part of state_dict but not a parameter
        # Shape: (1, max_len, d_model) for broadcasting with batch_first=True
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    """
    Standard Encoder-Decoder Transformer architecture for Sequence-to-Sequence tasks.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_idx: int,
        tgt_pad_idx: int,
        d_model: int = Config.D_MODEL,
        nhead: int = Config.NHEAD,
        num_encoder_layers: int = Config.NUM_ENCODER_LAYERS,
        num_decoder_layers: int = Config.NUM_DECODER_LAYERS,
        dim_feedforward: int = Config.DIM_FEEDFORWARD,
        dropout: float = Config.DROPOUT,
    ):
        super(Seq2SeqTransformer, self).__init__()

        self.d_model = d_model
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        # Embeddings
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

        # Transformer
        # batch_first=True means input/output tensors are (batch, seq, feature)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # Output Generator
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize parameters with Xavier Uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        """
        Forward pass of the Transformer.

        Args:
            src: Source sequence tensor (batch_size, src_len)
            tgt: Target sequence tensor (batch_size, tgt_len)

        Returns:
            Logits tensor (batch_size, tgt_len, tgt_vocab_size)
        """
        # Create masks
        # src_key_padding_mask: (batch, src_len) - True where pad
        src_key_padding_mask = src == self.src_pad_idx
        tgt_key_padding_mask = tgt == self.tgt_pad_idx

        # tgt_mask: (tgt_len, tgt_len) - Causal mask to prevent look-ahead
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(
            tgt.device
        )

        # Embeddings (scaled by sqrt(d_model))
        src_emb = self.positional_encoding(
            self.src_tok_emb(src) * math.sqrt(self.d_model)
        )
        tgt_emb = self.positional_encoding(
            self.tgt_tok_emb(tgt) * math.sqrt(self.d_model)
        )

        # Transformer Pass
        # memory_key_padding_mask ensures decoder doesn't attend to source padding
        outs = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        return self.generator(outs)

    def encode(self, src: torch.Tensor):
        """
        Encodes the source sequence. Useful for inference.

        Args:
            src: (batch_size, src_len)

        Returns:
            memory: (batch_size, src_len, d_model)
        """
        src_key_padding_mask = src == self.src_pad_idx
        src_emb = self.positional_encoding(
            self.src_tok_emb(src) * math.sqrt(self.d_model)
        )
        return self.transformer.encoder(
            src_emb, src_key_padding_mask=src_key_padding_mask
        )

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor = None,
    ):
        """
        Decodes one step given the memory (encoded source). Useful for inference.

        Args:
            tgt: (batch_size, tgt_len) - The target sequence generated so far
            memory: (batch_size, src_len, d_model) - Output from encoder
            memory_key_padding_mask: (batch_size, src_len) - Mask for source padding

        Returns:
            output: (batch_size, tgt_len, d_model)
        """
        tgt_key_padding_mask = tgt == self.tgt_pad_idx
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(
            tgt.device
        )
        tgt_emb = self.positional_encoding(
            self.tgt_tok_emb(tgt) * math.sqrt(self.d_model)
        )

        return self.transformer.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
