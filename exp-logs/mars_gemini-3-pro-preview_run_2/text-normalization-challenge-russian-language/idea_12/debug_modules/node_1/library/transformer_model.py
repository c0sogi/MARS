import torch
import torch.nn as nn
import math
from library.config import ModelConfig


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
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class CharToBPESeq2Seq(nn.Module):
    """
    Transformer model with Character-level Encoder and BPE-level Decoder.
    Designed for the Heterogeneous Granularity Tier 2 of the Hybrid Cascade.
    """

    def __init__(
        self,
        config: ModelConfig,
        char_vocab_size: int,
        bpe_vocab_size: int,
        src_pad_idx: int,
        tgt_pad_idx: int,
    ):
        super(CharToBPESeq2Seq, self).__init__()

        self.d_model = config.d_model
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        # 1. Embeddings
        # Encoder takes characters
        self.src_embedding = nn.Embedding(char_vocab_size, self.d_model)
        # Decoder takes BPE subwords
        self.tgt_embedding = nn.Embedding(bpe_vocab_size, self.d_model)

        # 2. Positional Encoding
        # Max length buffer covers both encoder and decoder max lengths
        max_len = max(config.max_enc_len, config.max_dec_len) + 50
        self.pos_encoder = PositionalEncoding(
            self.d_model, config.dropout, max_len=max_len
        )

        # 3. Transformer Core
        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
        )

        # 4. Output Projection
        self.fc_out = nn.Linear(self.d_model, bpe_vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize parameters with Xavier Uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz):
        """Generates the causal mask for the decoder."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def create_masks(self, src, tgt):
        """
        Creates masks for Transformer training.
        src: [batch, src_len]
        tgt: [batch, tgt_len]
        """
        tgt_seq_len = tgt.shape[1]

        # Target causal mask (prevent looking ahead)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(src.device)

        # Padding masks (True indicates position should be ignored)
        src_padding_mask = src == self.src_pad_idx
        tgt_padding_mask = tgt == self.tgt_pad_idx

        return tgt_mask, src_padding_mask, tgt_padding_mask

    def forward(self, src, tgt):
        """
        Forward pass for training.

        Args:
            src: [batch_size, src_len] (Char IDs)
            tgt: [batch_size, tgt_len] (BPE IDs)

        Returns:
            logits: [batch_size, tgt_len, bpe_vocab_size]
        """
        # Create masks
        tgt_mask, src_padding_mask, tgt_padding_mask = self.create_masks(src, tgt)

        # Embeddings
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)

        # Apply Positional Encoding
        # PE expects [seq_len, batch, dim], so we transpose, apply, then transpose back
        src_emb = self.pos_encoder(src_emb.transpose(0, 1)).transpose(0, 1)
        tgt_emb = self.pos_encoder(tgt_emb.transpose(0, 1)).transpose(0, 1)

        # Transformer Pass
        outs = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,  # Mask encoder output padding in decoder attention
        )

        # Projection to Vocabulary
        logits = self.fc_out(outs)
        return logits

    def encode(self, src):
        """
        Encodes the source sequence. Used during inference.

        Args:
            src: [batch_size, src_len]

        Returns:
            memory: [batch_size, src_len, d_model]
            src_padding_mask: [batch_size, src_len]
        """
        src_padding_mask = src == self.src_pad_idx

        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb.transpose(0, 1)).transpose(0, 1)

        memory = self.transformer.encoder(
            src_emb, src_key_padding_mask=src_padding_mask
        )
        return memory, src_padding_mask

    def decode(self, tgt, memory, memory_key_padding_mask):
        """
        Decodes one step. Used during inference.

        Args:
            tgt: [batch_size, tgt_len] (Sequence generated so far)
            memory: [batch_size, src_len, d_model] (Encoder output)
            memory_key_padding_mask: [batch_size, src_len]

        Returns:
            logits: [batch_size, tgt_len, bpe_vocab_size]
        """
        tgt_seq_len = tgt.shape[1]
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)

        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb.transpose(0, 1)).transpose(0, 1)

        out = self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        return self.fc_out(out)
