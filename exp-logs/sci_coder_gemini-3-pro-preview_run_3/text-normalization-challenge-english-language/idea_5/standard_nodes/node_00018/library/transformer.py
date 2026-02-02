import math
import torch
import torch.nn as nn
from torch.nn import Transformer
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
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    """
    Embedding layer that scales the weights by sqrt(d_model).
    """

    def __init__(self, vocab_size: int, d_model: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, tokens):
        return self.embedding(tokens) * math.sqrt(self.d_model)


class Seq2SeqTransformer(nn.Module):
    """
    Standard Seq2Seq Transformer model.
    """

    def __init__(
        self,
        num_tokens: int,
        d_model: int = Config.D_MODEL,
        nhead: int = Config.NHEAD,
        num_encoder_layers: int = Config.NUM_ENCODER_LAYERS,
        num_decoder_layers: int = Config.NUM_DECODER_LAYERS,
        dim_feedforward: int = Config.DIM_FEEDFORWARD,
        dropout: float = Config.DROPOUT,
    ):
        super(Seq2SeqTransformer, self).__init__()
        self.model_type = "Transformer"
        self.d_model = d_model

        # Embeddings
        self.src_tok_emb = TokenEmbedding(num_tokens, d_model)
        self.tgt_tok_emb = TokenEmbedding(num_tokens, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

        # Transformer
        self.transformer = Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Output Generator
        self.generator = nn.Linear(d_model, num_tokens)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src,
        tgt,
        src_mask=None,
        tgt_mask=None,
        src_padding_mask=None,
        tgt_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        """
        Forward pass for training.
        Input tensors are expected to be (batch_size, seq_len) if batch_first=False in Transformer,
        but PyTorch Transformer default is (seq_len, batch_size).
        We will permute inputs to match (seq_len, batch_size).
        """
        # Permute to (seq_len, batch_size)
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        # Embeddings + Positional Encoding
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))

        # Transformer Forward
        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=None,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        # Generator -> Logits
        return self.generator(outs).transpose(
            0, 1
        )  # Return (batch_size, seq_len, vocab)

    def encode(self, src, src_mask=None, src_padding_mask=None):
        """
        Encodes the source sequence.
        """
        src = src.transpose(0, 1)  # (seq_len, batch_size)
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        memory = self.transformer.encoder(
            src_emb, mask=src_mask, src_key_padding_mask=src_padding_mask
        )
        return memory

    def decode(self, tgt, memory, tgt_mask=None, memory_key_padding_mask=None):
        """
        Decodes the target sequence given the encoded memory.
        """
        tgt = tgt.transpose(0, 1)  # (seq_len, batch_size)
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        outs = self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.generator(outs).transpose(0, 1)  # (batch_size, seq_len, vocab)


def generate_square_subsequent_mask(sz: int, device: torch.device):
    """
    Generates a mask to prevent the decoder from looking ahead.
    Returns a tensor of shape (sz, sz) with -inf in the upper triangle (excluding diagonal).
    """
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(src, tgt, pad_idx, device):
    """
    Creates source and target masks for the transformer.
    """
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = src == pad_idx
    tgt_padding_mask = tgt == pad_idx

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
