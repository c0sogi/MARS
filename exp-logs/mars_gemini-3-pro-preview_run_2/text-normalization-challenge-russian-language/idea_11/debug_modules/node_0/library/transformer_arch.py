import math
import torch
import torch.nn as nn
from library.config import Config


class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as
    the embeddings, so that the two can be summed.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        # Register buffer to be part of state_dict but not a parameter
        # Shape: (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Slice the positional encoding to match the sequence length
        # Transpose pe from [seq_len, 1, d_model] to [1, seq_len, d_model] for broadcasting
        x = x + self.pe[: x.size(1)].transpose(0, 1)
        return self.dropout(x)


class CharToSubwordTransformer(nn.Module):
    """
    A Transformer-based Encoder-Decoder model for normalizing text.
    Encoder: Character-level input (to handle complex number formatting).
    Decoder: Subword-level output (BPE) (to generate valid Russian morphology).
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
        max_src_len: int = Config.MAX_SRC_LEN,
        max_tgt_len: int = Config.MAX_TGT_LEN,
    ):
        super().__init__()
        self.d_model = d_model
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        # 1. Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # 2. Positional Encoding
        # Ensure max_len covers the longest possible sequence plus some buffer
        max_len = max(max_src_len, max_tgt_len) + 50
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_len)

        # 3. Transformer Core
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Input/Output shape: (Batch, Seq, Feature)
        )

        # 4. Output Projection
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize parameters with Xavier Uniform.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generates a causal mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def create_mask(
        self, src: torch.Tensor, tgt: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Creates masks for Source and Target sequences.

        Returns:
            src_mask: None (Encoder attends to all positions)
            tgt_mask: Causal mask for Decoder
            src_padding_mask: Bool mask for Source padding
            tgt_padding_mask: Bool mask for Target padding
        """
        tgt_seq_len = tgt.shape[1]

        # Causal mask for decoder (prevent looking ahead)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(src.device)

        # Padding masks (True where pad token is present)
        src_padding_mask = src == self.src_pad_idx
        tgt_padding_mask = tgt == self.tgt_pad_idx

        return None, tgt_mask, src_padding_mask, tgt_padding_mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer.

        Args:
            src: Source sequence indices [Batch, SrcLen]
            tgt: Target sequence indices [Batch, TgtLen]

        Returns:
            logits: Output logits [Batch, TgtLen, TgtVocabSize]
        """
        # Generate masks
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(
            src, tgt
        )

        # Apply embeddings and positional encoding
        # Multiply by sqrt(d_model) as per "Attention Is All You Need"
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))

        # Transformer Pass
        # Note: memory_key_padding_mask prevents decoder from attending to source pads
        outs = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=None,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )

        # Project to vocabulary
        return self.generator(outs)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """
        Encodes the source sequence. Useful for inference.

        Args:
            src: Source sequence indices [Batch, SrcLen]

        Returns:
            memory: Encoder output [Batch, SrcLen, DModel]
        """
        src_padding_mask = src == self.src_pad_idx
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask)

    def decode(
        self, tgt: torch.Tensor, memory: torch.Tensor, memory_key_padding_mask=None
    ) -> torch.Tensor:
        """
        Decodes the target sequence given the encoder memory. Useful for inference.

        Args:
            tgt: Target sequence indices so far [Batch, TgtLen]
            memory: Encoder output [Batch, SrcLen, DModel]
            memory_key_padding_mask: Mask for source padding (optional, should match encode)

        Returns:
            output: Decoder output [Batch, TgtLen, DModel]
        """
        tgt_seq_len = tgt.shape[1]
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)
        tgt_padding_mask = tgt == self.tgt_pad_idx

        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))

        return self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
