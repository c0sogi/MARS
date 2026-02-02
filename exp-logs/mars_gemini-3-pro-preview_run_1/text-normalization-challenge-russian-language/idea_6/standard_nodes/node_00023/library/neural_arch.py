import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

from library.config import Config


class PositionalEncoding(nn.Module):
    """
    Standard Sinusoidal Positional Encoding.
    Injects information about the relative or absolute position of the tokens in the sequence.
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
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class DualGranularityTransformer(nn.Module):
    """
    A Hybrid Neuro-Symbolic Transformer model.

    It fuses two granularities of input:
    1. Context words (Left/Right) encoded via BPE (Byte-Pair Encoding) to capture syntax/morphology.
    2. The Target token (Center) encoded via Characters to preserve precise orthography.

    The Encoder input sequence is constructed as:
    [BPE_Left, SEP, Char_Target, SEP, BPE_Right]

    The Decoder generates the normalized text using BPE tokens.
    """

    def __init__(self, config: Config, bpe_pad_id: int = 0, char_pad_id: int = 0):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.bpe_pad_id = bpe_pad_id
        self.char_pad_id = char_pad_id

        # --- Embeddings ---
        # 1. BPE Embedding: Used for Context (Left/Right) and Decoder Input/Output
        self.bpe_embedding = nn.Embedding(config.bpe_vocab_size, config.d_model)

        # 2. Char Embedding: Used for the specific token being normalized
        self.char_embedding = nn.Embedding(config.max_char_vocab_size, config.d_model)

        # 3. Separator Embedding: Learnable vector to delimit context and target
        self.sep_embedding = nn.Parameter(torch.randn(1, 1, config.d_model))

        # --- Positional Encoding ---
        self.pos_encoder = PositionalEncoding(
            config.d_model, config.dropout, max_len=config.max_seq_len * 2
        )

        # --- Transformer Backbone ---
        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,  # We prefer [Batch, Seq, Dim]
        )

        # --- Output Head ---
        # Projects back to BPE vocab for text generation
        self.fc_out = nn.Linear(config.d_model, config.bpe_vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize parameters with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(
        self, sz: int, device: torch.device
    ) -> torch.Tensor:
        """Generates a causal mask for the decoder (prevent looking ahead)."""
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def _construct_encoder_input(
        self, src_left: torch.Tensor, src_target: torch.Tensor, src_right: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuses the three input components into a single sequence and creates the padding mask.

        Args:
            src_left: [Batch, Len_L] (BPE IDs)
            src_target: [Batch, Len_T] (Char IDs)
            src_right: [Batch, Len_R] (BPE IDs)

        Returns:
            src_emb: [Batch, Total_Len, d_model]
            src_key_padding_mask: [Batch, Total_Len] (Bool, True where padded)
        """
        batch_size = src_left.size(0)
        device = src_left.device

        # 1. Embeddings
        # Scale embeddings by sqrt(d_model) as per Attention is All You Need
        emb_left = self.bpe_embedding(src_left) * math.sqrt(self.d_model)
        emb_target = self.char_embedding(src_target) * math.sqrt(self.d_model)
        emb_right = self.bpe_embedding(src_right) * math.sqrt(self.d_model)

        # Expand SEP embedding to [Batch, 1, d_model]
        sep = self.sep_embedding.expand(batch_size, -1, -1)

        # Concatenate: Left -> SEP -> Target -> SEP -> Right
        src_emb = torch.cat([emb_left, sep, emb_target, sep, emb_right], dim=1)

        # 2. Padding Masks
        # True indicates the position should be ignored (is padding)
        mask_left = src_left == self.bpe_pad_id
        mask_target = src_target == self.char_pad_id
        mask_right = src_right == self.bpe_pad_id

        # SEP tokens are never padded (False)
        mask_sep = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)

        src_key_padding_mask = torch.cat(
            [mask_left, mask_sep, mask_target, mask_sep, mask_right], dim=1
        )

        return src_emb, src_key_padding_mask

    def forward(
        self,
        src_left: torch.Tensor,
        src_target: torch.Tensor,
        src_right: torch.Tensor,
        tgt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            src_left: Context left BPE IDs [Batch, Len_L]
            src_target: Target token Char IDs [Batch, Len_T]
            src_right: Context right BPE IDs [Batch, Len_R]
            tgt: Target normalized text BPE IDs [Batch, Len_Out] (Input to Decoder)

        Returns:
            Logits: [Batch, Len_Out, BPE_Vocab]
        """
        # 1. Prepare Encoder Input
        src_emb, src_key_padding_mask = self._construct_encoder_input(
            src_left, src_target, src_right
        )

        # Apply Positional Encoding to Source
        # Permute to [Seq, Batch, Dim] for PE, then back to [Batch, Seq, Dim]
        src_emb = self.pos_encoder(src_emb.transpose(0, 1)).transpose(0, 1)

        # 2. Prepare Decoder Input
        tgt_emb = self.bpe_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb.transpose(0, 1)).transpose(0, 1)

        # Masks for Decoder
        tgt_seq_len = tgt.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len, tgt.device)
        tgt_key_padding_mask = tgt == self.bpe_pad_id

        # 3. Transformer Pass
        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=None,  # Encoder is fully visible
            tgt_mask=tgt_mask,  # Decoder is causal
            memory_mask=None,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        # 4. Output Projection
        logits = self.fc_out(output)
        return logits

    def encode(
        self, src_left: torch.Tensor, src_target: torch.Tensor, src_right: torch.Tensor
    ) -> torch.Tensor:
        """
        Helper for inference: Runs only the encoder.
        Returns the memory (encoded representation).
        """
        src_emb, src_key_padding_mask = self._construct_encoder_input(
            src_left, src_target, src_right
        )
        src_emb = self.pos_encoder(src_emb.transpose(0, 1)).transpose(0, 1)

        memory = self.transformer.encoder(
            src=src_emb, src_key_padding_mask=src_key_padding_mask
        )
        return memory, src_key_padding_mask

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Helper for inference: Runs one step of the decoder.
        """
        tgt_emb = self.bpe_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb.transpose(0, 1)).transpose(0, 1)

        tgt_seq_len = tgt.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len, tgt.device)

        output = self.transformer.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.fc_out(output)
