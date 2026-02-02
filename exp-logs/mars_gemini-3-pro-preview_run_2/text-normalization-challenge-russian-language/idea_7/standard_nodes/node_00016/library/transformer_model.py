import torch
import torch.nn as nn
import math
from library.config import Config


class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as the embeddings,
    so that the two can be summed.
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
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    """
    Helper class to embed tokens and scale by sqrt(d_model).
    """

    def __init__(self, vocab_size: int, emb_size: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class CharToSubwordTransformer(nn.Module):
    """
    A Transformer-based Seq2Seq model for text normalization.
    Encoder: Character-level input (contextualized).
    Decoder: Subword-level output (BPE).
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        pad_idx_src: int,
        pad_idx_tgt: int,
        bos_idx: int,
        eos_idx: int,
    ):
        super(CharToSubwordTransformer, self).__init__()

        self.pad_idx_src = pad_idx_src
        self.pad_idx_tgt = pad_idx_tgt
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

        # Embeddings
        self.src_tok_emb = TokenEmbedding(src_vocab_size, Config.D_MODEL)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, Config.D_MODEL)
        self.positional_encoding = PositionalEncoding(
            Config.D_MODEL, dropout=Config.DROPOUT
        )

        # Transformer Core
        # Note: batch_first=True requires PyTorch >= 1.10, but is generally preferred.
        # If running on older versions, ensure inputs are transposed.
        self.transformer = nn.Transformer(
            d_model=Config.D_MODEL,
            nhead=Config.NHEAD,
            num_encoder_layers=Config.NUM_ENCODER_LAYERS,
            num_decoder_layers=Config.NUM_DECODER_LAYERS,
            dim_feedforward=Config.DIM_FEEDFORWARD,
            dropout=Config.DROPOUT,
            batch_first=True,
        )

        # Output Generator
        self.generator = nn.Linear(Config.D_MODEL, tgt_vocab_size)

        # Weight Initialization
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Runs the encoder pass.
        """
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)),
            src_key_padding_mask=src_mask,
        )

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Runs the decoder pass.
        """
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)),
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training.

        Args:
            src: Source tensor (batch, src_len)
            tgt: Target tensor (batch, tgt_len) - includes BOS and EOS

        Returns:
            Logits (batch, tgt_len, tgt_vocab_size)
        """
        # Create Padding Masks (True indicates padding location)
        src_key_padding_mask = src == self.pad_idx_src
        tgt_key_padding_mask = tgt == self.pad_idx_tgt

        # Create Causal Mask for Target
        tgt_seq_len = tgt.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(
            src.device
        )

        # Embeddings + Positional Encoding
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))

        # Transformer Pass
        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
            tgt_mask=tgt_mask,
        )

        return self.generator(outs)

    def predict(self, src: torch.Tensor, max_len: int = None) -> torch.Tensor:
        """
        Inference using Greedy Decoding.

        Args:
            src: Source tensor (batch, src_len)
            max_len: Maximum length of generated sequence

        Returns:
            Generated target indices (batch, generated_len)
        """
        if max_len is None:
            max_len = Config.MAX_OUTPUT_LEN

        self.eval()
        device = src.device
        batch_size = src.size(0)

        # Create source mask
        src_mask = src == self.pad_idx_src

        # Encode
        memory = self.encode(src, src_mask)

        # Initialize decoder input with BOS token
        ys = torch.full((batch_size, 1), self.bos_idx, dtype=torch.long, device=device)

        # Track finished sequences
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for i in range(max_len):
            # Generate causal mask for current sequence length
            tgt_mask = self.transformer.generate_square_subsequent_mask(ys.size(1)).to(
                device
            )

            # Decode
            out = self.decode(
                ys, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_mask
            )

            # Get logits for the last token
            prob = self.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)

            # If a sequence is already finished, force padding
            next_word = next_word.masked_fill(finished, self.pad_idx_tgt)

            # Append to sequence
            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)

            # Update finished status
            finished |= next_word == self.eos_idx

            if finished.all():
                break

        return ys
