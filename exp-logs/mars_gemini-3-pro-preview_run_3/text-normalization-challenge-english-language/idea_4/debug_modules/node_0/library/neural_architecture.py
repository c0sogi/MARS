import math
import torch
import torch.nn as nn
from library.config import Config


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for Transformer models.
    Injects information about the relative or absolute position of tokens in the sequence.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant 'pe' matrix with values dependent on pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a learnable parameter, but part of state_dict)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Add positional encoding to embeddings
        # x.size(1) is the sequence length
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class RAGTransformer(nn.Module):
    """
    Retrieval-Augmented Character Transformer.

    A Seq2Seq Transformer model designed to normalize text by leveraging
    retrieved examples in the input sequence.

    Input format: [Context] <SEP> [Retrieved_Src] <SEP> [Retrieved_Tgt] <SEP> [Target_Raw]
    Output: Normalized text sequence.
    """

    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        d_model: int = Config.EMBED_DIM,
        nhead: int = Config.N_HEADS,
        num_encoder_layers: int = Config.N_ENCODER_LAYERS,
        num_decoder_layers: int = Config.N_DECODER_LAYERS,
        dim_feedforward: int = Config.HIDDEN_DIM,
        dropout: float = Config.DROPOUT,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(
            d_model, dropout, max_len=Config.MAX_SEQ_LEN * 2
        )

        # Transformer Core
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

        # Output Projection
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        """Initialize parameters with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _create_masks(
        self, src: torch.Tensor, tgt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Creates masks for the Transformer.

        1. src_padding_mask: Ignored positions in source (True where pad).
        2. tgt_padding_mask: Ignored positions in target (True where pad).
        3. tgt_mask: Causal mask to prevent attending to future tokens in decoder.
        """
        # Padding masks: (Batch, Seq_Len), True where value is pad_token_id
        src_padding_mask = src == self.pad_token_id
        tgt_padding_mask = tgt == self.pad_token_id

        # Causal mask for decoder: (Tgt_Seq_Len, Tgt_Seq_Len)
        # -inf where we shouldn't attend (upper triangular), 0 otherwise
        tgt_seq_len = tgt.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(
            src.device
        )

        return src_padding_mask, tgt_padding_mask, tgt_mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training.

        Args:
            src: Source sequence indices (Batch, Src_Len)
            tgt: Target sequence indices (Batch, Tgt_Len) - usually shifted right for teacher forcing

        Returns:
            Logits: (Batch, Tgt_Len, Vocab_Size)
        """
        # Create masks
        src_padding_mask, tgt_padding_mask, tgt_mask = self._create_masks(src, tgt)

        # Embed and add positional encoding
        # Scale embeddings by sqrt(d_model) as per Attention is All You Need
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))

        # Transformer Pass
        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,  # Mask encoder output for decoder attention
        )

        # Project to vocabulary
        logits = self.fc_out(output)
        return logits

    def generate(
        self,
        src: torch.Tensor,
        sos_token_id: int,
        eos_token_id: int,
        max_len: int = 128,
    ) -> torch.Tensor:
        """
        Performs greedy decoding for inference.

        Args:
            src: Source sequence indices (Batch, Src_Len)
            sos_token_id: Start of Sequence token ID
            eos_token_id: End of Sequence token ID
            max_len: Maximum length of generated sequence

        Returns:
            Generated sequence indices (Batch, Generated_Len)
        """
        batch_size = src.size(0)
        device = src.device

        # 1. Encode Source
        src_padding_mask = src == self.pad_token_id
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))

        # We use the transformer's encoder directly to save computation in the loop
        memory = self.transformer.encoder(
            src_emb, src_key_padding_mask=src_padding_mask
        )

        # 2. Initialize Decoder Input
        # Start with <sos> for every sample in batch
        ys = torch.full((batch_size, 1), sos_token_id, dtype=torch.long, device=device)

        # 3. Autoregressive Loop
        # We track which sequences have finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            # Embed decoder input
            tgt_emb = self.pos_encoder(self.embedding(ys) * math.sqrt(self.d_model))

            # Create causal mask for current length
            tgt_mask = self.transformer.generate_square_subsequent_mask(ys.size(1)).to(
                device
            )

            # Decode
            # Note: We pass the pre-computed memory (encoder output)
            out = self.transformer.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_padding_mask,
            )

            # Get logits for the last token only
            last_token_logits = self.fc_out(out[:, -1, :])

            # Greedy selection
            _, next_word = torch.max(last_token_logits, dim=1)
            next_word = next_word.unsqueeze(1)

            # Append to sequence
            ys = torch.cat([ys, next_word], dim=1)

            # Update finished status
            # If a sequence generated EOS, mark as finished
            # (We continue generating for others, but could optimize to stop early if all finished)
            just_finished = next_word.squeeze(1) == eos_token_id
            finished = finished | just_finished

            if finished.all():
                break

        return ys
