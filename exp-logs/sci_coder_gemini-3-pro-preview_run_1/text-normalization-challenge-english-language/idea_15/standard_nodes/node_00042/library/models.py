import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Tuple, Dict, Optional

from library.config import Config
from library.utils import get_logger

logger = get_logger("models")


class CharCNN(nn.Module):
    """
    Character-level CNN for extracting morphological features from tokens.
    Input: (Batch, Seq_Len, Char_Len) -> Output: (Batch, Seq_Len, Embed_Dim)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        filters: int,
        kernel_size: int,
        output_dim: int,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,  # Same padding logic
        )
        self.relu = nn.ReLU()
        # Project to target dimension (usually LSTM hidden size)
        self.projection = nn.Linear(filters, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, char_len)
        batch_size, seq_len, char_len = x.shape

        # Flatten to (batch * seq_len, char_len) for processing
        x_flat = x.view(-1, char_len)

        # Embed: (batch * seq_len, char_len, embed_dim)
        emb = self.embedding(x_flat)

        # Transpose for Conv1d: (batch * seq_len, embed_dim, char_len)
        emb = emb.permute(0, 2, 1)

        # CNN -> ReLU -> MaxPool
        # Conv output: (batch * seq_len, filters, char_len)
        conv_out = self.relu(self.conv(emb))

        # Max Pool over character dimension: (batch * seq_len, filters)
        pooled, _ = torch.max(conv_out, dim=2)

        # Project to output dim
        out = self.projection(pooled)

        # Reshape back to sequence format: (batch_size, seq_len, output_dim)
        return out.view(batch_size, seq_len, -1)


class GatedBiLSTMTagger(nn.Module):
    """
    Stage 1: Gated Multi-Granularity Bi-LSTM Tagger.
    Fuses Word, BPE, Char-CNN, Regex, and Prior features using a gated mechanism.
    """

    def __init__(
        self,
        word_vocab_size: int,
        bpe_vocab_size: int,
        char_vocab_size: int,
        class_vocab_size: int,
        num_regex_feats: int,
        num_classes: int,
    ):
        super().__init__()

        self.hidden_dim = Config.TAGGER_LSTM_HIDDEN

        # --- 1. Feature Extractors ---

        # Word Embedding
        self.word_embedding = nn.Embedding(
            word_vocab_size, Config.TAGGER_EMBED_DIM_WORD, padding_idx=0
        )
        self.word_proj = nn.Linear(Config.TAGGER_EMBED_DIM_WORD, self.hidden_dim)

        # BPE Embedding (Average Pooling)
        self.bpe_embedding = nn.Embedding(
            bpe_vocab_size, Config.TAGGER_EMBED_DIM_BPE, padding_idx=0
        )
        self.bpe_proj = nn.Linear(Config.TAGGER_EMBED_DIM_BPE, self.hidden_dim)

        # Char CNN
        self.char_cnn = CharCNN(
            vocab_size=char_vocab_size,
            embed_dim=Config.TAGGER_EMBED_DIM_CHAR,
            filters=Config.TAGGER_CNN_FILTERS,
            kernel_size=Config.TAGGER_CNN_KERNEL_SIZE,
            output_dim=self.hidden_dim,
            padding_idx=0,
        )

        # Regex Features
        self.regex_proj = nn.Linear(num_regex_feats, self.hidden_dim)

        # Prior Features (Probabilities)
        # Input dimension is num_classes (size of probability vector)
        self.prior_proj = nn.Linear(num_classes, self.hidden_dim)
        self.prior_dropout = nn.Dropout(Config.TAGGER_PRIOR_DROPOUT)

        # --- 2. Gated Fusion ---

        # Gating Network: Takes concatenated [Mem, Gen] -> Scalar weight
        # Input dim is 2 * hidden_dim because we concat Mem and Gen vectors
        self.gate_net = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid(),
        )

        # --- 3. Backbone & Head ---

        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=Config.TAGGER_LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.TAGGER_DROPOUT if Config.TAGGER_LSTM_LAYERS > 1 else 0,
        )

        self.dropout = nn.Dropout(Config.TAGGER_DROPOUT)

        # Output is 2 * hidden_dim due to bidirectional LSTM
        self.classifier = nn.Linear(2 * self.hidden_dim, num_classes)

    def forward(
        self,
        word_ids: torch.Tensor,
        bpe_ids: torch.Tensor,
        char_ids: torch.Tensor,
        regex_feats: torch.Tensor,
        prior_feats: torch.Tensor,
    ) -> torch.Tensor:

        # --- Feature Projection ---

        # 1. Word: (B, L) -> (B, L, H)
        h_word = self.word_proj(self.word_embedding(word_ids))

        # 2. BPE: (B, L, BPE_Len) -> (B, L, H)
        # Embed -> Mean over BPE_Len dimension
        bpe_emb = self.bpe_embedding(bpe_ids)  # (B, L, BPE_Len, Emb)
        bpe_mean = torch.mean(bpe_emb, dim=2)  # (B, L, Emb)
        h_bpe = self.bpe_proj(bpe_mean)

        # 3. Char CNN: (B, L, Char_Len) -> (B, L, H)
        h_char = self.char_cnn(char_ids)

        # 4. Regex: (B, L, Feats) -> (B, L, H)
        h_regex = self.regex_proj(regex_feats)

        # 5. Priors: (B, L, Classes) -> (B, L, H)
        # Apply specific dropout to priors to force reliance on other feats
        h_prior = self.prior_proj(self.prior_dropout(prior_feats))

        # --- Gated Fusion ---

        # Group 1: Memorization (Word + Priors)
        h_mem = h_word + h_prior

        # Group 2: Generalization (BPE + Char + Regex)
        h_gen = h_bpe + h_char + h_regex

        # Compute Gate
        # Concatenate along feature dimension
        combined = torch.cat([h_mem, h_gen], dim=-1)  # (B, L, 2*H)
        gate = self.gate_net(combined)  # (B, L, 1)

        # Apply Gate
        # z * Mem + (1-z) * Gen
        h_fused = gate * h_mem + (1 - gate) * h_gen

        # --- LSTM Backbone ---

        # Pack sequence for efficiency and correctness with padding
        # Lengths based on non-padding word_ids
        lengths = (word_ids != 0).sum(dim=1).cpu()

        # Handle case where a sequence might be length 0 (unlikely but safe to check)
        # or if all are padding. We assume valid data.

        packed_input = nn.utils.rnn.pack_padded_sequence(
            h_fused, lengths, batch_first=True, enforce_sorted=False
        )

        packed_output, _ = self.lstm(packed_input)

        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=word_ids.size(1)
        )

        # --- Classification Head ---

        logits = self.classifier(self.dropout(output))
        return logits


class Seq2SeqFallback(nn.Module):
    """
    Stage 2: Character-Level LSTM Seq2Seq.
    Conditioned on the predicted class embedding.
    """

    def __init__(
        self,
        char_vocab_size: int,
        class_vocab_size: int,
        sos_idx: int,
        eos_idx: int,
        max_seq_len: int = 128,
    ):
        super().__init__()

        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.max_seq_len = max_seq_len

        embed_dim = Config.SEQ2SEQ_EMBED_DIM
        hidden_dim = Config.SEQ2SEQ_HIDDEN_DIM

        # Embeddings
        self.char_embedding = nn.Embedding(char_vocab_size, embed_dim, padding_idx=0)
        self.class_embedding = nn.Embedding(class_vocab_size, embed_dim)

        # Encoder
        self.encoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=Config.SEQ2SEQ_LAYERS,
            batch_first=True,
            dropout=Config.SEQ2SEQ_DROPOUT if Config.SEQ2SEQ_LAYERS > 1 else 0,
        )

        # Decoder
        # Input to decoder is: Char_Emb + Class_Emb
        self.decoder_cell = nn.LSTMCell(
            input_size=embed_dim + embed_dim, hidden_size=hidden_dim
        )

        self.output_layer = nn.Linear(hidden_dim, char_vocab_size)
        self.dropout = nn.Dropout(Config.SEQ2SEQ_DROPOUT)

    def forward(
        self,
        src_ids: torch.Tensor,
        class_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Forward pass for training.
        Returns logits: (Batch, Seq_Len, Vocab_Size)
        """
        batch_size = src_ids.size(0)
        target_len = tgt_ids.size(1)
        vocab_size = self.output_layer.out_features

        # --- Encode ---
        src_emb = self.dropout(self.char_embedding(src_ids))
        _, (hidden, cell) = self.encoder(src_emb)

        # Prepare Class Embedding (Conditioning)
        # (Batch, Embed_Dim)
        class_emb = self.class_embedding(class_ids)

        # --- Decode ---
        # Initialize outputs container
        outputs = torch.zeros(batch_size, target_len, vocab_size).to(src_ids.device)

        # First input is <sos> token (assumed to be at tgt_ids[:, 0])
        input_token = tgt_ids[:, 0]

        # Decoder hidden state initialized with Encoder final state
        # If multi-layer, we take the last layer for LSTMCell (which is single layer usually in this simple impl)
        # For simplicity in this fallback, we assume 1 layer or handle shape matching.
        # Config says SEQ2SEQ_LAYERS = 1.
        dec_hidden = hidden[-1]
        dec_cell = cell[-1]

        for t in range(1, target_len):
            # Embed input char
            char_emb = self.char_embedding(input_token)  # (Batch, Emb)

            # Concatenate with Class Embedding
            decoder_input = torch.cat([char_emb, class_emb], dim=1)  # (Batch, 2*Emb)
            decoder_input = self.dropout(decoder_input)

            # Step
            dec_hidden, dec_cell = self.decoder_cell(
                decoder_input, (dec_hidden, dec_cell)
            )

            # Project to Vocab
            prediction = self.output_layer(dec_hidden)  # (Batch, Vocab)
            outputs[:, t, :] = prediction

            # Teacher Forcing
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            if use_teacher_forcing:
                input_token = tgt_ids[:, t]
            else:
                input_token = prediction.argmax(1)

        return outputs

    def generate(self, src_ids: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
        """
        Inference (Greedy Decoding).
        Returns predicted token indices: (Batch, Max_Len)
        """
        batch_size = src_ids.size(0)

        # Encode
        src_emb = self.char_embedding(src_ids)
        _, (hidden, cell) = self.encoder(src_emb)

        class_emb = self.class_embedding(class_ids)

        dec_hidden = hidden[-1]
        dec_cell = cell[-1]

        # Start with <sos>
        input_token = torch.full(
            (batch_size,), self.sos_idx, dtype=torch.long, device=src_ids.device
        )

        predictions = []

        for _ in range(self.max_seq_len):
            char_emb = self.char_embedding(input_token)
            decoder_input = torch.cat([char_emb, class_emb], dim=1)

            dec_hidden, dec_cell = self.decoder_cell(
                decoder_input, (dec_hidden, dec_cell)
            )
            logits = self.output_layer(dec_hidden)

            # Greedy
            input_token = logits.argmax(1)
            predictions.append(input_token.unsqueeze(1))

            # Optimization: Can stop if all batches generated <eos>, but fixed length is simpler for batching

        return torch.cat(predictions, dim=1)
