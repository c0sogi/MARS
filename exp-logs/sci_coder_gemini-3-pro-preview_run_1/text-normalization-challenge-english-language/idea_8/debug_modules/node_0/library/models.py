import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from library.config import Config


class PositionalEncoding(nn.Module):
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
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (Batch, Seq_Len, D_Model)
        # Slice pe to the current sequence length
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class BiLSTMTagger(nn.Module):
    def __init__(
        self,
        vocab_size,
        char_vocab_size,
        num_classes,
        token_pad_idx=None,
        char_pad_idx=None,
    ):
        """
        Bi-LSTM Tagger with Hybrid Word + Character CNN inputs.
        """
        super(BiLSTMTagger, self).__init__()

        self.token_pad_idx = token_pad_idx if token_pad_idx is not None else 0
        self.char_pad_idx = char_pad_idx if char_pad_idx is not None else 0

        # 1. Word Embeddings
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=Config.TAGGER_EMBED_DIM,
            padding_idx=self.token_pad_idx,
        )

        # 2. Character Embeddings & CNN
        self.char_embedding = nn.Embedding(
            num_embeddings=char_vocab_size,
            embedding_dim=Config.TAGGER_CHAR_EMBED_DIM,
            padding_idx=self.char_pad_idx,
        )

        # CNN to extract morphological features from characters
        # Kernel size 3, Padding 1 preserves length (L_out = L_in)
        self.char_cnn = nn.Conv1d(
            in_channels=Config.TAGGER_CHAR_EMBED_DIM,
            out_channels=Config.TAGGER_CNN_FILTERS,
            kernel_size=Config.TAGGER_CNN_KERNEL_SIZE,
            padding=1,
        )

        # 3. Bi-LSTM
        # Input size = Word Emb Dim + Char CNN Filters
        lstm_input_dim = Config.TAGGER_EMBED_DIM + Config.TAGGER_CNN_FILTERS

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=Config.TAGGER_HIDDEN_DIM,
            num_layers=Config.TAGGER_NUM_LAYERS,
            bidirectional=True,
            batch_first=True,
            dropout=Config.TAGGER_DROPOUT if Config.TAGGER_NUM_LAYERS > 1 else 0,
        )

        self.dropout = nn.Dropout(Config.TAGGER_DROPOUT)

        # 4. Classifier
        # Bidirectional LSTM outputs 2 * hidden_dim
        self.fc = nn.Linear(Config.TAGGER_HIDDEN_DIM * 2, num_classes)

    def forward(self, token_ids, char_ids, lengths):
        """
        Args:
            token_ids: (Batch, Seq_Len)
            char_ids: (Batch, Seq_Len, Char_Len)
            lengths: (Batch) - Actual lengths of sequences for packing
        """
        batch_size, seq_len = token_ids.size()

        # --- Process Characters ---
        # Flatten to apply CNN to all tokens at once: (Batch * Seq_Len, Char_Len)
        char_inputs = char_ids.view(-1, char_ids.size(-1))

        # Embed: (Batch * Seq_Len, Char_Len, Char_Dim)
        char_embs = self.char_embedding(char_inputs)

        # Transpose for CNN (expects channels first): (Batch * Seq_Len, Char_Dim, Char_Len)
        char_embs = char_embs.permute(0, 2, 1)

        # CNN: (Batch * Seq_Len, Filters, Char_Len)
        char_feats = self.char_cnn(char_embs)

        # Max Pool over time (character dimension): (Batch * Seq_Len, Filters)
        # Squeeze removes the dimension of size 1
        char_feats = F.max_pool1d(char_feats, kernel_size=char_feats.size(2)).squeeze(2)

        # Reshape back to sequence format: (Batch, Seq_Len, Filters)
        char_feats = char_feats.view(batch_size, seq_len, -1)

        # --- Process Tokens ---
        # Embed: (Batch, Seq_Len, Token_Dim)
        token_embs = self.token_embedding(token_ids)

        # --- Concatenate Features ---
        # (Batch, Seq_Len, Token_Dim + Filters)
        combined_input = torch.cat([token_embs, char_feats], dim=2)
        combined_input = self.dropout(combined_input)

        # --- Bi-LSTM ---
        # Pack sequence for efficient RNN processing (ignores padding)
        # lengths must be on CPU for pack_padded_sequence
        lengths_cpu = lengths.cpu()
        packed_input = nn.utils.rnn.pack_padded_sequence(
            combined_input, lengths_cpu, batch_first=True, enforce_sorted=False
        )

        packed_output, _ = self.lstm(packed_input)

        # Unpack back to padded sequence
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=seq_len
        )

        lstm_output = self.dropout(lstm_output)

        # --- Classifier ---
        logits = self.fc(lstm_output)

        return logits


class TransformerSeq2Seq(nn.Module):
    def __init__(
        self, char_vocab_size, num_classes, pad_idx=None, sos_idx=None, eos_idx=None
    ):
        """
        Transformer Seq2Seq model for generating normalized text.
        Conditioned on the class via a learned embedding prepended to the encoder input.
        """
        super(TransformerSeq2Seq, self).__init__()

        self.d_model = Config.SEQ2SEQ_D_MODEL
        self.pad_idx = pad_idx if pad_idx is not None else 0
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.char_vocab_size = char_vocab_size

        # Embeddings
        self.char_embedding = nn.Embedding(
            char_vocab_size, self.d_model, padding_idx=self.pad_idx
        )
        self.class_embedding = nn.Embedding(num_classes, self.d_model)

        self.pos_encoder = PositionalEncoding(
            self.d_model, dropout=Config.SEQ2SEQ_DROPOUT
        )

        # Transformer
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=Config.SEQ2SEQ_NHEAD,
            num_encoder_layers=Config.SEQ2SEQ_NUM_ENCODER_LAYERS,
            num_decoder_layers=Config.SEQ2SEQ_NUM_DECODER_LAYERS,
            dim_feedforward=Config.SEQ2SEQ_DIM_FEEDFORWARD,
            dropout=Config.SEQ2SEQ_DROPOUT,
            batch_first=True,
        )

        # Output Head
        self.fc_out = nn.Linear(self.d_model, char_vocab_size)

    def generate_square_subsequent_mask(self, sz):
        """Generates a causal mask for the decoder."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, src_ids, tgt_ids, class_ids):
        """
        Args:
            src_ids: (Batch, Src_Len) - Source characters
            tgt_ids: (Batch, Tgt_Len) - Target characters (input to decoder)
            class_ids: (Batch) - Class labels for conditioning
        """
        device = src_ids.device

        # --- Prepare Source (Encoder Input) ---
        # Embed chars: (Batch, Src_Len, D_Model)
        src_emb = self.char_embedding(src_ids)

        # Embed class: (Batch, 1, D_Model)
        cls_emb = self.class_embedding(class_ids).unsqueeze(1)

        # Prepend Class Embedding to Source: (Batch, 1 + Src_Len, D_Model)
        # This allows the encoder to attend to the class type globally
        src_full_emb = torch.cat([cls_emb, src_emb], dim=1)
        src_full_emb = self.pos_encoder(src_full_emb)

        # --- Prepare Target (Decoder Input) ---
        # Embed tgt: (Batch, Tgt_Len, D_Model)
        tgt_emb = self.char_embedding(tgt_ids)
        tgt_emb = self.pos_encoder(tgt_emb)

        # --- Masks ---
        # Src Padding Mask: (Batch, 1 + Src_Len)
        # The class token (index 0) is never padding, so mask is False
        cls_mask = torch.zeros((src_ids.size(0), 1), dtype=torch.bool, device=device)
        src_pad_mask = src_ids == self.pad_idx
        src_key_padding_mask = torch.cat([cls_mask, src_pad_mask], dim=1)

        # Tgt Padding Mask: (Batch, Tgt_Len)
        tgt_key_padding_mask = tgt_ids == self.pad_idx

        # Tgt Causal Mask: (Tgt_Len, Tgt_Len)
        tgt_mask = self.generate_square_subsequent_mask(tgt_ids.size(1)).to(device)

        # --- Transformer Forward ---
        output = self.transformer(
            src=src_full_emb,
            tgt=tgt_emb,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_mask=tgt_mask,
        )

        # --- Output Projection ---
        logits = self.fc_out(output)
        return logits

    def predict(self, src_ids, class_ids, max_len=128):
        """
        Greedy decoding for inference.
        """
        device = src_ids.device
        batch_size = src_ids.size(0)

        # --- Encode ---
        src_emb = self.char_embedding(src_ids)
        cls_emb = self.class_embedding(class_ids).unsqueeze(1)
        src_full_emb = torch.cat([cls_emb, src_emb], dim=1)
        src_full_emb = self.pos_encoder(src_full_emb)

        # Source Mask
        cls_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
        src_pad_mask = src_ids == self.pad_idx
        src_key_padding_mask = torch.cat([cls_mask, src_pad_mask], dim=1)

        # Run Encoder once
        memory = self.transformer.encoder(
            src_full_emb, src_key_padding_mask=src_key_padding_mask
        )

        # --- Decode Loop ---
        # Start with SOS token
        ys = torch.full((batch_size, 1), self.sos_idx, dtype=torch.long, device=device)

        # Track finished sequences
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for i in range(max_len):
            tgt_emb = self.char_embedding(ys)
            tgt_emb = self.pos_encoder(tgt_emb)

            tgt_mask = self.generate_square_subsequent_mask(ys.size(1)).to(device)

            # Decoder step
            out = self.transformer.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )

            # Get logits for the last token only
            last_token_logits = self.fc_out(out[:, -1, :])

            # Greedy selection
            _, next_word = torch.max(last_token_logits, dim=1)

            # Append prediction
            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)

            # Check EOS
            if self.eos_idx is not None:
                finished |= next_word == self.eos_idx
                if finished.all():
                    break

        # Remove SOS token from start
        return ys[:, 1:]
