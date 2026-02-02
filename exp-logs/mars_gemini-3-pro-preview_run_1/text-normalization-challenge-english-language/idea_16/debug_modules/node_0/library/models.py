import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from library.config import Config


class MorphoBiLSTMTagger(nn.Module):
    """
    Tri-Hybrid Tagger: Combines Word Embeddings, Character CNN, and Regex Features
    fed into a Bi-LSTM for robust token classification.
    """

    def __init__(self, word_vocab_size, class_vocab_size, char_vocab_size):
        super(MorphoBiLSTMTagger, self).__init__()

        # 1. Word Embeddings
        self.word_embedding = nn.Embedding(
            num_embeddings=word_vocab_size,
            embedding_dim=Config.EMBED_DIM_WORD,
            padding_idx=0,
        )

        # 2. Character CNN Components
        self.char_embedding = nn.Embedding(
            num_embeddings=char_vocab_size,
            embedding_dim=Config.EMBED_DIM_CHAR,
            padding_idx=0,
        )
        self.char_conv = nn.Conv1d(
            in_channels=Config.EMBED_DIM_CHAR,
            out_channels=Config.CHAR_CNN_FILTERS,
            kernel_size=Config.CHAR_CNN_KERNEL_SIZE,
            padding=1,  # 'same' padding approximation
        )

        # 3. Regex Projection
        # Project sparse binary regex features to a dense vector matching CNN output size
        self.regex_projection = nn.Linear(
            Config.NUM_REGEX_FEATURES, Config.CHAR_CNN_FILTERS
        )

        # 4. Bi-LSTM Backbone
        # Input: Word(256) + Char(64) + Regex(64) = 384 (if using default config)
        lstm_input_dim = (
            Config.EMBED_DIM_WORD + Config.CHAR_CNN_FILTERS + Config.CHAR_CNN_FILTERS
        )

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=Config.HIDDEN_DIM_TAGGER,
            num_layers=Config.NUM_LAYERS_TAGGER,
            bidirectional=True,
            batch_first=True,
            dropout=Config.DROPOUT_TAGGER if Config.NUM_LAYERS_TAGGER > 1 else 0,
        )

        self.dropout = nn.Dropout(Config.DROPOUT_TAGGER)

        # 5. Classification Head
        self.classifier = nn.Linear(
            Config.HIDDEN_DIM_TAGGER * 2, class_vocab_size  # *2 for Bidirectional
        )

    def forward(self, word_ids, char_features, regex_features, lengths):
        """
        Args:
            word_ids: [Batch, SeqLen]
            char_features: [Batch, SeqLen, CharLen]
            regex_features: [Batch, SeqLen, NumRegex]
            lengths: [Batch]
        """
        batch_size, seq_len = word_ids.size()

        # --- Word Path ---
        word_emb = self.word_embedding(word_ids)  # [Batch, Seq, WordEmb]

        # --- Character Path ---
        # Collapse Batch and Seq dims to process all tokens in parallel
        char_in = char_features.view(-1, Config.MAX_CHAR_LEN)  # [Batch*Seq, CharLen]
        char_emb = self.char_embedding(char_in)  # [Batch*Seq, CharLen, CharEmb]

        # Permute for CNN: [N, Emb, L]
        char_emb = char_emb.permute(0, 2, 1)

        # Conv + Max Pool
        char_conv = self.char_conv(char_emb)  # [Batch*Seq, Filters, L_out]
        char_pool, _ = torch.max(char_conv, dim=2)  # [Batch*Seq, Filters]

        # Reshape back to sequence structure
        char_out = char_pool.view(batch_size, seq_len, -1)  # [Batch, Seq, Filters]

        # --- Regex Path ---
        regex_proj = self.regex_projection(regex_features)  # [Batch, Seq, Filters]
        regex_out = F.relu(regex_proj)

        # --- Feature Fusion ---
        combined_input = torch.cat([word_emb, char_out, regex_out], dim=2)
        combined_input = self.dropout(combined_input)

        # --- Sequence Modeling (Bi-LSTM) ---
        # Move lengths to CPU for pack_padded_sequence
        lengths_cpu = lengths.cpu()

        packed_input = pack_padded_sequence(
            combined_input, lengths_cpu, batch_first=True, enforce_sorted=False
        )

        packed_output, _ = self.lstm(packed_input)

        lstm_out, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=seq_len
        )

        lstm_out = self.dropout(lstm_out)

        # --- Classification ---
        logits = self.classifier(lstm_out)  # [Batch, Seq, NumClasses]

        return logits


class CharSeq2Seq(nn.Module):
    """
    LSTM-based Encoder-Decoder for text normalization fallback.
    Conditioned on the Class Embedding predicted by the Tagger.
    """

    def __init__(self, char_vocab_size, num_classes):
        super(CharSeq2Seq, self).__init__()

        # Embeddings
        self.embedding = nn.Embedding(
            num_embeddings=char_vocab_size,
            embedding_dim=Config.EMBED_DIM_SEQ2SEQ,
            padding_idx=0,
        )

        # Class Embedding (Conditioning)
        self.class_embedding = nn.Embedding(
            num_embeddings=num_classes, embedding_dim=Config.HIDDEN_DIM_SEQ2SEQ
        )

        # Encoder (Unidirectional for simplicity in fallback)
        self.encoder = nn.LSTM(
            input_size=Config.EMBED_DIM_SEQ2SEQ,
            hidden_size=Config.HIDDEN_DIM_SEQ2SEQ,
            num_layers=Config.NUM_LAYERS_SEQ2SEQ,
            batch_first=True,
        )

        # Decoder
        self.decoder = nn.LSTM(
            input_size=Config.EMBED_DIM_SEQ2SEQ,
            hidden_size=Config.HIDDEN_DIM_SEQ2SEQ,
            num_layers=Config.NUM_LAYERS_SEQ2SEQ,
            batch_first=True,
        )

        self.fc = nn.Linear(Config.HIDDEN_DIM_SEQ2SEQ, char_vocab_size)
        self.dropout = nn.Dropout(Config.DROPOUT_SEQ2SEQ)

    def forward(self, src_ids, src_lens, class_ids, tgt_ids=None):
        """
        Args:
            src_ids: [Batch, SrcLen]
            src_lens: [Batch]
            class_ids: [Batch]
            tgt_ids: [Batch, TgtLen] (Optional, if provided runs training mode)

        Returns:
            If tgt_ids is provided: Logits [Batch, TgtLen-1, Vocab]
            If tgt_ids is None: Predicted Indices [Batch, MaxGenLen]
        """
        batch_size = src_ids.size(0)

        # --- Encoder ---
        src_emb = self.embedding(src_ids)
        src_emb = self.dropout(src_emb)

        packed_src = pack_padded_sequence(
            src_emb, src_lens.cpu(), batch_first=True, enforce_sorted=False
        )

        _, (hidden, cell) = self.encoder(packed_src)

        # --- Conditioning ---
        # Retrieve class embedding
        class_emb = self.class_embedding(class_ids)  # [Batch, Hidden]

        # Add class info to the encoder final states to initialize decoder
        # hidden: [NumLayers, Batch, Hidden]
        # We broadcast add: [1, B, H] + [B, H] -> [1, B, H]
        hidden = hidden + class_emb.unsqueeze(0)
        cell = cell + class_emb.unsqueeze(0)

        # --- Decoder ---
        if tgt_ids is not None:
            # Training Mode (Teacher Forcing)
            # Input: <SOS>...<LastChar> (tgt_ids[:, :-1])
            dec_input = tgt_ids[:, :-1]
            dec_emb = self.embedding(dec_input)
            dec_emb = self.dropout(dec_emb)

            dec_out, _ = self.decoder(dec_emb, (hidden, cell))
            logits = self.fc(dec_out)
            return logits

        else:
            # Inference Mode (Greedy Search)
            # SOS ID is 2 (based on dataset.py specials)
            sos_id = 2

            # Initial input: <SOS>
            curr_input = torch.full(
                (batch_size, 1), sos_id, dtype=torch.long, device=src_ids.device
            )

            outputs = []
            curr_hidden, curr_cell = hidden, cell

            for _ in range(Config.MAX_GEN_LEN):
                dec_emb = self.embedding(curr_input)  # [Batch, 1, Emb]

                dec_out, (curr_hidden, curr_cell) = self.decoder(
                    dec_emb, (curr_hidden, curr_cell)
                )

                logits = self.fc(dec_out)  # [Batch, 1, Vocab]

                # Greedy selection
                top1 = logits.argmax(dim=2)  # [Batch, 1]
                outputs.append(top1)

                # Next input
                curr_input = top1

                # Optional: Break if all batches produced EOS (not implemented for speed)

            return torch.cat(outputs, dim=1)
