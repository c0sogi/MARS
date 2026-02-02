import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class CharCNN(nn.Module):
    """
    Character-level CNN to extract morphological features from tokens.
    Input: (Batch, Seq_Len, Char_Len)
    Output: (Batch, Seq_Len, CNN_Filters)
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        filters,
        kernel_size,
        padding_idx=0,
        dropout=0.0,
    ):
        super(CharCNN, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )
        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,  # Same padding-ish strategy
        )
        self.dropout = nn.Dropout(dropout)
        self.padding_idx = padding_idx

    def forward(self, x):
        # x shape: (batch_size, seq_len, char_len)
        batch_size, seq_len, char_len = x.size()

        # Flatten batch and seq dimensions to treat each token as an independent sample
        # Shape: (batch_size * seq_len, char_len)
        x_flat = x.view(-1, char_len)

        # Embedding
        # Shape: (batch_size * seq_len, char_len, embedding_dim)
        emb = self.embedding(x_flat)

        # Transpose for Conv1d (expects N, C, L)
        # Shape: (batch_size * seq_len, embedding_dim, char_len)
        emb = emb.transpose(1, 2)

        # Convolution
        # Shape: (batch_size * seq_len, filters, char_len_out)
        conv_out = self.conv(emb)

        # Activation
        conv_out = F.relu(conv_out)

        # Global Max Pooling over character dimension
        # Shape: (batch_size * seq_len, filters)
        pooled, _ = torch.max(conv_out, dim=2)

        pooled = self.dropout(pooled)

        # Reshape back to (batch_size, seq_len, filters)
        return pooled.view(batch_size, seq_len, -1)


class MultiGranularityTagger(nn.Module):
    """
    Multi-Granularity Bi-LSTM Tagger.
    Fuses Word Embeddings, CharCNN features, and BPE Embeddings.
    """

    def __init__(
        self,
        word_vocab_size,
        char_vocab_size,
        bpe_vocab_size,
        class_vocab_size,
        # Dimensions from Config
        word_emb_dim=Config.TAGGER_WORD_EMBEDDING_DIM,
        char_emb_dim=Config.TAGGER_CHAR_EMBEDDING_DIM,
        bpe_emb_dim=Config.TAGGER_BPE_EMBEDDING_DIM,
        cnn_filters=Config.TAGGER_CNN_FILTERS,
        cnn_kernel=Config.TAGGER_CNN_KERNEL_SIZE,
        hidden_dim=Config.TAGGER_HIDDEN_DIM,
        num_layers=Config.TAGGER_NUM_LAYERS,
        dropout=Config.TAGGER_DROPOUT,
        pad_idx=Config.PAD_IDX,
    ):
        super(MultiGranularityTagger, self).__init__()

        self.pad_idx = pad_idx

        # 1. Word Embedding Stream
        self.word_embedding = nn.Embedding(
            word_vocab_size, word_emb_dim, padding_idx=pad_idx
        )

        # 2. Character CNN Stream
        self.char_cnn = CharCNN(
            vocab_size=char_vocab_size,
            embedding_dim=char_emb_dim,
            filters=cnn_filters,
            kernel_size=cnn_kernel,
            padding_idx=pad_idx,
            dropout=dropout,
        )

        # 3. BPE Embedding Stream
        self.bpe_embedding = nn.Embedding(
            bpe_vocab_size, bpe_emb_dim, padding_idx=pad_idx
        )

        # Feature Fusion
        # Input size to LSTM is sum of all feature dimensions
        self.input_dim = word_emb_dim + cnn_filters + bpe_emb_dim

        # Backbone: Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Classification Head
        # Bidirectional LSTM outputs 2 * hidden_dim
        self.fc_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, class_vocab_size)

    def forward(self, word_ids, char_ids, bpe_ids, mask=None):
        """
        Args:
            word_ids: (batch, seq_len)
            char_ids: (batch, seq_len, char_len)
            bpe_ids: (batch, seq_len, bpe_len)
            mask: (batch, seq_len) - True for valid tokens, False for padding
        """
        # 1. Word Features
        # Shape: (batch, seq_len, word_emb_dim)
        word_emb = self.word_embedding(word_ids)

        # 2. Char Features
        # Shape: (batch, seq_len, cnn_filters)
        char_feat = self.char_cnn(char_ids)

        # 3. BPE Features
        # bpe_ids shape: (batch, seq_len, bpe_len)
        # Look up embeddings: (batch, seq_len, bpe_len, bpe_emb_dim)
        bpe_emb = self.bpe_embedding(bpe_ids)

        # Create mask for BPE padding to ignore pad tokens in averaging
        # Shape: (batch, seq_len, bpe_len, 1)
        bpe_mask = (bpe_ids != self.pad_idx).unsqueeze(-1).float()

        # Sum embeddings over bpe_len dimension
        # Shape: (batch, seq_len, bpe_emb_dim)
        bpe_sum = torch.sum(bpe_emb * bpe_mask, dim=2)

        # Count non-pad tokens
        # Shape: (batch, seq_len, 1)
        bpe_counts = torch.sum(bpe_mask, dim=2)

        # Avoid division by zero
        bpe_counts = torch.clamp(bpe_counts, min=1.0)

        # Average pooling
        bpe_feat = bpe_sum / bpe_counts

        # 4. Concatenate Features
        # Shape: (batch, seq_len, input_dim)
        combined_features = torch.cat([word_emb, char_feat, bpe_feat], dim=-1)

        # 5. LSTM Backbone
        # Use pack_padded_sequence if mask is provided for efficiency and correctness
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()
            # Enforce lengths > 0 just in case
            lengths = torch.clamp(lengths, min=1)

            # Pack
            packed_input = nn.utils.rnn.pack_padded_sequence(
                combined_features, lengths, batch_first=True, enforce_sorted=False
            )

            packed_output, _ = self.lstm(packed_input)

            # Unpack
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True, total_length=word_ids.size(1)
            )
        else:
            lstm_out, _ = self.lstm(combined_features)

        # 6. Classification Head
        lstm_out = self.fc_dropout(lstm_out)
        logits = self.fc(lstm_out)

        return logits
