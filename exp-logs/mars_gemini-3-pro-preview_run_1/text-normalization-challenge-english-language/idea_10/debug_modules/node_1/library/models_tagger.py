import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class QuadHybridBiLSTM(nn.Module):
    """
    A Quad-Hybrid Bi-LSTM Tagger for text normalization classification.

    Integrates four input streams:
    1. Word Embeddings (Semantic)
    2. Character CNN (Morphological)
    3. BPE Embeddings (Subword/OOV robustness)
    4. Explicit Regex Features (Structural/Rule-based)

    Backbone: Bi-directional LSTM
    Head: Linear Classification Layer
    """

    def __init__(self, num_classes, vocab_words, vocab_chars, vocab_bpe_size):
        """
        Args:
            num_classes (int): Number of target classes.
            vocab_words (Vocabulary): Vocabulary object for words.
            vocab_chars (Vocabulary): Vocabulary object for characters.
            vocab_bpe_size (int): Size of the BPE vocabulary.
        """
        super(QuadHybridBiLSTM, self).__init__()
        self.config = Config()

        # =====================================================================
        # Hyperparameters
        # =====================================================================
        self.word_embed_dim = self.config.TAGGER_WORD_EMBED_DIM
        self.char_embed_dim = self.config.TAGGER_CHAR_EMBED_DIM
        self.bpe_embed_dim = self.config.TAGGER_BPE_EMBED_DIM
        self.hidden_dim = self.config.TAGGER_HIDDEN_DIM
        self.num_layers = self.config.TAGGER_NUM_LAYERS
        self.dropout_p = self.config.TAGGER_DROPOUT

        self.cnn_filters = self.config.CNN_FILTERS
        self.cnn_kernel = self.config.CNN_KERNEL_SIZE
        self.num_regex_feats = self.config.NUM_REGEX_FEATURES

        # =====================================================================
        # 1. Word Embeddings
        # =====================================================================
        self.word_vocab_size = len(vocab_words)
        self.word_pad_idx = vocab_words.get_id("<PAD>")
        self.word_embedding = nn.Embedding(
            self.word_vocab_size, self.word_embed_dim, padding_idx=self.word_pad_idx
        )

        # =====================================================================
        # 2. Character CNN
        # =====================================================================
        self.char_vocab_size = len(vocab_chars)
        self.char_pad_idx = vocab_chars.get_id("<PAD>")
        self.char_embedding = nn.Embedding(
            self.char_vocab_size, self.char_embed_dim, padding_idx=self.char_pad_idx
        )
        # Conv1d: Input (N, C_in, L), Output (N, C_out, L_out)
        self.char_conv = nn.Conv1d(
            in_channels=self.char_embed_dim,
            out_channels=self.cnn_filters,
            kernel_size=self.cnn_kernel,
            padding=1,  # Padding to handle short tokens
        )

        # =====================================================================
        # 3. BPE Embeddings
        # =====================================================================
        self.bpe_vocab_size = vocab_bpe_size
        self.bpe_pad_idx = 0  # SentencePiece pad ID is typically 0
        self.bpe_embedding = nn.Embedding(
            self.bpe_vocab_size, self.bpe_embed_dim, padding_idx=self.bpe_pad_idx
        )

        # =====================================================================
        # 4. Explicit Feature Projection
        # =====================================================================
        # Project binary features to a dense vector to allow learnable importance
        self.feat_proj_dim = 32
        self.feature_projection = nn.Linear(self.num_regex_feats, self.feat_proj_dim)

        # =====================================================================
        # Backbone & Head
        # =====================================================================
        # Calculate total input dimension for LSTM
        self.lstm_input_dim = (
            self.word_embed_dim
            + self.cnn_filters
            + self.bpe_embed_dim
            + self.feat_proj_dim
        )

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_p if self.num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(self.dropout_p)

        # Bidirectional LSTM outputs hidden_dim * 2
        self.classifier = nn.Linear(self.hidden_dim * 2, num_classes)

    def forward(self, word_ids, char_ids, bpe_ids, features):
        """
        Forward pass of the tagger.

        Args:
            word_ids (torch.Tensor): (Batch, Seq_Len)
            char_ids (torch.Tensor): (Batch, Seq_Len, Char_Len)
            bpe_ids (torch.Tensor): (Batch, Seq_Len, BPE_Len)
            features (torch.Tensor): (Batch, Seq_Len, Num_Regex_Feats)

        Returns:
            logits (torch.Tensor): (Batch, Seq_Len, Num_Classes)
        """
        batch_size, seq_len = word_ids.size()

        # ---------------------------------------------------------------------
        # 1. Process Word Embeddings
        # ---------------------------------------------------------------------
        word_emb = self.word_embedding(word_ids)  # (B, S, Word_Dim)

        # ---------------------------------------------------------------------
        # 2. Process Character CNN
        # ---------------------------------------------------------------------
        # Flatten batch and sequence dimensions to process tokens in parallel
        char_len = char_ids.size(2)
        char_ids_flat = char_ids.view(-1, char_len)  # (B*S, Char_Len)

        char_emb = self.char_embedding(char_ids_flat)  # (B*S, Char_Len, Char_Dim)

        # Permute for Conv1d: needs (N, Channels, Length)
        char_emb = char_emb.permute(0, 2, 1)  # (B*S, Char_Dim, Char_Len)

        cnn_out = self.char_conv(char_emb)  # (B*S, Filters, L_out)

        # Global Max Pooling over character dimension
        # Returns (B*S, Filters)
        cnn_out, _ = torch.max(cnn_out, dim=2)

        # Reshape back to sequence structure
        char_feat = cnn_out.view(batch_size, seq_len, self.cnn_filters)

        # ---------------------------------------------------------------------
        # 3. Process BPE Embeddings (Mean Pooling)
        # ---------------------------------------------------------------------
        bpe_emb = self.bpe_embedding(bpe_ids)  # (B, S, BPE_Len, BPE_Dim)

        # Create mask for padding (0 is pad) to ignore in average
        bpe_mask = (bpe_ids != 0).float().unsqueeze(-1)  # (B, S, BPE_Len, 1)

        # Sum embeddings
        bpe_sum = torch.sum(bpe_emb * bpe_mask, dim=2)  # (B, S, BPE_Dim)

        # Count non-pad tokens
        bpe_counts = torch.sum(bpe_mask, dim=2)  # (B, S, 1)
        bpe_counts = torch.clamp(bpe_counts, min=1.0)  # Avoid div by zero

        bpe_feat = bpe_sum / bpe_counts  # (B, S, BPE_Dim)

        # ---------------------------------------------------------------------
        # 4. Process Explicit Features
        # ---------------------------------------------------------------------
        # Project and apply non-linearity
        explicit_feat = self.feature_projection(features)  # (B, S, Proj_Dim)
        explicit_feat = F.relu(explicit_feat)

        # ---------------------------------------------------------------------
        # Aggregation & Classification
        # ---------------------------------------------------------------------
        # Concatenate all features
        combined_input = torch.cat(
            [word_emb, char_feat, bpe_feat, explicit_feat], dim=2
        )  # (B, S, Total_Dim)

        # Pass through Bi-LSTM
        lstm_out, _ = self.lstm(combined_input)  # (B, S, Hidden*2)

        lstm_out = self.dropout(lstm_out)

        # Classification Head
        logits = self.classifier(lstm_out)  # (B, S, Num_Classes)

        return logits
