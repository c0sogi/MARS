import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class QuestionEncoder(nn.Module):
    """
    Encodes the question text into a single context vector using 1D Convolution
    and Global Max Pooling.
    """

    def __init__(self, embedding_dim, filters, kernel_size):
        super(QuestionEncoder, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,  # 'same' padding for odd kernels
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Emb_Dim)
        # Permute for Conv1d: (Batch, Emb_Dim, Seq_Len)
        x = x.permute(0, 2, 1)

        x = self.conv(x)
        x = self.activation(x)

        # Global Max Pooling over sequence dimension
        # Shape: (Batch, Filters)
        x = F.adaptive_max_pool1d(x, 1).squeeze(2)
        return x


class CandidateEncoder(nn.Module):
    """
    Encodes the candidate text into a sequence of feature maps using 1D Convolution.
    Preserves sequence length for span prediction.
    """

    def __init__(self, embedding_dim, filters, kernel_size):
        super(CandidateEncoder, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,  # 'same' padding
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Emb_Dim)
        # Permute: (Batch, Emb_Dim, Seq_Len)
        x = x.permute(0, 2, 1)

        x = self.conv(x)
        x = self.activation(x)

        # Return sequence: (Batch, Filters, Seq_Len)
        return x


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation Layer.
    Uses the question context to scale and shift the candidate feature maps.
    """

    def __init__(self, context_dim, feature_dim):
        super(FiLMLayer, self).__init__()
        # Project context to scale (gamma) and shift (beta)
        self.fc_gamma = nn.Linear(context_dim, feature_dim)
        self.fc_beta = nn.Linear(context_dim, feature_dim)

    def forward(self, candidate_features, question_context):
        """
        Args:
            candidate_features: (Batch, Channels, Seq_Len)
            question_context: (Batch, Context_Dim)
        """
        # Generate modulation parameters
        gamma = self.fc_gamma(question_context)  # (Batch, Channels)
        beta = self.fc_beta(question_context)  # (Batch, Channels)

        # Unsqueeze to broadcast over sequence length: (Batch, Channels, 1)
        gamma = gamma.unsqueeze(2)
        beta = beta.unsqueeze(2)

        # Apply FiLM: (Features * Gamma) + Beta
        modulated = (candidate_features * gamma) + beta

        return modulated


class RankingHead(nn.Module):
    """
    Predicts if the candidate is the correct long answer.
    """

    def __init__(self, input_dim, dropout_rate):
        super(RankingHead, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: Modulated features (Batch, Channels, Seq_Len)
        # Global Max Pool -> (Batch, Channels)
        pooled = F.adaptive_max_pool1d(x, 1).squeeze(2)

        out = self.dropout(pooled)
        logits = self.fc(out)  # (Batch, 1)
        return logits


class SpanHead(nn.Module):
    """
    Predicts start and end token indices for the short answer.
    """

    def __init__(self, input_dim, kernel_size):
        super(SpanHead, self).__init__()
        # Lightweight convolutions for span prediction
        self.conv_start = nn.Conv1d(input_dim, 1, kernel_size, padding=kernel_size // 2)
        self.conv_end = nn.Conv1d(input_dim, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        # x: Modulated features (Batch, Channels, Seq_Len)

        # Start Logits: (Batch, 1, Seq_Len) -> (Batch, Seq_Len)
        start_logits = self.conv_start(x).squeeze(1)

        # End Logits: (Batch, 1, Seq_Len) -> (Batch, Seq_Len)
        end_logits = self.conv_end(x).squeeze(1)

        return start_logits, end_logits


class YesNoHead(nn.Module):
    """
    Predicts YES, NO, or NONE.
    """

    def __init__(self, input_dim, dropout_rate):
        super(YesNoHead, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        # 3 classes: 0=NONE, 1=YES, 2=NO (Mapped in dataset.py)
        self.fc = nn.Linear(input_dim, 3)

    def forward(self, x):
        # x: Modulated features (Batch, Channels, Seq_Len)
        # Global Max Pool -> (Batch, Channels)
        pooled = F.adaptive_max_pool1d(x, 1).squeeze(2)

        out = self.dropout(pooled)
        logits = self.fc(out)  # (Batch, 3)
        return logits


class FiLMNetwork(nn.Module):
    """
    Main model architecture combining encoders, FiLM modulation, and output heads.
    """

    def __init__(self, embedding_matrix):
        super(FiLMNetwork, self).__init__()

        vocab_size, emb_dim = embedding_matrix.shape

        # 1. Embedding Layer (Frozen)
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            freeze=True,
            padding_idx=0,  # Assuming 0 is PAD based on Vocabulary class
        )

        # 2. Encoders
        self.q_encoder = QuestionEncoder(
            embedding_dim=emb_dim,
            filters=Config.CNN_FILTERS,
            kernel_size=Config.CNN_KERNEL_SIZE,
        )

        self.cand_encoder = CandidateEncoder(
            embedding_dim=emb_dim,
            filters=Config.CNN_FILTERS,
            kernel_size=Config.CNN_KERNEL_SIZE,
        )

        # 3. Modulation
        self.film_layer = FiLMLayer(
            context_dim=Config.CNN_FILTERS, feature_dim=Config.CNN_FILTERS
        )

        # 4. Heads
        self.ranking_head = RankingHead(
            input_dim=Config.CNN_FILTERS, dropout_rate=Config.DROPOUT
        )

        self.span_head = SpanHead(
            input_dim=Config.CNN_FILTERS, kernel_size=Config.CNN_KERNEL_SIZE
        )

        self.yesno_head = YesNoHead(
            input_dim=Config.CNN_FILTERS, dropout_rate=Config.DROPOUT
        )

    def forward(self, q_input, cand_input):
        """
        Args:
            q_input: (Batch, Q_Len)
            cand_input: (Batch, Ctx_Len)

        Returns:
            dict containing logits for ranking, span_start, span_end, and yes_no
        """
        # Embed inputs
        q_emb = self.embedding(q_input)  # (Batch, Q_Len, Emb_Dim)
        cand_emb = self.embedding(cand_input)  # (Batch, Ctx_Len, Emb_Dim)

        # Encode Question -> Context Vector
        q_context = self.q_encoder(q_emb)  # (Batch, Filters)

        # Encode Candidate -> Feature Sequence
        cand_feats = self.cand_encoder(cand_emb)  # (Batch, Filters, Ctx_Len)

        # Apply FiLM Modulation
        mod_feats = self.film_layer(cand_feats, q_context)  # (Batch, Filters, Ctx_Len)

        # Apply Heads
        rank_logits = self.ranking_head(mod_feats)  # (Batch, 1)
        start_logits, end_logits = self.span_head(mod_feats)  # (Batch, Ctx_Len) each
        yesno_logits = self.yesno_head(mod_feats)  # (Batch, 3)

        return {
            "rank_logits": rank_logits,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "yesno_logits": yesno_logits,
        }
