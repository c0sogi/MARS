import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SiameseRanker(nn.Module):
    """
    Siamese Bi-LSTM Ranker for Long Answer Selection.
    Encodes Question and Candidate Paragraph into vectors and computes Cosine Similarity.
    """

    def __init__(
        self,
        vocab_size=Config.VOCAB_SIZE,
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        lstm_layers=Config.LSTM_LAYERS,
        dropout=Config.DROPOUT,
    ):
        super(SiameseRanker, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Shared Bi-Directional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)

    def forward_one(self, x):
        """
        Encodes a single sequence (Question or Context).
        Args:
            x: Tensor of shape [batch_size, seq_len]
        Returns:
            pooled: Tensor of shape [batch_size, hidden_dim * 2]
        """
        # [batch, seq_len, emb_dim]
        embeds = self.embedding(x)
        embeds = self.dropout(embeds)

        # LSTM Output: [batch, seq_len, hidden_dim * 2]
        output, _ = self.lstm(embeds)

        # Max-Over-Time Pooling
        # Takes the maximum value across the sequence dimension
        pooled, _ = torch.max(output, dim=1)

        return pooled

    def forward(self, q_input, ctx_input):
        """
        Computes similarity between Question and Context.
        Args:
            q_input: [batch_size, q_len]
            ctx_input: [batch_size, ctx_len]
        Returns:
            scores: [batch_size] (Cosine Similarity)
        """
        q_vec = self.forward_one(q_input)
        ctx_vec = self.forward_one(ctx_input)

        # Compute Cosine Similarity
        scores = F.cosine_similarity(q_vec, ctx_vec, dim=1)
        return scores


class ConditionalReader(nn.Module):
    """
    Conditional LSTM Reader for Short Answer Extraction.
    Conditions the Paragraph sequence processing on the Question encoding.
    """

    def __init__(
        self,
        vocab_size=Config.VOCAB_SIZE,
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        lstm_layers=Config.LSTM_LAYERS,
        dropout=Config.DROPOUT,
    ):
        super(ConditionalReader, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        # Question Encoder: Bi-LSTM
        self.q_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # Context Encoder: Bi-LSTM
        # Input size is Embedding Dim + Question Vector Size (2 * hidden_dim)
        self.ctx_lstm = nn.LSTM(
            input_size=embedding_dim + (hidden_dim * 2),
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # Output Classifiers
        # Project LSTM output to scalar logit for each token
        self.start_head = nn.Linear(hidden_dim * 2, 1)
        self.end_head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, q_input, ctx_input):
        """
        Predicts start and end logits for the answer span.
        Args:
            q_input: [batch_size, q_len]
            ctx_input: [batch_size, ctx_len]
        Returns:
            start_logits: [batch_size, ctx_len]
            end_logits: [batch_size, ctx_len]
        """
        # 1. Embedding
        q_embed = self.embedding(q_input)  # [batch, q_len, emb_dim]
        ctx_embed = self.embedding(ctx_input)  # [batch, ctx_len, emb_dim]

        q_embed = self.dropout(q_embed)
        ctx_embed = self.dropout(ctx_embed)

        # 2. Encode Question
        # We use the final hidden states (h_n)
        # h_n shape: [num_layers * num_directions, batch, hidden_dim]
        _, (h_n, _) = self.q_lstm(q_embed)

        # Concatenate the last layer's forward and backward hidden states
        # h_n[-2] is forward, h_n[-1] is backward for the last layer
        q_vec = torch.cat((h_n[-2], h_n[-1]), dim=1)  # [batch, hidden_dim * 2]

        # 3. Context Conditioning
        seq_len = ctx_embed.size(1)
        # Replicate q_vec for every token in context
        q_vec_expanded = q_vec.unsqueeze(1).expand(
            -1, seq_len, -1
        )  # [batch, seq_len, hidden_dim * 2]

        # Concatenate Context Embeddings with Question Vector
        combined_input = torch.cat(
            (ctx_embed, q_vec_expanded), dim=2
        )  # [batch, seq_len, emb_dim + 2*hidden]

        # 4. Sequence Modeling
        ctx_output, _ = self.ctx_lstm(
            combined_input
        )  # [batch, seq_len, hidden_dim * 2]
        ctx_output = self.dropout(ctx_output)

        # 5. Output Projection
        start_logits = self.start_head(ctx_output).squeeze(-1)  # [batch, seq_len]
        end_logits = self.end_head(ctx_output).squeeze(-1)  # [batch, seq_len]

        return start_logits, end_logits
