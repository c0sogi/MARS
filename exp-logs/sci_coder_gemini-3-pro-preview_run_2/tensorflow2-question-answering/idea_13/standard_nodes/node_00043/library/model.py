import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class WindowMaxPoolingNetwork(nn.Module):
    """
    Window-Based Max-Pooling Network for Natural Questions.

    Architecture:
    1. Frozen Embedding Layer (Pre-trained)
    2. Question Encoder (Mean Pooling + Projection)
    3. Window Encoder (Mean Pooling + Projection)
    4. Local Scoring Module (MLP) -> Window Relevance Score
    5. Span Prediction Heads -> Start/End Logits
    6. Yes/No Classifier -> Class Logits
    """

    def __init__(self, embedding_matrix, config=Config):
        super(WindowMaxPoolingNetwork, self).__init__()
        self.config = config

        # 1. Embedding Layer
        # Convert numpy matrix to tensor
        if not torch.is_tensor(embedding_matrix):
            embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)

        num_embeddings, embedding_dim = embedding_matrix.shape
        # Cite debug_lesson_3: Never Freeze Randomly Initialized Embeddings
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=False, padding_idx=0
        )

        # 2. Encoders (Projection layers after pooling)
        self.question_proj = nn.Linear(embedding_dim, config.HIDDEN_DIM)
        self.window_proj = nn.Linear(embedding_dim, config.HIDDEN_DIM)

        # 3. Local Scoring Module (Relevance Scorer)
        # Input: Concatenated Question Vector + Window Vector
        self.scorer = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM * 2, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, 1),
        )

        # 4. Span Prediction Heads
        # Input: Unpooled Window Embeddings (Emb Dim) + Expanded Question Vector (Hidden Dim)
        span_input_dim = embedding_dim + config.HIDDEN_DIM

        self.start_head = nn.Linear(span_input_dim, 1)
        self.end_head = nn.Linear(span_input_dim, 1)

        # 5. Yes/No Classifier
        # Input: Concatenated Question Vector + Window Vector (same as scorer)
        self.yes_no_head = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM * 2, config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM // 2, config.NUM_YES_NO_CLASSES),
        )

        self.dropout = nn.Dropout(config.DROPOUT)

    def _masked_mean_pooling(self, embeddings, input_ids):
        """
        Performs mean pooling ignoring padding tokens.

        Args:
            embeddings: (Batch, Seq_Len, Dim)
            input_ids: (Batch, Seq_Len)

        Returns:
            pooled: (Batch, Dim)
        """
        # Create mask: 1 for real tokens, 0 for padding (index 0)
        mask = (input_ids != 0).float().unsqueeze(-1)  # (B, L, 1)

        # Sum embeddings
        sum_embeddings = torch.sum(embeddings * mask, dim=1)  # (B, D)

        # Count non-pad tokens
        sum_mask = torch.sum(mask, dim=1)  # (B, 1)

        # Avoid division by zero
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

    def forward(self, input_ids, question_ids):
        """
        Forward pass.

        Args:
            input_ids: (Batch, Window_Size) - Token indices for the window
            question_ids: (Batch, Question_Len) - Token indices for the question

        Returns:
            window_score: (Batch, 1) - Relevance logit
            start_logits: (Batch, Window_Size) - Logits for span start
            end_logits: (Batch, Window_Size) - Logits for span end
            yes_no_logits: (Batch, Num_Classes) - Logits for Yes/No
        """
        # --- 1. Embeddings ---
        # (B, L_w, E)
        window_emb = self.embedding(input_ids)
        # (B, L_q, E)
        question_emb = self.embedding(question_ids)

        # --- 2. Pooling & Encoding ---
        # Pool
        window_pooled = self._masked_mean_pooling(window_emb, input_ids)  # (B, E)
        question_pooled = self._masked_mean_pooling(
            question_emb, question_ids
        )  # (B, E)

        # Project
        window_vec = F.relu(self.window_proj(window_pooled))  # (B, H)
        question_vec = F.relu(self.question_proj(question_pooled))  # (B, H)

        # Apply dropout
        window_vec = self.dropout(window_vec)
        question_vec = self.dropout(question_vec)

        # --- 3. Relevance Scoring ---
        # Concatenate vectors for interaction
        combined_vec = torch.cat([question_vec, window_vec], dim=1)  # (B, 2*H)
        window_score = self.scorer(combined_vec)  # (B, 1)

        # --- 4. Span Prediction ---
        # We need to combine the Question Vector with every token in the Window.
        # Expand question vector to match window length: (B, 1, H) -> (B, L_w, H)
        seq_len = window_emb.size(1)
        question_expanded = question_vec.unsqueeze(1).expand(-1, seq_len, -1)

        # Concatenate unpooled window embeddings with pooled question vector
        # (B, L_w, E) cat (B, L_w, H) -> (B, L_w, E+H)
        span_input = torch.cat([window_emb, question_expanded], dim=2)
        span_input = self.dropout(span_input)

        # Predict logits
        start_logits = self.start_head(span_input).squeeze(-1)  # (B, L_w)
        end_logits = self.end_head(span_input).squeeze(-1)  # (B, L_w)

        # --- 5. Yes/No Prediction ---
        yes_no_logits = self.yes_no_head(combined_vec)  # (B, Num_Classes)

        return window_score, start_logits, end_logits, yes_no_logits
