import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SiameseBiLSTM(nn.Module):
    """
    A Siamese Recurrent Dual-Encoder with Trainable Span Heads.

    Architecture:
    1. Frozen Embedding Layer (Pre-trained)
    2. Shared Bi-Directional LSTM Encoder
    3. Ranking Head: Global Max Pooling + Cosine Similarity
    4. Span Head: Context-conditioned Linear Layers
    5. Yes/No Head: MLP Classifier
    """

    def __init__(
        self,
        embedding_matrix,
        hidden_size=Config.HIDDEN_SIZE,
        dropout_rate=Config.DROPOUT_RATE,
    ):
        """
        Args:
            embedding_matrix (np.ndarray): Pre-trained embedding weights.
            hidden_size (int): Hidden dimension of the LSTM.
            dropout_rate (float): Dropout probability.
        """
        super(SiameseBiLSTM, self).__init__()

        num_embeddings, embedding_dim = embedding_matrix.shape

        # 1. Embedding Layer
        # Load pre-trained weights and freeze them
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.embedding.weight = nn.Parameter(
            torch.tensor(embedding_matrix, dtype=torch.float32), requires_grad=False
        )

        # 2. Shared Encoder (Bi-LSTM)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # The output dimension of a Bi-LSTM is hidden_size * 2 (forward + backward)
        self.lstm_out_dim = hidden_size * 2

        self.dropout = nn.Dropout(dropout_rate)

        # 3. Span Heads
        # Input: Concatenation of Candidate LSTM output (lstm_out_dim) and Repeated Pooled Question (lstm_out_dim)
        # Total input dimension: lstm_out_dim * 2
        self.span_start_classifier = nn.Linear(self.lstm_out_dim * 2, 1)
        self.span_end_classifier = nn.Linear(self.lstm_out_dim * 2, 1)

        # 4. Yes/No Head
        # Input: Concatenation of Pooled Question (lstm_out_dim) and Pooled Candidate (lstm_out_dim)
        # Total input dimension: lstm_out_dim * 2
        self.yn_classifier = nn.Sequential(
            nn.Linear(self.lstm_out_dim * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 3),  # Classes: NONE, YES, NO
        )

    def forward(self, q_input_ids, c_input_ids):
        """
        Args:
            q_input_ids (torch.Tensor): Question indices (batch_size, q_len).
            c_input_ids (torch.Tensor): Candidate indices (batch_size, c_len).

        Returns:
            dict: Contains 'rank_score', 'span_start_logits', 'span_end_logits', 'yn_logits'.
        """
        # --- Embedding ---
        # q_embed: (batch, q_len, embed_dim)
        # c_embed: (batch, c_len, embed_dim)
        q_embed = self.embedding(q_input_ids)
        c_embed = self.embedding(c_input_ids)

        # --- Shared Encoder ---
        # H_q: (batch, q_len, lstm_out_dim)
        # H_c: (batch, c_len, lstm_out_dim)
        H_q, _ = self.lstm(q_embed)
        H_c, _ = self.lstm(c_embed)

        # Apply dropout
        H_q = self.dropout(H_q)
        H_c = self.dropout(H_c)

        # --- Pooling (Global Max Pooling) ---
        # v_q: (batch, lstm_out_dim)
        # v_c: (batch, lstm_out_dim)
        # Max pooling over the sequence dimension (dim=1)
        v_q, _ = torch.max(H_q, dim=1)
        v_c, _ = torch.max(H_c, dim=1)

        # --- Ranking Head ---
        # Compute Cosine Similarity -> range [-1, 1]
        cosine_sim = F.cosine_similarity(v_q, v_c, dim=1)

        # Normalize to [0, 1] for Binary Cross Entropy Loss
        rank_prob = (cosine_sim + 1.0) / 2.0

        # --- Span Head ---
        # Condition the candidate representation on the question context.
        # Expand v_q to match the candidate sequence length.
        # v_q_expanded: (batch, c_len, lstm_out_dim)
        c_len = H_c.size(1)
        v_q_expanded = v_q.unsqueeze(1).expand(-1, c_len, -1)

        # Concatenate candidate hidden states with expanded question vector
        # span_input: (batch, c_len, lstm_out_dim * 2)
        span_input = torch.cat([H_c, v_q_expanded], dim=2)

        # Predict logits for start and end positions
        # Squeeze the last dimension to get (batch, c_len)
        start_logits = self.span_start_classifier(span_input).squeeze(-1)
        end_logits = self.span_end_classifier(span_input).squeeze(-1)

        # --- Yes/No Head ---
        # Concatenate pooled vectors for classification
        # yn_input: (batch, lstm_out_dim * 2)
        yn_input = torch.cat([v_q, v_c], dim=1)
        yn_logits = self.yn_classifier(yn_input)

        return {
            "rank_score": rank_prob,  # Shape: (batch,)
            "span_start_logits": start_logits,  # Shape: (batch, c_len)
            "span_end_logits": end_logits,  # Shape: (batch, c_len)
            "yn_logits": yn_logits,  # Shape: (batch, 3)
        }
