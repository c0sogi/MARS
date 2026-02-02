import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class QuestionEncoder(nn.Module):
    """
    Encodes the question tokens into a single global context vector using Mean Pooling.
    """

    def __init__(self):
        super(QuestionEncoder, self).__init__()

    def forward(self, q_embeddings):
        """
        Args:
            q_embeddings: Tensor of shape (batch_size, q_seq_len, embedding_dim)
        Returns:
            q_vector: Tensor of shape (batch_size, embedding_dim)
        """
        # Mean pooling across the sequence length dimension (dim=1)
        # The embedding layer handles padding_idx=0 mapping to zero vectors,
        # so they do not contribute to the sum, effectively handling variable lengths
        # in a fixed-size tensor structure.
        return torch.mean(q_embeddings, dim=1)


class AttentiveCandidateEncoder(nn.Module):
    """
    Encodes the candidate text into a single vector using attention weights derived
    from the question vector.
    """

    def __init__(self):
        super(AttentiveCandidateEncoder, self).__init__()

    def forward(self, c_embeddings, q_vector):
        """
        Args:
            c_embeddings: Tensor of shape (batch_size, c_seq_len, embedding_dim)
            q_vector: Tensor of shape (batch_size, embedding_dim)
        Returns:
            c_vector: Tensor of shape (batch_size, embedding_dim)
            attn_weights: Tensor of shape (batch_size, c_seq_len)
        """
        # 1. Calculate Relevance Weights
        # Expand q_vector to (batch_size, embedding_dim, 1) for batch matrix multiplication
        q_vector_expanded = q_vector.unsqueeze(2)

        # Dot product: (batch, seq, dim) @ (batch, dim, 1) -> (batch, seq, 1)
        # This computes the similarity between every candidate token and the question vector
        scores = torch.bmm(c_embeddings, q_vector_expanded).squeeze(2)

        # Normalize scores to get probabilities (attention weights)
        attn_weights = F.softmax(scores, dim=1)  # (batch_size, c_seq_len)

        # 2. Weighted Sum
        # Expand weights to (batch_size, 1, c_seq_len)
        weights_expanded = attn_weights.unsqueeze(1)

        # Compute weighted sum: (batch, 1, seq) @ (batch, seq, dim) -> (batch, 1, dim)
        context = torch.bmm(weights_expanded, c_embeddings).squeeze(1)

        return context, attn_weights


class RankingHead(nn.Module):
    """
    Predicts the likelihood of the candidate being the correct long answer.
    """

    def __init__(self, input_dim, hidden_dim, dropout_prob):
        super(RankingHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, combined_features):
        return self.mlp(combined_features)


class YesNoHead(nn.Module):
    """
    Predicts the Yes/No label (NONE, YES, NO).
    """

    def __init__(self, input_dim, hidden_dim, dropout_prob):
        super(YesNoHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, 3),  # 3 classes: NONE=0, YES=1, NO=2
        )

    def forward(self, combined_features):
        return self.mlp(combined_features)


class AGBoEModel(nn.Module):
    """
    Attention-Guided Bag-of-Embeddings Model.
    """

    def __init__(self, embedding_matrix):
        """
        Args:
            embedding_matrix: Numpy array of shape (vocab_size, embedding_dim)
        """
        super(AGBoEModel, self).__init__()

        # Load pre-trained embeddings and freeze them
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            freeze=True,
            padding_idx=0,  # Assuming index 0 is PAD based on vocab implementation
        )

        self.embed_dim = Config.EMBEDDING_DIM
        self.hidden_dim = Config.HIDDEN_DIM
        self.dropout_prob = Config.DROPOUT

        # Components
        self.q_encoder = QuestionEncoder()
        self.c_encoder = AttentiveCandidateEncoder()

        # Input dimension for heads: Q_vec + C_vec + (Q_vec * C_vec)
        # 3 vectors of size embed_dim
        head_input_dim = 3 * self.embed_dim

        self.ranking_head = RankingHead(
            head_input_dim, self.hidden_dim, self.dropout_prob
        )
        self.yesno_head = YesNoHead(head_input_dim, self.hidden_dim, self.dropout_prob)

    def forward(self, q_indices, c_indices):
        """
        Args:
            q_indices: (batch_size, q_seq_len)
            c_indices: (batch_size, c_seq_len)
        Returns:
            ranking_logits: (batch_size,) - Raw scores for long answer ranking
            yesno_logits: (batch_size, 3) - Raw scores for Yes/No classification
            attn_weights: (batch_size, c_seq_len) - Attention weights for auxiliary loss/span extraction
        """
        # 1. Embedding Lookup
        q_emb = self.embedding(q_indices)  # (batch, q_len, dim)
        c_emb = self.embedding(c_indices)  # (batch, c_len, dim)

        # 2. Encode Question
        q_vec = self.q_encoder(q_emb)  # (batch, dim)

        # 3. Encode Candidate with Attention
        c_vec, attn_weights = self.c_encoder(
            c_emb, q_vec
        )  # (batch, dim), (batch, c_len)

        # 4. Feature Interaction
        # Element-wise product to capture similarity/interaction
        interaction = q_vec * c_vec

        # Concatenate features
        combined = torch.cat([q_vec, c_vec, interaction], dim=1)  # (batch, 3*dim)

        # 5. Prediction Heads
        ranking_logits = self.ranking_head(combined).squeeze(1)  # (batch,)
        yesno_logits = self.yesno_head(combined)  # (batch, 3)

        return ranking_logits, yesno_logits, attn_weights
