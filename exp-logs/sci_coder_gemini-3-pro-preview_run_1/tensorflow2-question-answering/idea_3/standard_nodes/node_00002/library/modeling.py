import torch
import torch.nn as nn
import torch.nn.functional as F


class DanRanker(nn.Module):
    """
    Deep Averaging Network (DAN) for Long Answer Ranking.
    Encodes Question and Candidate by averaging token embeddings, then classifies their relationship.
    """

    def __init__(self, embedding_dim, hidden_dim=64):
        super(DanRanker, self).__init__()
        # Input features: [Q_avg; C_avg; Q_avg * C_avg] -> 3 * embedding_dim
        self.input_dim = 3 * embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, q_embeds, c_embeds, q_mask, c_mask):
        """
        Args:
            q_embeds: Tensor of shape (batch_size, q_len, embedding_dim)
            c_embeds: Tensor of shape (batch_size, c_len, embedding_dim)
            q_mask: Tensor of shape (batch_size, q_len), 1 for valid tokens, 0 for pad
            c_mask: Tensor of shape (batch_size, c_len), 1 for valid tokens, 0 for pad
        Returns:
            logits: Tensor of shape (batch_size, 1)
            q_mean: Tensor of shape (batch_size, embedding_dim)
        """
        # Compute masked means to ignore padding
        # Sum embeddings along sequence dimension
        q_sum = torch.sum(q_embeds * q_mask.unsqueeze(-1), dim=1)
        c_sum = torch.sum(c_embeds * c_mask.unsqueeze(-1), dim=1)

        # Count valid tokens (avoid division by zero)
        q_counts = torch.clamp(q_mask.sum(dim=1, keepdim=True), min=1e-9)
        c_counts = torch.clamp(c_mask.sum(dim=1, keepdim=True), min=1e-9)

        q_mean = q_sum / q_counts
        c_mean = c_sum / c_counts

        # Create interaction vector
        # Concatenate Q and C means, and add element-wise product for interaction
        interaction_vector = torch.cat([q_mean, c_mean, q_mean * c_mean], dim=1)

        # Classify
        logits = self.mlp(interaction_vector)

        return logits, q_mean


class TqpExtractor(nn.Module):
    """
    Token-Query Projection (TQP) for Short Answer Extraction.
    Projects each candidate token combined with the Question Mean to predict Start/End/Neither.
    """

    def __init__(self, embedding_dim):
        super(TqpExtractor, self).__init__()
        # Input features per token: [Token_Embedding; Q_avg] -> 2 * embedding_dim
        self.input_dim = 2 * embedding_dim
        # Output classes: 0=Neither, 1=Start, 2=End
        self.classifier = nn.Linear(self.input_dim, 3)

    def forward(self, c_embeds, q_mean):
        """
        Args:
            c_embeds: Tensor of shape (batch_size, c_len, embedding_dim)
            q_mean: Tensor of shape (batch_size, embedding_dim)
        Returns:
            logits: Tensor of shape (batch_size, c_len, 3)
        """
        batch_size, c_len, _ = c_embeds.size()

        # Expand q_mean to match candidate sequence length
        # (batch, dim) -> (batch, 1, dim) -> (batch, c_len, dim)
        q_mean_expanded = q_mean.unsqueeze(1).expand(-1, c_len, -1)

        # Concatenate each token embedding with the global question context
        combined_features = torch.cat([c_embeds, q_mean_expanded], dim=2)

        # Predict logits for each token
        logits = self.classifier(combined_features)

        return logits


class DanTqpModel(nn.Module):
    """
    Wrapper model combining the DAN Ranker and TQP Extractor.
    Manages the shared embedding layer.
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=100,
        hidden_dim=64,
        padding_idx=0,
        pretrained_embeddings=None,
    ):
        super(DanTqpModel, self).__init__()

        # Initialize Embedding Layer
        if pretrained_embeddings is not None:
            # If pretrained embeddings are provided (numpy array or tensor)
            self.embedding = nn.Embedding.from_pretrained(
                torch.tensor(pretrained_embeddings, dtype=torch.float32),
                freeze=False,  # Allow fine-tuning
                padding_idx=padding_idx,
            )
            embedding_dim = self.embedding.embedding_dim
        else:
            self.embedding = nn.Embedding(
                vocab_size, embedding_dim, padding_idx=padding_idx
            )

        self.padding_idx = padding_idx

        # Sub-modules
        self.ranker = DanRanker(embedding_dim, hidden_dim)
        self.extractor = TqpExtractor(embedding_dim)

    def forward(self, q_input_ids, c_input_ids):
        """
        Args:
            q_input_ids: Tensor (batch, q_len) containing token indices
            c_input_ids: Tensor (batch, c_len) containing token indices
        Returns:
            ranker_logits: Tensor (batch, 1) - Score for Long Answer
            extractor_logits: Tensor (batch, c_len, 3) - Scores for Short Answer spans
        """
        # Generate masks based on padding index
        q_mask = (q_input_ids != self.padding_idx).float()
        c_mask = (c_input_ids != self.padding_idx).float()

        # Lookup Embeddings
        q_embeds = self.embedding(q_input_ids)
        c_embeds = self.embedding(c_input_ids)

        # 1. Long Answer Ranking
        # Get ranker score and the computed question mean vector
        ranker_logits, q_mean = self.ranker(q_embeds, c_embeds, q_mask, c_mask)

        # 2. Short Answer Extraction
        # Pass candidate embeddings and question mean to extractor
        extractor_logits = self.extractor(c_embeds, q_mean)

        return ranker_logits, extractor_logits
