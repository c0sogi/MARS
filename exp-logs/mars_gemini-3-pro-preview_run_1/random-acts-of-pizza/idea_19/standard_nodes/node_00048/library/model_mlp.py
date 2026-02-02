import torch
import torch.nn as nn
import torch.nn.functional as F


class DotProductAttention(nn.Module):
    """
    Computes Scaled Dot-Product Attention:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """

    def __init__(self, dropout_rate=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (Batch, Dim) - The request embedding
            key:   (Batch, Seq, Dim) - History embeddings
            value: (Batch, Seq, Dim) - History embeddings
            mask:  (Batch, Seq) - 1 for valid tokens, 0 for padding

        Returns:
            context: (Batch, Dim) - Weighted sum of values
            weights: (Batch, Seq) - Attention weights
        """
        # Expand query to (Batch, 1, Dim) for broadcasting
        query = query.unsqueeze(1)

        d_k = query.size(-1)

        # Compute scores: (Batch, 1, Dim) @ (Batch, Dim, Seq) -> (Batch, 1, Seq)
        scores = torch.bmm(query, key.transpose(1, 2)) / (d_k**0.5)

        if mask is not None:
            # Expand mask to (Batch, 1, Seq)
            mask = mask.unsqueeze(1)
            # Apply mask: set padding positions to -infinity
            scores = scores.masked_fill(mask == 0, -1e9)

        # Compute weights
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # Compute context: (Batch, 1, Seq) @ (Batch, Seq, Dim) -> (Batch, 1, Dim)
        context = torch.bmm(weights, value)

        # Squeeze back to (Batch, Dim)
        return context.squeeze(1), weights.squeeze(1)


class CredibilityGatedMLP(nn.Module):
    """
    Neural Network Architecture:
    1. Semantic Branch: Request + Attended History -> Semantic Vector
    2. Metadata Branch: Metadata -> Credibility Gate
    3. Fusion: Semantic Vector * Credibility Gate -> Prediction
    """

    def __init__(
        self, input_dim_meta, embedding_dim=384, hidden_dim=128, dropout_rate=0.3
    ):
        super().__init__()

        # --- Branch 1 & 2: Semantic Processing ---
        self.attention = DotProductAttention(dropout_rate=dropout_rate)

        # Projects the concatenated [Request; Context] vector (384 + 384) to hidden_dim
        self.semantic_proj = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # --- Branch 3: Metadata Credibility Gate ---
        # Projects metadata to a gate vector of size hidden_dim
        self.meta_gate = nn.Sequential(
            nn.Linear(input_dim_meta, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),  # Sigmoid ensures output is between 0 and 1 (Gating)
        )

        # --- Final Prediction Head ---
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, req_emb, hist_emb, meta, mask=None):
        """
        Args:
            req_emb: (Batch, 384)
            hist_emb: (Batch, Seq, 384)
            meta: (Batch, MetaDim)
            mask: (Batch, Seq)
        """
        # 1. Attention Mechanism
        # Extract relevant history context based on the current request
        context, _ = self.attention(req_emb, hist_emb, hist_emb, mask)

        # 2. Semantic Representation
        # Combine explicit request semantics with implicit historical context
        combined_sem = torch.cat([req_emb, context], dim=1)  # (Batch, 768)
        h_sem = self.semantic_proj(combined_sem)  # (Batch, 128)

        # 3. Credibility Gate
        # Generate gating vector from metadata
        gate = self.meta_gate(meta)  # (Batch, 128)

        # 4. Gated Fusion
        # Modulate semantic signal by credibility
        h_fused = h_sem * gate  # (Batch, 128)

        # 5. Output
        logits = self.classifier(h_fused)  # (Batch, 1)

        return logits
