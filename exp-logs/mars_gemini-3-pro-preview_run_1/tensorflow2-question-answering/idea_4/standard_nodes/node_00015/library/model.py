import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SoftAlignment(nn.Module):
    """
    Computes a Similarity Matrix between every Question token and every Candidate token.
    Generates an 'aligned question vector' for each candidate token via weighted sum.
    """

    def __init__(self):
        super(SoftAlignment, self).__init__()

    def forward(self, q_emb, c_emb, q_mask):
        """
        Args:
            q_emb: [Batch, MaxQ, Dim]
            c_emb: [Batch, MaxC, Dim]
            q_mask: [Batch, MaxQ] (1 for valid, 0 for pad)
        Returns:
            aligned_q: [Batch, MaxC, Dim]
        """
        # Similarity Matrix: [B, Lc, Lq]
        # Dot product between candidate tokens and question tokens
        sim_matrix = torch.bmm(c_emb, q_emb.transpose(1, 2))

        # Masking: Set similarity for padding question tokens to -inf
        if q_mask is not None:
            # Expand mask to [B, 1, Lq] then broadcast to [B, Lc, Lq]
            mask = q_mask.unsqueeze(1).expand(-1, c_emb.size(1), -1)
            sim_matrix = sim_matrix.masked_fill(mask == 0, -1e9)

        # Attention Weights: [B, Lc, Lq]
        # For each candidate token, distribution over question tokens
        attn_weights = F.softmax(sim_matrix, dim=-1)

        # Aligned Question Vector: [B, Lc, Dim]
        # Weighted sum of question embeddings
        aligned_q = torch.bmm(attn_weights, q_emb)

        return aligned_q


class CompareFFN(nn.Module):
    """
    Feed-Forward Network to process the combined Candidate and Aligned Question representation.
    Shared across all tokens (Local Inference).
    """

    def __init__(self, input_dim, hidden_dim, dropout):
        super(CompareFFN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [Batch, MaxC, InputDim]
        return self.net(x)


class AggregationHead(nn.Module):
    """
    Aggregates token-level match vectors to predict Long Answer probability.
    """

    def __init__(self, hidden_dim, dropout):
        super(AggregationHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, match_vectors, c_mask):
        """
        Args:
            match_vectors: [Batch, MaxC, HiddenDim]
            c_mask: [Batch, MaxC]
        Returns:
            logits: [Batch, 1]
        """
        # Mask out padding tokens in the candidate sequence before summing
        mask = c_mask.unsqueeze(-1).expand_as(match_vectors)
        masked_vectors = match_vectors * mask

        # Sum aggregation: [Batch, HiddenDim]
        global_vec = torch.sum(masked_vectors, dim=1)

        # Binary Classification Logits
        logits = self.classifier(global_vec)
        return logits


class SpanHead(nn.Module):
    """
    Predicts Start and End token logits for Short Answer extraction.
    """

    def __init__(self, hidden_dim):
        super(SpanHead, self).__init__()
        self.start_layer = nn.Linear(hidden_dim, 1)
        self.end_layer = nn.Linear(hidden_dim, 1)

    def forward(self, match_vectors):
        """
        Args:
            match_vectors: [Batch, MaxC, HiddenDim]
        Returns:
            start_logits: [Batch, MaxC]
            end_logits: [Batch, MaxC]
        """
        start_logits = self.start_layer(match_vectors).squeeze(-1)
        end_logits = self.end_layer(match_vectors).squeeze(-1)
        return start_logits, end_logits


class DAAN(nn.Module):
    """
    Decomposable Attention and Alignment Network.
    Synthesizes Context-Query Attention with Retrieval-style logic.
    """

    def __init__(self, embedding_matrix):
        super(DAAN, self).__init__()

        vocab_size, emb_dim = embedding_matrix.shape

        # Embedding Layer
        # Initialize with pre-trained matrix, allow fine-tuning
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            freeze=False,
            padding_idx=0,
        )

        # 1. Soft Alignment
        self.alignment = SoftAlignment()

        # 2. Compare (Input is concat of Candidate + Aligned Question)
        compare_input_dim = emb_dim * 2
        self.compare = CompareFFN(compare_input_dim, Config.HIDDEN_DIM, Config.DROPOUT)

        # 3. Aggregation (Long Answer)
        self.aggregation = AggregationHead(Config.HIDDEN_DIM, Config.DROPOUT)

        # 4. Span Prediction (Short Answer)
        self.span_head = SpanHead(Config.HIDDEN_DIM)

    def forward(self, q_input, c_input):
        """
        Args:
            q_input: [Batch, MaxQ] (Token IDs)
            c_input: [Batch, MaxC] (Token IDs)
        Returns:
            la_logits: [Batch, 1]
            start_logits: [Batch, MaxC]
            end_logits: [Batch, MaxC]
        """
        # Generate masks (assuming 0 is PAD_TOKEN ID)
        q_mask = (q_input != 0).float()
        c_mask = (c_input != 0).float()

        # Embeddings
        q_emb = self.embedding(q_input)  # [B, Lq, D]
        c_emb = self.embedding(c_input)  # [B, Lc, D]

        # Step 1: Soft Alignment
        # Align question parts to candidate text
        aligned_q = self.alignment(q_emb, c_emb, q_mask)  # [B, Lc, D]

        # Step 2: Comparison
        # Concatenate original candidate embedding with aligned question vector
        combined = torch.cat([c_emb, aligned_q], dim=-1)  # [B, Lc, 2D]

        # Compute match vectors for every token
        match_vectors = self.compare(combined)  # [B, Lc, H]

        # Step 3: Long Answer Prediction
        # Aggregate token vectors to predict if candidate contains answer
        la_logits = self.aggregation(match_vectors, c_mask)

        # Step 4: Short Answer Prediction
        # Predict start/end span indices
        start_logits, end_logits = self.span_head(match_vectors)

        # Mask span logits for padding tokens in candidate
        # Set pad positions to large negative value to prevent selection by softmax/argmax
        start_logits = start_logits.masked_fill(c_mask == 0, -1e9)
        end_logits = end_logits.masked_fill(c_mask == 0, -1e9)

        return la_logits, start_logits, end_logits
