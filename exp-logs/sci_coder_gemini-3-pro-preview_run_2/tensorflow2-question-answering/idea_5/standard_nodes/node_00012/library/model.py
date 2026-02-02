import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class FeedForwardDecomposableAttention(nn.Module):
    """
    Feed-Forward Decomposable Attention Network.

    Implements a lightweight attention mechanism to align question and candidate text,
    followed by a feed-forward comparison layer and aggregation for classification.
    """

    def __init__(self, config: Config, embedding_matrix: torch.Tensor):
        super(FeedForwardDecomposableAttention, self).__init__()
        self.config = config

        # 1. Embedding Layer
        # Load pre-trained embeddings and freeze them to save resources
        # We assume index 0 is padding based on DataProcessor logic (PAD_TOKEN)
        # Cite debug_lesson_3: Never Freeze Randomly Initialized Embeddings
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=False, padding_idx=0
        )

        self.embed_dim = config.EMBED_DIM
        self.hidden_dim = config.HIDDEN_DIM

        # 2. Attention-Based Alignment
        # Projects embeddings to hidden dimension for calculating similarity scores
        self.q_proj = nn.Linear(self.embed_dim, self.hidden_dim, bias=False)
        self.c_proj = nn.Linear(self.embed_dim, self.hidden_dim, bias=False)

        # 3. Comparison Layer
        # Input: Concatenation of Candidate Embedding and Aligned Question Vector
        # We use original embeddings for the comparison features
        self.comparison_input_dim = self.embed_dim * 2

        self.comparison_mlp = nn.Sequential(
            nn.Linear(self.comparison_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
        )

        # 4. Output Heads

        # Long Answer Ranking Head: Aggregated -> Score
        self.ranking_head = nn.Linear(self.hidden_dim, 1)

        # Span Prediction Heads: Sequence -> Logits
        self.span_start_head = nn.Linear(self.hidden_dim, 1)
        self.span_end_head = nn.Linear(self.hidden_dim, 1)

        # Yes/No Head: Aggregated -> Class Logits
        self.yes_no_head = nn.Linear(self.hidden_dim, config.NUM_CLASSES_YN)

    def forward(self, q_input, c_input):
        """
        Forward pass of the model.

        Args:
            q_input (torch.Tensor): Question indices, shape (Batch, Q_Len)
            c_input (torch.Tensor): Candidate indices, shape (Batch, C_Len)

        Returns:
            dict: Dictionary containing logits for all tasks.
        """
        # Create masks for padding (0 is pad index)
        # q_mask: (Batch, Q_Len)
        q_mask = (q_input != 0).float()
        # c_mask: (Batch, C_Len)
        c_mask = (c_input != 0).float()

        # --- 1. Embeddings ---
        # (Batch, Seq_Len, Embed_Dim)
        q_embed = self.embedding(q_input)
        c_embed = self.embedding(c_input)

        # --- 2. Attention-Based Alignment ---
        # Project to hidden space
        q_proj = self.q_proj(q_embed)
        c_proj = self.c_proj(c_embed)

        # Compute Similarity Matrix: S = C_proj * Q_proj^T
        # (Batch, C_Len, Hidden) x (Batch, Hidden, Q_Len) -> (Batch, C_Len, Q_Len)
        similarity = torch.bmm(c_proj, q_proj.transpose(1, 2))

        # Mask Question Padding in Attention
        # Expand q_mask to (Batch, 1, Q_Len) to broadcast over C_Len
        q_mask_expanded = q_mask.unsqueeze(1)
        similarity = similarity.masked_fill(q_mask_expanded == 0, -1e9)

        # Softmax over Question dimension to get alignment weights
        # (Batch, C_Len, Q_Len)
        attn_weights = F.softmax(similarity, dim=-1)

        # Compute Soft-Aligned Question Vector
        # (Batch, C_Len, Q_Len) x (Batch, Q_Len, Embed_Dim) -> (Batch, C_Len, Embed_Dim)
        aligned_q = torch.bmm(attn_weights, q_embed)

        # --- 3. Comparison Layer ---
        # Concatenate Candidate Embeddings with Aligned Question Vectors
        # (Batch, C_Len, Embed_Dim * 2)
        combined = torch.cat([c_embed, aligned_q], dim=-1)

        # Process through MLP
        # (Batch, C_Len, Hidden_Dim)
        comparison_vecs = self.comparison_mlp(combined)

        # --- 4. Aggregation ---
        # Mask Candidate Padding before aggregation to avoid summing noise/bias
        # c_mask_expanded: (Batch, C_Len, 1)
        c_mask_expanded = c_mask.unsqueeze(-1)
        masked_comparison = comparison_vecs * c_mask_expanded

        # Sum over sequence length
        # (Batch, Hidden_Dim)
        aggregated_vec = torch.sum(masked_comparison, dim=1)

        # --- 5. Output Heads ---

        # Ranking (Long Answer)
        # (Batch, 1)
        ranking_logits = self.ranking_head(aggregated_vec)

        # Span Prediction
        # Apply linear layer to sequence of comparison vectors
        # (Batch, C_Len, 1) -> (Batch, C_Len)
        start_logits = self.span_start_head(comparison_vecs).squeeze(-1)
        end_logits = self.span_end_head(comparison_vecs).squeeze(-1)

        # Mask span logits for padded candidate tokens
        # Set to large negative value so they aren't selected by argmax/softmax
        start_logits = start_logits.masked_fill(c_mask == 0, -1e9)
        end_logits = end_logits.masked_fill(c_mask == 0, -1e9)

        # Yes/No Prediction
        # (Batch, Num_Classes)
        yn_logits = self.yes_no_head(aggregated_vec)

        return {
            "ranking_logits": ranking_logits,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "yn_logits": yn_logits,
        }
