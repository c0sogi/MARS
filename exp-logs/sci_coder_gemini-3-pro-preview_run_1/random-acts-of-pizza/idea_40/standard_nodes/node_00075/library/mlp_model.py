import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config


class DualQueryAttention(nn.Module):
    """
    Computes attention context for a query against user history.
    """

    def __init__(self, embed_dim):
        super(DualQueryAttention, self).__init__()
        self.scale = embed_dim**-0.5

    def forward(self, query, history, mask):
        """
        Args:
            query: (Batch, EmbedDim)
            history: (Batch, SeqLen, EmbedDim)
            mask: (Batch, SeqLen) - 1.0 for valid, 0.0 for padding
        Returns:
            context: (Batch, EmbedDim)
        """
        # (Batch, 1, EmbedDim)
        q = query.unsqueeze(1)

        # (Batch, 1, EmbedDim) @ (Batch, EmbedDim, SeqLen) -> (Batch, 1, SeqLen)
        scores = torch.matmul(q, history.transpose(-2, -1)) * self.scale

        # Apply mask: fill 0s with -inf
        # mask needs to be broadcastable to (Batch, 1, SeqLen)
        mask_expanded = mask.unsqueeze(1)
        scores = scores.masked_fill(mask_expanded == 0, float("-inf"))

        # Softmax over sequence dimension
        attn_weights = F.softmax(scores, dim=-1)

        # Handle NaN if mask blocked all values (empty history)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # (Batch, 1, SeqLen) @ (Batch, SeqLen, EmbedDim) -> (Batch, 1, EmbedDim)
        context = torch.matmul(attn_weights, history)

        return context.squeeze(1)


class SkipGatedDualQueryMLP(nn.Module):
    """
    Hybrid Ensemble Stream B Model.
    Features:
    - Dual-Query Attention (Title/Body -> History)
    - Alignment Injection (Query-Context Similarity)
    - Skip-Gated Fusion (Metadata gates semantics but also bypasses)
    """

    def __init__(self, input_meta_dim):
        super(SkipGatedDualQueryMLP, self).__init__()

        params = config.MLP_PARAMS
        self.embed_dim = params["embedding_dim"]
        hidden_fusion = params["hidden_dim_fusion"]
        dropout_emb = params["dropout_emb"]
        dropout_dense = params["dropout_dense"]

        # --- Branches 1, 2, 3: Semantic Processing ---
        self.att_title = DualQueryAttention(self.embed_dim)
        self.att_body = DualQueryAttention(self.embed_dim)

        self.dropout_emb = nn.Dropout(dropout_emb)

        # Semantic Vector Composition:
        # 1. Title Emb (384)
        # 2. Body Emb (384)
        # 3. Title Context (384)
        # 4. Body Context (384)
        # 5. External Consistency Scalars (2) - from FeatureEngineer (Title-Centroid, Body-Centroid)
        # 6. Internal Alignment Scalars (2) - Computed dynamically (Title-Context, Body-Context)
        self.semantic_dim = (self.embed_dim * 4) + 2 + 2

        # --- Branch 4: Metadata & Gating ---
        # Gate projects Metadata -> Semantic Dim
        # Sigmoid(Dense(Metadata))
        self.meta_gate = nn.Linear(input_meta_dim, self.semantic_dim)

        # --- Fusion ---
        # Input: Gated Semantic (semantic_dim) + Raw Metadata (input_meta_dim)
        # This implements the "Wide & Deep Skip-Connection"
        fusion_input_dim = self.semantic_dim + input_meta_dim

        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_fusion),
            nn.ReLU(),
            nn.Dropout(dropout_dense),
            nn.Linear(hidden_fusion, 1),
        )

    def forward(self, title_emb, body_emb, hist_seq, hist_mask, meta, cons):
        """
        Args:
            title_emb: (B, 384)
            body_emb: (B, 384)
            hist_seq: (B, Seq, 384)
            hist_mask: (B, Seq)
            meta: (B, MetaDim)
            cons: (B, 2)
        """
        # 1. Dropout on Embeddings
        t_emb = self.dropout_emb(title_emb)
        b_emb = self.dropout_emb(body_emb)
        h_seq = self.dropout_emb(hist_seq)

        # 2. Dual-Query Attention
        ctx_title = self.att_title(t_emb, h_seq, hist_mask)
        ctx_body = self.att_body(b_emb, h_seq, hist_mask)

        # 3. Internal Alignment Injection
        # Calculate Cosine Similarity between Query and Attended Context
        # This captures how well the history supports the specific request topic/narrative
        sim_title = F.cosine_similarity(t_emb, ctx_title, dim=1).unsqueeze(1)
        sim_body = F.cosine_similarity(b_emb, ctx_body, dim=1).unsqueeze(1)

        # 4. Construct Semantic Vector
        semantic_vec = torch.cat(
            [t_emb, b_emb, ctx_title, ctx_body, cons, sim_title, sim_body], dim=1
        )

        # 5. Skip-Gated Fusion
        # Compute Gate from Metadata
        gate = torch.sigmoid(self.meta_gate(meta))

        # Apply Gate (Multiplicative interaction)
        gated_semantic = semantic_vec * gate

        # Concatenate with Metadata (Additive Skip Connection)
        # This ensures metadata can act as both a filter and a direct predictor
        fused_vec = torch.cat([gated_semantic, meta], dim=1)

        # 6. Classification
        logits = self.fusion_mlp(fused_vec)

        return logits.squeeze(1)
