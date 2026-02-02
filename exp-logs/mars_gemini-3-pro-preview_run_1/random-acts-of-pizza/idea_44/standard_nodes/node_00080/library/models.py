import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class OrthogonalSkipMLP(nn.Module):
    """
    Orthogonal Skip-Gated MLP Architecture.

    Features:
    - Dual-Query History Attention (Title & Body queries).
    - Global Persona Injection (Centroid).
    - Orthogonal Gating: Gate derived purely from Metadata.
    - Skip Connection: Raw Metadata concatenated to output.
    """

    def __init__(self):
        super().__init__()

        # Dimensions
        self.sbert_dim = Config.SBERT_EMBEDDING_DIM
        self.meta_dim = 9  # Based on FeatureEngineer numeric_cols
        self.hidden_dim = Config.MLP_HIDDEN_DIM
        self.consistency_dim = 2

        # Regularization
        self.drop_emb = nn.Dropout(Config.MLP_DROPOUT_EMBEDDINGS)
        self.drop_dense = nn.Dropout(Config.MLP_DROPOUT_DENSE)

        # --- Branch 1 & 2: Semantic Encoders ---
        # Projects raw SBERT embeddings to hidden space
        self.title_proj = nn.Linear(self.sbert_dim, self.hidden_dim)
        self.body_proj = nn.Linear(self.sbert_dim, self.hidden_dim)

        # --- Branch 3: Dual-Query History Attention ---
        # Shared attention mechanism
        # Queries: Title, Body
        # Keys/Values: History Sequence
        self.attention = nn.MultiheadAttention(
            embed_dim=self.sbert_dim, num_heads=4, batch_first=True, dropout=0.1
        )
        # Project attention outputs to hidden space
        self.ctx_title_proj = nn.Linear(self.sbert_dim, self.hidden_dim)
        self.ctx_body_proj = nn.Linear(self.sbert_dim, self.hidden_dim)

        # --- Branch 4: Global Persona Injection ---
        self.centroid_proj = nn.Linear(self.sbert_dim, self.hidden_dim)

        # --- Branch 5: Metadata Gate (Orthogonal Control) ---
        # Strictly processes metadata to generate a scalar reliability gate
        self.gate_net = nn.Sequential(
            nn.Linear(self.meta_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # --- Fusion Layer ---
        # Semantic Vector Components:
        # 1. Title Projection (Hidden)
        # 2. Body Projection (Hidden)
        # 3. Attended History (Title Query) (Hidden)
        # 4. Attended History (Body Query) (Hidden)
        # 5. Centroid Projection (Hidden)
        # 6. Consistency Scores (2 dims)
        self.semantic_dim = (self.hidden_dim * 5) + self.consistency_dim

        # Final Classifier
        # Input: Gated Semantic Vector + Raw Metadata (Skip Connection)
        self.classifier = nn.Sequential(
            nn.Linear(self.semantic_dim + self.meta_dim, self.hidden_dim),
            nn.ReLU(),
            self.drop_dense,
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, batch):
        """
        Forward pass of the model.

        Args:
            batch (dict): Dictionary containing:
                - title: (B, 384)
                - body: (B, 384)
                - history: (B, Seq, 384)
                - history_mask: (B, Seq) Boolean mask (True=Padding)
                - centroid: (B, 384)
                - meta: (B, 9)
                - consistency: (B, 2)
        """
        # Unpack inputs
        title = batch["title"]
        body = batch["body"]
        history = batch["history"]
        mask = batch["history_mask"]
        centroid = batch["centroid"]
        meta = batch["meta"]
        consistency = batch["consistency"]

        # 1. Semantic Projections (Branches 1, 2, 4)
        p_title = self.drop_emb(F.relu(self.title_proj(title)))
        p_body = self.drop_emb(F.relu(self.body_proj(body)))
        p_cent = self.drop_emb(F.relu(self.centroid_proj(centroid)))

        # 2. Dual-Query Attention (Branch 3)
        # Prepare Queries: (B, 1, Dim)
        q_title = title.unsqueeze(1)
        q_body = body.unsqueeze(1)

        # Attention Pass 1: Topic Context (Query = Title)
        # key_padding_mask expects True for ignored positions
        ctx_t, _ = self.attention(q_title, history, history, key_padding_mask=mask)

        # Attention Pass 2: Narrative Context (Query = Body)
        ctx_b, _ = self.attention(q_body, history, history, key_padding_mask=mask)

        # Handle potential NaNs if a user has no history (all keys masked)
        ctx_t = torch.nan_to_num(ctx_t, nan=0.0)
        ctx_b = torch.nan_to_num(ctx_b, nan=0.0)

        # Squeeze sequence dim (B, 1, D) -> (B, D) and project
        p_ctx_t = self.drop_emb(F.relu(self.ctx_title_proj(ctx_t.squeeze(1))))
        p_ctx_b = self.drop_emb(F.relu(self.ctx_body_proj(ctx_b.squeeze(1))))

        # 3. Construct Semantic Vector
        semantic_vector = torch.cat(
            [p_title, p_body, p_ctx_t, p_ctx_b, p_cent, consistency], dim=1
        )

        # 4. Orthogonal Gating (Branch 5)
        # Gate depends ONLY on metadata (Credibility Signal)
        gate = self.gate_net(meta)  # (B, 1)

        # Apply Gate to Semantic Vector
        # "Trust this semantic content only if the user is credible"
        gated_semantic = semantic_vector * gate

        # 5. Skip Connection & Final Fusion
        # Concatenate Gated Semantics with Raw Metadata
        # Allows model to use metadata directly regardless of gate
        combined_vec = torch.cat([gated_semantic, meta], dim=1)

        # 6. Classification
        logits = self.classifier(combined_vec)

        return logits
