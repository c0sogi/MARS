import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import MLP_PARAMS, EMBEDDING_DIM


class DualQueryAttention(nn.Module):
    """
    Computes attention over user history using two distinct queries:
    1. Title Query -> Topic Context
    2. Body Query -> Narrative Context
    """

    def __init__(self, embedding_dim, dropout=0.1):
        super(DualQueryAttention, self).__init__()
        self.scale = embedding_dim**-0.5
        self.dropout = nn.Dropout(dropout)

        # Optional: Projections for Q, K, V.
        # Given the "Raw SBERT" instruction, we can keep identity or use linear.
        # We'll use identity for simplicity and to preserve the raw semantic space
        # as per the "Raw SBERT embedding" note in the idea.

    def forward(self, history_seq, history_mask, query_title, query_body):
        """
        Args:
            history_seq: (B, SeqLen, EmbDim) - Keys/Values
            history_mask: (B, SeqLen) - 1 for valid, 0 for pad
            query_title: (B, EmbDim)
            query_body: (B, EmbDim)
        Returns:
            topic_context: (B, EmbDim)
            narrative_context: (B, EmbDim)
        """
        # Expand queries for broadcasting: (B, 1, EmbDim)
        q_title = query_title.unsqueeze(1)
        q_body = query_body.unsqueeze(1)

        # Calculate raw scores: (B, 1, SeqLen)
        # Q * K^T
        scores_title = torch.bmm(q_title, history_seq.transpose(1, 2)) * self.scale
        scores_body = torch.bmm(q_body, history_seq.transpose(1, 2)) * self.scale

        # Apply Masking
        # mask is 1 for valid, 0 for pad. We want to set pad positions to -inf.
        # (B, 1, SeqLen)
        mask_expanded = history_mask.unsqueeze(1)

        # Create a large negative number tensor
        neg_inf = torch.tensor(-1e9).to(history_seq.device)

        scores_title = torch.where(mask_expanded == 1, scores_title, neg_inf)
        scores_body = torch.where(mask_expanded == 1, scores_body, neg_inf)

        # Softmax
        attn_title = F.softmax(scores_title, dim=-1)
        attn_body = F.softmax(scores_body, dim=-1)

        attn_title = self.dropout(attn_title)
        attn_body = self.dropout(attn_body)

        # Weighted Sum: (B, 1, SeqLen) * (B, SeqLen, EmbDim) -> (B, 1, EmbDim)
        context_title = torch.bmm(attn_title, history_seq).squeeze(1)
        context_body = torch.bmm(attn_body, history_seq).squeeze(1)

        # Handle case where history is empty (all masked) -> context will be 0 (softmax on all -inf is uniform or nan, but masking usually handles it if implemented carefully.
        # With -1e9, softmax is close to 0. If all are -1e9, it distributes evenly.
        # We multiply by the mask sum check to ensure zeroing out if absolutely no history.
        has_history = (history_mask.sum(dim=1, keepdim=True) > 0).float()
        context_title = context_title * has_history
        context_body = context_body * has_history

        return context_title, context_body


class OrthogonalSkipGate(nn.Module):
    """
    Implements the Orthogonal Skip-Gated Fusion.

    Logic:
    1. Gate G is derived ONLY from the Control Vector (Metadata).
    2. Semantic Vector S is modulated by G: S_gated = S * G.
    3. Final output is concatenation of [S_gated, Control Vector].

    This prevents semantic information from leaking into the gate (contamination),
    ensuring the gate represents pure 'reliability/credibility'.
    """

    def __init__(self, semantic_dim, control_dim):
        super(OrthogonalSkipGate, self).__init__()

        # Gate generator: Control -> Gate (size of semantic)
        self.gate_fc = nn.Sequential(nn.Linear(control_dim, semantic_dim), nn.Sigmoid())

    def forward(self, semantic_vector, control_vector):
        """
        Args:
            semantic_vector: (B, SemDim)
            control_vector: (B, CtrlDim)
        Returns:
            fused_vector: (B, SemDim + CtrlDim)
        """
        # Generate Gate
        gate = self.gate_fc(control_vector)

        # Modulate Semantic
        semantic_gated = semantic_vector * gate

        # Orthogonal Skip Connection
        fused = torch.cat([semantic_gated, control_vector], dim=1)

        return fused


class PizzaNet(nn.Module):
    """
    Hybrid Ensemble MLP (Stream B).

    Architecture:
    - Branch 1 (Title): Raw SBERT
    - Branch 2 (Body): Raw SBERT
    - Branch 3 (History): DualQueryAttention (Topic + Narrative Contexts)
    - Branch 4 (Persona): History Centroid
    - Internal: Consistency Scalars (Cosine Sim)
    - Branch 5 (Metadata): Control Vector

    Fusion: OrthogonalSkipGate
    """

    def __init__(self, input_metadata_dim):
        super(PizzaNet, self).__init__()

        self.embedding_dim = EMBEDDING_DIM
        self.hidden_dim = MLP_PARAMS["hidden_dim"]
        self.dropout_emb_p = MLP_PARAMS["dropout_embedding"]
        self.dropout_dense_p = MLP_PARAMS["dropout_dense"]

        # --- Feature Extractors ---
        self.dropout_emb = nn.Dropout(self.dropout_emb_p)

        # History Attention
        self.history_attn = DualQueryAttention(
            self.embedding_dim, dropout=self.dropout_emb_p
        )

        # --- Semantic Vector Assembly ---
        # Components:
        # 1. Title (384)
        # 2. Body (384)
        # 3. Topic Context (384)
        # 4. Narrative Context (384)
        # 5. Centroid (384)
        # 6. Consistency Scalars (2: Title-Centroid, Body-Centroid)
        self.semantic_dim = (self.embedding_dim * 5) + 2

        # --- Fusion ---
        self.fusion_gate = OrthogonalSkipGate(self.semantic_dim, input_metadata_dim)

        # --- Classifier Head ---
        # Input size is Semantic + Metadata (due to skip connection)
        self.head_input_dim = self.semantic_dim + input_metadata_dim

        self.classifier = nn.Sequential(
            nn.Linear(self.head_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_dense_p),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_dense_p),
            nn.Linear(self.hidden_dim // 2, 1),
        )

    def forward(self, batch_dict):
        """
        Args:
            batch_dict: Dictionary containing tensors from PizzaDataset
        Returns:
            logits: (B, 1)
        """
        # Unpack
        metadata = batch_dict["metadata"]  # (B, MetaDim)
        title_emb = batch_dict["title_emb"]  # (B, 384)
        body_emb = batch_dict["body_emb"]  # (B, 384)
        history_seq = batch_dict["history_seq"]  # (B, Seq, 384)
        history_mask = batch_dict["history_mask"]  # (B, Seq)
        centroid = batch_dict["history_centroid"]  # (B, 384)

        # 1. Apply Dropout to Embeddings
        title_emb = self.dropout_emb(title_emb)
        body_emb = self.dropout_emb(body_emb)
        history_seq = self.dropout_emb(history_seq)
        centroid = self.dropout_emb(centroid)

        # 2. History Attention (Dual Query)
        # Note: We use the dropped-out versions for attention calculation
        topic_ctx, narrative_ctx = self.history_attn(
            history_seq, history_mask, title_emb, body_emb
        )

        # 3. Compute Consistency Scalars (On-the-fly)
        # We use the original (non-dropped) or dropped?
        # Usually consistent usage is better. We used dropped for attention, let's use dropped here.
        # Cosine Similarity: (B, D) vs (B, D) -> (B,)
        # Add epsilon to avoid div by zero if dropout zeros out everything (unlikely but safe)
        sim_title = F.cosine_similarity(title_emb, centroid, dim=1, eps=1e-8).unsqueeze(
            1
        )
        sim_body = F.cosine_similarity(body_emb, centroid, dim=1, eps=1e-8).unsqueeze(1)

        # 4. Assemble Semantic Vector
        # [Title, Body, TopicCtx, NarrativeCtx, Centroid, SimTitle, SimBody]
        semantic_vector = torch.cat(
            [
                title_emb,
                body_emb,
                topic_ctx,
                narrative_ctx,
                centroid,
                sim_title,
                sim_body,
            ],
            dim=1,
        )

        # 5. Orthogonal Skip-Gated Fusion
        # Control vector is Metadata
        fused_vector = self.fusion_gate(semantic_vector, metadata)

        # 6. Classifier
        logits = self.classifier(fused_vector)

        return logits
