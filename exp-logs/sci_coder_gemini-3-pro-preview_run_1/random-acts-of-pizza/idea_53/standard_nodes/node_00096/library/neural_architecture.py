import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DualQueryAttention(nn.Module):
    """
    Implements Scaled Dot-Product Attention for querying user history.
    Used to generate context vectors for Title (Topic) and Body (Narrative).
    """

    def __init__(self, embed_dim):
        super(DualQueryAttention, self).__init__()
        self.scale = embed_dim**-0.5

    def forward(self, query, keys, mask=None):
        """
        Args:
            query: (Batch, Embed_Dim)
            keys: (Batch, Seq_Len, Embed_Dim)
            mask: (Batch, Seq_Len) - 1 for valid, 0 for padding

        Returns:
            context: (Batch, Embed_Dim) - Weighted sum of keys
        """
        # Expand query to (Batch, 1, Embed_Dim) for broadcasting
        q = query.unsqueeze(1)

        # Calculate attention scores: (Batch, 1, Seq_Len)
        scores = torch.bmm(q, keys.transpose(1, 2)) * self.scale

        if mask is not None:
            # Expand mask to (Batch, 1, Seq_Len)
            m = mask.unsqueeze(1)
            # Apply additive masking: set padded positions to -inf
            scores = scores.masked_fill(m == 0, float("-inf"))

        # Softmax over sequence dimension
        attn_weights = F.softmax(scores, dim=-1)

        # Compute context: (Batch, 1, Seq_Len) * (Batch, Seq_Len, Embed_Dim) -> (Batch, 1, Embed_Dim)
        context = torch.bmm(attn_weights, keys)

        # Squeeze back to (Batch, Embed_Dim)
        return context.squeeze(1)


class OrthogonalSkipGatedMLP(nn.Module):
    """
    Hybrid architecture combining semantic embeddings and metadata via
    Orthogonal Skip-Gated Fusion.
    """

    def __init__(self, metadata_input_dim):
        super(OrthogonalSkipGatedMLP, self).__init__()

        self.embed_dim = Config.EMBEDDING_DIM
        self.hidden_dim = Config.MLP_HIDDEN_DIM

        # --- Components ---

        # 1. Dual-Query Attention Mechanism
        self.attention = DualQueryAttention(self.embed_dim)

        # 2. Metadata Branch (Branch 5)
        # Processes raw metadata into a latent representation
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(Config.MLP_DROPOUT_DENSE),
        )

        # 3. Gate Generator
        # Projects Metadata latent -> Gate (Dimension of Semantic Vector)
        # Semantic Vector Size = Title(384) + Body(384) + TitleContext(384) + BodyContext(384) + Centroid(384) + Scalars(2)
        self.semantic_vector_dim = (self.embed_dim * 5) + 2
        self.gate_proj = nn.Linear(self.hidden_dim, self.semantic_vector_dim)

        # 4. Final Classifier
        # Input: Gated Semantic Vector + Metadata Skip Connection
        fusion_dim = self.semantic_vector_dim + self.hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(Config.MLP_DROPOUT_DENSE),
            nn.Linear(self.hidden_dim, 1),
        )

        # Dropout for embeddings
        self.emb_dropout = nn.Dropout(Config.MLP_DROPOUT_EMBEDDING)

    def forward(self, inputs):
        """
        Args:
            inputs: Dictionary containing:
                - title_embs: (B, 384)
                - body_embs: (B, 384)
                - history_seq_embs: (B, Seq, 384)
                - history_mask: (B, Seq)
                - centroid_embs: (B, 384)
                - topic_consistency: (B, 1)
                - narrative_consistency: (B, 1)
                - metadata_mlp: (B, MetaDim)
        """
        # --- 1. Extract and Regularize Semantic Inputs ---
        title = self.emb_dropout(inputs["title_embs"])
        body = self.emb_dropout(inputs["body_embs"])
        history = self.emb_dropout(inputs["history_seq_embs"])
        centroid = self.emb_dropout(inputs["centroid_embs"])

        mask = inputs["history_mask"]

        # Scalars (B, 1)
        topic_sim = inputs["topic_consistency"]
        narr_sim = inputs["narrative_consistency"]
        scalars = torch.cat([topic_sim, narr_sim], dim=1)  # (B, 2)

        # --- 2. Dual-Query Attention (Branch 3) ---
        # Query 1: Topic Context (Title -> History)
        topic_context = self.attention(title, history, mask)

        # Query 2: Narrative Context (Body -> History)
        narrative_context = self.attention(body, history, mask)

        # --- 3. Construct Semantic Vector (S) ---
        # Concatenate all semantic signals
        # [Title, Body, TopicContext, NarrContext, Centroid, Scalars]
        semantic_vector = torch.cat(
            [title, body, topic_context, narrative_context, centroid, scalars], dim=1
        )  # (B, 1922)

        # --- 4. Process Metadata (Branch 5) ---
        meta_input = inputs["metadata_mlp"]
        meta_latent = self.metadata_encoder(meta_input)  # (B, Hidden)

        # --- 5. Orthogonal Skip-Gated Fusion ---

        # A. Generate Control Gate (G) strictly from Metadata
        # Sigmoid ensures range [0, 1]
        gate = torch.sigmoid(self.gate_proj(meta_latent))  # (B, 1922)

        # B. Apply Gate to Semantic Vector
        # "Credibility modulates Relevance"
        gated_semantic = semantic_vector * gate

        # C. Fusion with Skip Connection
        # We concatenate the Gated Semantic Vector with the Metadata Latent
        # This preserves the direct predictive power of metadata (Skip)
        # while using it to filter the text signals (Gate).
        fused_vector = torch.cat([gated_semantic, meta_latent], dim=1)

        # --- 6. Classification ---
        logits = self.classifier(fused_vector)

        return logits
