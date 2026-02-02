import torch
import torch.nn as nn
from library.config import Config


class DualStreamReasoningModule(nn.Module):
    """
    Dual-Stream Reasoning Module for Study-Level Classification.

    This module processes the output of the DINO decoder to predict study-level labels.
    It consists of two streams:
    1. Semantic Stream: Processes object query embeddings via Self-Attention to capture
       visual/texture dependencies (e.g., ground-glass vs consolidation).
    2. Geometric Stream: Processes predicted bounding boxes via an MLP to capture
       spatial configurations (e.g., bilateral, peripheral).

    The outputs of both streams are fused to predict the 4 study categories.
    """

    def __init__(self):
        super(DualStreamReasoningModule, self).__init__()

        # Hyperparameters from Config
        self.hidden_dim = Config.HIDDEN_DIM
        self.reasoning_dim = Config.REASONING_HIDDEN_DIM
        self.num_classes = Config.NUM_STUDY_CLASSES
        self.dropout_p = Config.REASONING_DROPOUT

        # ---------------------------------------------------------------------
        # 1. Semantic Stream
        # Input: Object Query Embeddings [Batch, Num_Queries, Hidden_Dim]
        # ---------------------------------------------------------------------
        # Self-Attention to model dependencies between detected objects
        # We use 8 heads as a standard default for 256 dim (32 dim per head)
        self.semantic_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            dropout=self.dropout_p,
            batch_first=True,
        )
        self.semantic_norm = nn.LayerNorm(self.hidden_dim)

        # Projection to reasoning dimension
        self.semantic_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.reasoning_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
        )

        # ---------------------------------------------------------------------
        # 2. Geometric Stream
        # Input: Predicted Boxes [Batch, Num_Queries, 4] (cx, cy, w, h)
        # ---------------------------------------------------------------------
        # Lightweight MLP to embed spatial coordinates
        self.geo_mlp = nn.Sequential(
            nn.Linear(4, self.reasoning_dim),
            nn.ReLU(),
            nn.Linear(self.reasoning_dim, self.reasoning_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
        )

        # ---------------------------------------------------------------------
        # 3. Fusion Head
        # ---------------------------------------------------------------------
        # Concatenates pooled outputs from both streams
        self.fusion_head = nn.Sequential(
            nn.Linear(self.reasoning_dim * 2, self.reasoning_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.reasoning_dim, self.num_classes),
        )

    def forward(self, query_embeds, pred_boxes):
        """
        Forward pass of the reasoning module.

        Args:
            query_embeds (Tensor): Object query embeddings from DINO decoder.
                                   Shape: [Batch, Num_Queries, Hidden_Dim]
            pred_boxes (Tensor): Predicted bounding boxes (sigmoid outputs).
                                 Shape: [Batch, Num_Queries, 4] (cx, cy, w, h)

        Returns:
            logits (Tensor): Study-level class logits.
                             Shape: [Batch, Num_Study_Classes]
        """
        # --- Semantic Stream Processing ---
        # 1. Self-Attention: Allow queries to reason about each other
        # query_embeds is used for query, key, and value
        attn_out, _ = self.semantic_attn(query_embeds, query_embeds, query_embeds)

        # 2. Residual Connection + Norm
        semantic_feat = self.semantic_norm(query_embeds + attn_out)

        # 3. Global Average Pooling over queries
        # Aggregates the semantic information from all potential findings
        semantic_feat = torch.mean(semantic_feat, dim=1)  # [Batch, Hidden_Dim]

        # 4. Project to Reasoning Dimension
        semantic_out = self.semantic_proj(semantic_feat)  # [Batch, Reasoning_Dim]

        # --- Geometric Stream Processing ---
        # 1. Embed Coordinates via MLP
        # pred_boxes contains normalized coordinates [0, 1]
        geo_feat = self.geo_mlp(pred_boxes)  # [Batch, Num_Queries, Reasoning_Dim]

        # 2. Max Pooling over queries
        # Captures the existence of specific spatial features (e.g., is there a box at top-left?)
        # Max pooling is permutation invariant and effective for set-based features
        geo_out = torch.max(geo_feat, dim=1)[0]  # [Batch, Reasoning_Dim]

        # --- Fusion & Prediction ---
        # Concatenate semantic and geometric representations
        combined_feat = torch.cat(
            [semantic_out, geo_out], dim=1
        )  # [Batch, Reasoning_Dim * 2]

        # Final Classification
        logits = self.fusion_head(combined_feat)  # [Batch, Num_Classes]

        return logits
