import torch
import torch.nn as nn
from library.config import Config
from library.layers import DenseTCNStack


class DDFRN(nn.Module):
    """
    Decoupled Dense-Feedback Recurrent Network (DDF-RN).

    Architecture:
    1. Static Backbone: Dense Dilated TCN processing sequence/structure features.
    2. Feedback Module: Lightweight Dense TCN processing recycled predictions.
    3. Interaction Head: Partner-aware gathering + BiGRU aggregation.
    """

    def __init__(self):
        super(DDFRN, self).__init__()

        # ----------------------------------------------------------------------
        # 1. Static Backbone
        # ----------------------------------------------------------------------
        # Input Channels:
        # Sequence (4) + Structure (3) + Loop Type (7) + Partner Identity (4)
        self.input_dim = 4 + 3 + 7 + 4

        self.backbone = DenseTCNStack(
            in_channels=self.input_dim,
            growth_rate=Config.BACKBONE_GROWTH_RATE,
            kernel_size=Config.BACKBONE_KERNEL_SIZE,
            dilations=Config.BACKBONE_DILATIONS,
            dropout=Config.BACKBONE_DROPOUT,
        )

        # Project backbone features to Latent Dim Z
        self.latent_proj = nn.Conv1d(
            self.backbone.out_channels, Config.LATENT_DIM, kernel_size=1
        )

        # ----------------------------------------------------------------------
        # 2. Feedback Module
        # ----------------------------------------------------------------------
        # Inputs: Recycled Predictions (5) + Topology Features (Struct 3 + Loop 7 = 10)
        self.feedback_in_dim = 5 + 3 + 7

        self.feedback_encoder = DenseTCNStack(
            in_channels=self.feedback_in_dim,
            growth_rate=Config.FEEDBACK_GROWTH_RATE,
            kernel_size=Config.FEEDBACK_KERNEL_SIZE,
            dilations=Config.FEEDBACK_DILATIONS,
            dropout=0.0,  # Lightweight, so minimal dropout
        )

        # Project feedback features to Embedding Dim E_fb
        self.feedback_proj = nn.Conv1d(
            self.feedback_encoder.out_channels, Config.FEEDBACK_EMBED_DIM, kernel_size=1
        )

        # ----------------------------------------------------------------------
        # 3. Interaction Head
        # ----------------------------------------------------------------------
        # Node Feature: [Z, E_fb]
        node_dim = Config.LATENT_DIM + Config.FEEDBACK_EMBED_DIM

        # Fusion Input: [Node_Self, Node_Partner]
        rnn_input_dim = node_dim * 2

        self.rnn = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=Config.RNN_HIDDEN_DIM,
            num_layers=Config.RNN_LAYERS,
            batch_first=True,
            bidirectional=Config.RNN_BIDIRECTIONAL,
        )

        rnn_out_dim = (
            Config.RNN_HIDDEN_DIM * 2
            if Config.RNN_BIDIRECTIONAL
            else Config.RNN_HIDDEN_DIM
        )

        self.head = nn.Linear(rnn_out_dim, 5)

    def forward(self, inputs, partner_indices):
        """
        Args:
            inputs (torch.Tensor): Shape (Batch, Seq_Len, 18)
            partner_indices (torch.Tensor): Shape (Batch, Seq_Len)

        Returns:
            y_1 (torch.Tensor): Pass 1 predictions (Batch, Seq_Len, 5)
            y_2 (torch.Tensor): Pass 2 predictions (Batch, Seq_Len, 5)
        """
        B, L, _ = inputs.shape

        # Permute for Conv1d: (B, C, L)
        x = inputs.transpose(1, 2)

        # ------------------------------------------------------------------
        # Step 1: Static Backbone
        # ------------------------------------------------------------------
        # Compute Z once
        backbone_feat = self.backbone(x)
        z = self.latent_proj(backbone_feat)  # (B, Latent, L)

        # Extract Topology Features for Feedback (Struct + Loop)
        # Indices in input: Seq(0-3), Struct(4-6), Loop(7-13), PartnerID(14-17)
        # We need indices 4 to 13 (inclusive range 4:14)
        topo_features = x[:, 4:14, :]  # (B, 10, L)

        # ------------------------------------------------------------------
        # Step 2: Pass 1 (Zero Feedback)
        # ------------------------------------------------------------------
        # Initial prediction y_0 is all zeros
        y_0 = torch.zeros(B, 5, L, device=inputs.device)

        # Compute Feedback Embeddings E_fb_0
        e_fb_0 = self._compute_feedback(y_0, topo_features)

        # Run Interaction Head -> y_1
        y_1 = self._run_head(z, e_fb_0, partner_indices)

        # ------------------------------------------------------------------
        # Step 3: Pass 2 (Refined Feedback)
        # ------------------------------------------------------------------
        # Prepare feedback: Detach and Mask
        y_fb = y_1.detach().transpose(1, 2)  # (B, 5, L)

        # Mask unscored positions (indices >= seq_scored)
        if Config.SEQ_SCORED < L:
            # Create a mask of shape (1, 1, L) or (B, 5, L)
            mask = torch.ones_like(y_fb)
            mask[:, :, Config.SEQ_SCORED :] = 0.0
            y_fb = y_fb * mask

        # Compute Feedback Embeddings E_fb_1
        e_fb_1 = self._compute_feedback(y_fb, topo_features)

        # Run Interaction Head -> y_2
        y_2 = self._run_head(z, e_fb_1, partner_indices)

        return y_1, y_2

    def _compute_feedback(self, y_pred, topo_features):
        """
        Computes feedback embeddings from predictions and topology.

        Args:
            y_pred: (B, 5, L)
            topo_features: (B, 10, L)

        Returns:
            e_fb: (B, Feedback_Dim, L)
        """
        # Concatenate predictions with topology
        fb_input = torch.cat([y_pred, topo_features], dim=1)  # (B, 15, L)

        # Encode
        fb_feat = self.feedback_encoder(fb_input)
        e_fb = self.feedback_proj(fb_feat)

        return e_fb

    def _run_head(self, z, e_fb, partner_indices):
        """
        Runs the interaction head (Gather -> Fusion -> RNN -> Projection).

        Args:
            z: (B, Latent, L)
            e_fb: (B, Feedback_Dim, L)
            partner_indices: (B, L)

        Returns:
            logits: (B, L, 5)
        """
        B, _, L = z.shape

        # 1. Construct Node Vector [Z, E_fb]
        h = torch.cat([z, e_fb], dim=1)  # (B, C_total, L)
        C_total = h.shape[1]

        # 2. Gather Partner Vector
        # partner_indices has -1 for unpaired.
        # We clamp -1 to 0 for gathering, then mask the result.

        # Clone to avoid modifying original tensor
        p_idx_clamped = partner_indices.clone()
        mask_unpaired = partner_indices == -1  # Boolean mask (B, L)
        p_idx_clamped[mask_unpaired] = 0

        # Expand indices for gather: (B, C, L)
        # We gather along dim 2 (sequence length)
        gather_idx = p_idx_clamped.unsqueeze(1).expand(-1, C_total, -1)

        # Gather
        h_partner = torch.gather(h, 2, gather_idx)

        # Apply mask to zero out vectors gathered from index 0 if they were actually unpaired
        mask_expanded = mask_unpaired.unsqueeze(1).expand(-1, C_total, -1)
        h_partner[mask_expanded] = 0.0

        # 3. Fusion: Concatenate Self + Partner
        combined = torch.cat([h, h_partner], dim=1)  # (B, 2*C_total, L)

        # 4. RNN Aggregation
        # Permute to (B, L, C)
        rnn_in = combined.transpose(1, 2)

        rnn_out, _ = self.rnn(rnn_in)

        # 5. Projection
        logits = self.head(rnn_out)  # (B, L, 5)

        return logits
