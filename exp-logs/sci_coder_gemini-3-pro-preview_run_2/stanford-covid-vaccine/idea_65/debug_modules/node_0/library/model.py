import torch
import torch.nn as nn
import torch.nn.functional as F

from library.config import (
    BACKBONE_CHANNELS,
    FEEDBACK_CHANNELS,
    HIDDEN_DIM,
    DROPOUT,
    SEQ_LEN,
    SCORED_LEN,
)
from library.modules import SpatialStem, DilatedResidualBlock, FeedbackTCN


class SSRFN(nn.Module):
    """
    Spatial-Stem Residual-Feedback Network (SS-RFN).

    Features:
    - Spatial Input Stem for robust initial feature mixing.
    - Fixed-Width Dilated Residual Backbone to prevent overfitting.
    - Iterative Refinement with Feedback TCN.
    - Explicit Partner Interaction via gathering and fusion.
    """

    def __init__(self):
        super().__init__()

        # --- Embeddings ---
        # Sequence: 4 bases (A, G, U, C)
        self.emb_seq = nn.Embedding(4, 4)
        # Structure: 3 types ((, ), .)
        self.emb_struct = nn.Embedding(3, 3)
        # Loop Type: 7 types (S, M, I, B, H, E, X)
        self.emb_loop = nn.Embedding(7, 7)
        # Partner Identity: 4 bases + 1 padding (index 4)
        self.emb_pid = nn.Embedding(5, 4)

        # Total input channels after concatenation
        in_dim = 4 + 3 + 7 + 4

        # --- Spatial Stem ---
        self.stem = SpatialStem(in_channels=in_dim, out_channels=BACKBONE_CHANNELS)

        # --- Backbone (Fixed-Width Residual Dilated TCN) ---
        # Exponential dilation: 1, 2, 4, 8, 16, 32
        self.backbone = nn.Sequential(
            *[
                DilatedResidualBlock(
                    channels=BACKBONE_CHANNELS, dilation=2**i, dropout=DROPOUT
                )
                for i in range(6)
            ]
        )

        # --- Feedback Module ---
        self.feedback_net = FeedbackTCN(
            in_channels=5, out_channels=FEEDBACK_CHANNELS, dropout=DROPOUT
        )

        # --- Interaction & Aggregation ---
        # Project backbone features to lower dim before fusion
        self.proj_z = nn.Linear(BACKBONE_CHANNELS, HIDDEN_DIM)

        # Fusion Dimension:
        # Self Vector = Z_proj (64) + Feedback (32) = 96
        # Partner Vector = 96
        # Total GRU Input = 96 + 96 = 192
        fusion_dim = (HIDDEN_DIM + FEEDBACK_CHANNELS) * 2

        self.gru = nn.GRU(
            input_size=fusion_dim,
            hidden_size=HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Head: 2 * HIDDEN_DIM (Bidirectional) -> 5 Targets
        self.head = nn.Linear(HIDDEN_DIM * 2, 5)

    def forward_backbone(self, inputs):
        """Computes static backbone features."""
        # 1. Embed Inputs
        s = self.emb_seq(inputs["seq"])  # (B, L, 4)
        st = self.emb_struct(inputs["struct"])  # (B, L, 3)
        l = self.emb_loop(inputs["loop"])  # (B, L, 7)
        p = self.emb_pid(inputs["pid"])  # (B, L, 4)

        # 2. Concatenate
        x = torch.cat([s, st, l, p], dim=2)  # (B, L, 18)
        x = x.transpose(1, 2)  # (B, 18, L)

        # 3. Stem
        x = self.stem(x)  # (B, 128, L)

        # 4. Backbone
        x = self.backbone(x)  # (B, 128, L)

        return x

    def forward_head(self, z, feedback, partner_idx):
        """
        Processes backbone features and feedback to generate predictions.

        Args:
            z: Backbone features (B, 128, L)
            feedback: Recycled predictions (B, 5, L)
            partner_idx: Indices of paired bases (B, L)
        """
        B, _, L = z.shape

        # 1. Process Feedback
        fb_emb = self.feedback_net(feedback)  # (B, 32, L)

        # 2. Prepare for Interaction (Transpose to B, L, C)
        z_t = z.transpose(1, 2)  # (B, L, 128)
        fb_t = fb_emb.transpose(1, 2)  # (B, L, 32)

        # 3. Project Backbone
        z_proj = self.proj_z(z_t)  # (B, L, 64)

        # 4. Create Self Vector
        self_vec = torch.cat([z_proj, fb_t], dim=2)  # (B, L, 96)

        # 5. Gather Partner Vector
        # Flatten for gathering
        C_self = self_vec.shape[2]
        self_vec_flat = self_vec.reshape(B * L, C_self)

        # Calculate gather indices
        # Offset each batch index by L
        batch_offsets = (torch.arange(B, device=z.device) * L).unsqueeze(1)

        # Handle unpaired (-1): replace with 0 temporarily
        # We clone partner_idx to avoid modifying the input tensor in place if it's reused
        safe_p_idx = partner_idx.clone()
        unpaired_mask = safe_p_idx == -1
        safe_p_idx[unpaired_mask] = 0

        # Add offsets
        gather_indices = (safe_p_idx + batch_offsets).view(-1)

        # Gather
        partner_vec_flat = self_vec_flat[gather_indices]

        # Mask unpaired positions (set vector to 0)
        # unpaired_mask is (B, L), flatten it
        partner_vec_flat[unpaired_mask.view(-1)] = 0

        # Reshape back
        partner_vec = partner_vec_flat.view(B, L, C_self)

        # 6. Fusion
        combined = torch.cat([self_vec, partner_vec], dim=2)  # (B, L, 192)

        # 7. Global Aggregation (GRU)
        gru_out, _ = self.gru(combined)  # (B, L, 128)

        # 8. Head
        preds = self.head(gru_out)  # (B, L, 5)

        return preds

    def forward(self, inputs, prev_preds=None):
        """
        Orchestrates the iterative refinement loop.

        If prev_preds is None, executes the full 2-pass loop:
          Pass 1: Feedback = 0 -> Y1
          Pass 2: Feedback = Masked(Y1.detach()) -> Y2

        Returns:
            If training and prev_preds is None: (Y2, Y1)
            Otherwise: Y2 (or single pass output)
        """
        # 1. Compute Static Backbone Features
        z = self.forward_backbone(inputs)
        B, _, L = z.shape

        # If prev_preds is explicitly provided (e.g. manual loop control), run single pass
        if prev_preds is not None:
            # Ensure feedback is (B, 5, L)
            if prev_preds.shape[1] != 5:
                prev_preds = prev_preds.transpose(1, 2)

            # Apply Masking logic just in case, or assume caller handled it.
            # Here we assume caller provides raw preds, so we mask.
            mask = torch.tensor([1, 1, 0, 1, 0], device=z.device, dtype=z.dtype).view(
                1, 5, 1
            )
            masked_feedback = prev_preds * mask

            return self.forward_head(z, masked_feedback, inputs["partner_idx"])

        # --- Internal Iterative Refinement Loop ---

        # Pass 1: Zero Feedback
        zero_feedback = torch.zeros((B, 5, L), device=z.device, dtype=z.dtype)
        y1 = self.forward_head(z, zero_feedback, inputs["partner_idx"])

        # Pass 2: Feedback from Pass 1
        # Detach gradients from Pass 1 to stop gradient flow through the feedback loop itself
        # (We only want to train the network to correct the error, not optimize y1 to be good feedback)
        y1_detached = y1.detach().transpose(1, 2)  # (B, 5, L)

        # Mask unscored channels (indices 2 and 4)
        # 0: Reactivity, 1: Mg_pH10, 2: pH10 (Unscored), 3: Mg_50C, 4: 50C (Unscored)
        mask = torch.tensor([1, 1, 0, 1, 0], device=z.device, dtype=z.dtype).view(
            1, 5, 1
        )
        masked_feedback = y1_detached * mask

        y2 = self.forward_head(z, masked_feedback, inputs["partner_idx"])

        # Return logic
        if self.training:
            return y2, y1
        else:
            return y2
