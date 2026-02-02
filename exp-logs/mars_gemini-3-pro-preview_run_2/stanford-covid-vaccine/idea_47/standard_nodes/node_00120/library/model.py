import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.layers import DenseTCN, LayerNormChannels


class PFDRN(nn.Module):
    """
    Pure-Feedback Dense-Refined Network (PF-DRN).

    Implements a two-pass iterative refinement architecture:
    1. Static Backbone: Extracts robust structural features.
    2. Pure-Feedback Module: Encodes strictly masked predictions from the previous pass.
    3. Interaction & Aggregation: Fuses self and partner features via BiGRU.
    """

    def __init__(self):
        super().__init__()

        # =====================================================================
        # 1. Main Backbone (Static)
        # =====================================================================
        # Input channels: 18 (4 Seq + 3 Struct + 7 Loop + 4 PartnerID)
        self.backbone = DenseTCN(
            in_channels=18,
            growth_rate=Config.BACKBONE_GROWTH_RATE,
            dilations=Config.BACKBONE_DILATIONS,
            kernel_size=Config.BACKBONE_KERNEL_SIZE,
            dropout=Config.DROPOUT,
        )

        # Calculate output channels of DenseNet: In + (Layers * Growth)
        backbone_out_channels = (
            18 + len(Config.BACKBONE_DILATIONS) * Config.BACKBONE_GROWTH_RATE
        )

        # Projection to Latent Dim Z
        self.backbone_proj = nn.Sequential(
            LayerNormChannels(backbone_out_channels),
            nn.SiLU(),
            nn.Conv1d(backbone_out_channels, Config.LATENT_DIM, kernel_size=1),
        )

        # =====================================================================
        # 2. Pure-Feedback Module
        # =====================================================================
        # Input: 5 prediction channels
        self.feedback_module = DenseTCN(
            in_channels=5,
            growth_rate=Config.FEEDBACK_GROWTH_RATE,
            dilations=Config.FEEDBACK_DILATIONS,
            kernel_size=3,  # Standard kernel size
            dropout=Config.DROPOUT,
        )

        feedback_out_channels = (
            5 + len(Config.FEEDBACK_DILATIONS) * Config.FEEDBACK_GROWTH_RATE
        )

        # Projection to Feedback Embedding E_fb
        self.feedback_proj = nn.Sequential(
            LayerNormChannels(feedback_out_channels),
            nn.SiLU(),
            nn.Conv1d(feedback_out_channels, Config.FEEDBACK_DIM, kernel_size=1),
        )

        # =====================================================================
        # 3. Interaction & Aggregation
        # =====================================================================
        # Feature size per position = Z (64) + E_fb (32) = 96
        self.feature_dim = Config.LATENT_DIM + Config.FEEDBACK_DIM

        # RNN Input = Self Vector (96) + Partner Vector (96) = 192
        rnn_input_dim = self.feature_dim * 2

        self.gru = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=Config.RNN_HIDDEN_SIZE,
            num_layers=Config.RNN_LAYERS,
            batch_first=True,
            bidirectional=Config.RNN_BIDIRECTIONAL,
            dropout=Config.DROPOUT if Config.RNN_LAYERS > 1 else 0,
        )

        # Head: BiGRU outputs 2 * HiddenSize
        self.head = nn.Linear(Config.RNN_HIDDEN_SIZE * 2, 5)

    def _process_pass(self, z, feedback_preds, partner_indices, pairing_mask):
        """
        Executes one pass of the refinement loop.

        Args:
            z: (Batch, Latent_Dim, Seq_Len) Static backbone features.
            feedback_preds: (Batch, 5, Seq_Len) Predictions from prev pass (or zeros).
            partner_indices: (Batch, Seq_Len) Indices of paired bases.
            pairing_mask: (Batch, Seq_Len) 1.0 if paired, 0.0 otherwise.

        Returns:
            preds: (Batch, Seq_Len, 5) New predictions.
        """
        B, _, L = z.shape

        # 1. Process Feedback
        # (B, 5, L) -> (B, Feedback_Dim, L)
        e_fb = self.feedback_module(feedback_preds)
        e_fb = self.feedback_proj(e_fb)

        # 2. Construct Self Vector
        # Concatenate Z and E_fb along channel dim: (B, 96, L)
        self_features = torch.cat([z, e_fb], dim=1)

        # 3. Gather Partner Vector
        # Expand partner indices for gathering: (B, 1, L) -> (B, 96, L)
        # We need to gather along the sequence dimension (dim 2)
        expanded_indices = partner_indices.unsqueeze(1).expand(-1, self.feature_dim, -1)

        # Gather: out[i][j][k] = input[i][j][index[i][j][k]]
        partner_features = torch.gather(self_features, 2, expanded_indices)

        # 4. Null-Masking
        # Zero out partner features where there is no partner
        # pairing_mask: (B, L) -> (B, 1, L)
        mask = pairing_mask.unsqueeze(1)
        partner_features = partner_features * mask

        # 5. Fusion
        # Concatenate Self and Partner: (B, 192, L)
        combined = torch.cat([self_features, partner_features], dim=1)

        # 6. RNN Aggregation
        # Permute to (B, L, C) for RNN
        rnn_in = combined.permute(0, 2, 1)
        rnn_out, _ = self.gru(rnn_in)

        # 7. Head
        # (B, L, Hidden*2) -> (B, L, 5)
        preds = self.head(rnn_out)

        return preds

    def forward(self, inputs, partner_indices, pairing_mask):
        """
        Args:
            inputs: (Batch, 18, Seq_Len)
            partner_indices: (Batch, Seq_Len)
            pairing_mask: (Batch, Seq_Len)

        Returns:
            If training: (preds_pass1, preds_pass2)
            If eval: preds_pass2
        """
        # 1. Static Backbone Computation (Done once)
        # (B, 18, L) -> (B, Latent_Dim, L)
        z = self.backbone(inputs)
        z = self.backbone_proj(z)

        B, _, L = z.shape

        # =====================================================================
        # Pass 1: Zero Feedback
        # =====================================================================
        # Initialize feedback with zeros
        feedback_0 = torch.zeros((B, 5, L), device=inputs.device, dtype=inputs.dtype)

        preds_1 = self._process_pass(z, feedback_0, partner_indices, pairing_mask)

        # =====================================================================
        # Pass 2: Refinement with Strict Masking
        # =====================================================================
        # Detach gradients to stop backprop through time (simple iterative refinement)
        # Transpose preds_1 to (B, 5, L) for feedback module
        feedback_1 = preds_1.detach().permute(0, 2, 1)

        # Strict Masking: Zero out unscored targets (indices 2 and 4)
        # Scored indices are [0, 1, 3] (reactivity, deg_Mg_pH10, deg_Mg_50C)
        # We create a mask tensor
        strict_mask = torch.zeros_like(feedback_1)
        strict_mask[:, [0, 1, 3], :] = 1.0

        feedback_1 = feedback_1 * strict_mask

        preds_2 = self._process_pass(z, feedback_1, partner_indices, pairing_mask)

        if self.training:
            return preds_1, preds_2
        else:
            return preds_2
