import torch
import torch.nn as nn
from library.config import Config
from library.model_components import (
    DilatedDenseBackbone,
    FeedbackModule,
    InteractionLayer,
)


class HC_SDRN(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. Spatial Input Stem
        # Processes concatenated One-Hot features (Input Dim 18)
        # Conv1d(k=3) -> LayerNorm -> SiLU
        # This ensures immediate mixing of adjacent nucleotides (n-grams)
        self.stem = nn.Sequential(
            nn.Conv1d(
                Config.INPUT_DIM, Config.BACKBONE_GROWTH, kernel_size=3, padding=1
            ),
            nn.LayerNorm(Config.BACKBONE_GROWTH),
            nn.SiLU(),
        )

        # 2. Main Backbone (High-Capacity Dense Dilated TCN)
        # Uses DecoupledDenseBlocks with dense connections and exponential dilation
        self.backbone = DilatedDenseBackbone(
            in_channels=Config.BACKBONE_GROWTH,
            growth_rate=Config.BACKBONE_GROWTH,
            dilations=Config.BACKBONE_DILATIONS,
            latent_dim=Config.LATENT_DIM,
        )

        # 3. Global-Context Pure-Feedback Module
        # Processes recycled predictions (targets) into feedback embeddings
        self.feedback_net = FeedbackModule(
            in_channels=Config.FEEDBACK_CHANNELS, growth_rate=Config.FEEDBACK_GROWTH
        )

        # 4. Interaction & Aggregation
        # Fuses Backbone(Z) and Feedback(E_fb) features
        # Gathers partner features based on structure and aggregates via Bi-GRU
        self.interaction = InteractionLayer(
            latent_dim=Config.LATENT_DIM,
            feedback_dim=16,  # Output dim of FeedbackModule is fixed at 16 in component
            rnn_hidden=Config.RNN_HIDDEN,
        )

        # 5. Output Head
        # Projects GRU output (Hidden * 2) to 5 target channels
        self.head = nn.Linear(Config.RNN_HIDDEN * 2, 5)

    def forward_backbone(self, x):
        """
        Computes the static latent representation Z from the input sequence.
        """
        # x: (N, L, Input_Dim) -> Permute to (N, Input_Dim, L) for Conv1d
        x = x.permute(0, 2, 1)

        # Stem
        out = self.stem[0](x)  # Conv
        out = out.permute(0, 2, 1)  # (N, L, C) for LN
        out = self.stem[1](out)  # LN
        out = out.permute(0, 2, 1)  # (N, C, L)
        out = self.stem[2](out)  # SiLU

        # Backbone
        # Returns (N, L, Latent)
        z = self.backbone(out)
        return z

    def forward_head(self, z, prev_pred, partner_indices):
        """
        Computes predictions using the latent representation and feedback.
        """
        # z: (N, L, Latent)
        # prev_pred: (N, L, 5)

        # Strict Channel Masking for Feedback
        # Scored indices: 0, 1, 3 (reactivity, deg_Mg_pH10, deg_Mg_50C)
        # We zero out unscored columns (2, 4) to prevent unsupervised noise injection.
        mask = torch.zeros_like(prev_pred)
        mask[:, :, [0, 1, 3]] = 1.0
        masked_pred = prev_pred * mask

        # Feedback Embedding
        # FeedbackModule expects (N, L, 5) input
        e_fb = self.feedback_net(masked_pred)  # (N, L, 16)

        # Interaction & Aggregation
        # Fuses Z, E_fb, and Partner features, then runs GRU
        rnn_out = self.interaction(z, e_fb, partner_indices)

        # Head
        logits = self.head(rnn_out)
        return logits

    def forward(self, x, partner_indices, prev_pred=None):
        """
        Main forward pass with iterative refinement (recycling).
        """
        # 1. Compute Static Backbone Features
        z = self.forward_backbone(x)

        # Initialize prev_pred if not provided (Pass 1 start)
        if prev_pred is None:
            prev_pred = torch.zeros((x.shape[0], Config.SEQ_LEN, 5), device=x.device)

        # 2. Pass 1: Zero Feedback (or external prev_pred)
        y1 = self.forward_head(z, prev_pred, partner_indices)

        # 3. Pass 2: Recycled Feedback
        # Detach y1 to stop gradients flowing through the generation of the target for the first pass
        # This implements the "Pure-Feedback" logic where the feedback is treated as a static hint.
        y2 = self.forward_head(z, y1.detach(), partner_indices)

        return y1, y2
