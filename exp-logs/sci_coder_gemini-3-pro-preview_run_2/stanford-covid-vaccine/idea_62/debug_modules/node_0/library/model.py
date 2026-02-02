import torch
import torch.nn as nn
from library.config import (
    GROWTH_RATE,
    LATENT_DIM,
    DILATIONS,
    DROPOUT,
    FEEDBACK_GROWTH,
    FEEDBACK_OUT_DIM,
    RNN_HIDDEN,
    NUM_TARGETS,
)
from library.layers import (
    HybridInputStem,
    DenseTCN,
    FeedbackProcessor,
    InteractionModule,
)


class HS_GFN(nn.Module):
    """
    Hybrid-Stem Global-Feedback Network (HS-GFN).

    Architecture:
    1. Hybrid Input Stem: Processes static features (Identity + Context).
    2. Main Backbone: Post-Activation Dense Dilated TCN to extract latent Z.
    3. Feedback Module: Processes previous predictions to extract feedback embeddings E_fb.
    4. Interaction Module: Fuses Z and E_fb, gathers partner features, and aggregates global context via BiGRU.
    5. Output Head: Linear projection to targets.
    """

    def __init__(self, in_channels=19):
        """
        Args:
            in_channels (int): Number of input feature channels.
                               Default 19 (4 Seq + 3 Struct + 7 Loop + 5 Partner).
        """
        super().__init__()

        # 1. Hybrid Input Stem
        # Input: (B, C, L) -> Output: (B, 2*C, L)
        self.stem = HybridInputStem(in_channels=in_channels)

        # 2. Main Backbone (Dense Dilated TCN)
        # Input: (B, 2*C, L) -> Output: (B, Latent_Dim, L)
        # Note: Input channels are doubled by the hybrid stem
        self.backbone = DenseTCN(
            in_channels=in_channels * 2,
            growth_rate=GROWTH_RATE,
            dilations=DILATIONS,
            dropout=DROPOUT,
            out_channels=LATENT_DIM,
        )

        # 3. Global-Context Feedback Module
        # Input: (B, 5, L) -> Output: (B, Feedback_Dim, L)
        self.feedback_processor = FeedbackProcessor(
            input_channels=NUM_TARGETS,
            growth_rate=FEEDBACK_GROWTH,
            out_channels=FEEDBACK_OUT_DIM,
            dilations=DILATIONS,
            dropout=DROPOUT,
        )

        # 4. Interaction & Aggregation Module
        # Input: Z (Latent) + E_fb (Feedback) -> Output: (B, L, 2*RNN_Hidden)
        self.interaction = InteractionModule(
            z_dim=LATENT_DIM,
            fb_dim=FEEDBACK_OUT_DIM,
            rnn_hidden=RNN_HIDDEN,
        )

        # 5. Output Head
        # Input: (B, L, 2*RNN_Hidden) -> Output: (B, L, Num_Targets)
        self.head = nn.Linear(2 * RNN_HIDDEN, NUM_TARGETS)

    def forward(self, x, partner_indices, y_prev=None):
        """
        Forward pass of the HS-GFN.

        Args:
            x (torch.Tensor): Static input features of shape (B, In_Channels, L).
            partner_indices (torch.Tensor): Partner index map of shape (B, L).
            y_prev (torch.Tensor, optional): Previous predictions of shape (B, Num_Targets, L).
                                             If None, feedback embeddings are initialized to zero.

        Returns:
            torch.Tensor: Predicted values of shape (B, L, Num_Targets).
        """
        # 1. Static Feature Processing (Backbone)
        # x: (B, 19, L) -> stem: (B, 38, L)
        stem_out = self.stem(x)

        # stem_out: (B, 38, L) -> z: (B, 64, L)
        z = self.backbone(stem_out)

        # 2. Feedback Processing
        if y_prev is None:
            # First pass: No feedback available, use zeros
            B, _, L = z.shape
            e_fb = torch.zeros((B, FEEDBACK_OUT_DIM, L), device=z.device, dtype=z.dtype)
        else:
            # Subsequent passes: Process previous predictions
            # y_prev: (B, 5, L) -> e_fb: (B, 32, L)
            e_fb = self.feedback_processor(y_prev)

        # 3. Interaction & Aggregation
        # Fuses self/partner features and runs BiGRU
        # Returns: (B, L, 128)
        rnn_out = self.interaction(z, e_fb, partner_indices)

        # 4. Output Projection
        # (B, L, 128) -> (B, L, 5)
        preds = self.head(rnn_out)

        return preds
