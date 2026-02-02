import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.layers import SpatialStem, DenseTCN


class FeedbackModule(nn.Module):
    """
    Processes recycled predictions.
    Applies channel masking to unscored targets before encoding.
    """

    def __init__(self, in_channels, hidden_dim, out_dim):
        super().__init__()
        # Spatial Stem for immediate local mixing of reactivity motifs
        self.stem = SpatialStem(in_channels, hidden_dim)

        # Lightweight Dense TCN for feedback context
        # Using smaller growth rate and fewer dilations as per design
        self.backbone = DenseTCN(
            in_channels=hidden_dim,
            growth_rate=Config.FEEDBACK_GROWTH_RATE,
            dilations=[
                1,
                2,
                4,
                8,
            ],  # Smaller receptive field sufficient for feedback refinement
            dropout=Config.DROPOUT,
        )

        # Projection to feedback embedding dimension
        self.proj = nn.Conv1d(self.backbone.out_channels, out_dim, kernel_size=1)

    def forward(self, x):
        # x: (B, NumTargets, L) - already masked in the caller
        x = self.stem(x)
        x = self.backbone(x)
        x = self.proj(x)
        return x


class GCSDNModel(nn.Module):
    """
    Global-Context Spatial-Dense Network (GC-SDN).
    Combines a static dense backbone with a dynamic feedback loop and explicit
    structural interaction gathering.
    """

    def __init__(self):
        super().__init__()

        # 1. Static Encoder
        # Input: Sequence(4) + Structure(3) + Loop(7) + PartnerId(4) = 18
        self.static_stem = SpatialStem(Config.INPUT_CHANNELS, Config.HIDDEN_DIM)

        self.static_backbone = DenseTCN(
            in_channels=Config.HIDDEN_DIM,
            growth_rate=Config.HIDDEN_DIM,
            dilations=Config.DILATIONS,
            dropout=Config.DROPOUT,
        )

        self.static_proj = nn.Conv1d(
            self.static_backbone.out_channels, Config.LATENT_DIM, kernel_size=1
        )

        # 2. Feedback Module
        self.feedback_module = FeedbackModule(
            in_channels=Config.NUM_TARGETS,
            hidden_dim=Config.HIDDEN_DIM,
            out_dim=Config.FEEDBACK_DIM,
        )

        # 3. Interaction & Aggregation
        # Input to RNN: Self(Latent + Feedback) + Partner(Latent + Feedback)
        rnn_input_dim = (Config.LATENT_DIM + Config.FEEDBACK_DIM) * 2

        self.rnn = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=Config.LATENT_DIM,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        # Head: Bidirectional RNN outputs 2 * hidden_size
        self.head = nn.Linear(Config.LATENT_DIM * 2, Config.NUM_TARGETS)

    def forward_backbone(self, x):
        """
        Computes the static latent representation Z.
        """
        x = self.static_stem(x)
        x = self.static_backbone(x)
        z = self.static_proj(x)  # (B, Latent, L)
        return z

    def forward_pass(self, z, y_prev, partner_indices):
        """
        Executes one pass of the feedback loop + interaction + aggregation.

        Args:
            z: Static latent features (B, Latent, L)
            y_prev: Previous predictions (B, Targets, L)
            partner_indices: Structural pairings (B, L)
        """
        # 1. Channel Masking for Feedback
        # Scored cols: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
        # Unscored: deg_pH10(2), deg_50C(4)
        # Mask: [1, 1, 0, 1, 0]
        mask_vec = torch.tensor(
            [1, 1, 0, 1, 0], device=z.device, dtype=torch.float32
        ).view(1, -1, 1)
        y_masked = y_prev * mask_vec

        # 2. Compute Feedback Embeddings
        e_fb = self.feedback_module(y_masked)  # (B, FeedbackDim, L)

        # 3. Interaction (Gather)
        # Combine Static Z and Feedback E_fb
        combined = torch.cat([z, e_fb], dim=1)  # (B, Latent+Feedback, L)
        combined_t = combined.permute(0, 2, 1)  # (B, L, C) for gather

        batch_size, seq_len, channels = combined_t.shape

        # Handle partner indices
        # -1 indicates unpaired. We replace -1 with 0 for gather safety, then mask the result.
        p_idx_safe = partner_indices.clone()
        mask_unpaired = p_idx_safe == -1
        p_idx_safe[mask_unpaired] = 0

        # Expand indices for gather: (B, L, C)
        p_idx_expanded = p_idx_safe.unsqueeze(-1).expand(-1, -1, channels)

        # Gather partner vectors
        partner_vec = torch.gather(combined_t, 1, p_idx_expanded)

        # Zero out vectors for unpaired bases
        # mask_unpaired is (B, L), expand to (B, L, C)
        mask_unpaired_expanded = mask_unpaired.unsqueeze(-1).expand(-1, -1, channels)
        partner_vec[mask_unpaired_expanded] = 0.0

        # Concatenate Self and Partner
        # Shape: (B, L, C*2)
        rnn_in = torch.cat([combined_t, partner_vec], dim=2)

        # 4. Global Aggregation (RNN)
        rnn_out, _ = self.rnn(rnn_in)  # (B, L, Hidden*2)

        # 5. Prediction Head
        pred = self.head(rnn_out)  # (B, L, Targets)

        return pred.permute(0, 2, 1)  # (B, Targets, L)

    def forward(self, x, partner_indices, y_init=None):
        """
        Full forward pass with iterative refinement.
        """
        # Compute Static Latent
        z = self.forward_backbone(x)

        # Pass 1: Zero Feedback (or provided init)
        if y_init is None:
            y_init = torch.zeros(
                (x.size(0), Config.NUM_TARGETS, x.size(2)),
                device=x.device,
                dtype=x.dtype,
            )

        y1 = self.forward_pass(z, y_init, partner_indices)

        # Pass 2: Feedback from Pass 1
        # In a training loop, y1 might be detached before being passed back
        # to stop gradients from flowing through the feedback generation step indefinitely,
        # but here we simply feed it. The caller (training loop) controls detach.
        y2 = self.forward_pass(z, y1, partner_indices)

        return y1, y2
