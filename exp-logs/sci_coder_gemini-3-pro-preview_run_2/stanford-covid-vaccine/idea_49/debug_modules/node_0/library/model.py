import torch
import torch.nn as nn
from library.config import Config
from library.model_components import InputEmbeddingStem, DenseTCN


class EIPFN(nn.Module):
    """
    Embedded-Input Pure-Feedback Network (EI-PFN).

    Features:
    - Input Embedding Stem: Projects sparse categorical inputs to dense latent space.
    - Static Backbone: DenseTCN processing sequence/structure features.
    - Pure-Feedback Module: Lightweight DenseTCN processing recycled predictions.
    - Interaction Head: Augmented Gather (Self + Partner) + BiGRU.
    """

    def __init__(self):
        super(EIPFN, self).__init__()

        # Hyperparameters from Config
        self.hidden_dim = Config.HIDDEN_DIM  # 64
        self.latent_dim = Config.LATENT_DIM  # 64
        self.fb_dim = Config.FEEDBACK_DIM  # 16
        self.fb_embed_dim = Config.FEEDBACK_EMBED_DIM  # 32
        self.dropout = Config.DROPOUT
        self.dilations = Config.DILATIONS  # [1, 2, 4, 8, 16, 32]
        self.kernel_size = Config.KERNEL_SIZE  # 3
        self.num_targets = Config.NUM_TARGETS  # 5
        self.input_channels = Config.INPUT_CHANNELS  # 18

        # 1. Input Embedding Stem
        # Projects 18 -> 64
        self.input_stem = InputEmbeddingStem(
            in_channels=self.input_channels, out_channels=self.hidden_dim
        )

        # 2. Backbone (Static Dense TCN)
        # Input: 64, Growth: 64
        self.backbone = DenseTCN(
            in_channels=self.hidden_dim,
            growth_rate=self.hidden_dim,
            kernel_size=self.kernel_size,
            dilations=self.dilations,
            dropout=self.dropout,
        )

        # Calculate backbone output channels: In + (Layers * Growth)
        # 64 + (6 * 64) = 448
        backbone_out_channels = self.hidden_dim + len(self.dilations) * self.hidden_dim

        # Project back to Latent Dim (64)
        self.backbone_proj = nn.Conv1d(
            backbone_out_channels, self.latent_dim, kernel_size=1
        )

        # 3. Pure-Feedback Module
        # Embeds 5 targets -> 16
        self.fb_input_proj = nn.Conv1d(self.num_targets, self.fb_dim, kernel_size=1)

        # Lightweight DenseTCN: Input 16, Growth 16
        self.fb_module = DenseTCN(
            in_channels=self.fb_dim,
            growth_rate=self.fb_dim,
            kernel_size=self.kernel_size,
            dilations=self.dilations,
            dropout=self.dropout,
        )

        # Calculate feedback output channels: 16 + (6 * 16) = 112
        fb_out_channels = self.fb_dim + len(self.dilations) * self.fb_dim

        # Project to Feedback Embed Dim (32)
        self.fb_proj = nn.Conv1d(fb_out_channels, self.fb_embed_dim, kernel_size=1)

        # 4. Interaction & Aggregation
        # Input to RNN: (Latent + Feedback) * 2 (for Self and Partner)
        # (64 + 32) * 2 = 192
        rnn_input_dim = (self.latent_dim + self.fb_embed_dim) * 2

        self.gru = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Head: 64 * 2 (BiDirectional) -> 5 Targets
        self.head = nn.Linear(self.hidden_dim * 2, self.num_targets)

    def forward(self, inputs, partner_indices, y_prev=None):
        """
        Args:
            inputs: (B, L, 18) - Sequence, Structure, Loop, Partner Identity
            partner_indices: (B, L) - Indices of paired bases (-1 if unpaired)
            y_prev: (B, L, 5) - Recycled predictions from previous pass (optional)
        """
        B, L, _ = inputs.shape
        device = inputs.device

        # --- 1. Static Backbone ---
        # InputEmbeddingStem handles (B, L, C) -> (B, C, L) permutation internally
        x = self.input_stem(inputs)  # (B, 64, L)

        feat = self.backbone(x)  # (B, 448, L)
        z = self.backbone_proj(feat)  # (B, 64, L)

        # Permute to (B, L, 64) for concatenation with feedback and gathering
        z = z.permute(0, 2, 1)

        # --- 2. Feedback Loop ---
        if y_prev is None:
            y_prev = torch.zeros((B, L, self.num_targets), device=device)

        # Strict Masking: Only scored columns (0, 1, 3) drive the feedback
        # Create mask: [1, 1, 0, 1, 0]
        mask = torch.zeros((1, 1, self.num_targets), device=device)
        mask[:, :, Config.TARGET_INDICES] = 1.0

        y_masked = y_prev * mask  # (B, L, 5)

        # Process Feedback
        # Permute for Conv1d: (B, L, 5) -> (B, 5, L)
        fb_in = self.fb_input_proj(y_masked.permute(0, 2, 1))  # (B, 16, L)
        fb_feat = self.fb_module(fb_in)  # (B, 112, L)
        e_fb = self.fb_proj(fb_feat)  # (B, 32, L)

        # Permute to (B, L, 32)
        e_fb = e_fb.permute(0, 2, 1)

        # --- 3. Interaction (Augmented Gather) ---
        # Self Vector: Concatenate Static Latent + Feedback Embedding
        self_vec = torch.cat([z, e_fb], dim=2)  # (B, L, 96)

        # Partner Vector: Gather self_vec from partner positions
        # Create batch index grid
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, L)

        # Handle unpaired bases (-1): Replace with 0 for safe gathering, then mask result
        mask_unpaired = partner_indices == -1
        safe_indices = partner_indices.clone()
        safe_indices[mask_unpaired] = 0

        # Gather
        partner_vec = self_vec[batch_idx, safe_indices]  # (B, L, 96)

        # Zero out vectors for unpaired positions
        partner_vec[mask_unpaired] = 0.0

        # Fuse Self and Partner
        combined = torch.cat([self_vec, partner_vec], dim=2)  # (B, L, 192)

        # --- 4. Aggregation & Head ---
        out, _ = self.gru(combined)  # (B, L, 128)
        preds = self.head(out)  # (B, L, 5)

        return preds
