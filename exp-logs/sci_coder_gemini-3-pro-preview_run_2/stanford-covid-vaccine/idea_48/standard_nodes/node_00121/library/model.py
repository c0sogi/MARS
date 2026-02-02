import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.model_components import (
    InputEmbeddingStem,
    DenseDilatedBlock,
    PureFeedbackModule,
)


class InteractionAggregation(nn.Module):
    """
    Handles the interaction between self and partner bases, followed by
    sequence modeling and output projection.

    Logic:
    1. Construct Self Vector: [Z_i, E_fb_i]
    2. Construct Partner Vector: [Z_j, E_fb_j] (via gather)
    3. Apply Null-Mask to Partner Vector (zero if unpaired)
    4. Fuse: Concatenate [Self, Partner]
    5. Aggregate: Bidirectional GRU
    6. Project: Linear -> Targets
    """

    def __init__(self, input_dim, rnn_hidden_size, num_layers, num_targets):
        super().__init__()

        # Input dim is (Z_dim + E_fb_dim) * 2 because of Self+Partner concatenation
        self.rnn_input_dim = input_dim * 2

        self.rnn = nn.GRU(
            input_size=self.rnn_input_dim,
            hidden_size=rnn_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Bidirectional output is 2 * hidden_size
        self.proj = nn.Linear(rnn_hidden_size * 2, num_targets)

    def forward(self, z, e_fb, partner_map):
        """
        Args:
            z: Static backbone features (N, C_z, L)
            e_fb: Feedback embeddings (N, C_fb, L)
            partner_map: Indices of partners (N, L)
        """
        # 1. Concatenate Self Features: (N, C_z + C_fb, L)
        features = torch.cat([z, e_fb], dim=1)

        N, C, L = features.shape

        # 2. Gather Partner Features
        # Expand partner_map to (N, C, L) for gathering
        partner_map_expanded = partner_map.unsqueeze(1).expand(-1, C, -1)

        # Gather: partner_features[b, c, i] = features[b, c, partner_map[b, i]]
        partner_features = torch.gather(features, 2, partner_map_expanded)

        # 3. Null-Masking
        # Identify unpaired bases. In data loader, unpaired indices are set to self (i).
        # We mask partner features where partner_index == current_index
        curr_indices = torch.arange(L, device=features.device).view(1, 1, L)
        # Mask is 0 if unpaired (index == self), 1 otherwise
        mask = (partner_map.unsqueeze(1) != curr_indices).float()

        partner_features = partner_features * mask

        # 4. Fusion (Self + Partner) -> (N, 2*C, L)
        combined = torch.cat([features, partner_features], dim=1)

        # 5. RNN Aggregation
        # Permute to (N, L, Channels) for RNN
        combined = combined.permute(0, 2, 1)

        rnn_out, _ = self.rnn(combined)

        # 6. Projection -> (N, L, 5)
        out = self.proj(rnn_out)

        # Permute back to (N, 5, L)
        return out.permute(0, 2, 1)


class EIPFN(nn.Module):
    """
    Embedded-Input Pure-Feedback Network (EI-PFN).

    Architecture:
    1. Input Embedding Stem: Projects discrete inputs to continuous space.
    2. Static Backbone: Dense Dilated TCN extracting structural features (Z).
    3. Pure-Feedback Loop:
       - Pass 1: Predict using Z and zero-initialized feedback.
       - Pass 2: Predict using Z and masked feedback from Pass 1.
    """

    def __init__(self):
        super().__init__()

        # =====================================================================
        # 1. Input Embedding Stem
        # =====================================================================
        self.stem = InputEmbeddingStem(
            in_channels=Config.INPUT_CHANNELS, out_channels=Config.NUM_INIT_FEATURES
        )

        # =====================================================================
        # 2. Main Backbone (Static Dense Dilated TCN)
        # =====================================================================
        self.backbone_blocks = nn.ModuleList()
        current_dim = Config.NUM_INIT_FEATURES

        for d in Config.DILATIONS:
            block = DenseDilatedBlock(
                in_channels=current_dim,
                growth_rate=Config.GROWTH_RATE,
                kernel_size=Config.KERNEL_SIZE,
                dilation=d,
                dropout=Config.DROPOUT,
            )
            self.backbone_blocks.append(block)
            # Dense connection: input to next layer grows
            current_dim += Config.GROWTH_RATE

        # Latent Projection: Project dense output to Z
        self.latent_proj = nn.Conv1d(current_dim, Config.LATENT_DIM, kernel_size=1)

        # =====================================================================
        # 3. Pure-Feedback Module
        # =====================================================================
        # We choose a modest hidden dim for the feedback embedding start
        fb_hidden_dim = 32

        self.feedback_module = PureFeedbackModule(
            in_channels=Config.NUM_TARGETS,
            hidden_dim=fb_hidden_dim,
            growth_rate=Config.FEEDBACK_GROWTH_RATE,
            out_channels=Config.FEEDBACK_OUT_CHANNELS,
            dilations=Config.DILATIONS,  # Use same receptive field pattern
            dropout=Config.DROPOUT,
        )

        # =====================================================================
        # 4. Interaction & Aggregation
        # =====================================================================
        # Input to interaction is Z + E_fb
        interaction_input_dim = Config.LATENT_DIM + Config.FEEDBACK_OUT_CHANNELS

        self.aggregator = InteractionAggregation(
            input_dim=interaction_input_dim,
            rnn_hidden_size=Config.RNN_HIDDEN_SIZE,
            num_layers=Config.RNN_LAYERS,
            num_targets=Config.NUM_TARGETS,
        )

        # Mask for feedback (Keep: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3))
        # Zero out: deg_pH10(2), deg_50C(4)
        # Shape (1, 5, 1)
        self.register_buffer(
            "feedback_mask",
            torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0], dtype=torch.float32).view(1, 5, 1),
        )

    def forward(self, x, partner_map):
        """
        Args:
            x: Input features (N, 19, L)
            partner_map: Partner indices (N, L)

        Returns:
            y_pred_1: Prediction from Pass 1 (Zero Feedback)
            y_pred_2: Prediction from Pass 2 (Refined Feedback)
        """
        # =====================================================================
        # Step 1: Static Backbone (Compute Z)
        # =====================================================================
        # Embed inputs
        features = self.stem(x)

        # Pass through Dense Blocks
        for block in self.backbone_blocks:
            out = block(features)
            # Dense connection: Concatenate along channel dimension
            features = torch.cat([features, out], dim=1)

        # Project to Latent Z
        z = self.latent_proj(features)  # (N, LATENT_DIM, L)

        # =====================================================================
        # Step 2: Pass 1 (Zero Feedback)
        # =====================================================================
        batch_size, _, seq_len = x.shape

        # Initialize feedback inputs as zeros
        y_initial = torch.zeros(
            (batch_size, Config.NUM_TARGETS, seq_len), device=x.device, dtype=x.dtype
        )

        # Generate Feedback Embeddings
        e_fb_0 = self.feedback_module(y_initial)

        # Interaction & Prediction
        y_pred_1 = self.aggregator(z, e_fb_0, partner_map)

        # =====================================================================
        # Step 3: Pass 2 (Refined Feedback)
        # =====================================================================
        # Detach first pass prediction to stop gradients flowing into the feedback generation
        # This treats the feedback as a fixed external signal
        feedback_input = y_pred_1.detach()

        # Apply Strict Masking: Zero out unscored columns
        feedback_input = feedback_input * self.feedback_mask

        # Generate Refined Feedback Embeddings
        e_fb_1 = self.feedback_module(feedback_input)

        # Interaction & Prediction
        y_pred_2 = self.aggregator(z, e_fb_1, partner_map)

        return y_pred_1, y_pred_2
