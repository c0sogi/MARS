import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.layers import DenseDilatedTCN


class InputEmbeddingStem(nn.Module):
    """
    Projects sparse one-hot inputs into a dense embedding space
    before entering the pre-activation backbone.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.stem = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.stem(x)


class PureFeedbackModule(nn.Module):
    """
    Lightweight Dense TCN for processing recycled predictions.
    """

    def __init__(self):
        super().__init__()
        self.embedding = nn.Conv1d(
            Config.FEEDBACK_INPUT_DIM, Config.FEEDBACK_GROWTH_RATE, kernel_size=1
        )

        dilations = [2**i for i in range(Config.FEEDBACK_LAYERS)]
        self.backbone = DenseDilatedTCN(
            in_channels=Config.FEEDBACK_GROWTH_RATE,
            growth_rate=Config.FEEDBACK_GROWTH_RATE,
            dilations=dilations,
            dropout=0.1,
        )

        # Calculate output dimension of DenseNet: Input + Layers * Growth
        dense_out_dim = Config.FEEDBACK_GROWTH_RATE + (
            len(dilations) * Config.FEEDBACK_GROWTH_RATE
        )
        self.out_proj = nn.Conv1d(
            dense_out_dim, Config.FEEDBACK_EMBED_DIM, kernel_size=1
        )

    def forward(self, x):
        # x: (B, 5, L)
        x = self.embedding(x)
        x = self.backbone(x)
        x = self.out_proj(x)
        return x


class REIDFN(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. Input Embedding Stem
        # Input channels: 18 (4 seq + 3 struct + 7 loop + 4 partner)
        self.input_stem = InputEmbeddingStem(18, Config.BACKBONE_GROWTH_RATE)

        # 2. Static Backbone (Dense Dilated TCN)
        self.backbone = DenseDilatedTCN(
            in_channels=Config.BACKBONE_GROWTH_RATE,
            growth_rate=Config.BACKBONE_GROWTH_RATE,
            dilations=Config.BACKBONE_DILATIONS,
            dropout=0.1,
        )

        # Calculate backbone output dimension
        # Input (Growth) + Layers * Growth
        backbone_out_dim = Config.BACKBONE_GROWTH_RATE + (
            len(Config.BACKBONE_DILATIONS) * Config.BACKBONE_GROWTH_RATE
        )

        self.latent_proj = nn.Conv1d(backbone_out_dim, Config.LATENT_DIM, kernel_size=1)

        # 3. Pure-Feedback Module
        self.feedback_module = PureFeedbackModule()

        # 4. Interaction & Aggregation
        # Input to RNN: (Latent + Feedback) * 2 (Self + Partner)
        rnn_input_dim = (Config.LATENT_DIM + Config.FEEDBACK_EMBED_DIM) * 2

        self.rnn = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=Config.RNN_HIDDEN,
            num_layers=Config.RNN_LAYERS,
            batch_first=True,
            bidirectional=True,
        )

        # Head
        self.head = nn.Linear(Config.RNN_HIDDEN * 2, 5)

    def forward(self, x, pair_indices, prev_preds=None):
        """
        Args:
            x: (B, 18, L) Input features
            pair_indices: (B, L) Indices of paired bases (-1 if unpaired)
            prev_preds: (B, L, 5) Previous predictions for feedback loop
        """
        B, C, L = x.shape
        device = x.device

        # --- 1. Static Backbone Processing ---
        # Embed sparse inputs
        embed = self.input_stem(x)

        # Run Backbone
        backbone_features = self.backbone(embed)

        # Project to Latent Z
        z = self.latent_proj(backbone_features)  # (B, Latent, L)
        z = z.transpose(1, 2)  # (B, L, Latent)

        # --- 2. Feedback Processing ---
        if prev_preds is None:
            prev_preds = torch.zeros((B, L, 5), device=device)

        # Strict Masking: Only keep reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
        # Mask: [1, 1, 0, 1, 0]
        mask = torch.tensor([1, 1, 0, 1, 0], device=device).view(1, 1, 5)
        masked_preds = prev_preds * mask

        # Feedback Module expects (B, 5, L)
        fb_in = masked_preds.transpose(1, 2)
        e_fb = self.feedback_module(fb_in).transpose(1, 2)  # (B, L, FB_Dim)

        # --- 3. Interaction (Gathering) ---
        # Construct Self Vector
        h_self = torch.cat([z, e_fb], dim=2)  # (B, L, Latent+FB)

        # Partner Vector Gathering
        # Create batch indices grid
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, L)

        # Handle -1 indices (unpaired) by clamping to 0 and masking later
        valid_mask = pair_indices != -1
        safe_indices = pair_indices.clone()
        safe_indices[~valid_mask] = 0

        # Gather partner features
        h_partner = h_self[batch_idx, safe_indices]  # (B, L, Dim)

        # Zero-out features for unpaired bases
        h_partner = h_partner * valid_mask.unsqueeze(2).float()

        # Fusion
        h_combined = torch.cat([h_self, h_partner], dim=2)  # (B, L, Dim*2)

        # --- 4. Aggregation & Head ---
        rnn_out, _ = self.rnn(h_combined)
        logits = self.head(rnn_out)  # (B, L, 5)

        return logits
