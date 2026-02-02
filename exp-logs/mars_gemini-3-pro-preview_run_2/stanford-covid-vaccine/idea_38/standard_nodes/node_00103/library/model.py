import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DenseDilatedBlock(nn.Module):
    """
    Single-Layer Dilated Block with Pre-activation (LN -> ReLU -> Conv -> Dropout).
    Designed for Dense Connectivity where in_channels grows.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        # Padding to maintain sequence length: dilation * (kernel_size // 2)
        padding = dilation * (kernel_size // 2)

        self.net = nn.Sequential(
            # LayerNorm is applied in forward after permutation
            nn.ReLU(),
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.Dropout(dropout),
        )
        self.ln = nn.LayerNorm(in_channels, eps=1e-6)

    def forward(self, x):
        # x: (N, C, L)
        # Apply LayerNorm on channels -> Permute to (N, L, C)
        x_ln = x.permute(0, 2, 1)
        x_ln = self.ln(x_ln)
        x_ln = x_ln.permute(0, 2, 1)  # Back to (N, C, L)

        # Apply rest of the block
        out = self.net(x_ln)
        return out


class DenseDilatedTCN(nn.Module):
    """
    Temporal Convolutional Network with Dense Connections.
    Input to each block is the concatenation of the original input and all prior block outputs.
    """

    def __init__(
        self, in_dim, growth_rate, layers, kernel_size, latent_dim, dilations=None
    ):
        super().__init__()
        self.blocks = nn.ModuleList()

        current_dim = in_dim

        # Default dilations (powers of 2) if not provided
        if dilations is None:
            dilations = [2**i for i in range(layers)]
        else:
            dilations = dilations[:layers]

        for d in dilations:
            block = DenseDilatedBlock(
                in_channels=current_dim,
                out_channels=growth_rate,
                kernel_size=kernel_size,
                dilation=d,
                dropout=Config.DROPOUT,
            )
            self.blocks.append(block)
            # In DenseNet, the next block receives everything concatenated
            current_dim += growth_rate

        # Final projection to latent dimension
        self.projection = nn.Conv1d(current_dim, latent_dim, 1)

    def forward(self, x):
        # x: (N, C, L)
        features = [x]

        for block in self.blocks:
            # Concatenate all previous features along channel dim
            inp = torch.cat(features, dim=1)
            out = block(inp)
            features.append(out)

        # Final concatenation of all features
        total_features = torch.cat(features, dim=1)

        # Project to latent dim
        z = self.projection(total_features)
        return z


class DF_DCN(nn.Module):
    """
    Dense-Feedback Dense-Context Network.
    Features:
    - Static Dense TCN Backbone for sequence/structure features.
    - Dynamic Dense TCN Backbone for iterative feedback predictions.
    - Explicit Partner Interaction (Self + Partner features).
    - BiGRU Aggregator.
    """

    def __init__(self):
        super().__init__()

        # =====================================================================
        # 1. Main Backbone (Static)
        # =====================================================================
        # Input channels: 18
        # (4 Sequence + 3 Structure + 7 Loop Type + 4 Partner Identity)
        self.input_dim = 18

        self.main_backbone = DenseDilatedTCN(
            in_dim=self.input_dim,
            growth_rate=Config.MAIN_GROWTH_RATE,
            layers=Config.MAIN_LAYERS,
            kernel_size=Config.MAIN_KERNEL_SIZE,
            latent_dim=Config.MAIN_LATENT_DIM,
            dilations=[2**i for i in range(Config.MAIN_LAYERS)],
        )

        # =====================================================================
        # 2. Feedback Backbone (Dynamic)
        # =====================================================================
        # Input channels: 5 (The 5 target columns)
        self.fb_backbone = DenseDilatedTCN(
            in_dim=Config.NUM_TARGETS,
            growth_rate=Config.FB_GROWTH_RATE,
            layers=Config.FB_LAYERS,
            kernel_size=Config.FB_KERNEL_SIZE,
            latent_dim=Config.FB_LATENT_DIM,
            dilations=[2**i for i in range(Config.FB_LAYERS)],
        )

        # =====================================================================
        # 3. Aggregator & Head
        # =====================================================================
        # Input to RNN is (Z + E_fb) for Self AND Partner -> * 2
        rnn_input_dim = (Config.MAIN_LATENT_DIM + Config.FB_LATENT_DIM) * 2

        self.rnn = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=Config.RNN_HIDDEN_SIZE,
            num_layers=Config.RNN_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.RNN_DROPOUT if Config.RNN_LAYERS > 1 else 0.0,
        )

        # Head projects from RNN output (Hidden * 2) to Targets
        self.head = nn.Linear(Config.RNN_HIDDEN_SIZE * 2, Config.NUM_TARGETS)

        # =====================================================================
        # 4. Helpers
        # =====================================================================
        # Mask to zero out unscored targets in feedback loop
        # Scored: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
        # Mask: [1, 1, 0, 1, 0]
        mask_val = torch.tensor([1, 1, 0, 1, 0], dtype=torch.float32).view(1, 5, 1)
        self.register_buffer("fb_mask", mask_val)

    def forward_backbone(self, x):
        """
        Runs the heavy static backbone.
        Args:
            x (torch.Tensor): Input features (N, L, 18)
        Returns:
            torch.Tensor: Latent features Z (N, L, LatentDim)
        """
        # Permute to (N, C, L) for Conv1d
        x = x.permute(0, 2, 1)

        # Run Backbone
        z = self.main_backbone(x)  # (N, Latent, L)

        # Permute back to (N, L, Latent)
        return z.permute(0, 2, 1)

    def forward_head(self, z, partner_idx, prev_preds=None):
        """
        Runs the feedback backbone, interaction, RNN, and head.
        Args:
            z (torch.Tensor): Precomputed static features (N, L, MainLatent)
            partner_idx (torch.Tensor): Partner indices (N, L)
            prev_preds (torch.Tensor, optional): Previous predictions (N, L, 5)
        Returns:
            torch.Tensor: New predictions (N, L, 5)
        """
        N, L, _ = z.shape
        device = z.device

        # 1. Process Feedback
        if prev_preds is None:
            # First pass: No feedback, use zeros
            e_fb = torch.zeros(N, L, Config.FB_LATENT_DIM, device=device)
        else:
            # Permute preds to (N, 5, L)
            prev_preds_t = prev_preds.permute(0, 2, 1)

            # Apply Mask (Zero out unscored targets)
            masked_preds = prev_preds_t * self.fb_mask

            # Run Feedback Backbone
            e_fb_t = self.fb_backbone(masked_preds)  # (N, FB_Latent, L)
            e_fb = e_fb_t.permute(0, 2, 1)  # (N, L, FB_Latent)

        # 2. Interaction (Gathering)
        # Concatenate Z and E_fb -> Self Features
        self_feat = torch.cat([z, e_fb], dim=2)  # (N, L, Main+FB)

        # Prepare indices for gathering
        # partner_idx is (N, L). -1 indicates unpaired.
        # We replace -1 with 0 for gathering, then mask the result.
        gather_idx = partner_idx.clone()
        mask_unpaired = gather_idx == -1
        gather_idx[mask_unpaired] = 0

        # Expand indices to match feature dimension
        feat_dim = self_feat.size(2)
        # gather_idx: (N, L) -> (N, L, 1) -> (N, L, feat_dim)
        gather_idx_exp = gather_idx.unsqueeze(-1).expand(-1, -1, feat_dim)

        # Gather partner features
        partner_feat = torch.gather(self_feat, 1, gather_idx_exp)

        # Apply mask to set unpaired positions to 0
        partner_feat[mask_unpaired] = 0.0

        # 3. Fusion
        # Concatenate Self and Partner features
        fused = torch.cat([self_feat, partner_feat], dim=2)  # (N, L, (Main+FB)*2)

        # 4. Aggregation (RNN)
        rnn_out, _ = self.rnn(fused)  # (N, L, Hidden*2)

        # 5. Projection
        logits = self.head(rnn_out)  # (N, L, 5)

        return logits

    def forward(self, x, partner_idx, prev_preds=None):
        """
        End-to-end forward pass.
        """
        z = self.forward_backbone(x)
        return self.forward_head(z, partner_idx, prev_preds)
