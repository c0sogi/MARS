import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class HybridInputStem(nn.Module):
    """
    Hybrid Input Stem:
    Branch A: Identity (Linear projection of raw input).
    Branch B: Context (Conv1d(k=3) -> LayerNorm -> SiLU).
    Concatenates A and B to resolve the Spatial Context vs. Raw Identity conflict.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Split output channels for the two branches
        self.branch_a_dim = out_channels // 2
        self.branch_b_dim = out_channels - self.branch_a_dim

        # Branch A: Identity (Pointwise linear to preserve raw features)
        self.branch_a = nn.Linear(in_channels, self.branch_a_dim)

        # Branch B: Context (Spatial mixing)
        self.branch_b = nn.Sequential(
            nn.Conv1d(in_channels, self.branch_b_dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, self.branch_b_dim),  # LayerNorm equivalent for (N, C, L)
            nn.SiLU(),
        )

    def forward(self, x):
        # x: (B, L, C_in)

        # Branch A processing (Linear expects last dim)
        out_a = self.branch_a(x)  # (B, L, C_out/2)

        # Branch B processing (Conv1d expects (B, C, L))
        x_perm = x.permute(0, 2, 1)
        out_b = self.branch_b(x_perm)  # (B, C_out/2, L)
        out_b = out_b.permute(0, 2, 1)  # (B, L, C_out/2)

        # Concatenate
        return torch.cat([out_a, out_b], dim=-1)


class PostActDenseBlock(nn.Module):
    """
    Post-Activation Dense Dilated Block.
    Input: Concatenation of all previous features.
    Structure: Dilated Conv(k=3) -> LN -> SiLU -> Pointwise Conv -> LN -> SiLU -> Dropout.
    """

    def __init__(self, in_channels, growth_rate, dilation, dropout):
        super().__init__()
        self.net = nn.Sequential(
            # Spatial mixing with dilation
            nn.Conv1d(
                in_channels,
                growth_rate,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.GroupNorm(1, growth_rate),
            nn.SiLU(),
            # Channel mixing
            nn.Conv1d(growth_rate, growth_rate, kernel_size=1),
            nn.GroupNorm(1, growth_rate),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, C_in, L)
        return self.net(x)


class DenseTCNBackbone(nn.Module):
    """
    Stack of PostActDenseBlocks with Dense Connections.
    """

    def __init__(self, in_channels, growth_rate, dilations, dropout, latent_dim):
        super().__init__()
        self.blocks = nn.ModuleList()

        current_in_channels = in_channels

        for d in dilations:
            blk = PostActDenseBlock(current_in_channels, growth_rate, d, dropout)
            self.blocks.append(blk)
            # In DenseNet, next input is concat of all previous.
            # Here we accumulate outputs. The input to the next block will be
            # the concatenation of everything so far.
            current_in_channels += growth_rate

        self.final_proj = nn.Conv1d(current_in_channels, latent_dim, kernel_size=1)

    def forward(self, x):
        # x: (B, L, C) -> permute to (B, C, L)
        x = x.permute(0, 2, 1)

        features = [x]

        for block in self.blocks:
            # Dense connection: concatenate all previous features
            inp = torch.cat(features, dim=1)
            out = block(inp)
            features.append(out)

        # Final projection on the concatenation of all features
        all_features = torch.cat(features, dim=1)
        z = self.final_proj(all_features)  # (B, Latent, L)

        return z.permute(0, 2, 1)  # (B, L, Latent)


class FeedbackModule(nn.Module):
    """
    Global-Context Pure-Feedback Module.
    Processes recycled predictions.
    """

    def __init__(self, num_targets, growth_rate, latent_dim):
        super().__init__()
        # Lightweight Dense TCN
        # We'll use a simplified version: 3 layers, dilations 1, 2, 4
        self.stem = nn.Conv1d(num_targets, growth_rate, kernel_size=1)

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        growth_rate, growth_rate, kernel_size=3, padding=d, dilation=d
                    ),
                    nn.GroupNorm(1, growth_rate),
                    nn.SiLU(),
                )
                for d in [1, 2, 4]
            ]
        )

        self.proj = nn.Conv1d(growth_rate, latent_dim, kernel_size=1)

    def forward(self, y_prev):
        # y_prev: (B, L, 5)

        # Channel Masking: Zero out unscored targets
        # Config.TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
        # Unscored are indices 2 (deg_pH10) and 4 (deg_50C)
        mask = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0], device=y_prev.device).view(
            1, 1, 5
        )
        y_masked = y_prev * mask

        x = y_masked.permute(0, 2, 1)  # (B, 5, L)
        x = self.stem(x)

        for layer in self.layers:
            x = x + layer(x)  # Residual connection for stability in feedback

        x = self.proj(x)
        return x.permute(0, 2, 1)  # (B, L, FB_Latent)


class AHDRNModel(nn.Module):
    """
    Anchored Hybrid-Dense Recurrent Network (AHD-RN).
    """

    def __init__(self):
        super().__init__()

        # Calculate input channels
        # 4 (seq) + 3 (struct) + 7 (loop) + 4 (partner identity) = 18
        self.in_channels = 18

        # 1. Hybrid Input Stem
        # We project to an initial width equal to growth rate for consistency
        self.stem = HybridInputStem(self.in_channels, Config.GROWTH_RATE)

        # 2. Backbone
        self.backbone = DenseTCNBackbone(
            in_channels=Config.GROWTH_RATE,
            growth_rate=Config.GROWTH_RATE,
            dilations=Config.DILATIONS,
            dropout=Config.DROPOUT,
            latent_dim=Config.LATENT_DIM,
        )

        # 3. Feedback Module
        self.feedback_mod = FeedbackModule(
            num_targets=Config.NUM_TARGETS,
            growth_rate=Config.FB_GROWTH_RATE,
            latent_dim=Config.FB_LATENT_DIM,
        )

        # 4. Aggregation (RNN)
        # Input dim = (Z_dim + FB_dim) * 2 (Self + Partner)
        rnn_input_dim = (Config.LATENT_DIM + Config.FB_LATENT_DIM) * 2

        self.rnn = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=Config.RNN_HIDDEN_SIZE,
            num_layers=Config.RNN_LAYERS,
            batch_first=True,
            bidirectional=Config.RNN_BIDIRECTIONAL,
        )

        rnn_out_dim = (
            Config.RNN_HIDDEN_SIZE * 2
            if Config.RNN_BIDIRECTIONAL
            else Config.RNN_HIDDEN_SIZE
        )
        self.head = nn.Linear(rnn_out_dim, Config.NUM_TARGETS)

    def forward(self, x, partner_indices):
        """
        Args:
            x: (B, L, 18) Input features
            partner_indices: (B, L) Indices of paired bases
        Returns:
            y_final: (B, L, 5) Final prediction
            y_aux: (B, L, 5) Auxiliary prediction (Pass 1)
        """
        batch_size, seq_len, _ = x.shape

        # --- Step 1: Static Backbone ---
        stem_out = self.stem(x)
        z = self.backbone(stem_out)  # (B, L, Latent)

        # --- Step 2: Iterative Refinement ---

        # Pass 1: Zero Feedback
        y_prev = torch.zeros(batch_size, seq_len, Config.NUM_TARGETS, device=x.device)
        e_fb_1 = self.feedback_mod(y_prev)  # (B, L, FB_Latent)
        y_1 = self._run_head(z, e_fb_1, partner_indices)

        # Pass 2: Feedback from Pass 1 (Detached)
        # Detach gradients to stop backprop through time (simulating inference refinement)
        y_prev_2 = y_1.detach()
        e_fb_2 = self.feedback_mod(y_prev_2)
        y_2 = self._run_head(z, e_fb_2, partner_indices)

        if self.training:
            return y_2, y_1
        else:
            return y_2

    def _run_head(self, z, e_fb, partner_indices):
        """
        Helper to run Interaction -> RNN -> Head
        """
        batch_size, seq_len, _ = z.shape

        # Concatenate Self features
        self_feat = torch.cat([z, e_fb], dim=-1)  # (B, L, Z+FB)

        # Gather Partner features
        # partner_indices is (B, L). We need to gather from self_feat.
        # Handle -1 indices (unpaired) by clamping to 0 and masking later or
        # relying on the fact that gather will pull from 0.
        # A robust way is to append a zero-vector at the end and map -1 to that index,
        # but here we can just gather and mask or assume partner_indices are valid 0..L-1
        # for paired and self-index or specific logic for unpaired.
        # Given the dataset, unpaired usually map to -1.

        # Strategy: Create a dummy zero row for index -1
        # (B, L+1, Feat)
        dummy = torch.zeros(batch_size, 1, self_feat.shape[-1], device=z.device)
        feat_expanded = torch.cat([self_feat, dummy], dim=1)

        # Adjust indices: -1 becomes L (the dummy index)
        gather_idx = partner_indices.clone()
        gather_idx[gather_idx == -1] = seq_len

        # Expand gather_idx for gathering: (B, L, Feat)
        gather_idx_expanded = gather_idx.unsqueeze(-1).expand(
            -1, -1, self_feat.shape[-1]
        )

        partner_feat = torch.gather(feat_expanded, 1, gather_idx_expanded)

        # Interaction: Concat Self + Partner
        rnn_in = torch.cat([self_feat, partner_feat], dim=-1)  # (B, L, (Z+FB)*2)

        # RNN
        rnn_out, _ = self.rnn(rnn_in)

        # Head
        logits = self.head(rnn_out)

        return logits
