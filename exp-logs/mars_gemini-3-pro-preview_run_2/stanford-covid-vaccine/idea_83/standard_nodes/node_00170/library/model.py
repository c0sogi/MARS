import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class HybridInputStem(nn.Module):
    """
    Hybrid Input Stem:
    Branch A: Identity (Raw features)
    Branch B: Context (Spatial Conv -> LN -> SiLU)
    Concatenates A and B to preserve both exact identity and local context.
    """

    def __init__(self, in_channels, context_channels=32):
        super(HybridInputStem, self).__init__()
        self.branch_a_dim = in_channels

        # Branch B: Spatial Context
        self.branch_b = nn.Sequential(
            nn.Conv1d(in_channels, context_channels, kernel_size=3, padding=1),
            nn.LayerNorm(
                context_channels
            ),  # LayerNorm over channels requires permutation in forward if using standard LN
            nn.SiLU(),
        )
        self.out_channels = in_channels + context_channels

    def forward(self, x):
        # x: (Batch, Channels, Seq)

        # Branch A: Identity
        out_a = x

        # Branch B: Context
        # LayerNorm expects (Batch, Seq, Channels), Conv1d outputs (Batch, Channels, Seq)
        out_b = self.branch_b[0](x)
        out_b = out_b.permute(0, 2, 1)  # to (B, L, C)
        out_b = self.branch_b[1](out_b)
        out_b = self.branch_b[2](out_b)
        out_b = out_b.permute(0, 2, 1)  # back to (B, C, L)

        return torch.cat([out_a, out_b], dim=1)


class DenseDilatedBlock(nn.Module):
    """
    Single-Layer Dilated Block with Dense Connections.
    Structure: Conv(k=3) -> LN -> SiLU -> Conv(k=1) -> LN -> SiLU -> Dropout
    """

    def __init__(self, in_channels, growth_rate, dilation, dropout=0.1):
        super(DenseDilatedBlock, self).__init__()
        self.net = nn.Sequential(
            # Spatial Aggregation
            nn.Conv1d(
                in_channels,
                growth_rate,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            # Permute for LN handled in forward
            nn.LayerNorm(growth_rate),
            nn.SiLU(),
            # Channel Mixing
            nn.Conv1d(growth_rate, growth_rate, kernel_size=1),
            nn.LayerNorm(growth_rate),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (Batch, In_Channels, Seq)
        out = self.net[0](x)

        # LN 1
        out = out.permute(0, 2, 1)
        out = self.net[1](out)
        out = out.permute(0, 2, 1)

        # SiLU 1
        out = self.net[2](out)

        # Conv 1x1
        out = self.net[3](out)

        # LN 2
        out = out.permute(0, 2, 1)
        out = self.net[4](out)
        out = out.permute(0, 2, 1)

        # SiLU 2 + Dropout
        out = self.net[5](out)
        out = self.net[6](out)

        return out


class DenseBackbone(nn.Module):
    """
    Stack of DenseDilatedBlocks.
    Maintains dense connections by concatenating all prior outputs.
    """

    def __init__(self, in_channels, growth_rate, dilations, latent_dim):
        super(DenseBackbone, self).__init__()
        self.blocks = nn.ModuleList()
        current_dim = in_channels

        for d in dilations:
            block = DenseDilatedBlock(
                current_dim, growth_rate, dilation=d, dropout=Config.DROPOUT
            )
            self.blocks.append(block)
            current_dim += growth_rate

        # Final projection to Latent Dim Z
        self.projector = nn.Conv1d(current_dim, latent_dim, kernel_size=1)

    def forward(self, x):
        # x: (Batch, In_Channels, Seq)
        features = [x]

        for block in self.blocks:
            # Concatenate all previous features
            dense_input = torch.cat(features, dim=1)
            out = block(dense_input)
            features.append(out)

        # Final concatenation
        total_concat = torch.cat(features, dim=1)
        z = self.projector(total_concat)
        return z


class FeedbackModule(nn.Module):
    """
    Global-Context Pure-Feedback Module.
    Processes recycled predictions with strict channel masking but full sequence context.
    """

    def __init__(self, in_channels=5, out_channels=32, growth_rate=16):
        super(FeedbackModule, self).__init__()

        # Channel Masking Indices
        # We want to keep indices [0, 1, 3] and zero out [2, 4]
        self.register_buffer(
            "channel_mask",
            torch.tensor([1, 1, 0, 1, 0], dtype=torch.float32).view(1, 5, 1),
        )

        # Spatial Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, growth_rate, kernel_size=3, padding=1),
            nn.LayerNorm(growth_rate),
            nn.SiLU(),
        )

        # Lightweight Backbone
        # Using same dilations but smaller growth rate
        self.backbone = DenseBackbone(
            in_channels=growth_rate,
            growth_rate=growth_rate,
            dilations=Config.DILATIONS,
            latent_dim=out_channels,
        )

    def forward(self, prev_pred):
        # prev_pred: (Batch, Seq, 5) -> needs transpose for Conv1d
        x = prev_pred.permute(0, 2, 1)  # (B, 5, L)

        # Apply Channel Mask
        x = x * self.channel_mask

        # Stem
        out = self.stem[0](x)
        out = out.permute(0, 2, 1)
        out = self.stem[1](out)
        out = self.stem[2](out)
        out = out.permute(0, 2, 1)  # (B, Stem_Dim, L)

        # Backbone
        e_fb = self.backbone(out)  # (B, Out_Dim, L)

        return e_fb


class InteractionModule(nn.Module):
    """
    Interaction & Aggregation Module.
    Gathers partner features and fuses with self features.
    """

    def __init__(self, latent_dim, feedback_dim, rnn_hidden=64):
        super(InteractionModule, self).__init__()

        self.input_dim = latent_dim + feedback_dim

        # RNN Aggregator
        # Input to RNN is (Self + Partner) = 2 * input_dim
        self.rnn = nn.GRU(
            input_size=self.input_dim * 2,
            hidden_size=rnn_hidden,
            batch_first=True,
            bidirectional=True,
        )

        # Output Head
        self.head = nn.Linear(rnn_hidden * 2, 5)

    def forward(self, z, e_fb, partner_indices):
        # z: (B, Latent, L)
        # e_fb: (B, FB, L)
        # partner_indices: (B, L)

        # 1. Construct Self Vector
        h_self = torch.cat([z, e_fb], dim=1)  # (B, Dim, L)

        # 2. Gather Partner Vector
        # partner_indices has -1 for unpaired. We need to handle this.
        # Replace -1 with 0 for gathering, then mask result.
        B, C, L = h_self.shape

        # Create batch indices
        batch_idx = torch.arange(B, device=z.device).view(B, 1).expand(B, L)

        # Handle -1 indices
        safe_indices = partner_indices.clone()
        mask_unpaired = safe_indices == -1
        safe_indices[mask_unpaired] = 0

        # Gather: h_self[b, :, partner_idx[b, l]]
        # We transpose to (B, L, C) for easier gathering if using gather on dim 1,
        # but here we are gathering along L dimension.

        # Expand indices for gather: (B, C, L)
        # It's easier to permute to (B, L, C) and gather
        h_self_t = h_self.permute(0, 2, 1)  # (B, L, C)

        # Gather
        # index shape needs to be (B, L, C)
        gather_idx = safe_indices.unsqueeze(-1).expand(B, L, C)
        h_partner_t = torch.gather(h_self_t, 1, gather_idx)

        # Apply mask (Zero out unpaired)
        h_partner_t[mask_unpaired] = 0.0

        # 3. Fusion
        h_combined = torch.cat([h_self_t, h_partner_t], dim=2)  # (B, L, 2*C)

        # 4. Global Aggregation (RNN)
        rnn_out, _ = self.rnn(h_combined)  # (B, L, 2*Hidden)

        # 5. Projection
        out = self.head(rnn_out)  # (B, L, 5)

        return out


class HC_HIGFN(nn.Module):
    """
    High-Capacity Hybrid-Input Global-Feedback Network.
    """

    def __init__(self):
        super(HC_HIGFN, self).__init__()

        # 1. Hybrid Input Stem
        # Input channels: 4(Seq) + 3(Struct) + 7(Loop) + 4(Partner) = 18
        self.input_stem = HybridInputStem(in_channels=18, context_channels=32)

        # 2. Main Backbone (High Capacity)
        # Input to backbone is output of stem
        self.main_backbone = DenseBackbone(
            in_channels=self.input_stem.out_channels,
            growth_rate=Config.MAIN_GROWTH_RATE,
            dilations=Config.DILATIONS,
            latent_dim=Config.LATENT_DIM,
        )

        # 3. Feedback Module
        self.feedback_module = FeedbackModule(
            in_channels=5,
            out_channels=Config.FEEDBACK_DIM,
            growth_rate=Config.FB_GROWTH_RATE,
        )

        # 4. Interaction & Head
        self.interaction = InteractionModule(
            latent_dim=Config.LATENT_DIM,
            feedback_dim=Config.FEEDBACK_DIM,
            rnn_hidden=Config.RNN_HIDDEN_DIM,
        )

    def forward(self, x, partner_indices, targets=None):
        """
        Args:
            x: (Batch, Seq, Channels) - Input features
            partner_indices: (Batch, Seq) - Partner indices
            targets: Unused in forward, but kept for signature compatibility
        Returns:
            If training: (y_pred_final, y_pred_aux)
            If eval: y_pred_final
        """
        # Permute x for Conv1d: (B, C, L)
        x = x.permute(0, 2, 1)

        # Step 1: Static Backbone Features
        stem_out = self.input_stem(x)
        z = self.main_backbone(stem_out)  # (B, Latent, L)

        # Step 2: Iterative Refinement Loop

        # --- Pass 1 ---
        # Initialize prev_pred with zeros
        B, _, L = z.shape
        prev_pred = torch.zeros((B, L, 5), device=z.device)

        # Feedback
        e_fb_0 = self.feedback_module(prev_pred)  # (B, FB, L)

        # Interaction & Prediction
        y_1 = self.interaction(z, e_fb_0, partner_indices)

        # --- Pass 2 ---
        # Detach gradients from Pass 1 predictions
        prev_pred_2 = y_1.detach()

        # Feedback
        e_fb_1 = self.feedback_module(prev_pred_2)

        # Interaction & Prediction
        y_2 = self.interaction(z, e_fb_1, partner_indices)

        if self.training:
            return y_2, y_1
        else:
            return y_2
