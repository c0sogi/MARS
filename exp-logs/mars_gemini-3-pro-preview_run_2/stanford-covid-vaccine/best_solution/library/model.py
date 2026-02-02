import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DilatedResidualBlock(nn.Module):
    """
    A single residual block with dilated convolution.
    Structure: Conv1d(dilated) -> ReLU -> Dropout -> Conv1d(1x1) -> ReLU
    Note: The residual connection is added in the forward pass if input/output dims match,
    or handled by the dense connection structure in the backbone.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv_dilated = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm1 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

        self.conv_pointwise = nn.Conv1d(out_channels, out_channels, 1)
        self.norm2 = nn.LayerNorm(out_channels)

        self.act = nn.SiLU()

    def forward(self, x):
        # x: (Batch, Channels, Seq)
        residual = x if x.shape[1] == self.conv_pointwise.out_channels else None

        out = self.conv_dilated(x)
        # Permute for LayerNorm: (B, C, L) -> (B, L, C)
        out = out.permute(0, 2, 1)
        out = self.norm1(out)
        out = self.act(out)
        out = out.permute(0, 2, 1)

        out = self.dropout(out)

        out = self.conv_pointwise(out)
        out = out.permute(0, 2, 1)
        out = self.norm2(out)
        out = self.act(out)
        out = out.permute(0, 2, 1)

        if residual is not None:
            out = out + residual

        return out


class DenseDilatedBackbone(nn.Module):
    """
    Stack of dilated residual blocks with dense connections.
    Processes static inputs once.
    """

    def __init__(self):
        super().__init__()
        self.input_dim = Config.INPUT_CHANNELS
        self.growth_rate = Config.GROWTH_RATE
        self.dilations = Config.DILATIONS
        self.dropout = Config.DROPOUT
        self.kernel_size = Config.KERNEL_SIZE

        self.blocks = nn.ModuleList()

        current_dim = self.input_dim

        for d in self.dilations:
            # Input to block is concatenation of all previous outputs
            block = DilatedResidualBlock(
                in_channels=current_dim,
                out_channels=self.growth_rate,
                kernel_size=self.kernel_size,
                dilation=d,
                dropout=self.dropout,
            )
            self.blocks.append(block)
            current_dim += self.growth_rate

        # Final projection to latent dimension Z
        self.final_proj = nn.Conv1d(current_dim, Config.EMBED_DIM, 1)

    def forward(self, x):
        # x: (Batch, Seq, Channels) -> Permute to (Batch, Channels, Seq)
        x = x.permute(0, 2, 1)

        features = [x]

        for block in self.blocks:
            # Concatenate all previous features
            dense_input = torch.cat(features, dim=1)
            out = block(dense_input)
            features.append(out)

        # Concatenate everything for final projection
        total_features = torch.cat(features, dim=1)
        z = self.final_proj(total_features)

        # Return to (Batch, Seq, Channels)
        return z.permute(0, 2, 1)


class FeedbackTCN(nn.Module):
    """
    Lightweight TCN to process recycled predictions.
    Applies masking to unscored targets before processing.
    """

    def __init__(self):
        super().__init__()
        self.in_channels = Config.NUM_TARGETS
        self.dim = Config.FEEDBACK_DIM

        # Mask for scored targets (reactivity, deg_Mg_pH10, deg_Mg_50C)
        # Indices: 0, 1, 3
        self.register_buffer(
            "target_mask", torch.tensor([1, 1, 0, 1, 0], dtype=torch.float32)
        )

        self.proj = nn.Conv1d(self.in_channels, self.dim, 1)

        self.block1 = DilatedResidualBlock(self.dim, self.dim, 3, 1, 0.1)
        self.block2 = DilatedResidualBlock(self.dim, self.dim, 3, 2, 0.1)

    def forward(self, prev_preds):
        # prev_preds: (Batch, Seq, 5)

        # Apply mask: Zero out unscored columns to avoid noise injection
        masked_preds = prev_preds * self.target_mask.view(1, 1, -1)

        # Permute to (Batch, 5, Seq)
        x = masked_preds.permute(0, 2, 1)

        x = self.proj(x)
        x = self.block1(x)
        x = self.block2(x)

        # Return (Batch, Seq, Dim)
        return x.permute(0, 2, 1)


class CF_DCN(nn.Module):
    """
    Contextualized-Feedback Dense Network.
    Wraps Backbone, FeedbackTCN, and Head for iterative refinement.
    """

    def __init__(self):
        super().__init__()
        self.backbone = DenseDilatedBackbone()
        self.feedback_tcn = FeedbackTCN()

        # Fusion Dimension: (Z_dim + Feedback_dim) * 2 (Self + Partner)
        fusion_dim = (Config.EMBED_DIM + Config.FEEDBACK_DIM) * 2

        self.rnn = nn.GRU(
            input_size=fusion_dim,
            hidden_size=Config.RNN_HIDDEN_DIM,
            num_layers=Config.RNN_LAYERS,
            batch_first=True,
            bidirectional=True,
        )

        # Output projection: BiGRU (hidden*2) -> 5 targets
        # Config.RNN_HIDDEN_DIM is EMBED_DIM // 2, so bidirectional output is EMBED_DIM
        self.head = nn.Linear(Config.RNN_HIDDEN_DIM * 2, Config.NUM_TARGETS)

    def forward_backbone(self, x):
        """
        Computes static latent representation Z.
        """
        return self.backbone(x)

    def forward_head(self, z, prev_preds, partner_indices):
        """
        Computes feedback, fuses features, and runs RNN head.

        Args:
            z: Static latent features (Batch, Seq, 64)
            prev_preds: Recycled predictions (Batch, Seq, 5)
            partner_indices: Indices of paired bases (Batch, Seq)
        """
        batch_size, seq_len, _ = z.shape

        # 1. Compute Contextualized Feedback
        e_ctx = self.feedback_tcn(prev_preds)  # (Batch, Seq, 32)

        # 2. Construct Self Vector
        self_vec = torch.cat([z, e_ctx], dim=-1)  # (Batch, Seq, 96)

        # 3. Construct Partner Vector
        # Handle -1 indices (unpaired) by clamping to 0 and masking later
        valid_mask = (partner_indices != -1).unsqueeze(-1)  # (Batch, Seq, 1)
        safe_indices = partner_indices.clone()
        safe_indices[partner_indices == -1] = 0

        # Gather partner features
        # Expand indices for gather: (Batch, Seq, 96)
        gather_indices = safe_indices.unsqueeze(-1).expand(-1, -1, self_vec.shape[-1])
        partner_vec = torch.gather(self_vec, 1, gather_indices)

        # Apply mask to zero out features for unpaired bases
        partner_vec = partner_vec * valid_mask.float()

        # 4. Fusion
        fused = torch.cat([self_vec, partner_vec], dim=-1)  # (Batch, Seq, 192)

        # 5. RNN Aggregation
        rnn_out, _ = self.rnn(fused)

        # 6. Projection
        preds = self.head(rnn_out)

        return preds

    def forward(self, x, partner_indices):
        """
        Full inference forward pass (2 iterations).
        Used for validation/inference.
        """
        # Step 1: Static Backbone
        z = self.forward_backbone(x)

        # Step 2: Pass 1 (Zero Feedback)
        batch_size, seq_len, _ = z.shape
        initial_preds = torch.zeros(
            batch_size, seq_len, Config.NUM_TARGETS, device=x.device, dtype=x.dtype
        )

        y_1 = self.forward_head(z, initial_preds, partner_indices)

        # Step 3: Pass 2 (Feedback from y_1)
        # We detach y_1 usually in training, but in inference it doesn't matter.
        # However, for consistency with the logic:
        y_2 = self.forward_head(z, y_1, partner_indices)

        return y_2
