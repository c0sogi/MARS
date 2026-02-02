import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    INPUT_DIM,
    HIDDEN_DIM,
    NUM_CLASSES,
    NUM_STAGES,
    DROPOUT,
    DILATIONS,
)


class MaskedSEBlock(nn.Module):
    """
    Squeeze-and-Excitation block that respects sequence masking.
    Performs Global Average Pooling only on valid time steps.
    """

    def __init__(self, channels, reduction=16):
        super(MaskedSEBlock, self).__init__()
        reduced_channels = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, mask=None):
        # x: (B, C, T)
        # mask: (B, T)

        if mask is not None:
            # Expand mask to match channel dims for broadcasting: (B, 1, T)
            mask_expanded = mask.unsqueeze(1)

            # Zero out padding in input
            x_masked = x * mask_expanded

            # Sum over time dimension: (B, C)
            x_sum = x_masked.sum(dim=2)

            # Count valid frames per sequence: (B, 1)
            mask_sum = mask_expanded.sum(dim=2)
            # Avoid division by zero
            mask_sum = torch.clamp(mask_sum, min=1e-9)

            # Masked Global Average Pooling
            y = x_sum / mask_sum
        else:
            # Standard GAP if no mask provided
            y = x.mean(dim=2)

        # Excitation: MLP
        y = self.fc(y)  # (B, C)
        y = y.unsqueeze(2)  # (B, C, 1)

        # Scale
        return x * y


class GatedRefinementBlock(nn.Module):
    """
    Gated Activation Block with Masked SE and Residual connection.
    NO Normalization is used.
    """

    def __init__(self, channels, kernel_size, dilation, dropout=0.3):
        super(GatedRefinementBlock, self).__init__()
        # Calculate padding to maintain temporal dimension
        padding = (kernel_size - 1) * dilation // 2

        self.filter_conv = nn.Conv1d(
            channels, channels, kernel_size, dilation=dilation, padding=padding
        )
        self.gate_conv = nn.Conv1d(
            channels, channels, kernel_size, dilation=dilation, padding=padding
        )
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(channels, channels, 1)
        self.se = MaskedSEBlock(channels)

    def forward(self, x, mask=None):
        # x: (B, C, T)

        # Gated Activation Unit
        f = torch.tanh(self.filter_conv(x))
        g = torch.sigmoid(self.gate_conv(x))
        z = f * g

        z = self.dropout(z)

        # 1x1 Projection
        h = self.projection(z)

        # Masked Channel Attention
        h = self.se(h, mask)

        # Residual Connection
        return x + h


class RefinementStage(nn.Module):
    """
    A single refinement stage consisting of input projection,
    stack of dilated gated blocks, and output projection.
    """

    def __init__(self, in_dim, hidden_dim, dilations, dropout):
        super(RefinementStage, self).__init__()
        self.input_proj = nn.Conv1d(in_dim, hidden_dim, 1)
        self.blocks = nn.ModuleList(
            [GatedRefinementBlock(hidden_dim, 3, d, dropout) for d in dilations]
        )
        self.output_proj = nn.Conv1d(hidden_dim, in_dim, 1)

    def forward(self, x, mask):
        # x: (B, T, C) -> Need transpose for Conv1d
        x = x.permute(0, 2, 1)  # (B, C, T)

        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x, mask)

        x = self.output_proj(x)

        # Transpose back: (B, T, C)
        x = x.permute(0, 2, 1)
        return x


class MCAGCN(nn.Module):
    """
    Masked Channel-Attentive Gated-Cascaded Network.
    Stage 1: Bi-LSTM Encoder
    Stage 2+: Refinement Stages with Gated Blocks and Masked SE
    """

    def __init__(
        self,
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        num_stages=NUM_STAGES,
        dropout=DROPOUT,
        dilations=DILATIONS,
    ):
        super(MCAGCN, self).__init__()

        self.num_classes = num_classes

        # --- Stage 1: Multi-Task Recurrent Encoder ---
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        # Heads for Stage 1
        self.s1_cls = nn.Linear(2 * hidden_dim, num_classes)
        self.s1_bnd = nn.Linear(2 * hidden_dim, 1)

        # --- Stages 2 & 3: Refinement ---
        self.refinement_stages = nn.ModuleList()
        # Input to refinement is concatenation of Class Probs and Boundary Prob
        refine_in_dim = num_classes + 1

        for _ in range(num_stages - 1):
            self.refinement_stages.append(
                RefinementStage(refine_in_dim, hidden_dim, dilations, dropout)
            )

    def forward(self, x, mask=None):
        # x: (B, T, InputDim)
        # mask: (B, T) - Binary mask (1 for valid, 0 for padding)

        outputs = []

        # --- Stage 1 Forward ---
        lstm_out, _ = self.lstm(x)  # (B, T, 2*H)

        # Apply mask to LSTM output
        if mask is not None:
            lstm_out = lstm_out * mask.unsqueeze(2)

        s1_cls_logits = self.s1_cls(lstm_out)
        s1_bnd_logits = self.s1_bnd(lstm_out)

        s1_cls_probs = F.softmax(s1_cls_logits, dim=2)
        s1_bnd_probs = torch.sigmoid(s1_bnd_logits)

        outputs.append(
            {
                "cls_logits": s1_cls_logits,
                "bnd_logits": s1_bnd_logits,
                "cls_probs": s1_cls_probs,
                "bnd_probs": s1_bnd_probs,
            }
        )

        # Prepare input for next stage: Concatenate probabilities
        current_input = torch.cat([s1_cls_probs, s1_bnd_probs], dim=2)

        # --- Refinement Stages Forward ---
        for stage in self.refinement_stages:
            # Explicit Inter-Stage Masking
            if mask is not None:
                current_input = current_input * mask.unsqueeze(2)

            # Refine
            refined_out = stage(current_input, mask)  # (B, T, C+1)

            # Split heads
            cls_logits = refined_out[:, :, : self.num_classes]
            bnd_logits = refined_out[:, :, self.num_classes :]

            cls_probs = F.softmax(cls_logits, dim=2)
            bnd_probs = torch.sigmoid(bnd_logits)

            outputs.append(
                {
                    "cls_logits": cls_logits,
                    "bnd_logits": bnd_logits,
                    "cls_probs": cls_probs,
                    "bnd_probs": bnd_probs,
                }
            )

            # Update input for next iteration
            current_input = torch.cat([cls_probs, bnd_probs], dim=2)

        return outputs
