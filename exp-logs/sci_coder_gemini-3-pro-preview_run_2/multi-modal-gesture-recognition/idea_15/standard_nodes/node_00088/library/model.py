import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DilatedResidualBlock(nn.Module):
    """
    Standard Dilated Residual Block for TCN.
    Conv1d(dilated) -> ReLU -> Dropout -> Conv1d(1x1) -> Residual
    Replaces GatedTCNBlock to improve stability (Cite Lesson 00082).
    """

    def __init__(self, channels, kernel_size, dilation, dropout):
        super(DilatedResidualBlock, self).__init__()
        self.padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )

        self.conv2 = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.dropout(out)
        return x + out


class RefinementStage(nn.Module):
    """
    A single stage of the Gated MS-TCN.
    Consists of an input projection, a stack of GatedTCNBlocks, and output projections.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers,
        kernel_size,
        dropout,
        num_classes,
    ):
        super(RefinementStage, self).__init__()

        # Input projection (1x1 Conv)
        self.conv_in = nn.Conv1d(in_channels, hidden_channels, 1)

        # Stack of Dilated Residual Blocks with increasing dilation
        self.layers = nn.ModuleList(
            [
                DilatedResidualBlock(
                    channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    dropout=dropout,
                )
                for i in range(num_layers)
            ]
        )

        # Output projections (1x1 Conv)
        # Class probabilities head
        self.conv_cls = nn.Conv1d(hidden_channels, num_classes, 1)
        # Boundary probability head
        self.conv_bnd = nn.Conv1d(hidden_channels, 1, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, In_Channels, Time)

        out = self.conv_in(x)
        out = self.dropout(out)

        for layer in self.layers:
            out = layer(out)

        logits_cls = self.conv_cls(out)
        logits_bnd = self.conv_bnd(out)

        return logits_cls, logits_bnd


class MultiTaskRecurrentEncoder(nn.Module):
    """
    Stage 1: Bi-Directional LSTM Encoder.
    Fuses multi-modal features and provides initial predictions.
    """

    def __init__(self):
        super(MultiTaskRecurrentEncoder, self).__init__()

        self.lstm = nn.LSTM(
            input_size=Config.INPUT_DIM,
            hidden_size=Config.LSTM_HIDDEN_DIM,
            num_layers=Config.LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.LSTM_DROPOUT if Config.LSTM_LAYERS > 1 else 0.0,
        )

        self.dropout = nn.Dropout(Config.LSTM_DROPOUT)

        # Projections
        # Input to linear is hidden_dim * 2 (bidirectional)
        self.fc_cls = nn.Linear(Config.LSTM_HIDDEN_DIM * 2, Config.NUM_CLASSES)
        self.fc_bnd = nn.Linear(Config.LSTM_HIDDEN_DIM * 2, 1)

    def forward(self, x):
        # x: (Batch, Time, InputDim)

        # LSTM output: (Batch, Time, Hidden*2)
        out, _ = self.lstm(x)
        out = self.dropout(out)

        logits_cls = self.fc_cls(out)
        logits_bnd = self.fc_bnd(out)

        return logits_cls, logits_bnd


class SGCRCN(nn.Module):
    """
    Supervised Gated-Cascaded Recurrent-Convolutional Network.

    Architecture:
    1. Stage 1: Bi-LSTM Encoder (Features -> Initial Probs)
    2. Stage 2: Gated MS-TCN Refinement (Initial Probs -> Refined Probs 1)
    3. Stage 3: Gated MS-TCN Refinement (Refined Probs 1 -> Final Probs)

    Uses Deep Supervision at all stages.
    """

    def __init__(self):
        super(SGCRCN, self).__init__()

        self.encoder = MultiTaskRecurrentEncoder()

        # Input to refinement stages: Class Probs (NUM_CLASSES) + Boundary Prob (1)
        refine_in_dim = Config.NUM_CLASSES + 1

        self.stage2 = RefinementStage(
            in_channels=refine_in_dim,
            hidden_channels=Config.MSTCN_CHANNELS,
            num_layers=Config.MSTCN_LAYERS,
            kernel_size=Config.MSTCN_KERNEL_SIZE,
            dropout=Config.MSTCN_DROPOUT,
            num_classes=Config.NUM_CLASSES,
        )

        self.stage3 = RefinementStage(
            in_channels=refine_in_dim,
            hidden_channels=Config.MSTCN_CHANNELS,
            num_layers=Config.MSTCN_LAYERS,
            kernel_size=Config.MSTCN_KERNEL_SIZE,
            dropout=Config.MSTCN_DROPOUT,
            num_classes=Config.NUM_CLASSES,
        )

    def forward(self, x, mask):
        """
        Args:
            x: (Batch, Time, InputDim)
            mask: (Batch, Time) - 1.0 for valid frames, 0.0 for padding
        Returns:
            Dictionary containing logits for all stages:
            - stageX_cls: (Batch, Time, NumClasses)
            - stageX_bnd: (Batch, Time, 1)
        """
        # --- Stage 1: Encoder ---
        s1_logits_cls, s1_logits_bnd = self.encoder(x)
        # s1_logits_cls: (B, T, C)
        # s1_logits_bnd: (B, T, 1)

        # Prepare input for Stage 2
        # Apply Softmax/Sigmoid to convert logits to probabilities
        s1_probs_cls = F.softmax(s1_logits_cls, dim=2)
        s1_probs_bnd = torch.sigmoid(s1_logits_bnd)

        # Concatenate: (B, T, C+1)
        s1_in = torch.cat([s1_probs_cls, s1_probs_bnd], dim=2)

        # Apply Mask to zero out padding noise (Inter-Stage Masking)
        s1_in = s1_in * mask.unsqueeze(2)

        # Transpose for TCN: (B, C+1, T)
        s1_in = s1_in.permute(0, 2, 1)

        # --- Stage 2: Refinement ---
        s2_logits_cls_t, s2_logits_bnd_t = self.stage2(s1_in)
        # Outputs are (B, C, T) and (B, 1, T)

        # Prepare input for Stage 3
        s2_probs_cls = F.softmax(s2_logits_cls_t, dim=1)
        s2_probs_bnd = torch.sigmoid(s2_logits_bnd_t)

        # Concatenate: (B, C+1, T)
        s2_in = torch.cat([s2_probs_cls, s2_probs_bnd], dim=1)

        # Apply Mask
        s2_in = s2_in * mask.unsqueeze(1)

        # --- Stage 3: Refinement ---
        s3_logits_cls_t, s3_logits_bnd_t = self.stage3(s2_in)

        # --- Formatting Outputs ---
        # Convert TCN outputs back to (B, T, C) for consistency with loss functions

        return {
            "stage1_cls": s1_logits_cls,
            "stage1_bnd": s1_logits_bnd,
            "stage2_cls": s2_logits_cls_t.permute(0, 2, 1),
            "stage2_bnd": s2_logits_bnd_t.permute(0, 2, 1),
            "stage3_cls": s3_logits_cls_t.permute(0, 2, 1),
            "stage3_bnd": s3_logits_bnd_t.permute(0, 2, 1),
        }
