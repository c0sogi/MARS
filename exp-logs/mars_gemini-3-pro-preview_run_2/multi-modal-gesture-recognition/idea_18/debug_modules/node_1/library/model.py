import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DualScaleTCNBlock(nn.Module):
    """
    Dual-Scale TCN Block containing parallel Local and Global branches.
    - Local Branch: Dilation = 1
    - Global Branch: Dilation = 2^k
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(DualScaleTCNBlock, self).__init__()

        self.padding_local = (kernel_size - 1) // 2
        self.padding_global = (kernel_size - 1) * dilation // 2

        # Local Branch (High frequency / Boundaries)
        self.local_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding_local,
            dilation=1,
        )

        # Global Branch (Low frequency / Context)
        self.global_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding_global,
            dilation=dilation,
        )

        # Fusion Layer
        # Concatenates outputs of both branches (2 * out_channels) and projects back
        self.fusion_conv = nn.Conv1d(out_channels * 2, out_channels, 1)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Residual connection handling
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        # x shape: (Batch, Channels, Time)

        # Branch 1: Local
        local_out = self.local_conv(x)
        local_out = self.relu(local_out)

        # Branch 2: Global
        global_out = self.global_conv(x)
        global_out = self.relu(global_out)

        # Fusion
        fused = torch.cat([local_out, global_out], dim=1)
        out = self.fusion_conv(fused)
        out = self.dropout(out)

        # Residual
        res = x if self.downsample is None else self.downsample(x)

        return self.relu(out + res)


class Stage1_BiLSTM(nn.Module):
    """
    Stage 1: Latent-Transition Recurrent Encoder.
    Processes raw features and outputs Class Logits + Latent Transition Logits.
    """

    def __init__(self):
        super(Stage1_BiLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=Config.INPUT_DIM,
            hidden_size=Config.LSTM_HIDDEN_SIZE,
            num_layers=Config.LSTM_NUM_LAYERS,
            batch_first=True,
            bidirectional=Config.LSTM_BIDIRECTIONAL,
        )

        # Bidirectional doubles the hidden size
        lstm_out_dim = (
            Config.LSTM_HIDDEN_SIZE * 2
            if Config.LSTM_BIDIRECTIONAL
            else Config.LSTM_HIDDEN_SIZE
        )

        # Head 1: Class Probabilities (Logits)
        self.cls_head = nn.Linear(lstm_out_dim, Config.NUM_CLASSES)

        # Head 2: Latent Transition Signal (Logits)
        self.trans_head = nn.Linear(lstm_out_dim, Config.TRANSITION_CHANNELS)

    def forward(self, x):
        # x shape: (Batch, Time, Features)

        lstm_out, _ = self.lstm(x)

        # Project to heads
        cls_logits = self.cls_head(lstm_out)  # (B, T, NumClasses)
        trans_logits = self.trans_head(lstm_out)  # (B, T, 1)

        # Concatenate: [Class Logits, Transition Logits]
        out = torch.cat([cls_logits, trans_logits], dim=2)

        return out


class RefinementStage(nn.Module):
    """
    Generic Refinement Stage using Dual-Scale TCN Blocks.
    Used for Stage 2 (Refinement) and Stage 3 (Sharpening).
    """

    def __init__(self, in_channels, out_channels):
        super(RefinementStage, self).__init__()

        # Project input (prob/logit space) to high-dim feature space
        self.input_proj = nn.Conv1d(in_channels, Config.TCN_CHANNELS, 1)

        layers = []
        num_layers = Config.TCN_NUM_LAYERS
        kernel_size = Config.TCN_KERNEL_SIZE
        dropout = Config.TCN_DROPOUT

        for i in range(num_layers):
            dilation = 2**i
            layers.append(
                DualScaleTCNBlock(
                    in_channels=Config.TCN_CHANNELS,
                    out_channels=Config.TCN_CHANNELS,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

        self.tcn_stack = nn.Sequential(*layers)

        # Project back to output space
        self.output_proj = nn.Conv1d(Config.TCN_CHANNELS, out_channels, 1)

    def forward(self, x):
        # x shape: (Batch, Time, Channels) -> Need (Batch, Channels, Time) for Conv1d
        x = x.transpose(1, 2)

        x = self.input_proj(x)
        x = self.tcn_stack(x)
        x = self.output_proj(x)

        # Transpose back: (Batch, Channels, Time) -> (Batch, Time, Channels)
        x = x.transpose(1, 2)
        return x


class DSL_CRCN(nn.Module):
    """
    Dual-Scale Latent-Cascaded Recurrent-Convolutional Network.

    Architecture:
    1. Stage 1: Bi-LSTM -> [Class Logits, Transition Logit]
    2. Stage 2: Dual-Scale TCN Refinement -> [Refined Class, Refined Transition]
    3. Stage 3: Dual-Scale TCN Sharpening -> [Final Class Logits]
    """

    def __init__(self):
        super(DSL_CRCN, self).__init__()

        # Stage 1
        self.stage1 = Stage1_BiLSTM()

        # Intermediate Dimension: NumClasses + TransitionChannel
        inter_dim = Config.NUM_CLASSES + Config.TRANSITION_CHANNELS

        # Stage 2: Refines both class and transition
        self.stage2 = RefinementStage(in_channels=inter_dim, out_channels=inter_dim)

        # Stage 3: Outputs final class logits only
        self.stage3 = RefinementStage(
            in_channels=inter_dim, out_channels=Config.NUM_CLASSES
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: (Batch, Time, Features)
            mask: (Batch, Time) - Boolean mask (True for valid frames)

        Returns:
            Tuple(stage1_out, stage2_out, stage3_out)
        """
        # --- Stage 1 ---
        s1_out = self.stage1(x)

        # Apply Inter-Stage Masking
        if mask is not None:
            # mask shape: (B, T) -> (B, T, 1)
            mask_expanded = mask.unsqueeze(-1).float()
            s1_masked = s1_out * mask_expanded
        else:
            s1_masked = s1_out

        # --- Stage 2 ---
        s2_out = self.stage2(s1_masked)

        # Apply Inter-Stage Masking
        if mask is not None:
            s2_masked = s2_out * mask_expanded
        else:
            s2_masked = s2_out

        # --- Stage 3 ---
        s3_out = self.stage3(s2_masked)

        # Apply Masking to final output (optional but good for consistency)
        if mask is not None:
            s3_out = s3_out * mask_expanded

        return s1_out, s2_out, s3_out
