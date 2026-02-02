import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    SKELETON_JOINTS,
    N_MFCC,
    NUM_CLASSES,
    HIDDEN_DIM,
    DROPOUT_AUDIO,
    DROPOUT_SKEL,
    DILATION_SCHEDULE,
    KERNEL_SIZE,
    TCN_DROPOUT,
)


class ModalityDropout(nn.Module):
    """
    Stochastically zeros out specific modalities (Audio or Skeleton) during training
    to force the network to learn robust features from both.
    """

    def __init__(self, p_audio=DROPOUT_AUDIO, p_skel=DROPOUT_SKEL):
        super(ModalityDropout, self).__init__()
        self.p_audio = p_audio
        self.p_skel = p_skel

        # Calculate indices based on config
        # Skeleton: 20 joints * 3 coords * 3 derivatives (pos, vel, acc) = 180 features
        self.skel_dim = SKELETON_JOINTS * 3 * 3
        # Audio: N_MFCC = 13
        self.audio_dim = N_MFCC
        self.total_dim = self.skel_dim + self.audio_dim

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Time, Features)
        Returns:
            torch.Tensor: Masked input tensor
        """
        if not self.training:
            return x

        batch_size, time_steps, num_features = x.shape

        # Ensure input dimension matches expectation
        # We allow num_features to be larger if needed, but we target specific ranges

        # Create a mask of ones
        mask = torch.ones_like(x)

        # Apply Audio Dropout (Indices: skel_dim to end)
        if torch.rand(1).item() < self.p_audio:
            mask[:, :, self.skel_dim :] = 0.0

        # Apply Skeleton Dropout (Indices: 0 to skel_dim)
        if torch.rand(1).item() < self.p_skel:
            mask[:, :, : self.skel_dim] = 0.0

        return x * mask


class GatedDilatedBlock(nn.Module):
    """
    A single block for the Temporal Refinement Stage.
    Structure: Residual + (DilatedConv -> GatedActivation -> Dropout -> 1x1Conv)
    """

    def __init__(
        self, in_channels, dilation, kernel_size=KERNEL_SIZE, dropout=TCN_DROPOUT
    ):
        super(GatedDilatedBlock, self).__init__()

        # Padding for centered (non-causal) convolution
        # padding = dilation * (kernel_size - 1) / 2
        # For k=3, padding = dilation
        padding = dilation

        # Dilated Convolution: Expands channels by 2 for Gating
        self.conv_dilated = nn.Conv1d(
            in_channels,
            in_channels * 2,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.dropout = nn.Dropout(dropout)

        # 1x1 Convolution for projection back to original channels
        self.conv_1x1 = nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input (Batch, Channels, Time)
        """
        residual = x

        # Dilated Conv
        out = self.conv_dilated(x)

        # Split for Gated Activation
        filter_out, gate_out = out.chunk(2, dim=1)

        # Gated Activation: tanh(f) * sigmoid(g)
        out = torch.tanh(filter_out) * torch.sigmoid(gate_out)

        # Dropout
        out = self.dropout(out)

        # 1x1 Conv
        out = self.conv_1x1(out)

        # Residual connection
        return out + residual


class TemporalRefinementStage(nn.Module):
    """
    Refines class probabilities using a stack of dilated convolutions.
    Implements Monotonic Non-Causal Refinement.
    """

    def __init__(self, num_classes=NUM_CLASSES, dilation_schedule=DILATION_SCHEDULE):
        super(TemporalRefinementStage, self).__init__()

        layers = []
        for dilation in dilation_schedule:
            layers.append(GatedDilatedBlock(num_classes, dilation))

        self.network = nn.Sequential(*layers)

    def forward(self, probabilities):
        """
        Args:
            probabilities (torch.Tensor): Input probabilities (Batch, Time, NumClasses)
        Returns:
            torch.Tensor: Refined Logits (Batch, Time, NumClasses)
        """
        # Transpose for Conv1d: (B, T, C) -> (B, C, T)
        x = probabilities.permute(0, 2, 1)

        # Apply TCN stack
        out = self.network(x)

        # Transpose back: (B, C, T) -> (B, T, C)
        return out.permute(0, 2, 1)


class RMDKN(nn.Module):
    """
    Robust Modality-Dropout Kinematic Network.
    Three-Stage Cascaded Network:
    1. Stochastic Modality-Dropout Encoder (Bi-GRU)
    2. Monotonic Non-Causal Refinement (TCN)
    3. Independent Iterative Refinement (TCN)
    """

    def __init__(self):
        super(RMDKN, self).__init__()

        # ==========================
        # Stage 1: Encoder
        # ==========================
        self.modality_dropout = ModalityDropout()

        # Input dim = 180 (Skel) + 13 (Audio) = 193
        input_dim = SKELETON_JOINTS * 3 * 3 + N_MFCC

        self.bigru = nn.GRU(
            input_size=input_dim,
            hidden_size=HIDDEN_DIM // 2,  # Bidirectional, so split hidden dim
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )

        self.stage1_fc = nn.Linear(HIDDEN_DIM, NUM_CLASSES)

        # ==========================
        # Stage 2: Refinement
        # ==========================
        # Input: Strictly class probabilities from Stage 1
        self.stage2_tcn = TemporalRefinementStage()

        # ==========================
        # Stage 3: Refinement
        # ==========================
        # Input: Strictly class probabilities from Stage 2
        # Independent weights from Stage 2
        self.stage3_tcn = TemporalRefinementStage()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features (Batch, Time, Features)

        Returns:
            tuple: (logits_1, logits_2, logits_3)
        """
        # --- Stage 1 ---
        # 1. Modality Dropout
        x_aug = self.modality_dropout(x)

        # 2. Bi-GRU Encoder
        # self.bigru returns (output, h_n)
        # output shape: (Batch, Time, HiddenDim)
        gru_out, _ = self.bigru(x_aug)

        # 3. Projection
        logits_1 = self.stage1_fc(gru_out)

        # --- Stage 2 ---
        # Input: Probabilities from Stage 1
        probs_1 = F.softmax(logits_1, dim=-1)

        # Refinement
        logits_2 = self.stage2_tcn(probs_1)

        # --- Stage 3 ---
        # Input: Probabilities from Stage 2
        probs_2 = F.softmax(logits_2, dim=-1)

        # Refinement
        logits_3 = self.stage3_tcn(probs_2)

        return logits_1, logits_2, logits_3
