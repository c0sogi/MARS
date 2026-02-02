import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class MultiTaskKinematicEncoder(nn.Module):
    """
    Stage 1: Bi-Directional GRU Encoder with Multi-Task Heads.
    Produces Classification Logits and Auxiliary Boundary Logits.
    """

    def __init__(self):
        super(MultiTaskKinematicEncoder, self).__init__()

        self.gru = nn.GRU(
            input_size=Config.TOTAL_INPUT_DIM,
            hidden_size=Config.HIDDEN_DIM,
            num_layers=Config.GRU_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.DROPOUT if Config.GRU_LAYERS > 1 else 0.0,
        )

        # Input to Linear is Hidden_Dim * 2 (Bidirectional)
        self.fc_cls = nn.Linear(Config.HIDDEN_DIM * 2, Config.NUM_CLASSES)
        self.fc_bnd = nn.Linear(Config.HIDDEN_DIM * 2, 1)
        self.dropout = nn.Dropout(Config.DROPOUT)

    def forward(self, x):
        # x: (Batch, Frames, Input_Dim)

        # Backbone
        out, _ = self.gru(x)
        out = self.dropout(out)

        # Dual Heads
        logits_cls = self.fc_cls(out)  # (Batch, Frames, Num_Classes)
        logits_bnd = self.fc_bnd(out)  # (Batch, Frames, 1)

        return logits_cls, logits_bnd


class GatedDilatedBlock(nn.Module):
    """
    Single block for the Refinement Unit.
    Uses Dilated Convolutions with Gated Activation (Tanh * Sigmoid).
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(GatedDilatedBlock, self).__init__()

        # Calculate padding to maintain temporal dimension
        # Padding = (Kernel - 1) * Dilation / 2 (assuming stride 1)
        padding = (kernel_size - 1) * dilation // 2

        self.conv_filter = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )

        self.conv_gate = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )

        self.conv_out = nn.Conv1d(out_channels, in_channels, 1)
        self.dropout = nn.Dropout(Config.DROPOUT)

    def forward(self, x):
        # x: (Batch, Channels, Frames)

        f = self.conv_filter(x)
        g = self.conv_gate(x)

        # Gating Mechanism
        out = torch.tanh(f) * torch.sigmoid(g)
        out = self.dropout(out)

        # 1x1 Projection
        out = self.conv_out(out)

        # Residual Connection
        return x + out


class GatedRefinementUnit(nn.Module):
    """
    Refinement Stage: Stack of Gated Dilated Blocks.
    Input: Class Probabilities from previous stage.
    """

    def __init__(self):
        super(GatedRefinementUnit, self).__init__()

        # Input Projection: Num_Classes -> TCN_Channels
        self.conv_in = nn.Conv1d(Config.NUM_CLASSES, Config.TCN_NUM_CHANNELS, 1)

        # Stack of Gated Blocks
        self.layers = nn.ModuleList()
        for dilation in Config.TCN_DILATIONS:
            self.layers.append(
                GatedDilatedBlock(
                    Config.TCN_NUM_CHANNELS,
                    Config.TCN_NUM_CHANNELS,
                    Config.TCN_KERNEL_SIZE,
                    dilation,
                )
            )

        # Output Projection: TCN_Channels -> Num_Classes
        self.conv_out = nn.Conv1d(Config.TCN_NUM_CHANNELS, Config.NUM_CLASSES, 1)
        self.dropout = nn.Dropout(Config.DROPOUT)

    def forward(self, x):
        # x: (Batch, Num_Classes, Frames)

        out = self.conv_in(x)
        out = self.dropout(out)

        for layer in self.layers:
            out = layer(out)

        out = self.conv_out(out)

        return out


class ASK_RN(nn.Module):
    """
    Auxiliary-Supervised Kinematic Refinement Network (ASK-RN).
    Stage 1: Multi-Task Encoder (Cls + Boundary)
    Stage 2: Gated Refinement (Input: S1 Probs)
    Stage 3: Gated Refinement (Input: S2 Probs)
    """

    def __init__(self):
        super(ASK_RN, self).__init__()

        self.encoder = MultiTaskKinematicEncoder()
        self.stage2 = GatedRefinementUnit()
        self.stage3 = GatedRefinementUnit()

    def forward(self, x):
        # x: (Batch, Frames, Input_Dim)

        # ----------------------------------------
        # Stage 1: Encoder
        # ----------------------------------------
        logits_s1, logits_bnd = self.encoder(x)
        # logits_s1: (B, T, Num_Classes)
        # logits_bnd: (B, T, 1)

        # Prepare for Stage 2
        # Use Softmax to get probabilities (P_cls)
        probs_s1 = F.softmax(logits_s1, dim=2)

        # Permute for TCN: (B, T, C) -> (B, C, T)
        probs_s1_t = probs_s1.permute(0, 2, 1)

        # ----------------------------------------
        # Stage 2: Refinement
        # ----------------------------------------
        # Input: Strictly probabilities from Stage 1
        logits_s2_t = self.stage2(probs_s1_t)  # (B, C, T)
        logits_s2 = logits_s2_t.permute(0, 2, 1)  # (B, T, C)

        # Prepare for Stage 3
        probs_s2 = F.softmax(logits_s2, dim=2)
        probs_s2_t = probs_s2.permute(0, 2, 1)

        # ----------------------------------------
        # Stage 3: Refinement
        # ----------------------------------------
        logits_s3_t = self.stage3(probs_s2_t)  # (B, C, T)
        logits_s3 = logits_s3_t.permute(0, 2, 1)  # (B, T, C)

        # Return all outputs for Deep Supervision Loss
        return {
            "logits_s1": logits_s1,
            "logits_bnd": logits_bnd,
            "logits_s2": logits_s2,
            "logits_s3": logits_s3,
        }
