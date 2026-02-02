import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DecoupledGatedInput(nn.Module):
    """
    Splits input into two paths:
    1. Norm Path: LayerNorm -> Linear -> Sigmoid (Gate)
    2. Signal Path: Raw Input (Preserves physical magnitude)
    Fusion: Signal * Gate
    """

    def __init__(self, input_dim):
        super(DecoupledGatedInput, self).__init__()
        self.ln = nn.LayerNorm(input_dim)
        self.gate_fc = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (Batch, Time, InputDim)
        norm_x = self.ln(x)
        gate = self.sigmoid(self.gate_fc(norm_x))
        return x * gate


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for 1D sequences.
    Performs Global Average Pooling over time to capture global context.
    """

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        reduced_channels = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (Batch, Channels, Time)
        b, c, t = x.size()
        # Squeeze: Global Average Pooling
        y = self.avg_pool(x).view(b, c)
        # Excitation: FC layers
        y = self.fc(y).view(b, c, 1)
        # Scale
        return x * y


class GatedDilatedTCNBlock(nn.Module):
    """
    Residual Block with Gated Dilated Convolution and SE Attention.
    Structure:
    Input -> Dilated Conv -> Split -> Tanh * Sigmoid -> SEBlock -> 1x1 Conv -> Dropout -> + Input
    """

    def __init__(self, channels, kernel_size, dilation, dropout=0.2):
        super(GatedDilatedTCNBlock, self).__init__()
        self.padding = (kernel_size - 1) * dilation // 2

        # Dilated Convolution mapping to 2x channels for gating
        self.conv_dilated = nn.Conv1d(
            channels, channels * 2, kernel_size, padding=self.padding, dilation=dilation
        )

        # Global Context Attention
        self.se = SEBlock(channels)

        # Projection back for residual addition
        self.conv_1x1 = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Channels, Time)
        residual = x

        # Dilated Conv
        out = self.conv_dilated(x)

        # Gated Activation
        P, Q = out.chunk(2, dim=1)
        out = torch.tanh(P) * torch.sigmoid(Q)

        # Global Context
        out = self.se(out)

        # Projection
        out = self.conv_1x1(out)
        out = self.dropout(out)

        return residual + out


class RefinementStage(nn.Module):
    """
    High-Capacity Refinement Stage.
    Input: Class Probabilities from previous stage.
    Architecture: Adapter -> Stack of GatedDilatedTCNBlocks -> Classifier
    """

    def __init__(self, num_classes, hidden_dim, kernel_size, dilations, dropout):
        super(RefinementStage, self).__init__()

        # Adapter: Project probabilities to high-dimensional feature space
        self.adapter = nn.Conv1d(num_classes, hidden_dim, 1)

        # TCN Stack
        layers = []
        for dilation in dilations:
            layers.append(
                GatedDilatedTCNBlock(hidden_dim, kernel_size, dilation, dropout)
            )
        self.tcn = nn.Sequential(*layers)

        # Classifier: Project back to class logits
        self.classifier = nn.Conv1d(hidden_dim, num_classes, 1)

    def forward(self, probs):
        # probs: (Batch, Time, NumClasses)
        # Permute for Conv1d: (Batch, NumClasses, Time)
        x = probs.permute(0, 2, 1)

        x = self.adapter(x)
        x = self.tcn(x)
        logits = self.classifier(x)

        # Permute back: (Batch, Time, NumClasses)
        return logits.permute(0, 2, 1)


class SKAGN(nn.Module):
    """
    Structural-Kinematic Attentive Gated Network.
    Stage 1: Bi-GRU Encoder with Decoupled Input Gating.
    Stage 2: Attentive Refinement on Stage 1 Probabilities.
    Stage 3: Independent Attentive Refinement on Stage 2 Probabilities.
    """

    def __init__(self):
        super(SKAGN, self).__init__()

        # ==========================================
        # Stage 1: Encoder
        # ==========================================
        self.input_gate = DecoupledGatedInput(Config.INPUT_DIM)

        # Bi-GRU Backbone (192 total hidden units)
        self.gru = nn.GRU(
            input_size=Config.INPUT_DIM,
            hidden_size=Config.HIDDEN_DIM // 2,  # 96 per direction
            num_layers=Config.GRU_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.DROPOUT,
        )

        self.fc_p1 = nn.Linear(Config.HIDDEN_DIM, Config.NUM_CLASSES)

        # ==========================================
        # Stage 2: Refinement 1
        # ==========================================
        self.stage2 = RefinementStage(
            num_classes=Config.NUM_CLASSES,
            hidden_dim=Config.HIDDEN_DIM,  # 192 channels (High Capacity)
            kernel_size=Config.TCN_KERNEL_SIZE,
            dilations=Config.TCN_DILATIONS,
            dropout=Config.DROPOUT,
        )

        # ==========================================
        # Stage 3: Refinement 2
        # ==========================================
        self.stage3 = RefinementStage(
            num_classes=Config.NUM_CLASSES,
            hidden_dim=Config.HIDDEN_DIM,  # 192 channels
            kernel_size=Config.TCN_KERNEL_SIZE,
            dilations=Config.TCN_DILATIONS,
            dropout=Config.DROPOUT,
        )

    def forward(self, x):
        # x: (Batch, Time, InputDim)

        # --- Stage 1 ---
        x_gated = self.input_gate(x)
        gru_out, _ = self.gru(x_gated)
        logits_1 = self.fc_p1(gru_out)
        probs_1 = F.softmax(logits_1, dim=2)

        # --- Stage 2 ---
        # Input: Strictly probabilities from Stage 1
        logits_2 = self.stage2(probs_1)
        probs_2 = F.softmax(logits_2, dim=2)

        # --- Stage 3 ---
        # Input: Strictly probabilities from Stage 2
        logits_3 = self.stage3(probs_2)

        # Return all logits for cascaded loss supervision
        return logits_1, logits_2, logits_3
