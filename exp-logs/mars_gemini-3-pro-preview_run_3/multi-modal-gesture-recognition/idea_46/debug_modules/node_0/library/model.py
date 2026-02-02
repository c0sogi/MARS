import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import ModelConfig, DataConfig


class GatedDilatedBlock(nn.Module):
    """
    Gated Dilated Temporal Convolutional Block.
    Structure:
        - Dilated Conv1d (splitting into Filter and Gate)
        - Activation: Tanh(Filter) * Sigmoid(Gate)
        - 1x1 Conv (Projection)
        - Dropout
        - Residual Connection
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(GatedDilatedBlock, self).__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation

        # Calculate padding to keep output length same as input length (Centered Padding)
        # For kernel_size=3, padding = dilation
        self.padding = (kernel_size - 1) * dilation // 2

        # Convolution for Filter and Gate (2 * out_channels)
        self.conv_filter_gate = nn.Conv1d(
            in_channels,
            2 * out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

        # 1x1 Conv for projection/residual integration
        self.conv_out = nn.Conv1d(out_channels, out_channels, 1)

        self.dropout = nn.Dropout(dropout)

        # Residual connection handling
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.downsample = None

    def forward(self, x):
        # x shape: (Batch, Channels, Time)
        residual = x

        # 1. Dilated Convolution
        out = self.conv_filter_gate(x)

        # 2. Split into Filter and Gate
        filter_conv, gate_conv = out.chunk(2, dim=1)

        # 3. Gated Activation
        out = torch.tanh(filter_conv) * torch.sigmoid(gate_conv)

        # 4. Projection
        out = self.conv_out(out)
        out = self.dropout(out)

        # 5. Residual Connection
        if self.downsample is not None:
            residual = self.downsample(residual)

        return out + residual


class RefinementStage(nn.Module):
    """
    Refinement Stage using Gated Dilated TCNs.
    Input: Class Probabilities from previous stage.
    Output: Refined Class Logits.
    """

    def __init__(self, num_classes, hidden_channels, dilations, kernel_size, dropout):
        super(RefinementStage, self).__init__()

        # Input projection: Probabilities -> Hidden
        self.conv_in = nn.Conv1d(num_classes, hidden_channels, 1)

        # Stack of Gated Dilated Blocks
        self.layers = nn.ModuleList()
        for d in dilations:
            self.layers.append(
                GatedDilatedBlock(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout,
                )
            )

        # Output projection: Hidden -> Logits
        self.conv_out = nn.Conv1d(hidden_channels, num_classes, 1)

    def forward(self, x):
        # x shape: (Batch, NumClasses, Time) - Expecting probabilities

        out = self.conv_in(x)

        for layer in self.layers:
            out = layer(out)

        out = self.conv_out(out)

        return out


class BiGRUEncoder(nn.Module):
    """
    Stage 1: Root-Centric Moderate-Capacity Encoder.
    Input: Concatenated Skeleton + Audio features.
    Output: Initial Class Logits.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout):
        super(BiGRUEncoder, self).__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Projection: Hidden * 2 (Bidirectional) -> NumClasses
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x shape: (Batch, Time, InputDim)

        self.gru.flatten_parameters()
        out, _ = self.gru(x)

        # out shape: (Batch, Time, HiddenDim * 2)
        logits = self.fc(out)

        # Permute to (Batch, Classes, Time) for consistency with TCN stages
        logits = logits.permute(0, 2, 1)

        return logits


class CKARFNet(nn.Module):
    """
    Central-Kinematic Aligned-Receptive-Field Network (CK-ARF-Net).
    A Three-Stage Cascaded Network.
    """

    def __init__(self):
        super(CKARFNet, self).__init__()

        # ---------------------------------------------------------
        # Configuration
        # ---------------------------------------------------------
        num_classes = DataConfig.NUM_CLASSES

        # Input Dimension Calculation
        # Skeleton: 20 joints * 3 coords * 3 (pos, vel, acc) = 180
        # Audio: 13 MFCC
        input_dim = (DataConfig.NUM_JOINTS * 3 * 3) + DataConfig.N_MFCC

        # ---------------------------------------------------------
        # Stage 1: Encoder
        # ---------------------------------------------------------
        self.stage1 = BiGRUEncoder(
            input_dim=input_dim,
            hidden_dim=ModelConfig.GRU_HIDDEN_SIZE,
            num_layers=ModelConfig.GRU_LAYERS,
            num_classes=num_classes,
            dropout=ModelConfig.GRU_DROPOUT,
        )

        # ---------------------------------------------------------
        # Stage 2: Refinement 1 (RF-Aligned)
        # ---------------------------------------------------------
        self.stage2 = RefinementStage(
            num_classes=num_classes,
            hidden_channels=ModelConfig.TCN_CHANNELS,
            dilations=ModelConfig.TCN_DILATIONS,
            kernel_size=ModelConfig.TCN_KERNEL_SIZE,
            dropout=ModelConfig.TCN_DROPOUT,
        )

        # ---------------------------------------------------------
        # Stage 3: Refinement 2 (Independent)
        # ---------------------------------------------------------
        self.stage3 = RefinementStage(
            num_classes=num_classes,
            hidden_channels=ModelConfig.TCN_CHANNELS,
            dilations=ModelConfig.TCN_DILATIONS,
            kernel_size=ModelConfig.TCN_KERNEL_SIZE,
            dropout=ModelConfig.TCN_DROPOUT,
        )

    def forward(self, x):
        """
        Forward pass with Deep Supervision.
        Args:
            x (torch.Tensor): Input features (Batch, Time, InputDim)
        Returns:
            tuple: (logits_stage1, logits_stage2, logits_stage3)
                   Each shape: (Batch, Time, NumClasses)
        """
        # ---------------------------------------------------------
        # Stage 1
        # ---------------------------------------------------------
        # Output: (Batch, Classes, Time)
        logits1 = self.stage1(x)

        # Prepare input for Stage 2: Probabilities (Softmax)
        probs1 = F.softmax(logits1, dim=1)

        # ---------------------------------------------------------
        # Stage 2
        # ---------------------------------------------------------
        logits2 = self.stage2(probs1)

        # Prepare input for Stage 3: Probabilities (Softmax)
        probs2 = F.softmax(logits2, dim=1)

        # ---------------------------------------------------------
        # Stage 3
        # ---------------------------------------------------------
        logits3 = self.stage3(probs2)

        # Permute back to (Batch, Time, Classes) for Loss calculation
        return (
            logits1.permute(0, 2, 1),
            logits2.permute(0, 2, 1),
            logits3.permute(0, 2, 1),
        )
