import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import config


class BiGRUEncoder(nn.Module):
    """
    Stage 1: Regularized High-Capacity Encoder.
    Uses a Bi-Directional GRU to capture temporal dynamics from raw kinematic and audio features.
    """

    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        # Project concatenated hidden states (forward + backward) to class logits
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Time, Features)
        self.gru.flatten_parameters()
        out, _ = self.gru(x)
        out = self.dropout(out)
        logits = self.fc(out)  # (Batch, Time, NumClasses)
        return logits


class GatedDilatedConvBlock(nn.Module):
    """
    Building block for the TCN Refinement stages.
    Implements a Dilated Convolution with Gated Activation (Tanh * Sigmoid) and Residual Connection.
    """

    def __init__(self, channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        # Centered padding to maintain sequence length
        self.padding = dilation * (kernel_size - 1) // 2

        # Convolution mapping input to 2x channels (for filter + gate split)
        self.conv = nn.Conv1d(
            channels, channels * 2, kernel_size, padding=self.padding, dilation=dilation
        )

        self.dropout = nn.Dropout(dropout)

        # 1x1 Convolution for mixing and residual alignment
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        # x: (Batch, Channels, Time)
        residual = x

        out = self.conv(x)

        # Split into filter and gate
        filter_out, gate_out = out.chunk(2, dim=1)

        # Gated Activation: tanh(filter) * sigmoid(gate)
        out = torch.tanh(filter_out) * torch.sigmoid(gate_out)
        out = self.dropout(out)

        # Projection
        out = self.proj(out)

        # Residual connection
        return out + residual


class TCNRefinementStage(nn.Module):
    """
    Stage 2 & 3: Monotonic Non-Causal Refinement.
    Takes class probabilities as input and refines them using a stack of dilated convolutions.
    """

    def __init__(
        self, num_classes, hidden_channels, kernel_size, dilations, dropout=0.2
    ):
        super().__init__()

        # Input projection: Map probabilities (num_classes) to hidden representation
        self.input_proj = nn.Conv1d(num_classes, hidden_channels, 1)

        # Stack of dilated blocks
        self.layers = nn.ModuleList()
        for d in dilations:
            self.layers.append(
                GatedDilatedConvBlock(
                    channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout,
                )
            )

        # Output projection: Map hidden representation back to logits
        self.output_proj = nn.Conv1d(hidden_channels, num_classes, 1)

    def forward(self, probs):
        # probs: (Batch, Time, NumClasses)
        # Transpose to (Batch, NumClasses, Time) for Conv1d
        x = probs.transpose(1, 2)

        x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x)

        logits = self.output_proj(x)

        # Transpose back to (Batch, Time, NumClasses)
        logits = logits.transpose(1, 2)
        return logits


class RHCKN(nn.Module):
    """
    Regularized High-Capacity Kinematic Network (RHC-KN).
    Three-Stage Cascaded Network:
    1. Bi-GRU Encoder (Raw Features -> Logits1)
    2. TCN Refinement 1 (Probs1 -> Logits2)
    3. TCN Refinement 2 (Probs2 -> Logits3)
    """

    def __init__(self):
        super().__init__()

        # Calculate Input Size
        # 20 joints * 9 (pos, vel, acc) = 180
        # Audio MFCC = 13 (assumed 13 coefficients)
        # Total = 193
        input_size = config.JOINTS_COUNT * 9 + 13

        # Stage 1: Encoder
        self.stage1 = BiGRUEncoder(
            input_size=input_size,
            hidden_size=config.RNN_HIDDEN_SIZE,
            num_layers=config.RNN_LAYERS,
            num_classes=config.NUM_CLASSES,
            dropout=config.DROPOUT_RNN,
        )

        # Stage 2: Refinement 1
        self.stage2 = TCNRefinementStage(
            num_classes=config.NUM_CLASSES,
            hidden_channels=config.TCN_CHANNELS,
            kernel_size=config.TCN_KERNEL_SIZE,
            dilations=config.TCN_DILATIONS,
            dropout=config.DROPOUT_TCN,
        )

        # Stage 3: Refinement 2 (Independent weights)
        self.stage3 = TCNRefinementStage(
            num_classes=config.NUM_CLASSES,
            hidden_channels=config.TCN_CHANNELS,
            kernel_size=config.TCN_KERNEL_SIZE,
            dilations=config.TCN_DILATIONS,
            dropout=config.DROPOUT_TCN,
        )

    def forward(self, x):
        # x: (Batch, Time, Features)

        # --- Stage 1 ---
        logits1 = self.stage1(x)
        # Apply Softmax to get probabilities for the bottleneck
        probs1 = torch.softmax(logits1, dim=-1)

        # --- Stage 2 ---
        # Input: Strictly the class probabilities from Stage 1
        logits2 = self.stage2(probs1)
        probs2 = torch.softmax(logits2, dim=-1)

        # --- Stage 3 ---
        # Input: Strictly the class probabilities from Stage 2
        logits3 = self.stage3(probs2)

        # Return all logits for Deep Supervision Loss
        return logits1, logits2, logits3
