import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config


class BiGRUEncoder(nn.Module):
    """
    Stage 1: Sequence Encoder using Bi-Directional GRU.
    Processes the raw multi-modal features to generate initial frame-wise predictions.
    """

    def __init__(self):
        super(BiGRUEncoder, self).__init__()
        self.input_dim = config.INPUT_DIM
        self.hidden_size = config.GRU_HIDDEN_SIZE
        self.num_layers = config.GRU_NUM_LAYERS
        self.num_classes = config.NUM_CLASSES
        self.dropout_p = config.GRU_DROPOUT

        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_p if self.num_layers > 1 else 0.0,
        )

        # Project from hidden_size * 2 (bidirectional) to num_classes
        self.fc = nn.Linear(self.hidden_size * 2, self.num_classes)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Time, InputDim)
        Returns:
            logits: Output tensor of shape (Batch, Classes, Time)
        """
        # GRU Output: (Batch, Time, Hidden * 2)
        self.gru.flatten_parameters()
        out, _ = self.gru(x)

        # Project to classes: (Batch, Time, Classes)
        logits = self.fc(out)

        # Transpose to (Batch, Classes, Time) for Conv1d compatibility and Loss
        logits = logits.permute(0, 2, 1)

        return logits


class DilatedResidualLayer(nn.Module):
    """
    A single dilated residual block for the TCN.
    Conv1d (Dilated) -> ReLU -> Dropout -> Conv1d (1x1) -> Residual Add
    """

    def __init__(self, channels, kernel_size, dilation, dropout):
        super(DilatedResidualLayer, self).__init__()

        # Calculate padding to keep temporal dimension same
        # For kernel_size 3:
        # dilation 1 -> pad 1
        # dilation 2 -> pad 2
        # dilation 4 -> pad 4
        padding = (kernel_size - 1) * dilation // 2

        self.conv_dilated = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, dilation=dilation
        )
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.conv_1x1 = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        out = self.conv_dilated(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv_1x1(out)
        return x + out


class DilatedRefinementStage(nn.Module):
    """
    Stage 2 & 3: Refinement Module using Dilated TCN.
    Takes class probabilities as input and outputs refined logits.
    """

    def __init__(self):
        super(DilatedRefinementStage, self).__init__()
        self.num_classes = config.NUM_CLASSES
        self.channels = config.TCN_NUM_CHANNELS  # List like [64, 64, 64, 64]
        self.kernel_size = config.TCN_KERNEL_SIZE
        self.dropout = config.TCN_DROPOUT

        # Input projection: Classes -> Hidden Channel (e.g., 21 -> 64)
        self.conv_in = nn.Conv1d(self.num_classes, self.channels[0], 1)

        # Stack of dilated residual layers
        layers = []
        num_levels = len(self.channels)
        for i in range(num_levels):
            dilation = 2**i
            layers.append(
                DilatedResidualLayer(
                    self.channels[i], self.kernel_size, dilation, self.dropout
                )
            )
        self.layers = nn.Sequential(*layers)

        # Output projection: Hidden Channel -> Classes (e.g., 64 -> 21)
        self.conv_out = nn.Conv1d(self.channels[-1], self.num_classes, 1)

    def forward(self, x):
        """
        Args:
            x: Input probabilities tensor of shape (Batch, Classes, Time)
        Returns:
            logits: Refined logits tensor of shape (Batch, Classes, Time)
        """
        out = self.conv_in(x)
        out = self.layers(out)
        out = self.conv_out(out)
        return out


class IterativeCascadedNet(nn.Module):
    """
    Three-Stage Hybrid Network:
    1. Bi-GRU Encoder
    2. Refinement TCN (Stage 2)
    3. Refinement TCN (Stage 3)
    """

    def __init__(self):
        super(IterativeCascadedNet, self).__init__()

        self.stage1 = BiGRUEncoder()
        self.stage2 = DilatedRefinementStage()
        self.stage3 = DilatedRefinementStage()

    def forward(self, x):
        """
        Args:
            x: Input features (Batch, Time, InputDim)

        Returns:
            [logits1, logits2, logits3]: List of logits from each stage.
            Each tensor has shape (Batch, Classes, Time).
        """
        # --- Stage 1: Encoder ---
        # Input: (B, T, D) -> Logits: (B, C, T)
        logits1 = self.stage1(x)

        # Prepare input for Stage 2: Softmax Probabilities
        # Detach is NOT used here because we want end-to-end training
        probs1 = F.softmax(logits1, dim=1)

        # --- Stage 2: Refinement ---
        # Input: (B, C, T) -> Logits: (B, C, T)
        logits2 = self.stage2(probs1)

        # Prepare input for Stage 3: Softmax Probabilities
        probs2 = F.softmax(logits2, dim=1)

        # --- Stage 3: Iterative Refinement ---
        # Input: (B, C, T) -> Logits: (B, C, T)
        logits3 = self.stage3(probs2)

        return [logits1, logits2, logits3]
