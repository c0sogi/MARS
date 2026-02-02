import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    INPUT_DIM,
    GRU_HIDDEN_DIM,
    GRU_NUM_LAYERS,
    DROPOUT,
    NUM_CLASSES,
    TCN_NUM_CHANNELS,
    TCN_KERNEL_SIZE,
    TCN_DROPOUT,
)


class BiGRUEncoder(nn.Module):
    """
    Stage 1: Sequence Encoder using Bi-directional GRU.
    Extracts local temporal dynamics and generates initial frame-wise class logits.
    """

    def __init__(self):
        super(BiGRUEncoder, self).__init__()

        self.gru = nn.GRU(
            input_size=INPUT_DIM,
            hidden_size=GRU_HIDDEN_DIM,
            num_layers=GRU_NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT if GRU_NUM_LAYERS > 1 else 0.0,
        )

        # Output layer: Maps from (Hidden * 2) to NumClasses
        self.fc = nn.Linear(GRU_HIDDEN_DIM * 2, NUM_CLASSES)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        # x shape: (Batch, Time, InputDim)

        # GRU Output: (Batch, Time, Hidden * 2)
        out, _ = self.gru(x)

        out = self.dropout(out)

        # Project to classes: (Batch, Time, NumClasses)
        logits = self.fc(out)

        return logits


class DilatedResidualLayer(nn.Module):
    """
    A single dilated convolutional layer with residual connection.
    Used in the TCN Refinement stage.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(DilatedResidualLayer, self).__init__()

        # Calculate padding to maintain temporal dimension (centered)
        # padding = (kernel_size - 1) * dilation / 2
        self.padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=self.padding,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (Batch, Channels, Time)
        out = self.conv(x)
        out = self.relu(out)
        out = self.dropout(out)

        # Residual connection
        return x + out


class TCNRefinement(nn.Module):
    """
    Stage 2: Temporal Refinement Module using Dilated TCN.
    Takes class probabilities from Stage 1 and refines them.
    """

    def __init__(self):
        super(TCNRefinement, self).__init__()

        layers = []
        num_channels = TCN_NUM_CHANNELS
        kernel_size = TCN_KERNEL_SIZE
        dropout = TCN_DROPOUT

        # Input projection: Map from NumClasses (probabilities) to TCN hidden dim
        # We assume the first layer in TCN_NUM_CHANNELS defines the hidden dim
        hidden_dim = num_channels[0]
        self.conv_in = nn.Conv1d(NUM_CLASSES, hidden_dim, 1)

        # Stack dilated layers
        for i, out_channels in enumerate(num_channels):
            dilation = 2**i  # 1, 2, 4, 8...
            layers.append(
                DilatedResidualLayer(
                    hidden_dim, out_channels, kernel_size, dilation, dropout
                )
            )
            hidden_dim = out_channels  # Update for next layer if channels vary

        self.layers = nn.ModuleList(layers)

        # Output projection: Map back to NumClasses
        self.conv_out = nn.Conv1d(hidden_dim, NUM_CLASSES, 1)

    def forward(self, x):
        # x shape: (Batch, NumClasses, Time) - These are probabilities from Stage 1

        out = self.conv_in(x)

        for layer in self.layers:
            out = layer(out)

        out = self.conv_out(out)

        return out


class CascadedRefinementNet(nn.Module):
    """
    Cascaded Recurrent-Convolutional Refinement Network.
    Combines BiGRUEncoder (Stage 1) and TCNRefinement (Stage 2).
    """

    def __init__(self):
        super(CascadedRefinementNet, self).__init__()
        self.stage1 = BiGRUEncoder()
        self.stage2 = TCNRefinement()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, Time, InputDim)

        Returns:
            stage1_logits (torch.Tensor): Output from BiGRU, shape (Batch, Time, NumClasses)
            stage2_logits (torch.Tensor): Output from TCN, shape (Batch, Time, NumClasses)
        """
        # --- Stage 1: Bi-GRU ---
        # Get logits: (Batch, Time, NumClasses)
        s1_logits = self.stage1(x)

        # Convert logits to probabilities for Stage 2 input
        # Detach ensures gradients don't flow back through the soft input to Stage 1
        # (optional, but often helpful for stability, though end-to-end is also valid.
        # Here we allow end-to-end training as per prompt description).
        s1_probs = F.softmax(s1_logits, dim=2)

        # --- Stage 2: TCN Refinement ---
        # TCN expects (Batch, Channels, Time)
        s1_probs_t = s1_probs.permute(0, 2, 1)

        # Forward through TCN
        s2_logits_t = self.stage2(s1_probs_t)

        # Permute back to (Batch, Time, NumClasses)
        s2_logits = s2_logits_t.permute(0, 2, 1)

        return s1_logits, s2_logits
