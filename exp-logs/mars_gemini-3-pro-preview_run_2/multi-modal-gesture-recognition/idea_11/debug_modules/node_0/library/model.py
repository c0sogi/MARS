import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class BiLSTMEncoder(nn.Module):
    """
    Stage 1: Recurrent Encoder using Bi-Directional LSTM.
    Generates initial frame-wise predictions from raw features.
    """

    def __init__(self):
        super(BiLSTMEncoder, self).__init__()

        self.lstm = nn.LSTM(
            input_size=Config.INPUT_DIM,
            hidden_size=Config.LSTM_HIDDEN_SIZE,
            num_layers=Config.LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.LSTM_DROPOUT if Config.LSTM_LAYERS > 1 else 0.0,
        )

        # Projection to class logits
        # Input to linear is hidden_size * 2 (bidirectional)
        self.classifier = nn.Linear(Config.LSTM_HIDDEN_SIZE * 2, Config.NUM_CLASSES)

    def forward(self, x):
        # x: (Batch, Time, Features)
        self.lstm.flatten_parameters()

        # LSTM output: (Batch, Time, Hidden*2)
        outputs, _ = self.lstm(x)

        # Project to logits: (Batch, Time, NumClasses)
        logits = self.classifier(outputs)

        return logits


class DilatedResidualLayer(nn.Module):
    """
    Single Residual Block for the TCN with Instance Normalization.
    Structure: DilatedConv -> InstanceNorm -> ReLU -> Dropout -> 1x1Conv -> InstanceNorm -> ReLU -> Dropout -> Residual
    """

    def __init__(self, channels, kernel_size, dilation, dropout):
        super(DilatedResidualLayer, self).__init__()

        # Padding to maintain temporal dimension
        # For kernel_size 3, padding = dilation
        padding = (kernel_size - 1) * dilation // 2

        self.conv_dilated = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, dilation=dilation
        )
        self.norm1 = nn.InstanceNorm1d(channels, affine=True)
        self.dropout1 = nn.Dropout(dropout)

        self.conv_1x1 = nn.Conv1d(channels, channels, 1)
        self.norm2 = nn.InstanceNorm1d(channels, affine=True)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Channels, Time)
        out = self.conv_dilated(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.dropout1(out)

        out = self.conv_1x1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.dropout2(out)

        return x + out


class InstanceNormTCN(nn.Module):
    """
    Stage 2 & 3: Temporal Convolutional Network for Refinement.
    Uses dilated convolutions with Instance Normalization to refine probabilities.
    """

    def __init__(self):
        super(InstanceNormTCN, self).__init__()

        num_layers = Config.TCN_NUM_LAYERS
        num_channels = Config.TCN_NUM_CHANNELS
        kernel_size = Config.TCN_KERNEL_SIZE
        dropout = Config.TCN_DROPOUT
        num_classes = Config.NUM_CLASSES

        # Entry layer: Map probabilities (num_classes) to hidden channels
        self.conv_in = nn.Conv1d(num_classes, num_channels, 1)

        # Stack of dilated residual layers
        layers = []
        for i in range(num_layers):
            dilation = 2**i
            layers.append(
                DilatedResidualLayer(num_channels, kernel_size, dilation, dropout)
            )
        self.layers = nn.ModuleList(layers)

        # Exit layer: Map hidden channels back to logits
        self.conv_out = nn.Conv1d(num_channels, num_classes, 1)

    def forward(self, x):
        # x: (Batch, Time, NumClasses) - Input probabilities

        # Transpose for Conv1d: (Batch, NumClasses, Time)
        x = x.permute(0, 2, 1)

        out = self.conv_in(x)

        for layer in self.layers:
            out = layer(out)

        out = self.conv_out(out)

        # Transpose back: (Batch, Time, NumClasses)
        logits = out.permute(0, 2, 1)

        return logits


class NMD_CRCN(nn.Module):
    """
    Normalized Masked Dual-Stage Cascaded Recurrent-Convolutional Network.

    Pipeline:
    1. Features -> BiLSTM -> Logits1
    2. Softmax(Logits1) * Mask -> TCN1 -> Logits2
    3. Softmax(Logits2) * Mask -> TCN2 -> Logits3
    """

    def __init__(self):
        super(NMD_CRCN, self).__init__()

        self.stage1 = BiLSTMEncoder()
        self.stage2 = InstanceNormTCN()
        self.stage3 = InstanceNormTCN()

    def forward(self, x, mask):
        """
        Args:
            x: (Batch, Time, Features)
            mask: (Batch, Time) - Binary mask (1 for valid frames, 0 for padding)

        Returns:
            dict: {'stage1': logits, 'stage2': logits, 'stage3': logits}
        """
        # Ensure mask has correct shape for broadcasting: (Batch, Time, 1)
        mask_expanded = mask.unsqueeze(-1)

        # --- Stage 1: Generation ---
        logits1 = self.stage1(x)

        # Prepare input for Stage 2: Masked Probabilities
        probs1 = F.softmax(logits1, dim=2)
        probs1_masked = probs1 * mask_expanded

        # --- Stage 2: Refinement ---
        logits2 = self.stage2(probs1_masked)

        # Prepare input for Stage 3: Masked Probabilities
        probs2 = F.softmax(logits2, dim=2)
        probs2_masked = probs2 * mask_expanded

        # --- Stage 3: Sharpening ---
        logits3 = self.stage3(probs2_masked)

        return {"stage1": logits1, "stage2": logits2, "stage3": logits3}
