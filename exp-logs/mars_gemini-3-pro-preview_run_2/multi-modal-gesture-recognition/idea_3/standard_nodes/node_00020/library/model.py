import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import (
    INPUT_SIZE,
    NUM_CLASSES,
    LSTM_HIDDEN_DIM,
    LSTM_LAYERS,
    LSTM_DROPOUT,
    TCN_NUM_CHANNELS,
    TCN_KERNEL_SIZE,
    TCN_DROPOUT,
)


class BiLSTMEncoder(nn.Module):
    """
    Stage 1: Bi-directional LSTM Encoder.
    Processes the raw input features (Skeleton + Audio) to generate initial frame-wise predictions.
    """

    def __init__(self):
        super(BiLSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=LSTM_HIDDEN_DIM,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=LSTM_DROPOUT if LSTM_LAYERS > 1 else 0.0,
        )
        self.fc = nn.Linear(LSTM_HIDDEN_DIM * 2, NUM_CLASSES)

    def forward(self, x, lengths=None):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, Time, InputDim)
            lengths (torch.Tensor, optional): Valid lengths of sequences for packing.
        Returns:
            torch.Tensor: Logits of shape (Batch, Time, NumClasses)
        """
        if lengths is not None:
            # Move lengths to CPU for pack_padded_sequence
            lengths_cpu = lengths.cpu()
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            out_packed, _ = self.lstm(x_packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)

            # If the batch max length is less than x.size(1) (due to padding in collate),
            # pad_packed_sequence might return a shorter tensor. We pad it back.
            if out.size(1) < x.size(1):
                diff = x.size(1) - out.size(1)
                out = F.pad(out, (0, 0, 0, diff))
        else:
            out, _ = self.lstm(x)

        logits = self.fc(out)
        return logits


class DilatedResidualLayer(nn.Module):
    """
    Building block for the TCN.
    Dilated Conv1d -> ReLU -> Dropout -> Conv1d(1x1) -> Residual Add
    """

    def __init__(self, channels, kernel_size, dilation, dropout):
        super(DilatedResidualLayer, self).__init__()
        # For kernel_size=3, padding=dilation ensures input length == output length
        # Formula: padding = (kernel_size - 1) * dilation / 2
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


class TemporalConvNet(nn.Module):
    """
    Stage 2: Temporal Convolutional Network (Refinement).
    Takes probabilities from Stage 1 and refines them using temporal context.
    """

    def __init__(self):
        super(TemporalConvNet, self).__init__()

        layers = []
        # Input to TCN is probabilities (NumClasses)
        # We project to hidden dim first
        self.conv_in = nn.Conv1d(NUM_CLASSES, TCN_NUM_CHANNELS[0], 1)

        # Stack dilated layers
        for i in range(len(TCN_NUM_CHANNELS)):
            dilation = 2**i
            channels = TCN_NUM_CHANNELS[i]
            # Assuming hidden channels stay constant across layers as per config list
            layers.append(
                DilatedResidualLayer(channels, TCN_KERNEL_SIZE, dilation, TCN_DROPOUT)
            )

        self.network = nn.Sequential(*layers)

        # Project back to classes
        self.conv_out = nn.Conv1d(TCN_NUM_CHANNELS[-1], NUM_CLASSES, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input probabilities of shape (Batch, Time, NumClasses)
        Returns:
            torch.Tensor: Refined logits of shape (Batch, Time, NumClasses)
        """
        # Permute to (Batch, Channels, Time) for Conv1d
        x = x.permute(0, 2, 1)

        out = self.conv_in(x)
        out = self.network(out)
        out = self.conv_out(out)

        # Permute back to (Batch, Time, Channels)
        out = out.permute(0, 2, 1)
        return out


class HybridGestureNet(nn.Module):
    """
    End-to-End Hybrid Model.
    Stage 1: Bi-LSTM
    Stage 2: TCN Refinement
    """

    def __init__(self):
        super(HybridGestureNet, self).__init__()
        self.stage1 = BiLSTMEncoder()
        self.stage2 = TemporalConvNet()

    def forward(self, x, lengths=None):
        """
        Args:
            x (torch.Tensor): Input features (Batch, Time, InputDim)
            lengths (torch.Tensor): Sequence lengths
        Returns:
            tuple: (logits_stage1, logits_stage2)
        """
        # Stage 1 Forward
        logits1 = self.stage1(x, lengths)

        # Convert logits to probabilities for Stage 2 input
        # We allow gradients to flow back to Stage 1
        probs1 = F.softmax(logits1, dim=2)

        # Stage 2 Forward
        logits2 = self.stage2(probs1)

        return logits1, logits2
