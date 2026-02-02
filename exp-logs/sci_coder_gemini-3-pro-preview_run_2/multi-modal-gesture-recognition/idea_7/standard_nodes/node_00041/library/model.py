import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DilatedResidualBlock(nn.Module):
    """
    A single dilated residual block for the TCN.
    Structure: DilatedConv(3x3) -> ReLU -> Dropout -> 1x1Conv -> Residual
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(DilatedResidualBlock, self).__init__()

        # For kernel_size=3 and dilation=d, padding=d ensures output length == input length
        # (L + 2*d - (d*(k-1) + 1)) + 1 = L
        # k=3, d=d -> effective_k = 2d + 1
        # L + 2d - 2d - 1 + 1 = L
        self.conv_dilated = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=dilation,
            dilation=dilation,
        )
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x, mask=None):
        """
        Args:
            x: (Batch, Channels, Time)
            mask: (Batch, 1, Time) - Boolean mask where 0 indicates padding
        """
        out = self.conv_dilated(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv_1x1(out)

        # Apply mask before residual addition to ensure padding remains zero
        if mask is not None:
            out = out * mask

        return x + out


class SingleStageTCN(nn.Module):
    """
    Single-Stage TCN for refinement.
    Input: Probabilities (Batch, NumClasses, Time)
    Output: Refined Probabilities (Batch, NumClasses, Time)
    """

    def __init__(self, num_classes, num_channels, kernel_size=3, dropout=0.3):
        super(SingleStageTCN, self).__init__()

        self.num_classes = num_classes
        layers = []

        # Input Projection: NumClasses -> HiddenDim
        # We assume num_channels list defines the hidden dim for each layer
        # Usually constant, e.g., [256, 256, ...]
        input_dim = num_classes
        hidden_dim = num_channels[0]

        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)

        # Stack Dilated Residual Blocks
        for i, channels in enumerate(num_channels):
            dilation = 2**i
            layers.append(
                DilatedResidualBlock(
                    in_channels=hidden_dim,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            hidden_dim = channels

        self.layers = nn.ModuleList(layers)

        # Output Projection: HiddenDim -> NumClasses
        self.output_proj = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, x, mask):
        """
        Args:
            x: (Batch, NumClasses, Time) - Input probabilities
            mask: (Batch, Time) - Boolean mask indicating valid frames
        Returns:
            probs: (Batch, NumClasses, Time) - Refined probabilities
        """
        # Prepare mask for broadcasting: (Batch, 1, Time)
        mask_expanded = mask.unsqueeze(1).float()

        # Mask Input
        x = x * mask_expanded

        # Input Projection
        out = self.input_proj(x)
        out = self.relu(out)  # Optional activation after projection

        # Apply Layers with Strict Masking
        for layer in self.layers:
            out = layer(out, mask_expanded)

        # Output Projection
        logits = self.output_proj(out)

        # Mask Logits before Softmax (ensure padding is strictly ignored/handled)
        # Though multiplying by mask after softmax is safer for the values
        logits = logits * mask_expanded

        # Compute Probabilities
        probs = F.softmax(logits, dim=1)

        # Final Masking to ensure strict zeroing of padding
        probs = probs * mask_expanded

        return probs

    def relu(self, x):
        return F.relu(x, inplace=True)


class BiLSTMEncoder(nn.Module):
    """
    Stage 1: Bi-LSTM Encoder
    Input: Raw Features (Batch, Time, InputDim)
    Output: Initial Probabilities (Batch, NumClasses, Time)
    """

    def __init__(self, input_dim, hidden_size, num_layers, num_classes):
        super(BiLSTMEncoder, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=Config.LSTM_DROPOUT if num_layers > 1 else 0,
        )

        # Linear projection from LSTM output to classes
        # Bidirectional -> hidden_size * 2
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, lengths):
        """
        Args:
            x: (Batch, Time, InputDim)
            lengths: (Batch,) - Actual lengths of sequences
        Returns:
            probs: (Batch, NumClasses, Time) - Permuted for TCN compatibility
        """
        # Pack sequence for LSTM
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_output, _ = self.lstm(packed_input)

        # Unpack sequence
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # output: (Batch, Time, Hidden*2)

        # Project to classes
        logits = self.classifier(output)  # (Batch, Time, NumClasses)

        # Compute Probabilities
        probs = F.softmax(logits, dim=2)

        # Permute to (Batch, NumClasses, Time) for TCN
        probs = probs.permute(0, 2, 1)

        return probs


class IDCRCN(nn.Module):
    """
    Iterative Deeply-Cascaded Recurrent-Convolutional Network
    Stage 1: BiLSTM -> P0
    Stage 2: TCN(P0) -> P1
    Stage 3: TCN(P1) -> P2
    """

    def __init__(self):
        super(IDCRCN, self).__init__()

        # Stage 1
        self.stage1 = BiLSTMEncoder(
            input_dim=Config.INPUT_DIM,
            hidden_size=Config.LSTM_HIDDEN_SIZE,
            num_layers=Config.LSTM_NUM_LAYERS,
            num_classes=Config.NUM_CLASSES,
        )

        # Stage 2
        self.stage2 = SingleStageTCN(
            num_classes=Config.NUM_CLASSES,
            num_channels=Config.TCN_NUM_CHANNELS,
            kernel_size=Config.TCN_KERNEL_SIZE,
            dropout=Config.TCN_DROPOUT,
        )

        # Stage 3
        self.stage3 = SingleStageTCN(
            num_classes=Config.NUM_CLASSES,
            num_channels=Config.TCN_NUM_CHANNELS,
            kernel_size=Config.TCN_KERNEL_SIZE,
            dropout=Config.TCN_DROPOUT,
        )

    def forward(self, features, mask, lengths):
        """
        Args:
            features: (Batch, Time, InputDim)
            mask: (Batch, Time) - Boolean mask
            lengths: (Batch,) - Sequence lengths
        Returns:
            dict: Outputs from all stages (Batch, NumClasses, Time)
        """
        # Stage 1: BiLSTM
        # Returns (Batch, NumClasses, Time)
        p0 = self.stage1(features, lengths)

        # Ensure p0 is masked correctly before passing to next stage
        # (Though LSTM padding output is usually zero or ignored, explicit masking is safer)
        mask_expanded = mask.unsqueeze(1).float()
        p0 = p0 * mask_expanded

        # Stage 2: Coarse Refinement
        p1 = self.stage2(p0, mask)

        # Stage 3: Fine Refinement
        p2 = self.stage3(p1, mask)

        return {"stage1": p0, "stage2": p1, "stage3": p2}
