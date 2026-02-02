import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.utils import set_seed

# Ensure reproducibility
set_seed()


class BiGRUEncoder(nn.Module):
    """
    Stage 1: Kinematic Sequence Encoder.
    Processes the Early Fusion features (Skeleton Kinematics + Audio MFCC)
    using a Bi-directional GRU to generate initial frame-wise predictions.
    """

    def __init__(
        self,
        input_dim=Config.INPUT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.GRU_LAYERS,
        num_classes=Config.NUM_CLASSES,
        dropout=Config.DROPOUT,
    ):
        super(BiGRUEncoder, self).__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Bidirectional GRU outputs 2 * hidden_dim
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, Time, InputDim)
        Returns:
            torch.Tensor: Logits of shape (Batch, Time, NumClasses)
        """
        self.gru.flatten_parameters()

        # out shape: (Batch, Time, HiddenDim * 2)
        out, _ = self.gru(x)
        out = self.dropout(out)

        # Project to classes
        logits = self.fc(out)
        return logits


class GatedDilatedBlock(nn.Module):
    """
    A single block for the Gated Dilated TCN.
    Consists of: Dilated Conv -> Gated Activation -> 1x1 Conv -> Residual
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, dilation, dropout=Config.DROPOUT
    ):
        super(GatedDilatedBlock, self).__init__()

        # Calculate padding to keep temporal dimension same
        # For kernel_size 3 (odd), padding = dilation * (kernel_size - 1) / 2
        self.padding = dilation * (kernel_size - 1) // 2

        # Conv1d outputs 2 * out_channels for the Gated Activation (Filter + Gate)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels * 2,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

        self.dropout = nn.Dropout(dropout)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features (Batch, Channels, Time)
        Returns:
            torch.Tensor: Output features (Batch, Channels, Time)
        """
        residual = x

        # Dilated Convolution
        out = self.conv(x)

        # Gated Activation Unit
        # Split into filter (tanh) and gate (sigmoid)
        filter_out, gate_out = out.chunk(2, dim=1)
        out = torch.tanh(filter_out) * torch.sigmoid(gate_out)

        out = self.dropout(out)

        # 1x1 Conv
        out = self.conv_1x1(out)

        # Residual Connection
        return out + residual


class GatedDilatedTCN(nn.Module):
    """
    Shared Refinement Module.
    A deep stack of Gated Dilated Convolutional blocks.
    Takes class probabilities as input and outputs refined class logits.
    """

    def __init__(
        self,
        num_classes=Config.NUM_CLASSES,
        channels=Config.TCN_CHANNELS,
        kernel_size=Config.TCN_KERNEL_SIZE,
        dilations=Config.TCN_DILATIONS,
    ):
        super(GatedDilatedTCN, self).__init__()

        # Project Input Probabilities to Feature Space
        self.input_proj = nn.Conv1d(num_classes, channels, 1)

        # Stack of Dilated Blocks
        self.layers = nn.ModuleList()
        for dilation in dilations:
            self.layers.append(
                GatedDilatedBlock(channels, channels, kernel_size, dilation)
            )

        # Project Features back to Class Logits
        self.output_proj = nn.Conv1d(channels, num_classes, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input probabilities (Batch, Time, NumClasses)
        Returns:
            torch.Tensor: Refined logits (Batch, Time, NumClasses)
        """
        # Permute to (Batch, Channels, Time) for Conv1d
        x = x.permute(0, 2, 1)

        out = self.input_proj(x)

        for layer in self.layers:
            out = layer(out)

        out = self.output_proj(out)

        # Permute back to (Batch, Time, Channels)
        out = out.permute(0, 2, 1)
        return out


class RS_KRN(nn.Module):
    """
    Recurrently-Shared Kinematic Refinement Network.
    Combines BiGRUEncoder with a recursively applied GatedDilatedTCN.
    """

    def __init__(self):
        super(RS_KRN, self).__init__()
        self.encoder = BiGRUEncoder()
        self.refinement = GatedDilatedTCN()
        self.num_stages = Config.NUM_REFINEMENT_STAGES

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features (Batch, Time, InputDim)
        Returns:
            list[torch.Tensor]: List of logits from [Encoder, Refinement_1, ..., Refinement_N]
                                Each tensor has shape (Batch, Time, NumClasses)
        """
        outputs = []

        # Stage 1: Initial Prediction via Encoder
        logits_0 = self.encoder(x)
        outputs.append(logits_0)

        # Prepare input for refinement (Softmax probabilities)
        current_input = torch.softmax(logits_0, dim=2)

        # Stage 2+: Recursive Refinement
        # The same self.refinement module is used multiple times (weight sharing)
        for _ in range(self.num_stages):
            logits_ref = self.refinement(current_input)
            outputs.append(logits_ref)

            # Update input for next iteration
            current_input = torch.softmax(logits_ref, dim=2)

        return outputs
