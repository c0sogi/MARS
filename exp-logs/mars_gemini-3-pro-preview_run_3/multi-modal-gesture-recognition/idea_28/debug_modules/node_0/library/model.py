import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class InputGating(nn.Module):
    """
    Feature-Wise Input Gating to suppress noisy sensor channels.
    Formula: X_tilde = X * sigmoid(W * X + b)
    """

    def __init__(self, input_dim):
        super(InputGating, self).__init__()
        self.gate = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (Batch, Time, InputDim)
        gates = self.sigmoid(self.gate(x))
        return x * gates


class KinematicEncoder(nn.Module):
    """
    Stage 1: Gated High-Capacity Kinematic Encoder.
    Consists of Input Gating, Bi-GRU Backbone, and initial classification head.
    """

    def __init__(self, input_dim, hidden_dim, num_classes):
        super(KinematicEncoder, self).__init__()
        self.input_gating = InputGating(input_dim)

        # Bi-directional GRU
        # hidden_dim is per direction, so total output is hidden_dim * 2
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=Config.ENCODER_LAYERS,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (Batch, Time, InputDim)
        x = self.input_gating(x)

        # GRU output: (Batch, Time, HiddenDim * 2)
        outputs, _ = self.gru(x)

        # Project to classes: (Batch, Time, NumClasses)
        logits = self.classifier(outputs)

        # Return probabilities for the next stage
        probabilities = F.softmax(logits, dim=2)
        return probabilities


class DilatedResidualLayer(nn.Module):
    """
    Single layer for the MS-TCN block.
    Uses standard convolution with centered padding (Non-Causal).
    Implements Gated Activation Unit (Tanh * Sigmoid).
    """

    def __init__(self, channels, kernel_size, dilation, dropout=0.2):
        super(DilatedResidualLayer, self).__init__()

        # Centered padding for kernel_size=3: padding = dilation
        padding = dilation

        # Feature convolution (for Tanh)
        self.conv_tanh = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )

        # Gating convolution (for Sigmoid)
        self.conv_sigmoid = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )

        self.dropout = nn.Dropout(dropout)
        # 1x1 conv for residual connection is not needed if channels don't change,
        # but standard MS-TCN implementation often uses a 1x1 conv output.
        # Here we follow the standard residual block: Out = In + Dropout(Conv(In))
        self.conv_out = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        # x: (Batch, Channels, Time)

        out_tanh = torch.tanh(self.conv_tanh(x))
        out_sigmoid = torch.sigmoid(self.conv_sigmoid(x))

        out = out_tanh * out_sigmoid
        out = self.conv_out(out)
        out = self.dropout(out)

        return x + out


class MSTCNStage(nn.Module):
    """
    Monotonic Non-Causal Gated Refinement Stage.
    Takes class probabilities, refines them using dilated convolutions.
    """

    def __init__(self, num_classes, hidden_channels, dilations):
        super(MSTCNStage, self).__init__()

        # Input projection: NumClasses -> HiddenChannels
        self.conv_in = nn.Conv1d(num_classes, hidden_channels, kernel_size=1)

        # Stack of dilated layers
        self.layers = nn.ModuleList(
            [
                DilatedResidualLayer(
                    channels=hidden_channels,
                    kernel_size=Config.TCN_KERNEL_SIZE,
                    dilation=d,
                )
                for d in dilations
            ]
        )

        # Output projection: HiddenChannels -> NumClasses
        self.conv_out = nn.Conv1d(hidden_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # x: (Batch, Time, NumClasses) -> Needs permute for Conv1d
        x = x.permute(0, 2, 1)  # (Batch, NumClasses, Time)

        out = self.conv_in(x)

        for layer in self.layers:
            out = layer(out)

        out = self.conv_out(out)

        # Permute back: (Batch, Time, NumClasses)
        out = out.permute(0, 2, 1)

        return F.softmax(out, dim=2)


class RGHC_MN(nn.Module):
    """
    Robust Gated High-Capacity Monotonic Network.
    Three-Stage Cascade: Encoder -> Refinement 1 -> Refinement 2.
    """

    def __init__(self):
        super(RGHC_MN, self).__init__()

        # Dimensions
        # Input: 20 joints * 9 channels + 13 MFCCs = 193
        self.input_dim = (
            Config.NUM_JOINTS * Config.CHANNELS_PER_JOINT + Config.AUDIO_FEATURES
        )
        self.num_classes = Config.NUM_CLASSES

        # Stage 1: Encoder
        self.stage1 = KinematicEncoder(
            input_dim=self.input_dim,
            hidden_dim=Config.ENCODER_HIDDEN_SIZE,
            num_classes=self.num_classes,
        )

        # Stage 2: Refinement
        self.stage2 = MSTCNStage(
            num_classes=self.num_classes,
            hidden_channels=Config.TCN_CHANNELS,
            dilations=Config.TCN_DILATIONS,
        )

        # Stage 3: Independent Refinement
        self.stage3 = MSTCNStage(
            num_classes=self.num_classes,
            hidden_channels=Config.TCN_CHANNELS,
            dilations=Config.TCN_DILATIONS,
        )

    def forward(self, x):
        """
        Forward pass returning outputs from all stages for Deep Supervision.
        x: (Batch, Time, InputDim)
        """
        # Stage 1
        p1 = self.stage1(x)

        # Stage 2 (Input is P1)
        p2 = self.stage2(p1)

        # Stage 3 (Input is P2)
        p3 = self.stage3(p2)

        return {"stage1": p1, "stage2": p2, "stage3": p3}
