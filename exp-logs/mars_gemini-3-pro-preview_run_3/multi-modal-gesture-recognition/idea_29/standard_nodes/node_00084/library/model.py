import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class FeatureGating(nn.Module):
    """
    Learnable feature gating mechanism to suppress noisy input channels.
    Equation: X_tilde = X * sigmoid(W * X + b)
    """

    def __init__(self, input_dim):
        super(FeatureGating, self).__init__()
        self.gate = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x: (Batch, Time, InputDim)
        gate_values = torch.sigmoid(self.gate(x))
        return x * gate_values


class TemporalConvLayer(nn.Module):
    """
    Single Gated Dilated Temporal Convolutional Layer.
    Uses centered padding for non-causal temporal modeling.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.0):
        super(TemporalConvLayer, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation

        # Padding for centered convolution (non-causal)
        # Output length L_out = L_in + 2*padding - dilation*(kernel_size-1) - 1 + 1
        # To keep L_out = L_in, 2*padding = dilation*(kernel_size-1)
        # With kernel_size=3, padding = dilation
        self.padding = dilation

        # Filter convolution (tanh)
        self.filter_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=self.padding,
        )

        # Gate convolution (sigmoid)
        self.gate_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=self.padding,
        )

        # 1x1 conv for residual connection if channels change
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (Batch, Channels, Time)

        # Apply convolutions
        filter_out = self.filter_conv(x)
        gate_out = self.gate_conv(x)

        # Gated Activation
        activation = torch.tanh(filter_out) * torch.sigmoid(gate_out)
        activation = self.dropout(activation)

        # Residual connection
        res = x if self.residual_conv is None else self.residual_conv(x)

        return activation + res


class SingleStageTCN(nn.Module):
    """
    Monotonic Non-Causal MS-TCN Stage.
    Stack of dilated temporal convolutions.
    """

    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageTCN, self).__init__()

        # Input projection: Probabilities -> Hidden Dim
        self.conv_1x1_in = nn.Conv1d(num_classes, num_f_maps, 1)

        # Stack of dilated layers
        self.layers = nn.ModuleList(
            [
                TemporalConvLayer(
                    in_channels=num_f_maps,
                    out_channels=num_f_maps,
                    kernel_size=Config.TCN_KERNEL_SIZE,
                    dilation=Config.TCN_DILATIONS[i % len(Config.TCN_DILATIONS)],
                    dropout=Config.TCN_DROPOUT,
                )
                for i in range(num_layers)
            ]
        )

        # Output projection: Hidden Dim -> Classes
        self.conv_1x1_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        # x: (Batch, Time, NumClasses) - Input probabilities

        # Permute for Conv1d: (Batch, NumClasses, Time)
        out = x.permute(0, 2, 1)

        out = self.conv_1x1_in(out)

        for layer in self.layers:
            out = layer(out)

        out = self.conv_1x1_out(out)

        # Permute back: (Batch, Time, NumClasses)
        out = out.permute(0, 2, 1)

        return F.softmax(out, dim=2)


class BiGRUEncoder(nn.Module):
    """
    Stage 1: Normalized Gated Kinematic Encoder.
    """

    def __init__(self, input_dim, hidden_dim, num_classes):
        super(BiGRUEncoder, self).__init__()

        self.feature_gating = FeatureGating(input_dim)

        # Bi-GRU: hidden_dim is total (so per direction is hidden_dim // 2)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=Config.GRU_HIDDEN_SIZE,
            num_layers=Config.GRU_NUM_LAYERS,
            batch_first=True,
            bidirectional=Config.GRU_BIDIRECTIONAL,
        )

        # Output projection
        # GRU output dim is hidden_size * 2 (if bidirectional)
        gru_out_dim = (
            Config.GRU_HIDDEN_SIZE * 2
            if Config.GRU_BIDIRECTIONAL
            else Config.GRU_HIDDEN_SIZE
        )
        self.fc = nn.Linear(gru_out_dim, num_classes)

    def forward(self, x):
        # x: (Batch, Time, InputDim)

        # 1. Feature Gating
        x = self.feature_gating(x)

        # 2. Bi-GRU
        # output: (Batch, Time, HiddenDim)
        out, _ = self.gru(x)

        # 3. Classification
        logits = self.fc(out)

        return F.softmax(logits, dim=2)


class NGKRN(nn.Module):
    """
    Normalized Gated-Kinematic Refinement Network (Idea 29).
    Three-Stage Cascaded Network:
    1. BiGRU Encoder
    2. TCN Refinement
    3. TCN Refinement (Independent)
    """

    def __init__(self):
        super(NGKRN, self).__init__()

        # Stage 1
        self.stage1 = BiGRUEncoder(
            input_dim=Config.INPUT_DIM,
            hidden_dim=Config.HIDDEN_DIM,
            num_classes=Config.NUM_CLASSES,
        )

        # Stage 2
        # Number of layers corresponds to length of dilation schedule
        num_layers = len(Config.TCN_DILATIONS)
        self.stage2 = SingleStageTCN(
            num_layers=num_layers,
            num_f_maps=Config.TCN_NUM_CHANNELS,
            dim=Config.NUM_CLASSES,  # Input dim is num_classes (probs)
            num_classes=Config.NUM_CLASSES,
        )

        # Stage 3
        self.stage3 = SingleStageTCN(
            num_layers=num_layers,
            num_f_maps=Config.TCN_NUM_CHANNELS,
            dim=Config.NUM_CLASSES,
            num_classes=Config.NUM_CLASSES,
        )

    def forward(self, x):
        # x: (Batch, Time, InputDim)

        # Stage 1
        p1 = self.stage1(x)

        # Stage 2 (Refinement)
        p2 = self.stage2(p1)

        # Stage 3 (Refinement)
        p3 = self.stage3(p2)

        # Return all stages for deep supervision
        return p1, p2, p3
