import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class InputGating(nn.Module):
    """
    Feature-wise Gating Layer.
    Formula: X_tilde = X * Sigmoid(W * X + b)
    Suppresses noisy sensor channels before processing.
    """

    def __init__(self, input_dim):
        super(InputGating, self).__init__()
        self.gate_fc = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, D)
        Returns:
            torch.Tensor: Gated input of shape (B, T, D)
        """
        # Compute gate values
        gate = torch.sigmoid(self.gate_fc(x))
        # Apply gate
        return x * gate


class DilatedResidualLayer(nn.Module):
    """
    Gated Dilated Temporal Convolutional Layer with Residual Connection.
    Uses centered padding for non-causal convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(DilatedResidualLayer, self).__init__()

        # Calculate padding for centered convolution to maintain sequence length
        # Padding = (Kernel_Size - 1) * Dilation / 2
        # For Kernel=3, Padding = Dilation
        self.padding = dilation
        self.kernel_size = kernel_size
        self.dilation = dilation

        # Dilated Convolution
        # We output 2 * out_channels to split into Filter and Gate paths
        self.conv_dilated = nn.Conv1d(
            in_channels,
            out_channels * 2,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

        # 1x1 Convolution for output projection
        self.conv_1x1 = nn.Conv1d(out_channels, in_channels, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input (B, C, T)
        Returns:
            torch.Tensor: Output (B, C, T)
        """
        out = self.conv_dilated(x)

        # Split into filter and gate
        filter_out, gate_out = torch.chunk(out, 2, dim=1)

        # Gated Activation: Tanh * Sigmoid
        out = torch.tanh(filter_out) * torch.sigmoid(gate_out)

        # 1x1 Conv
        out = self.conv_1x1(out)

        # Dropout
        out = self.dropout(out)

        # Residual Connection
        return x + out


class MonotonicTCNStage(nn.Module):
    """
    Refinement Stage using Monotonic Non-Causal MS-TCN.
    Stack of DilatedResidualLayers with increasing dilation.
    """

    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(MonotonicTCNStage, self).__init__()

        # Input projection: Probabilities -> Feature Space
        self.conv_1x1_in = nn.Conv1d(num_classes, num_f_maps, 1)

        # Stack of dilated layers
        layers = []
        for i in range(num_layers):
            dilation = 2**i  # Monotonic schedule: 1, 2, 4, 8, 16...
            layers.append(
                DilatedResidualLayer(
                    num_f_maps,
                    num_f_maps,
                    Config.MSTCN_KERNEL_SIZE,
                    dilation,
                    Config.DROPOUT,
                )
            )
        self.layers = nn.ModuleList(layers)

        # Output projection: Feature Space -> Logits
        self.conv_1x1_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input Probabilities (B, NumClasses, T)
        Returns:
            torch.Tensor: Output Logits (B, NumClasses, T)
        """
        out = self.conv_1x1_in(x)

        for layer in self.layers:
            out = layer(out)

        out = self.conv_1x1_out(out)
        return out


class KinematicEncoder(nn.Module):
    """
    Stage 1: Gated High-Capacity Kinematic Encoder.
    Input Gating + Bi-GRU + Linear Projection.
    """

    def __init__(self):
        super(KinematicEncoder, self).__init__()

        self.input_gating = InputGating(Config.INPUT_DIM)

        self.gru = nn.GRU(
            input_size=Config.INPUT_DIM,
            hidden_size=Config.ENCODER_HIDDEN_DIM // 2,  # Bidirectional split
            num_layers=Config.GRU_NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.DROPOUT if Config.GRU_NUM_LAYERS > 1 else 0,
        )

        self.dropout = nn.Dropout(Config.DROPOUT)
        self.fc = nn.Linear(Config.ENCODER_HIDDEN_DIM, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features (B, T, Input_Dim)
        Returns:
            torch.Tensor: Logits (B, NumClasses, T) - Permuted for TCN compatibility
        """
        # Apply Input Gating
        x = self.input_gating(x)

        # GRU Backbone
        # out: (B, T, Hidden_Dim)
        out, _ = self.gru(x)

        out = self.dropout(out)

        # Projection to Classes
        # out: (B, T, NumClasses)
        out = self.fc(out)

        # Permute to (B, C, T) for compatibility with TCN stages and Loss
        out = out.permute(0, 2, 1)

        return out


class GHCKRN(nn.Module):
    """
    Gated High-Capacity Kinematic Refinement Network.
    Composition:
    1. KinematicEncoder (Stage 1)
    2. MonotonicTCNStage (Stage 2)
    3. MonotonicTCNStage (Stage 3)
    """

    def __init__(self):
        super(GHCKRN, self).__init__()

        # Stage 1
        self.stage1 = KinematicEncoder()

        # Stage 2
        self.stage2 = MonotonicTCNStage(
            Config.MSTCN_LAYERS,
            Config.MSTCN_FILTERS,
            Config.MSTCN_FILTERS,
            Config.NUM_CLASSES,
        )

        # Stage 3
        self.stage3 = MonotonicTCNStage(
            Config.MSTCN_LAYERS,
            Config.MSTCN_FILTERS,
            Config.MSTCN_FILTERS,
            Config.NUM_CLASSES,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features (B, T, Input_Dim)
        Returns:
            list[torch.Tensor]: List of logits [Stage1, Stage2, Stage3]
                                Each tensor has shape (B, NumClasses, T)
        """
        outputs = []

        # --- Stage 1 ---
        out1 = self.stage1(x)
        outputs.append(out1)

        # --- Stage 2 ---
        # Input: Softmax Probabilities from Stage 1
        probs1 = F.softmax(out1, dim=1)
        out2 = self.stage2(probs1)
        outputs.append(out2)

        # --- Stage 3 ---
        # Input: Softmax Probabilities from Stage 2
        probs2 = F.softmax(out2, dim=1)
        out3 = self.stage3(probs2)
        outputs.append(out3)

        return outputs
