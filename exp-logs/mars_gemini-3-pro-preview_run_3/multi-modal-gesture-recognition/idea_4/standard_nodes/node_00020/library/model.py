import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config

# Set fixed seeds for reproducibility
torch.manual_seed(Config.SEED)


class BiGRUEncoder(nn.Module):
    """
    Stage 1: Contextual Encoder
    Processes frame-wise features using a Bi-Directional GRU.
    Outputs initial class logits and rich latent features.
    """

    def __init__(self):
        super(BiGRUEncoder, self).__init__()

        self.gru = nn.GRU(
            input_size=Config.INPUT_DIM,
            hidden_size=Config.GRU_HIDDEN_SIZE,
            num_layers=Config.GRU_NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=Config.GRU_DROPOUT if Config.GRU_NUM_LAYERS > 1 else 0,
        )

        # Projection layer to map hidden states to class logits
        # Input dim is Hidden * 2 because of bi-directionality
        self.fc = nn.Linear(Config.GRU_HIDDEN_SIZE * 2, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Args:
            x: (Batch, Time, Input_Dim)
        Returns:
            logits: (Batch, Time, Num_Classes)
            features: (Batch, Time, Hidden_Size * 2)
        """
        # GRU Output: (Batch, Time, Hidden_Size * Num_Directions)
        features, _ = self.gru(x)

        # Project to classes
        logits = self.fc(features)

        return logits, features


class TemporalBlock(nn.Module):
    """
    Building block for the TCN.
    Consists of two dilated 1D convolutions with ReLU, Dropout, and a Residual connection.
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()

        # Calculate padding to keep output length same as input (assuming odd kernel size)
        # For centered (non-causal) convolution:
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.relu1, self.dropout1, self.conv2, self.relu2, self.dropout2
        )

        # 1x1 Conv for residual connection if dimensions change
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: (Batch, Channels, Time)
        Returns:
            out: (Batch, Channels, Time)
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class HoloRefinementTCN(nn.Module):
    """
    Stage 2: Holo-Refinement Module
    Uses a Residual TCN to refine predictions based on Stage 1 logits and latent features.
    """

    def __init__(self):
        super(HoloRefinementTCN, self).__init__()

        input_dim = Config.REFINEMENT_INPUT_DIM
        num_channels = Config.TCN_CHANNELS
        kernel_size = Config.TCN_KERNEL_SIZE
        dropout = Config.TCN_DROPOUT

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = input_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)

        # Final projection to classes
        self.final_conv = nn.Conv1d(num_channels[-1], Config.NUM_CLASSES, 1)

    def forward(self, x):
        """
        Args:
            x: (Batch, Time, Input_Dim) - Note: Time is dim 1 here
        Returns:
            logits: (Batch, Time, Num_Classes)
        """
        # Permute to (Batch, Channels, Time) for Conv1d
        x = x.permute(0, 2, 1)

        # Pass through TCN
        y = self.network(x)

        # Final projection
        y = self.final_conv(y)

        # Permute back to (Batch, Time, Num_Classes)
        logits = y.permute(0, 2, 1)

        return logits


class CascadedNet(nn.Module):
    """
    End-to-End Feature-Informed Cascaded Refinement Network.
    Combines BiGRUEncoder and HoloRefinementTCN with a feature-forwarding skip connection.
    """

    def __init__(self):
        super(CascadedNet, self).__init__()
        self.stage1 = BiGRUEncoder()
        self.stage2 = HoloRefinementTCN()

    def forward(self, x):
        """
        Args:
            x: (Batch, Time, Input_Dim)
        Returns:
            s1_logits: Output from Stage 1 (Batch, Time, Num_Classes)
            s2_logits: Output from Stage 2 (Batch, Time, Num_Classes)
        """
        # Stage 1: Encode
        s1_logits, s1_features = self.stage1(x)

        # Feature Fusion: Concatenate Logits and Latent Features
        # s1_logits: (B, T, NumClasses)
        # s1_features: (B, T, Hidden*2)
        fused_input = torch.cat([s1_logits, s1_features], dim=2)

        # Stage 2: Refine
        s2_logits = self.stage2(fused_input)

        return s1_logits, s2_logits
