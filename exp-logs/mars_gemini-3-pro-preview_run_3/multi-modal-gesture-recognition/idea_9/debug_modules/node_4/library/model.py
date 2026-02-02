import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for Channel Attention.
    Recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels.
    """

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (Batch, Channel, Time)
        b, c, t = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class GatedTCNBlock(nn.Module):
    """
    Gated Dilated Temporal Convolutional Block with SE Attention.
    Uses dilated convolutions to expand receptive field and Gated activation (Tanh * Sigmoid).
    """

    def __init__(self, in_channels, kernel_size, dilation, dropout=0.3):
        super(GatedTCNBlock, self).__init__()

        # Padding to keep temporal dimension same: (k-1) * d // 2 for both sides if odd kernel
        # Assuming kernel_size is odd (e.g., 3)
        padding = (kernel_size - 1) * dilation // 2

        self.filter_conv = nn.Conv1d(
            in_channels, in_channels, kernel_size, dilation=dilation, padding=padding
        )
        self.gate_conv = nn.Conv1d(
            in_channels, in_channels, kernel_size, dilation=dilation, padding=padding
        )

        self.dropout = nn.Dropout(dropout)
        self.conv_1x1 = nn.Conv1d(in_channels, in_channels, 1)
        self.se_block = SEBlock(in_channels)

    def forward(self, x):
        # x: (Batch, Channel, Time)

        # Gated Activation
        filter_out = torch.tanh(self.filter_conv(x))
        gate_out = torch.sigmoid(self.gate_conv(x))
        x_gated = filter_out * gate_out

        x_gated = self.dropout(x_gated)
        x_projected = self.conv_1x1(x_gated)

        # Channel Attention
        x_att = self.se_block(x_projected)

        # Residual Connection
        return x + x_att


class Stage1_BiGRU(nn.Module):
    """
    Stage 1: Sequence Encoder using Bi-GRU.
    Input: Early Fusion Features (Kinematics + Audio).
    Output: Class Probabilities.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.3):
        super(Stage1_BiGRU, self).__init__()

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # Bi-directional doubles the hidden size
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (Batch, Time, InputDim) - Note: RNN expects (Batch, Time, Feats)

        self.gru.flatten_parameters()
        out, _ = self.gru(x)

        # out: (Batch, Time, Hidden*2)
        logits = self.classifier(out)

        # Apply Softmax to output strictly probabilities for the next stage
        probs = F.softmax(logits, dim=2)

        return probs


class StageRefinement(nn.Module):
    """
    Refinement Stage (Stage 2 & 3).
    Input: Class Probabilities from previous stage.
    Backbone: Stacked Gated TCNs with SE Blocks.
    """

    def __init__(self, num_classes, hidden_dim, num_layers, kernel_size, dropout=0.3):
        super(StageRefinement, self).__init__()

        # Project probabilities to hidden dimension
        self.conv_in = nn.Conv1d(num_classes, hidden_dim, 1)

        # Stack of TCN blocks with increasing dilation
        layers = []
        for i in range(num_layers):
            dilation = 2**i
            layers.append(GatedTCNBlock(hidden_dim, kernel_size, dilation, dropout))
        self.layers = nn.Sequential(*layers)

        # Project back to class probabilities
        self.conv_out = nn.Conv1d(hidden_dim, num_classes, 1)

    def forward(self, x):
        # x: (Batch, Time, NumClasses) -> Needs transpose for Conv1d
        x = x.transpose(1, 2)  # (Batch, NumClasses, Time)

        out = self.conv_in(x)
        out = self.layers(out)
        out = self.conv_out(out)

        # Transpose back
        out = out.transpose(1, 2)  # (Batch, Time, NumClasses)

        # Softmax for probabilities
        probs = F.softmax(out, dim=2)

        return probs


class VIARN(nn.Module):
    """
    View-Invariant Attentive Refinement Network (VI-ARN).
    Three-Stage Cascaded Network.
    """

    def __init__(self):
        super(VIARN, self).__init__()

        # Dimensions
        # Kinematics: 20 joints * 3 coords * 3 (pos, vel, acc) = 180
        # Audio: 13 MFCCs
        self.input_dim = 180 + 13
        self.num_classes = Config.NUM_CLASSES

        # Hyperparameters
        gru_hidden = Config.GRU_HIDDEN_SIZE
        gru_layers = Config.GRU_NUM_LAYERS
        tcn_channels = Config.TCN_NUM_CHANNELS[
            0
        ]  # Assuming uniform channel size for simplicity or first element
        tcn_kernel = Config.TCN_KERNEL_SIZE
        dropout = Config.DROPOUT_RATE

        # Stage 1
        self.stage1 = Stage1_BiGRU(
            self.input_dim, gru_hidden, gru_layers, self.num_classes, dropout
        )

        # Stage 2 (Refinement)
        # Number of layers in TCN stack determined by length of TCN_NUM_CHANNELS or fixed depth
        # Usually MS-TCN uses ~10 layers. Let's use a fixed reasonable depth like 8 or 10 based on typical TCN configs
        # or derive from config if available. Config has TCN_NUM_CHANNELS=[64, 64, 64], implies 3 layers?
        # That seems shallow for TCN. Let's assume TCN_NUM_CHANNELS defines the hidden dim,
        # and we want enough layers for receptive field.
        # Let's use 8 layers (dilation 1 to 128) as a robust default for refinement.
        self.refinement_layers = 8

        self.stage2 = StageRefinement(
            self.num_classes, tcn_channels, self.refinement_layers, tcn_kernel, dropout
        )

        # Stage 3 (Refinement)
        self.stage3 = StageRefinement(
            self.num_classes, tcn_channels, self.refinement_layers, tcn_kernel, dropout
        )

    def forward(self, x):
        """
        Forward pass returning outputs from all stages for cascaded loss.
        x: (Batch, Time, InputDim)
        """
        # Stage 1
        p1 = self.stage1(x)

        # Stage 2
        p2 = self.stage2(p1)

        # Stage 3
        p3 = self.stage3(p2)

        return {"stage1": p1, "stage2": p2, "stage3": p3}
