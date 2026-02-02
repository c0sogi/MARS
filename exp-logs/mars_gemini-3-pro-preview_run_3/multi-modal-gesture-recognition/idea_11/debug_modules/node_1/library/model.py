import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for 1D Temporal Data.
    Recalibrates channel-wise feature responses by explicitly modelling
    interdependencies between channels.
    """

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, t = x.size()
        # Squeeze: Global Information Embedding
        y = self.avg_pool(x).view(b, c)
        # Excitation: Adaptive Recalibration
        y = self.fc(y).view(b, c, 1)
        # Scale
        return x * y.expand_as(x)


class GatedDilatedTemporalBlock(nn.Module):
    """
    A single block for the TCN Refinement Stage.
    Contains:
    - Dilated Convolution
    - Gated Activation (Tanh * Sigmoid)
    - Squeeze-and-Excitation
    - 1x1 Convolution
    - Residual Connection
    - Dropout
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.3):
        super(GatedDilatedTemporalBlock, self).__init__()

        # We need 2 * out_channels for the Gated Activation (Filter + Gate)
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels * 2,
            kernel_size,
            padding=(kernel_size - 1) * dilation // 2,  # "Same" padding for dilation
            dilation=dilation,
        )

        self.se = SEBlock(out_channels)
        self.conv2 = nn.Conv1d(out_channels, in_channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        # Dilated Conv
        out = self.conv1(x)

        # Gated Activation
        filter_out, gate_out = out.chunk(2, dim=1)
        out = torch.tanh(filter_out) * torch.sigmoid(gate_out)

        # Attention
        out = self.se(out)

        # Projection back to residual dim
        out = self.conv2(out)
        out = self.dropout(out)

        return out + residual


class RefinementStage(nn.Module):
    """
    Stage 2 & 3: Attentive Boundary-Informed Refinement.
    Takes probability maps as input (Information Bottleneck), refines them
    using stacked Gated Dilated TCN blocks, and outputs refined probabilities.
    """

    def __init__(self, num_layers, num_f_maps, input_dim, num_classes):
        super(RefinementStage, self).__init__()

        # Input Projection: Probabilities -> Feature Maps
        self.conv_in = nn.Conv1d(input_dim, num_f_maps, 1)

        # Stacked Dilated Blocks
        layers = []
        for i in range(num_layers):
            dilation = 2**i
            layers.append(
                GatedDilatedTemporalBlock(
                    num_f_maps,
                    num_f_maps,
                    kernel_size=3,
                    dilation=dilation,
                    dropout=Config.DROPOUT,
                )
            )
        self.layers = nn.Sequential(*layers)

        # Output Heads
        self.conv_cls = nn.Conv1d(num_f_maps, num_classes, 1)
        self.conv_bnd = nn.Conv1d(num_f_maps, 1, 1)

    def forward(self, x):
        # x shape: (Batch, Channels, Time)
        out = self.conv_in(x)
        out = self.layers(out)

        cls_out = self.conv_cls(out)
        bnd_out = self.conv_bnd(out)

        return cls_out, bnd_out


class BiGRUEncoder(nn.Module):
    """
    Stage 1: Multi-Task Sequence Encoder.
    Uses Bi-GRU to process raw features and produce initial predictions.
    """

    def __init__(self, input_dim, hidden_dim, num_classes):
        super(BiGRUEncoder, self).__init__()

        self.project = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=Config.DROPOUT,
        )

        # Dual Heads
        # Input to heads is hidden_dim * 2 (bidirectional)
        self.fc_cls = nn.Linear(hidden_dim * 2, num_classes)
        self.fc_bnd = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        # x shape: (Batch, Time, InputDim)
        x = F.relu(self.project(x))

        # GRU Output: (Batch, Time, HiddenDim * 2)
        out, _ = self.gru(x)

        cls_out = self.fc_cls(out)
        bnd_out = self.fc_bnd(out)

        # Permute to (Batch, Channels, Time) for compatibility with TCN stages
        cls_out = cls_out.permute(0, 2, 1)
        bnd_out = bnd_out.permute(0, 2, 1)

        return cls_out, bnd_out


class BA_AKN(nn.Module):
    """
    Boundary-Aware Attentive Kinematic Network.
    A Three-Stage Multi-Task Network.
    """

    def __init__(self):
        super(BA_AKN, self).__init__()

        self.num_classes = Config.NUM_CLASSES

        # Stage 1: Bi-GRU Encoder
        self.stage1 = BiGRUEncoder(
            input_dim=Config.INPUT_DIM,
            hidden_dim=Config.HIDDEN_DIM,
            num_classes=self.num_classes,
        )

        # Information Bottleneck Dimension: Classes + Boundary
        self.bottleneck_dim = self.num_classes + 1

        # Stage 2: Refinement
        self.stage2 = RefinementStage(
            num_layers=Config.NUM_LAYERS,
            num_f_maps=Config.NUM_F_MAPS,
            input_dim=self.bottleneck_dim,
            num_classes=self.num_classes,
        )

        # Stage 3: Refinement
        self.stage3 = RefinementStage(
            num_layers=Config.NUM_LAYERS,
            num_f_maps=Config.NUM_F_MAPS,
            input_dim=self.bottleneck_dim,
            num_classes=self.num_classes,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, Time, InputDim)
        Returns:
            dict: Outputs from all stages for Deep Supervision.
        """
        outputs = {}

        # --- Stage 1 ---
        s1_cls, s1_bnd = self.stage1(x)
        outputs["stage1_cls"] = s1_cls
        outputs["stage1_bnd"] = s1_bnd

        # Prepare input for Stage 2 (Softmax/Sigmoid + Concat)
        # Detach is NOT used here; we want gradients to flow back through probabilities
        # to enforce the bottleneck representation.
        s1_cls_prob = F.softmax(s1_cls, dim=1)
        s1_bnd_prob = torch.sigmoid(s1_bnd)
        s2_in = torch.cat([s1_cls_prob, s1_bnd_prob], dim=1)

        # --- Stage 2 ---
        s2_cls, s2_bnd = self.stage2(s2_in)
        outputs["stage2_cls"] = s2_cls
        outputs["stage2_bnd"] = s2_bnd

        # Prepare input for Stage 3
        s2_cls_prob = F.softmax(s2_cls, dim=1)
        s2_bnd_prob = torch.sigmoid(s2_bnd)
        s3_in = torch.cat([s2_cls_prob, s2_bnd_prob], dim=1)

        # --- Stage 3 ---
        s3_cls, s3_bnd = self.stage3(s3_in)
        outputs["stage3_cls"] = s3_cls
        outputs["stage3_bnd"] = s3_bnd

        return outputs
