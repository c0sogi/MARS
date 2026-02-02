import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for Channel Attention.
    Recalibrates channel-wise feature responses by explicitly modelling
    interdependencies between channels.
    """

    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # Ensure reduction doesn't make hidden dim 0
        reduced_dim = max(1, channels // reduction)

        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (Batch, Channels, Time)
        b, c, t = x.size()

        # Squeeze: Global Average Pooling -> (Batch, Channels, 1)
        y = self.avg_pool(x).view(b, c)

        # Excitation: FC layers -> (Batch, Channels)
        y = self.fc(y).view(b, c, 1)

        # Scale: Channel-wise multiplication
        return x * y


class GatedTCNBlock(nn.Module):
    """
    Gated Dilated Temporal Convolutional Block with SE Attention.
    Uses dilated convolutions to expand receptive field and gated activations
    to control information flow.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(GatedTCNBlock, self).__init__()

        # Padding to maintain temporal dimension (Same padding)
        # Assuming kernel_size is odd (Config.TCN_KERNEL_SIZE is 3)
        self.padding = (kernel_size - 1) * dilation // 2

        # Dilated Convolution
        # Output channels is 2 * out_channels for Gated Activation (Value + Gate)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels * 2,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

        self.dropout = nn.Dropout(dropout)
        self.se = SEBlock(out_channels)

        # 1x1 Conv for residual connection if dimensions change
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        # x: (Batch, In_Channels, Time)
        residual = x

        # Dilated Conv
        out = self.conv(x)

        # Gated Activation Unit
        # Split into Value (P) and Gate (Q)
        p, q = torch.chunk(out, 2, dim=1)
        out = torch.tanh(p) * torch.sigmoid(q)

        out = self.dropout(out)

        # Channel Attention
        out = self.se(out)

        # Residual Connection
        if self.downsample is not None:
            residual = self.downsample(residual)

        return out + residual


class BiGRUEncoder(nn.Module):
    """
    Stage 1: Sequence Encoder using Bi-Directional GRU.
    Processes raw features (Skeleton + Audio) into initial class probabilities.
    """

    def __init__(self):
        super(BiGRUEncoder, self).__init__()

        self.gru = nn.GRU(
            input_size=Config.INPUT_DIM,
            hidden_size=Config.GRU_HIDDEN_DIM,
            num_layers=Config.GRU_NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
        )

        # Projection to Class Probabilities
        # Input to FC is hidden_dim * 2 (Bidirectional)
        self.fc = nn.Linear(Config.GRU_HIDDEN_DIM * 2, Config.NUM_CLASSES)

    def forward(self, x):
        # x: (Batch, Channels/Features, Time)
        # GRU expects (Batch, Time, Features)
        x = x.permute(0, 2, 1)

        self.gru.flatten_parameters()
        outputs, _ = self.gru(x)

        # Project to classes
        # outputs: (Batch, Time, Hidden*2)
        logits = self.fc(outputs)

        # Permute back to (Batch, Classes, Time) for TCN stages
        return logits.permute(0, 2, 1)


class RefinementStage(nn.Module):
    """
    Stage 2 & 3: Attentive Gated Refinement.
    Refines class probabilities using stacked Gated TCN blocks.
    """

    def __init__(self):
        super(RefinementStage, self).__init__()

        layers = []
        num_classes = Config.NUM_CLASSES
        channel_sizes = Config.TCN_NUM_CHANNELS  # e.g. [64, 64, 64]
        kernel_size = Config.TCN_KERNEL_SIZE
        dropout = Config.TCN_DROPOUT

        # 1. Projection Layer: Probabilities -> Hidden Dim
        # Input is previous stage probabilities (num_classes)
        self.entry_conv = nn.Conv1d(num_classes, channel_sizes[0], 1)

        # 2. Stacked Gated TCN Blocks
        # We iterate through the channel configuration.
        # Dilation increases as 2^i
        for i, out_channels in enumerate(channel_sizes):
            in_channels = channel_sizes[i - 1] if i > 0 else channel_sizes[0]
            dilation = 2**i

            layers.append(
                GatedTCNBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

        self.tcn_layers = nn.Sequential(*layers)

        # 3. Output Layer: Hidden Dim -> Class Probabilities
        self.exit_conv = nn.Conv1d(channel_sizes[-1], num_classes, 1)

    def forward(self, x):
        # x: (Batch, Num_Classes, Time) - Logits or Probs from previous stage
        # Note: Usually refinement stages take Softmax probabilities as input

        out = self.entry_conv(x)
        out = self.tcn_layers(out)
        out = self.exit_conv(out)

        return out


class VIARN(nn.Module):
    """
    View-Invariant Attentive Refinement Network (VI-ARN).
    Three-stage cascaded architecture with Deep Supervision.
    """

    def __init__(self):
        super(VIARN, self).__init__()

        self.stage1 = BiGRUEncoder()
        self.stage2 = RefinementStage()
        self.stage3 = RefinementStage()

    def forward(self, x):
        """
        Args:
            x: Input features (Batch, Input_Dim, Time)

        Returns:
            out1, out2, out3: Logits from each stage (Batch, Num_Classes, Time)
        """
        # Stage 1: Encoder
        out1 = self.stage1(x)

        # Stage 2: Refinement
        # Input is Softmax probabilities of Stage 1
        probs1 = F.softmax(out1, dim=1)
        out2 = self.stage2(probs1)

        # Stage 3: Refinement
        # Input is Softmax probabilities of Stage 2
        probs2 = F.softmax(out2, dim=1)
        out3 = self.stage3(probs2)

        return out1, out2, out3
