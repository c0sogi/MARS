import torch
import torch.nn as nn
from library import config


class ECA_Layer(nn.Module):
    """
    Efficient Channel Attention (ECA) Module.
    Performs a 1D convolution across the channel dimension to model local cross-channel interactions
    without dimensionality reduction.
    """

    def __init__(self, channels, kernel_size=3):
        super(ECA_Layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 1D convolution: input channels=1, output channels=1, kernel size=k
        # We treat the channel dimension C as the sequence length L.
        # Padding is calculated to maintain the same sequence length (C).
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (N, C, H, W)
        y = self.avg_pool(x)  # (N, C, 1, 1)

        # Reshape for Conv1d: (N, 1, C)
        # Squeeze the spatial dims, then unsqueeze dim 1 to be the 'input channel' for Conv1d
        y = y.squeeze(-1).transpose(-1, -2)  # (N, 1, C)

        y = self.conv(y)  # (N, 1, C)

        # Reshape back to (N, C, 1, 1)
        y = y.transpose(-1, -2).unsqueeze(-1)

        y = self.sigmoid(y)

        return x * y.expand_as(x)


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block for EAP-CNN.
    Structure: Conv2d (bias=True) -> BatchNorm2d -> LeakyReLU -> ECA -> MaxPool2d
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        pool_size=2,
        leaky_slope=0.1,
        eca_k=3,
    ):
        super(ConvBlock, self).__init__()

        # Padding to maintain spatial dimensions before pooling
        padding = (kernel_size - 1) // 2

        # Bias is explicitly retained as per architectural requirements
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
        self.eca = ECA_Layer(out_channels, kernel_size=eca_k)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.eca(x)
        x = self.pool(x)
        return x


class EAP_CNN(nn.Module):
    """
    Efficient-Attentive Plain CNN (EAP-CNN).
    A 4-stage CNN with ECA modules and Global Max Pooling.
    """

    def __init__(self):
        super(EAP_CNN, self).__init__()

        # Configuration
        in_channels = config.IN_CHANNELS
        channels = config.MODEL_CHANNELS  # [64, 128, 128, 128]
        slope = config.LEAKY_RELU_SLOPE
        eca_k = config.ECA_KERNEL_SIZE
        dropout_rate = config.DROPOUT_RATE

        # 4 Sequential Blocks
        self.block1 = ConvBlock(
            in_channels, channels[0], leaky_slope=slope, eca_k=eca_k
        )
        self.block2 = ConvBlock(
            channels[0], channels[1], leaky_slope=slope, eca_k=eca_k
        )
        self.block3 = ConvBlock(
            channels[1], channels[2], leaky_slope=slope, eca_k=eca_k
        )
        self.block4 = ConvBlock(
            channels[2], channels[3], leaky_slope=slope, eca_k=eca_k
        )

        # Global Max Pooling
        self.global_pool = nn.AdaptiveMaxPool2d(1)

        # Classification Head
        # Input: Final channel count + 1 (incidence angle)
        head_in_dim = channels[-1] + 1
        hidden_dim = 256  # Standard hidden size for this feature scale

        self.head = nn.Sequential(
            nn.Linear(head_in_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, inc_angle):
        # x: (N, 3, 75, 75)
        # inc_angle: (N,) or (N, 1)

        # Feature Extraction
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Global Max Pooling -> (N, C, 1, 1) -> (N, C)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Feature Fusion
        # Ensure inc_angle is (N, 1)
        inc_angle = inc_angle.view(-1, 1)
        x = torch.cat([x, inc_angle], dim=1)

        # Classification Head
        x = self.head(x)

        return x
