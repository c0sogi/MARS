import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ResBlock1D(nn.Module):
    """
    1D Residual Block with Batch Normalization and ReLU.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.1):
        super(ResBlock1D, self).__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class ResUNet1D(nn.Module):
    """
    1D Residual U-Net architecture.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_dim=64,
        depth=4,
        kernel_size=3,
        dropout=0.1,
    ):
        super(ResUNet1D, self).__init__()
        self.depth = depth

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        current_channels = in_channels
        next_channels = hidden_dim

        for _ in range(depth):
            self.encoder_blocks.append(
                ResBlock1D(current_channels, next_channels, kernel_size, dropout)
            )
            self.downsamples.append(nn.MaxPool1d(2))
            current_channels = next_channels
            next_channels *= 2

        # Bottleneck
        self.bottleneck = ResBlock1D(
            current_channels, next_channels, kernel_size, dropout
        )
        current_channels = next_channels

        # Decoder
        self.up_convs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        for _ in range(depth):
            out_ch = current_channels // 2
            self.up_convs.append(
                nn.ConvTranspose1d(current_channels, out_ch, kernel_size=2, stride=2)
            )
            # Input to decoder block will be cat(skip, upsampled) -> out_ch * 2
            self.decoder_blocks.append(
                ResBlock1D(out_ch * 2, out_ch, kernel_size, dropout)
            )
            current_channels = out_ch

        # Final Output Layer
        self.final_conv = nn.Conv1d(current_channels, out_channels, kernel_size=1)

    def forward(self, x):
        skips = []

        # Encoder
        for i in range(self.depth):
            x = self.encoder_blocks[i](x)
            skips.append(x)
            x = self.downsamples[i](x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for i in range(self.depth):
            x = self.up_convs[i](x)
            # Handle potential size mismatch due to odd input dimensions (though Config.SEQUENCE_LENGTH=128 is safe)
            skip = skips[-(i + 1)]
            if x.shape[2] != skip.shape[2]:
                x = F.interpolate(
                    x, size=skip.shape[2], mode="linear", align_corners=False
                )

            x = torch.cat([skip, x], dim=1)
            x = self.decoder_blocks[i](x)

        return self.final_conv(x)


class CascadedResUNet(nn.Module):
    """
    Cascaded 1D Residual U-Net.
    Stage 1: Coarse estimation.
    Stage 2: Refinement based on input + Stage 1 output.
    """

    def __init__(self):
        super(CascadedResUNet, self).__init__()

        # Hyperparameters from Config
        input_channels = Config.INPUT_CHANNELS
        hidden_dim = Config.HIDDEN_DIM
        depth = Config.DEPTH
        kernel_size = Config.KERNEL_SIZE
        dropout = Config.DROPOUT

        # Stage 1 Network
        self.stage1 = ResUNet1D(
            in_channels=input_channels,
            out_channels=2,  # East, North residuals
            hidden_dim=hidden_dim,
            depth=depth,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # Stage 2 Network
        # Input: Original Features (input_channels) + Stage 1 Output (2)
        self.stage2 = ResUNet1D(
            in_channels=input_channels + 2,
            out_channels=2,  # Refinement residuals
            hidden_dim=hidden_dim,
            depth=depth,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Channels, Length)
        Returns:
            out1: Output from Stage 1 (Coarse)
            final_out: Final output (Stage 1 + Stage 2)
        """
        # Stage 1 Inference
        out1 = self.stage1(x)

        # Prepare input for Stage 2 (Concatenation)
        # Detach out1? No, we want gradients to flow back through Stage 1 if using joint loss.
        # However, for stability, sometimes it's better to treat Stage 1 as fixed input to Stage 2
        # regarding the Stage 2 loss component, but standard cascaded training usually allows end-to-end.
        # Given the task description implies a single training loop, we keep gradients connected.
        x2 = torch.cat([x, out1], dim=1)

        # Stage 2 Inference
        out2 = self.stage2(x2)

        # Final Prediction is additive
        final_out = out1 + out2

        return out1, final_out
