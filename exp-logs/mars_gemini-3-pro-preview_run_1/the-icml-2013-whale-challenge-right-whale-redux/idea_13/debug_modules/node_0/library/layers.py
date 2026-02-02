import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention for Efficient Mobile Network Design.
    Decomposes channel attention into two 1D feature encoding processes to preserve
    positional information.
    """

    def __init__(self, inp, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # Pool along width -> (N, C, H, 1)
        x_h = self.pool_h(x)
        # Pool along height -> (N, C, 1, W) -> Permute to (N, C, W, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # Concatenate along the spatial dimension (H + W)
        y = torch.cat([x_h, x_w], dim=2)

        # Shared 1x1 Conv + BN + Non-linearity
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split back into H and W components
        x_h, x_w = torch.split(y, [h, w], dim=2)

        # Permute x_w back to (N, C, 1, W)
        x_w = x_w.permute(0, 1, 3, 2)

        # Separate 1x1 Convs + Sigmoid to generate attention maps
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # Apply attention
        out = identity * a_h * a_w
        return out


class SEBlock1D(nn.Module):
    """
    Squeeze-and-Excitation Block for 1D sequences (N, Channels, Time).
    Recalibrates channel importance.
    """

    def __init__(self, channel, reduction=16):
        super(SEBlock1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (N, C, T)
        b, c, _ = x.size()
        # Global Average Pooling -> (N, C)
        y = self.avg_pool(x).view(b, c)
        # Excitation -> (N, C, 1)
        y = self.fc(y).view(b, c, 1)
        # Scale
        return x * y


class AdaptiveSpectralFusion(nn.Module):
    """
    Hierarchically pools features from different backbone layers to specific frequency bins,
    concatenates them, and fuses them via a bottleneck and SE block.
    """

    def __init__(self, in_channels_list, pool_bins_list, fusion_channels=512):
        """
        Args:
            in_channels_list (list[int]): List of input channel counts for each layer (e.g., [128, 256, 512]).
            pool_bins_list (list[int]): List of frequency bins to pool to for each layer (e.g., [4, 2, 1]).
            fusion_channels (int): Number of channels after the bottleneck 1x1 conv.
        """
        super(AdaptiveSpectralFusion, self).__init__()

        assert len(in_channels_list) == len(
            pool_bins_list
        ), "Channels and bins lists must match length."

        self.pool_bins_list = pool_bins_list

        # Calculate total channels after flattening frequency bins
        # Example: 128*4 + 256*2 + 512*1 = 512 + 512 + 512 = 1536
        total_input_channels = 0
        for ch, bins in zip(in_channels_list, pool_bins_list):
            total_input_channels += ch * bins

        self.bottleneck = nn.Sequential(
            nn.Conv1d(total_input_channels, fusion_channels, kernel_size=1),
            nn.BatchNorm1d(fusion_channels),
            nn.ReLU(inplace=True),
        )

        self.se = SEBlock1D(fusion_channels, reduction=16)

    def forward(self, features):
        """
        Args:
            features (list[torch.Tensor]): List of 4D tensors [(N, C, F, T), ...].
        Returns:
            torch.Tensor: Fused features of shape (N, fusion_channels, T).
        """
        processed_feats = []

        # Iterate over features and corresponding bin counts
        for i, feat in enumerate(features):
            # feat: (N, C, F, T)
            n, c, f, t = feat.size()
            target_bins = self.pool_bins_list[i]

            # Adaptive Average Pooling along Frequency (H) dimension
            # We want output (N, C, target_bins, T)
            # adaptive_avg_pool2d takes output_size=(H_out, W_out)
            # Here H corresponds to Frequency, W to Time. We preserve Time (t).
            pooled = F.adaptive_avg_pool2d(feat, (target_bins, t))

            # Flatten Frequency bins into Channel dimension
            # (N, C, target_bins, T) -> (N, C * target_bins, T)
            flattened = pooled.view(n, c * target_bins, t)
            processed_feats.append(flattened)

        # Concatenate all processed features along channel dimension
        # (N, Total_Channels, T)
        concatenated = torch.cat(processed_feats, dim=1)

        # Apply Bottleneck Fusion (1x1 Conv1d)
        fused = self.bottleneck(concatenated)

        # Apply SE Attention
        out = self.se(fused)

        return out


class AttentionPooling(nn.Module):
    """
    Attention-based pooling layer to aggregate temporal sequence.
    Learns a weight for each time step to focus on the call.
    """

    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input sequence of shape (N, T, C).
        Returns:
            torch.Tensor: Aggregated embedding of shape (N, C).
        """
        # Calculate attention weights: (N, T, 1)
        weights = self.attention(x)

        # Weighted sum: (N, T, C) * (N, T, 1) -> sum over T -> (N, C)
        out = torch.sum(x * weights, dim=1)

        return out
