import torch
import torch.nn as nn
import timm
from library.config import Config


class AttentivePooling(nn.Module):
    """
    Attentive Pooling module.
    Learns a spatial attention map to weight the feature map before aggregation.
    Formula: y = sum(x_i * softmax(tanh(W * x_i + b)))
    """

    def __init__(self, in_channels):
        super(AttentivePooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1), nn.Tanh()
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: [Batch, Channels, Height, Width]
        b, c, h, w = x.size()

        # Calculate attention scores
        # [B, 1, H, W]
        attn = self.attention(x)

        # Flatten spatial dimensions for softmax over the spatial map
        # [B, 1, H*W]
        attn = attn.view(b, 1, h * w)
        attn = self.softmax(attn)

        # Flatten input features
        # [B, C, H*W]
        x_flat = x.view(b, c, h * w)

        # Weighted sum: sum(x * attn)
        # bmm: [B, C, H*W] @ [B, H*W, 1] -> [B, C, 1]
        out = torch.bmm(x_flat, attn.transpose(1, 2)).squeeze(2)

        return out


class DilatedEfficientNet(nn.Module):
    """
    Dilated EfficientNet-B2 with Attentive Pooling.

    Modifications:
    1. First Conv adapted for 1-channel input (weights averaged).
    2. Final stage uses Dilated Convolutions (stride=1, dilation=2) via output_stride=16.
    3. Global Average Pooling replaced with Attentive Pooling.
    """

    def __init__(self, config: Config):
        super(DilatedEfficientNet, self).__init__()
        self.config = config

        # Determine output stride
        # Standard EfficientNet has stride 32 (5 reductions).
        # We want to preserve resolution in the final stage, so we use stride 16.
        # This implies the last stage (stride 2) becomes stride 1 with dilation.
        output_stride = 16 if config.use_dilated_conv else 32

        # Load Pretrained Backbone
        # We load with in_chans=3 initially to get the RGB weights,
        # then we will manually adapt the first layer to average them.
        self.backbone = timm.create_model(
            config.model_name,
            pretrained=config.pretrained,
            in_chans=3,
            num_classes=0,  # Remove default classifier
            global_pool="",  # Remove default pooling
            output_stride=output_stride,
        )

        # Adapt first convolution for 1-channel input (Spectrogram)
        if config.in_channels == 1:
            self._adapt_first_conv()

        # Feature dimension
        self.num_features = self.backbone.num_features

        # Pooling Head
        if config.use_attentive_pooling:
            self.pool = AttentivePooling(self.num_features)
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.classifier = nn.Linear(self.num_features, config.num_classes)

    def _adapt_first_conv(self):
        """
        Replaces the first convolutional layer (3 channels) with a 1-channel version.
        Weights are initialized by averaging the original RGB weights.
        """
        # In EfficientNet, the first layer is named 'conv_stem'
        old_conv = self.backbone.conv_stem

        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # Average weights: [Out, 3, K, K] -> [Out, 1, K, K]
        with torch.no_grad():
            new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
            if old_conv.bias is not None:
                new_conv.bias.data = old_conv.bias.data

        self.backbone.conv_stem = new_conv

    def forward(self, x):
        # x shape: [Batch, 1, F, T] (Spectrogram)

        # Extract features
        # Shape: [Batch, C, F', T']
        features = self.backbone(x)

        # Pooling
        if not self.config.use_attentive_pooling:
            # Standard GAP
            out = self.pool(features).flatten(1)
        else:
            # Attentive Pooling
            out = self.pool(features)

        # Classification
        logits = self.classifier(out)

        return logits
