import torch
import torch.nn as nn
import timm
from library.config import Config


class AttentivePooling(nn.Module):
    """
    Attentive Pooling module for temporal data.
    Input: (Batch, Channels, Time)
    Output: (Batch, Channels)
    """

    def __init__(self, in_channels):
        super().__init__()
        # 1x1 Conv to project channels to a single attention score per time step
        self.attn_conv = nn.Conv1d(in_channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: (B, C, T)

        # Calculate attention scores
        # (B, 1, T)
        attn_scores = self.attn_conv(x)

        # Normalize scores across time dimension
        attn_weights = self.softmax(attn_scores)

        # Weighted sum: (B, C, T) * (B, 1, T) -> (B, C, T) -> sum(dim=-1) -> (B, C)
        x_pooled = (x * attn_weights).sum(dim=-1)

        return x_pooled


class DilatedEfficientNet(nn.Module):
    def __init__(self, config=Config):
        super().__init__()
        self.num_classes = config.NUM_CLASSES

        # 1. Load Pretrained Backbone
        # Load with 3 channels first to get original pretrained weights
        # output_stride=16 enables dilated convolutions in the final stages (Cite Lesson 17)
        self.backbone = timm.create_model(
            config.BACKBONE,
            pretrained=config.PRETRAINED,
            num_classes=0,
            in_chans=3,
            global_pool="",
            output_stride=16,
        )

        # 2. Manual Weight Averaging for 1-Channel Input (Cite Lesson 16)
        if config.IN_CHANNELS == 1:
            if hasattr(self.backbone, "conv_stem"):
                # EfficientNet style
                old_layer = self.backbone.conv_stem
                new_layer = nn.Conv2d(
                    1,
                    old_layer.out_channels,
                    kernel_size=old_layer.kernel_size,
                    stride=old_layer.stride,
                    padding=old_layer.padding,
                    bias=old_layer.bias is not None,
                )
                new_layer.weight.data = old_layer.weight.data.mean(dim=1, keepdim=True)
                if old_layer.bias is not None:
                    new_layer.bias.data = old_layer.bias.data
                self.backbone.conv_stem = new_layer
            elif hasattr(self.backbone, "stem"):
                # ConvNeXt style
                old_layer = self.backbone.stem[0]
                new_layer = nn.Conv2d(
                    1,
                    old_layer.out_channels,
                    kernel_size=old_layer.kernel_size,
                    stride=old_layer.stride,
                    padding=old_layer.padding,
                    bias=old_layer.bias is not None,
                )
                new_layer.weight.data = old_layer.weight.data.mean(dim=1, keepdim=True)
                if old_layer.bias is not None:
                    new_layer.bias.data = old_layer.bias.data
                self.backbone.stem[0] = new_layer
            else:
                raise AttributeError(
                    f"Unsupported backbone for 1-channel input: {config.BACKBONE}"
                )

        # 3. Determine Feature Dimension
        with torch.no_grad():
            dummy = torch.randn(1, config.IN_CHANNELS, 224, 224)
            features = self.backbone(dummy)
            self.embed_dim = features.shape[1]

        # 4. Pooling and Classification Head
        self.pool = AttentivePooling(self.embed_dim)
        self.fc = nn.Linear(self.embed_dim, self.num_classes)

    def forward(self, x):
        # x: (Batch, 1, Freq, Time)

        # 1. Backbone Feature Extraction
        x = self.backbone(x)  # Output: (Batch, Channels, F', T')

        # 2. Frequency Pooling
        x = x.mean(dim=2)  # Output: (Batch, Channels, T')

        # 3. Attentive Temporal Pooling (Cite Lesson 17)
        x = self.pool(x)  # Output: (Batch, Channels)

        # 4. Classification
        x = self.fc(x)  # Output: (Batch, NumClasses)

        return x


def get_model(config=Config):
    return DilatedEfficientNet(config)
