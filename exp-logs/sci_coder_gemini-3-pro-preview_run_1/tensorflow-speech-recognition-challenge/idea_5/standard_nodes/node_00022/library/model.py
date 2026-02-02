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


class DilatedConvNeXt(nn.Module):
    def __init__(self, config=Config):
        super().__init__()
        self.num_classes = config.NUM_CLASSES

        # 1. Load Pretrained Backbone
        # in_chans=1 automatically adapts the first conv layer weights
        # global_pool='' ensures we get the feature map (B, C, H, W) instead of a vector
        self.backbone = timm.create_model(
            config.BACKBONE,
            pretrained=config.PRETRAINED,
            num_classes=0,
            in_chans=config.IN_CHANNELS,
            global_pool="",
        )

        # 2. Modify Stages for Dilation
        # ConvNeXt has 4 stages (0, 1, 2, 3).
        # Standard downsampling: 4 (stem) * 1 (s0) * 2 (s1) * 2 (s2) * 2 (s3) = 32x
        # Target downsampling: 8x (preserve time resolution)
        # We modify Stage 2 and Stage 3 to have stride 1 and use dilation.

        # Modify Stage 2 (Index 2): Stride 1, Dilation 2
        self._modify_stage(self.backbone.stages[2], stride=1, dilation=2)

        # Modify Stage 3 (Index 3): Stride 1, Dilation 4
        self._modify_stage(self.backbone.stages[3], stride=1, dilation=4)

        # 3. Determine Feature Dimension
        # Run a dummy forward pass to get the output channel count
        with torch.no_grad():
            dummy = torch.randn(1, config.IN_CHANNELS, 224, 224)
            features = self.backbone(dummy)
            # features shape: (1, C, H, W)
            self.embed_dim = features.shape[1]

        # 4. Pooling and Classification Head
        self.pool = AttentivePooling(self.embed_dim)
        self.fc = nn.Linear(self.embed_dim, self.num_classes)

    def _modify_stage(self, stage, stride, dilation):
        """
        Modifies a ConvNeXt stage to use a specific stride and dilation.
        """
        # A. Modify Downsample Layer (if it exists)
        # In ConvNeXt, downsample is usually a Sequential(LayerNorm, Conv2d)
        if stage.downsample is not None:
            for module in stage.downsample.modules():
                if isinstance(module, nn.Conv2d):
                    module.stride = (stride, stride)

        # B. Modify Blocks (Depthwise Convolutions)
        # Each block is a ConvNeXtBlock containing a depthwise conv 'conv_dw'
        for block in stage.blocks:
            # Calculate new padding to maintain size with dilation
            # Padding = (KernelSize - 1) * Dilation / 2
            # ConvNeXt kernel size is 7
            kernel_size = 7
            new_padding = (kernel_size - 1) * dilation // 2

            # Apply changes
            block.conv_dw.dilation = (dilation, dilation)
            block.conv_dw.padding = (new_padding, new_padding)

    def forward(self, x):
        # x: (Batch, 1, Freq, Time)

        # 1. Backbone Feature Extraction
        x = self.backbone(x)  # Output: (Batch, Channels, F', T')

        # 2. Frequency Pooling
        # We average over the frequency dimension (H), preserving Time (W)
        x = x.mean(dim=2)  # Output: (Batch, Channels, T')

        # 3. Attentive Temporal Pooling
        x = self.pool(x)  # Output: (Batch, Channels)

        # 4. Classification
        x = self.fc(x)  # Output: (Batch, NumClasses)

        return x


def get_model(config=Config):
    return DilatedConvNeXt(config)
