import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class AttentivePooling(nn.Module):
    """
    Attentive Pooling Layer.
    Computes a weighted sum of the feature map based on a learned attention map.
    Allows the model to focus on active speech segments in the time-frequency domain.
    """

    def __init__(self, in_channels):
        super(AttentivePooling, self).__init__()
        # 1x1 Conv to calculate attention scores from features
        self.att_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        # x shape: (Batch, Channels, Freq, Time)

        # 1. Calculate raw attention scores
        attn_logits = self.att_conv(x)  # (B, 1, F, T)

        # 2. Flatten spatial dimensions for Softmax
        b, _, h, w = attn_logits.size()
        attn_logits = attn_logits.view(b, -1)  # (B, F*T)

        # 3. Compute Attention Weights
        attn_weights = F.softmax(attn_logits, dim=1)  # (B, F*T)

        # 4. Reshape back to spatial dimensions
        attn_weights = attn_weights.view(b, 1, h, w)  # (B, 1, F, T)

        # 5. Weighted Sum Pooling
        # Multiply features by attention weights and sum over spatial dims
        pooled = (x * attn_weights).sum(dim=(2, 3))  # (B, C)

        return pooled


class DilatedEfficientNet(nn.Module):
    """
    Dilated EfficientNet-B2 with Attentive Pooling.

    Modifications:
    1. Input Conv: Adapted for 1-channel input (weights averaged).
    2. Final Stage: Stride=1, Dilation=2 to preserve temporal resolution.
    3. Head: Attentive Pooling + Linear Classifier.
    """

    def __init__(self, config=Config):
        super(DilatedEfficientNet, self).__init__()
        self.num_classes = config.num_classes

        # Load Pretrained Backbone
        # num_classes=0 removes the FC layer
        # global_pool='' removes the default pooling
        self.backbone = timm.create_model(
            config.backbone,
            pretrained=True,
            num_classes=0,
            global_pool="",
            drop_path_rate=0.2,
        )

        # 1. Adapt Input Layer (3 channels -> 1 channel)
        self._adapt_input_conv(config.in_channels)

        # 2. Apply Dilated Convolutions in the Final Stage
        self._modify_last_stage_dilation()

        # Get feature dimension (EfficientNet-B2 usually 1408)
        self.num_features = self.backbone.num_features

        # 3. Define Head
        self.att_pool = AttentivePooling(self.num_features)
        self.fc = nn.Linear(self.num_features, self.num_classes)

    def _adapt_input_conv(self, in_channels):
        """
        Replaces the first convolution layer to accept 'in_channels' (1).
        Initializes weights by averaging the original RGB weights.
        """
        original_conv = self.backbone.conv_stem

        new_conv = nn.Conv2d(
            in_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

        # Average weights along the channel dimension
        with torch.no_grad():
            new_conv.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)

        self.backbone.conv_stem = new_conv

    def _modify_last_stage_dilation(self):
        """
        Modifies the last stage of EfficientNet to use dilated convolutions.
        Sets stride=1 and dilation=2 to prevent excessive downsampling.
        """
        # EfficientNet blocks are stored in a Sequential of Sequentials
        # blocks[-1] refers to the last stage
        last_stage = self.backbone.blocks[-1]

        for block in last_stage:
            # We target the depthwise convolution in MBConv blocks
            if hasattr(block, "conv_dw"):
                # 1. Remove Downsampling: If stride is 2, set to 1
                if block.conv_dw.stride == (2, 2):
                    block.conv_dw.stride = (1, 1)

                # 2. Apply Dilation
                dilation = 2
                block.conv_dw.dilation = (dilation, dilation)

                # 3. Adjust Padding
                # Padding must be increased to accommodate dilation while keeping dimensions 'same'
                # Formula: p = (kernel_size - 1) * dilation // 2
                kernel_size = block.conv_dw.kernel_size[0]
                padding = ((kernel_size - 1) * dilation) // 2
                block.conv_dw.padding = (padding, padding)

    def forward(self, x):
        # x: (Batch, 1, Freq, Time)

        # Extract features
        features = self.backbone(x)  # (Batch, C, H, W)

        # Attentive Pooling
        pooled = self.att_pool(features)  # (Batch, C)

        # Classification
        logits = self.fc(pooled)  # (Batch, Num_Classes)

        return logits
