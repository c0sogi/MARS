import torch
import torch.nn as nn
import timm
from library.config import Config


class AttentivePooling(nn.Module):
    """
    Attentive Pooling module that dynamically weights relevant features
    in the spatial (Time-Frequency) dimensions.
    """

    def __init__(self, in_channels):
        super(AttentivePooling, self).__init__()
        # Attention mechanism:
        # 1. Project to hidden dim
        # 2. Non-linearity
        # 3. Project to 1 channel (attention score)
        self.att_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Tanh(),
            nn.Conv2d(in_channels, 1, kernel_size=1),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.size()

        # Compute attention scores
        # (B, 1, H, W)
        att_scores = self.att_conv(x)

        # Flatten spatial dimensions for Softmax
        # (B, 1, H*W)
        att_scores = att_scores.view(b, 1, -1)
        att_weights = torch.softmax(att_scores, dim=-1)

        # Reshape back to (B, 1, H, W)
        att_weights = att_weights.view(b, 1, h, w)

        # Weighted sum
        # (B, C, H, W) * (B, 1, H, W) -> (B, C, H, W)
        weighted_features = x * att_weights

        # Sum over spatial dimensions -> (B, C)
        return torch.sum(weighted_features, dim=(2, 3))


class DilatedEfficientNet(nn.Module):
    """
    Dilated EfficientNet-B2 with Single-Channel Input.

    Architecture:
    - Backbone: EfficientNet-B2 (Pretrained)
    - Input: 1 Channel (Log-Mel Spectrogram)
    - Modification: First conv layer adapted for 1 channel.
    - Modification: Dilated Convolutions in the final stage.
    - Head: Attentive Pooling + Linear Classifier.
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=True):
        super(DilatedEfficientNet, self).__init__()

        # 1. Load Backbone
        # We disable the built-in global pooling and classifier
        self.backbone = timm.create_model(
            "efficientnet_b2", pretrained=pretrained, num_classes=0, global_pool=""
        )

        # 2. Modify Input Layer (3 channels -> 1 channel)
        self._modify_first_conv(pretrained)

        # 3. Apply Dilation to Final Stage
        self._apply_dilation_to_last_stage()

        # 4. Classification Head
        self.num_features = self.backbone.num_features
        self.att_pool = AttentivePooling(self.num_features)
        self.fc = nn.Linear(self.num_features, num_classes)

    def _modify_first_conv(self, pretrained):
        """
        Adapts the first convolution layer to accept 1 input channel.
        Initializes weights by averaging ImageNet weights.
        """
        old_stem = self.backbone.conv_stem
        new_stem = nn.Conv2d(
            in_channels=1,
            out_channels=old_stem.out_channels,
            kernel_size=old_stem.kernel_size,
            stride=old_stem.stride,
            padding=old_stem.padding,
            bias=old_stem.bias is not None,
        )

        if pretrained:
            with torch.no_grad():
                # old_weight: (Out, 3, K, K)
                old_weight = old_stem.weight
                # Average over RGB channels -> (Out, 1, K, K)
                avg_weight = torch.mean(old_weight, dim=1, keepdim=True)
                new_stem.weight.copy_(avg_weight)

        self.backbone.conv_stem = new_stem

    def _apply_dilation_to_last_stage(self):
        """
        Modifies the last stage of EfficientNet to use Dilated Convolutions.
        Sets stride=1 and dilation=2 for spatial preservation.
        """
        # In timm's EfficientNet, blocks are organized in stages.
        # blocks[-1] refers to the last stage.
        last_stage = self.backbone.blocks[-1]

        for block in last_stage:
            # EfficientNet blocks in timm (InvertedResidual) usually have a 'conv_dw' attribute
            # which is the depthwise spatial convolution.
            if hasattr(block, "conv_dw"):
                m = block.conv_dw
                if isinstance(m, nn.Conv2d):
                    # Set stride to 1 to preserve resolution
                    m.stride = (1, 1)

                    # Set dilation to 2
                    dilation = 2
                    m.dilation = (dilation, dilation)

                    # Adjust padding to maintain size with dilation
                    # padding = (kernel_size - 1) // 2 * dilation
                    kernel_size = m.kernel_size[0]
                    padding = (kernel_size - 1) // 2 * dilation
                    m.padding = (padding, padding)

    def forward(self, x):
        # x: (B, 2, F, T)

        # Backbone Feature Extraction
        # Output: (B, C_feat, H_feat, W_feat)
        features = self.backbone(x)

        # Attentive Pooling
        # Output: (B, C_feat)
        pooled = self.att_pool(features)

        # Classification
        # Output: (B, NumClasses)
        logits = self.fc(pooled)

        return logits
