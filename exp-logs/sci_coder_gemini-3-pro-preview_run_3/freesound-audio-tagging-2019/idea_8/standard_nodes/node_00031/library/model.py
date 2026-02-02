import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from library.config import Config


class AudioClassifier(nn.Module):
    """
    EfficientNet-B0 with Learnable Attention Pooling.

    Uses Late Fusion: Aggregates high-dimensional features using a non-linear
    attention mechanism before classification (Cite Lesson 29).
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=True):
        super().__init__()

        # 1. Load Backbone
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)

        # 2. Input Adaptation (1-channel)
        original_conv = self.backbone.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight.data = original_conv.weight.data.sum(dim=1, keepdim=True)
            if original_conv.bias is not None:
                new_conv.bias.data = original_conv.bias.data
        self.backbone.features[0][0] = new_conv

        # Remove unneeded heads
        del self.backbone.avgpool
        del self.backbone.classifier

        # 3. Attention Pooling (Cite Lesson 7, 14, 16)
        # Feature dim for EfficientNet-B0 is 1280
        self.feature_dim = 1280

        # Non-linear attention: Linear -> Tanh -> Linear -> Softmax
        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim, 512), nn.Tanh(), nn.Linear(512, 1)
        )

        # 4. Classifier
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        # x: (Batch, 1, F, T)
        x = self.backbone.features(x)  # (Batch, 1280, F', T')

        # Flatten spatial dimensions (Cite Lesson 16)
        # (Batch, 1280, N) where N = F' * T'
        batch_size, channels, _, _ = x.size()
        x = x.view(batch_size, channels, -1)

        # Permute for Attention: (Batch, N, 1280)
        x = x.permute(0, 2, 1)

        # Compute Attention Weights
        # attn_weights: (Batch, N, 1)
        attn_weights = torch.softmax(self.attention(x), dim=1)

        # Weighted Sum
        # (Batch, 1280)
        x = torch.sum(x * attn_weights, dim=1)

        # Classification
        logits = self.classifier(x)
        return logits
