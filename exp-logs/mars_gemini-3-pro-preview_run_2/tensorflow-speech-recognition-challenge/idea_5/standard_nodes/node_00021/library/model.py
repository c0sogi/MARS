import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Attention Pooling module to dynamically weight spatial features.

    Structure:
    Input (B, H, W, C) -> Flatten (B, N, C) -> MLP -> Scores (B, N, 1) -> Softmax -> Weighted Sum (B, C)
    """

    def __init__(self, in_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Tanh(),
            nn.Linear(in_dim // 2, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        # x shape: (B, H, W, C) from Swin Transformer
        B, H, W, C = x.shape

        # Flatten spatial dimensions: (B, N, C) where N = H * W
        x_flat = x.view(B, H * W, C)

        # Compute attention scores
        # weights shape: (B, N, 1)
        weights = self.attention(x_flat)

        # Weighted sum of features
        # out shape: (B, C)
        out = torch.sum(weights * x_flat, dim=1)

        return out


class EfficientNetAudioClassifier(nn.Module):
    """
    EfficientNet-B0 adapted for 1-channel audio spectrograms with Attention Pooling.
    Cite solution_lesson_node_00020 (CNN vs Transformer)
    Cite solution_lesson_node_00011 (Native Resolution)
    """

    def __init__(self, num_classes=Config.NUM_CLASSES):
        super(EfficientNetAudioClassifier, self).__init__()

        # 1. Load Pretrained EfficientNet-B0
        print("Loading EfficientNet-B0 backbone...")
        self.backbone = models.efficientnet_b0(weights="DEFAULT")

        # 2. Adapt Input Layer for 1-Channel Spectrograms
        original_conv = self.backbone.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

        # Initialize weights by summing the original RGB weights (Cite solution_lesson_node_00019)
        with torch.no_grad():
            new_conv.weight.data = original_conv.weight.data.sum(dim=1, keepdim=True)

        self.backbone.features[0][0] = new_conv

        # 3. Extract Backbone Features
        self.features = self.backbone.features

        # 4. Define Custom Head
        # EfficientNet-B0 output channels is 1280
        self.pool = AttentionPooling(1280)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        # Input x: (Batch, 1, 128, 101)

        # Pass through Backbone
        # Output x: (Batch, 1280, H, W)
        x = self.features(x)

        # Permute for Attention Pooling (Batch, H, W, C)
        x = x.permute(0, 2, 3, 1)

        # Apply Attention Pooling
        # Output x: (Batch, C)
        x = self.pool(x)

        # Final Classification
        x = self.classifier(x)

        return x
