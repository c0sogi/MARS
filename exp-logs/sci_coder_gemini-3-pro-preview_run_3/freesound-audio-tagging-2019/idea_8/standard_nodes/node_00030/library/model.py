import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from library.config import Config


class EfficientNetAttention(nn.Module):
    """
    EfficientNet-B0 with Learnable Attention Pooling.

    Replaces Class-Wise projection (Cite solution_lesson_node_00029) with
    Feature-Wise Attention Pooling (Cite solution_lesson_node_00014).
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=True):
        super().__init__()

        # 1. Load Backbone
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)

        # 2. Input Adaptation: Modify first layer for 1-channel input
        # (Cite solution_lesson_node_00015)
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

        # EfficientNet-B0 feature dim
        self.feature_dim = 1280

        # 3. Attention Pooling (Cite solution_lesson_node_00014)
        # Non-linear scoring: Linear -> Tanh -> Linear
        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1),
        )

        # 4. Classifier
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        # Extract features: (Batch, 1280, F, T)
        x = self.backbone.features(x)

        # Flatten spatial/spectral dimensions to preserve resolution
        # (Cite solution_lesson_node_00016)
        # Shape: (Batch, 1280, N) where N = F * T
        x = x.flatten(2)

        # Transpose for Linear layer: (Batch, N, 1280)
        x = x.transpose(1, 2)

        # Calculate Attention Weights: (Batch, N, 1)
        weights = self.attention(x)

        # Weighted Sum: (Batch, 1280)
        x = torch.sum(x * weights, dim=1)

        # Classification
        out = self.classifier(x)
        return out
