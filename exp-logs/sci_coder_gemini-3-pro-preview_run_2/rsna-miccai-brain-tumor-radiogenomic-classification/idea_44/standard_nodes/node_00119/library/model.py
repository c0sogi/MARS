import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class AsymmetricEfficientNet(nn.Module):
    def __init__(self):
        super(AsymmetricEfficientNet, self).__init__()

        # 1. Load Pre-trained Backbone
        # We use the default weights (IMAGENET1K_V1)
        weights = models.EfficientNet_B0_Weights.DEFAULT
        self.backbone = models.efficientnet_b0(weights=weights)

        # 2. Surgical Stem Replacement (Cite Lesson 00072)
        # The first layer in EfficientNet-B0 is within features[0][0]
        original_stem = self.backbone.features[0][0]

        # Create a new Conv2d layer
        # in_channels=12 (4 modalities * 3 slices)
        # groups=4 (Modality isolation) (Cite Lesson 00068)
        new_stem = nn.Conv2d(
            in_channels=Config.NUM_CHANNELS,
            out_channels=original_stem.out_channels,
            kernel_size=original_stem.kernel_size,
            stride=original_stem.stride,
            padding=original_stem.padding,
            bias=original_stem.bias is not None,
            groups=Config.STEM_GROUPS,
        )

        # 3. Direct Asymmetric Initialization (Cite Lesson 00095)
        # We copy the weights directly.
        with torch.no_grad():
            new_stem.weight.data = original_stem.weight.data.clone()

        # Replace the layer in the backbone
        self.backbone.features[0][0] = new_stem

        # 4. Replace Classifier
        # EfficientNet-B0 classifier is Sequential(Dropout, Linear)
        # We replace it to output 1 class
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=Config.DROPOUT_RATE), nn.Linear(in_features, 1)
        )

    def forward(self, x):
        """
        Forward Pass.
        Args:
            x: Tensor of shape (B, 12, 224, 224)
        Returns:
            logits: Tensor of shape (B, 1)
        """
        return self.backbone(x)
