import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library.config import Config


class SpatialPyramidPooling(nn.Module):
    """
    Spatial Pyramid Pooling (SPP) Layer.

    Aggregates features from feature maps using multiple grid sizes. This allows the network
    to maintain spatial (and thus temporal/frequency) context which is often lost with
    standard Global Average Pooling.

    Args:
        levels (list): A list of integers representing the grid sizes for pooling.
                       Default is [1, 2, 4], corresponding to 1x1, 2x2, and 4x4 grids.
    """

    def __init__(self, levels=[1, 2, 4]):
        super(SpatialPyramidPooling, self).__init__()
        self.levels = levels

    def forward(self, x):
        # x shape: (Batch_Size, Channels, Height, Width)
        N, C, H, W = x.size()
        outputs = []

        for level in self.levels:
            # We use Adaptive Max Pooling.
            # Rationale: Bird calls are often sparse, high-energy events in the spectrogram.
            # Max pooling captures the strongest activation within a grid cell, indicating presence,
            # whereas Average pooling might dilute the signal with background silence/noise.
            # Output shape for level 'n': (N, C, n, n)
            pooled = F.adaptive_max_pool2d(x, output_size=(level, level))

            # Flatten the spatial dimensions: (N, C * level * level)
            outputs.append(pooled.view(N, -1))

        # Concatenate all flattened levels along the feature dimension
        # Total features = C * sum(level^2 for level in levels)
        return torch.cat(outputs, dim=1)


class BirdResNetSPP(nn.Module):
    """
    ResNet-18 with Spatial Pyramid Pooling for Multi-Label Bird Species Classification.

    This architecture replaces the standard Global Average Pooling and Fully Connected layer
    of ResNet-18 with an SPP module and a new classifier.

    Args:
        pretrained (bool): Whether to load ImageNet pretrained weights.
        num_classes (int): Number of output classes (species).
        spp_levels (list): Grid levels for the SPP layer.
    """

    def __init__(
        self,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
        spp_levels=Config.SPP_LEVELS,
    ):
        super(BirdResNetSPP, self).__init__()

        # 1. Load Pretrained ResNet-18 Backbone
        # Handle different torchvision versions for weight loading
        try:
            from torchvision.models import ResNet18_Weights

            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
        except ImportError:
            # Fallback for older torchvision versions
            self.backbone = models.resnet18(pretrained=pretrained)

        # 2. Feature Extractor
        # Retain layers: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4
        # Remove the last two layers: avgpool and fc
        layers = list(self.backbone.children())[:-2]
        self.features = nn.Sequential(*layers)

        # 3. Spatial Pyramid Pooling
        self.spp = SpatialPyramidPooling(levels=spp_levels)

        # 4. Classifier Head
        # Calculate the input dimension for the linear layer
        # ResNet-18 layer4 outputs 512 channels
        backbone_out_channels = 512

        # SPP concatenates flattened vectors from all grids
        # e.g., for levels [1, 2, 4]: 1*1 + 2*2 + 4*4 = 21 regions
        # Total dim = 512 * 21 = 10752
        spp_out_dim = backbone_out_channels * sum([l * l for l in spp_levels])

        self.classifier = nn.Linear(spp_out_dim, num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 3, 224, 224).

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes).
        """
        # Extract features using ResNet backbone
        # Output shape: (Batch, 512, 7, 7) for 224x224 input
        x = self.features(x)

        # Apply Spatial Pyramid Pooling
        # Output shape: (Batch, spp_out_dim)
        x = self.spp(x)

        # Classification
        # Output shape: (Batch, num_classes)
        logits = self.classifier(x)

        return logits
