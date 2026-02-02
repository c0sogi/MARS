import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class IsovariantResNet18(nn.Module):
    def __init__(self):
        """
        Initializes the IsovariantResNet18 model.

        Architecture:
        - Backbone: ResNet18 (Pretrained)
        - Pooling: Global Average Pooling
        - Fusion: Concatenation of image features + incidence angle
        - Head: BatchNorm -> Dropout -> Linear
        """
        super(IsovariantResNet18, self).__init__()

        # 1. Load Backbone
        # Use 'weights' parameter as 'pretrained' is deprecated in modern torchvision
        if Config.PRETRAINED:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.backbone = models.resnet18(weights=weights)

        # 2. Extract Feature Layers
        # Remove the last two layers (avgpool and fc) to get the feature extractor
        # ResNet18 child layers: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc
        # We keep everything up to layer4.
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])

        # 3. Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # 4. Define Dimensions
        self.cnn_dim = 512
        self.angle_dim = 1
        self.fusion_dim = self.cnn_dim + self.angle_dim

        # 5. Minimalist Classification Head
        # We apply BatchNorm first to normalize the fused vector (image features + angle)
        self.head = nn.Sequential(
            nn.BatchNorm1d(self.fusion_dim),
            nn.Dropout(p=Config.DROPOUT_RATE),
            nn.Linear(self.fusion_dim, Config.NUM_CLASSES),
        )

        # 6. Initialize Head Weights
        self._init_head()

    def _init_head(self):
        """
        Applies Xavier initialization to the linear layer of the head.
        """
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, angle):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input images, shape (Batch, 3, 224, 224)
            angle (torch.Tensor): Normalized incidence angles, shape (Batch, 1)

        Returns:
            torch.Tensor: Logits, shape (Batch, 1)
        """
        # Feature Extraction
        x = self.features(x)  # (Batch, 512, 7, 7)

        # Global Average Pooling
        x = self.gap(x)  # (Batch, 512, 1, 1)
        x = torch.flatten(x, 1)  # (Batch, 512)

        # Late Fusion
        # Concatenate 512-dim image vector with 1-dim angle scalar
        x = torch.cat([x, angle], dim=1)  # (Batch, 513)

        # Classification Head
        x = self.head(x)  # (Batch, 1)

        return x
