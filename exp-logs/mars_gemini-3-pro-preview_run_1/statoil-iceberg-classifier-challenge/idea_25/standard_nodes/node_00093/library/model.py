import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import library.config as config


class GeMPooling(nn.Module):
    """
    Generalized Mean (GeM) Pooling layer.
    Formula: f = (1/N * sum(x^p))^(1/p)
    """

    def __init__(self, p=config.GEM_P_INIT, eps=1e-6, trainable=config.GEM_P_TRAINABLE):
        super(GeMPooling, self).__init__()
        # Initialize p as a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p, requires_grad=trainable)
        self.eps = eps

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)
        # Clamp min to eps to avoid numerical instability with pow
        x = x.clamp(min=self.eps).pow(self.p)

        # Apply Average Pooling on the powered tensor
        # Kernel size matches the spatial dimensions (H, W) to perform global pooling
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))

        # Inverse power
        x = x.pow(1.0 / self.p)

        return x


class IcebergResNet(nn.Module):
    """
    ResNet-18 based architecture with GeM Pooling and Late Fusion for Incidence Angle.
    """

    def __init__(
        self,
        backbone_name=config.BACKBONE,
        pretrained=config.PRETRAINED,
        dropout_rate=config.DROPOUT_RATE,
        num_classes=config.NUM_CLASSES,
        gem_p=config.GEM_P_INIT,
        gem_trainable=config.GEM_P_TRAINABLE,
    ):
        super(IcebergResNet, self).__init__()

        # 1. Initialize Backbone
        if backbone_name == "resnet18":
            if pretrained:
                # Attempt to use the modern weights API if available, fallback to legacy
                try:
                    from torchvision.models import ResNet18_Weights

                    weights = ResNet18_Weights.IMAGENET1K_V1
                    self.backbone = models.resnet18(weights=weights)
                except ImportError:
                    self.backbone = models.resnet18(pretrained=True)
            else:
                self.backbone = models.resnet18(weights=None)
        else:
            raise ValueError(f"Backbone {backbone_name} is not currently supported.")

        # 2. Extract Feature Extractor
        # ResNet18 layers: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc
        # We keep everything up to layer4 to get spatial features
        layers = list(self.backbone.children())[:-2]
        self.features = nn.Sequential(*layers)

        # Feature dimension for ResNet18 is 512
        self.n_features = 512

        # 3. Adaptive GeM Pooling
        self.gem = GeMPooling(p=gem_p, trainable=gem_trainable)

        # 4. Late Fusion & Classification Head
        # Input dimension = Image Features (512) + Incidence Angle (1)
        self.n_input_head = self.n_features + 1

        self.head = nn.Sequential(
            nn.BatchNorm1d(self.n_input_head),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.n_input_head, num_classes),
        )

    def forward(self, x, angle):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input images of shape (Batch, 3, H, W).
            angle (torch.Tensor): Normalized incidence angles of shape (Batch,).

        Returns:
            torch.Tensor: Logits of shape (Batch, num_classes).
        """
        # Feature Extraction
        x = self.features(
            x
        )  # Output: (B, 512, H/32, W/32) -> (B, 512, 7, 7) for 224x224 input

        # GeM Pooling
        x = self.gem(x)  # Output: (B, 512, 1, 1)

        # Flatten
        x = x.view(x.size(0), -1)  # Output: (B, 512)

        # Prepare Angle
        angle = angle.view(-1, 1)  # Ensure shape is (B, 1)

        # Late Fusion
        x = torch.cat([x, angle], dim=1)  # Output: (B, 513)

        # Classification Head
        logits = self.head(x)  # Output: (B, num_classes)

        return logits
