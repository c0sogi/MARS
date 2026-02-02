import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Formula: f = (1/N * sum(x^p))^(1/p)

    This layer allows the network to learn the optimal pooling strategy,
    interpolating between Average Pooling (p=1) and Max Pooling (p -> infinity).
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x: (batch, channels, height, width)
        # Clamp for numerical stability before power operation
        x = x.clamp(min=self.eps)

        # Calculate (1/N * sum(x^p))
        # We use avg_pool2d over the spatial dimensions (H, W) to compute the mean
        x_pow = x.pow(self.p)
        # The kernel size matches the spatial dimensions of the input feature map
        avg = F.avg_pool2d(x_pow, (x.size(-2), x.size(-1)))

        # Apply ( )^(1/p)
        return avg.pow(1.0 / self.p)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class IcebergResNet(nn.Module):
    """
    ResNet-18 based architecture with Adaptive GeM pooling and Late Fusion for incidence angle.
    """

    def __init__(
        self, pretrained=True, dropout_rate=Config.DROPOUT_RATE, gem_p=Config.GEM_P_INIT
    ):
        """
        Args:
            pretrained (bool): Whether to load ImageNet weights.
            dropout_rate (float): Dropout probability in the head.
            gem_p (float): Initial value for the GeM pooling parameter p.
        """
        super(IcebergResNet, self).__init__()

        # Load ResNet18 backbone
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None

        base_model = models.resnet18(weights=weights)

        # Remove the original Average Pooling and FC layer
        # ResNet18 structure: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc
        # We take everything before avgpool to keep spatial dimensions (N, 512, 7, 7)
        self.features = nn.Sequential(*list(base_model.children())[:-2])

        # ResNet18 output channels are 512
        self.num_features = 512

        # Adaptive Generalized Mean Pooling
        # Replaces standard Global Average Pooling
        self.gem = GeM(p=gem_p, eps=Config.GEM_EPS)

        # Classification Head
        # Inputs: 512 (Image Features) + 1 (Incidence Angle)
        self.head = nn.Sequential(
            nn.BatchNorm1d(self.num_features + 1),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.num_features + 1, 1),
        )

    def forward(self, x, angle):
        """
        Args:
            x (torch.Tensor): Image tensor (Batch, 3, 224, 224)
            angle (torch.Tensor): Incidence angle tensor (Batch,) or (Batch, 1)

        Returns:
            torch.Tensor: Logits (Batch, 1)
        """
        # 1. Backbone Feature Extraction
        x = self.features(x)  # Output: (Batch, 512, 7, 7)

        # 2. GeM Pooling
        x = self.gem(x)  # Output: (Batch, 512, 1, 1)

        # 3. Flatten
        x = x.view(x.size(0), -1)  # Output: (Batch, 512)

        # 4. Late Fusion
        # Ensure angle is (Batch, 1)
        if angle.dim() == 1:
            angle = angle.view(-1, 1)

        # Concatenate features and angle
        x = torch.cat([x, angle], dim=1)  # Output: (Batch, 513)

        # 5. Classification Head
        x = self.head(x)  # Output: (Batch, 1)

        return x
