import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    Computes the generalized mean of each channel in the feature map.

    Formula: f(X) = (1/|X| * sum(x^p))^(1/p)

    Attributes:
        p (torch.Tensor): The power parameter, learnable.
        eps (float): Small constant to avoid numerical instability.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        """
        Implementation of GeM pooling.

        Args:
            x (torch.Tensor): Input feature map (B, C, H, W).
            p (float or torch.Tensor): Power parameter.
            eps (float): Epsilon for numerical stability.

        Returns:
            torch.Tensor: Pooled feature vector (B, C, 1, 1).
        """
        # Explicitly cast to Float32 to prevent NaN during power operations
        # This addresses the numerical instability observed in mixed precision training
        x = x.to(torch.float32)

        # Clamp to avoid negative values or zeros if p < 1 (though usually p >= 1)
        x = x.clamp(min=eps)

        # Calculate average of x^p
        # F.avg_pool2d computes (1/N) * sum(input)
        x_pow = x.pow(p)
        avg_pool = F.avg_pool2d(x_pow, (x.size(-2), x.size(-1)))

        # Take the p-th root
        return avg_pool.pow(1.0 / p)


class DRModel(nn.Module):
    """
    Diabetic Retinopathy Classification/Regression Model.
    Uses an EfficientNetV2 backbone with GeM pooling and a linear regression head.
    """

    def __init__(
        self,
        backbone_name=Config.BACKBONE,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
        gem_p=Config.GEM_P,
    ):
        """
        Args:
            backbone_name (str): Name of the timm backbone (default: tf_efficientnetv2_m).
            pretrained (bool): Whether to load pretrained ImageNet weights.
            num_classes (int): Number of output neurons (1 for regression).
            gem_p (float): Initial value for GeM pooling power parameter.
        """
        super(DRModel, self).__init__()

        # Create backbone
        # num_classes=0 and global_pool='' returns the unpooled feature maps (B, C, H, W)
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Get the number of features from the backbone
        self.num_features = self.backbone.num_features

        # Pooling layer
        self.pooling = GeM(p=gem_p)

        # Regression Head
        # Maps feature dimension to scalar output
        self.fc = nn.Linear(self.num_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images (B, 3, H, W).

        Returns:
            torch.Tensor: Predicted scores (B, 1).
        """
        # 1. Feature Extraction
        features = self.backbone(x)  # Shape: (B, C, H_feat, W_feat)

        # 2. Pooling (GeM)
        pooled = self.pooling(features)  # Shape: (B, C, 1, 1)

        # 3. Flatten
        flattened = pooled.view(pooled.size(0), -1)  # Shape: (B, C)

        # 4. Regression Head
        output = self.fc(flattened)  # Shape: (B, 1)

        return output
