import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    Computes the generalized mean of the input tensor.

    Formula: f(X) = (1/|X| * sum(x^p))^(1/p)

    Args:
        p (float): Initial value for the power parameter. Default: 1.0 (Average Pooling).
        eps (float): Small constant to avoid numerical instability.
    """

    def __init__(self, p=1.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x shape: (N, C, H, W)
        # We perform pooling over the spatial dimensions (H, W)
        # clamp(min=eps) ensures non-negative base for power operation
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class IcebergResNet18GeM(nn.Module):
    """
    ResNet-18 with GeM Pooling and Late Fusion for Iceberg Classification.

    Architecture:
    1. Backbone: ResNet-18 (pretrained).
    2. Pooling: GeM (replaces standard AvgPool).
    3. Fusion: Concatenates pooled image features with incidence angle.
    4. Head: BatchNorm -> Dropout -> Linear.
    """

    def __init__(self, pretrained=True):
        super(IcebergResNet18GeM, self).__init__()

        # Load ResNet18 backbone
        # We use try-except to handle potential API differences, though modern torchvision uses 'weights'
        try:
            from torchvision.models import ResNet18_Weights

            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
        except ImportError:
            self.backbone = models.resnet18(pretrained=pretrained)

        # Replace the standard Global Average Pooling with GeM
        # ResNet's avgpool is typically AdaptiveAvgPool2d((1, 1))
        # GeM performs pooling over the entire spatial dimension
        self.backbone.avgpool = GeM(p=1.0)

        # Remove the original Fully Connected layer
        # We replace it with Identity to retrieve the flattened feature vector (512 dim)
        self.backbone.fc = nn.Identity()

        # ResNet-18 output feature dimension
        self.num_features = 512

        # Classification Head
        # Input dimension: 512 (Image Features) + 1 (Incidence Angle) = 513
        self.head_bn = nn.BatchNorm1d(self.num_features + 1)
        self.head_dropout = nn.Dropout(p=0.5)
        self.head_fc = nn.Linear(self.num_features + 1, 1)

    def forward(self, x, angle):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Image tensor of shape (Batch_Size, 3, Height, Width).
            angle (torch.Tensor): Incidence angle tensor of shape (Batch_Size,) or (Batch_Size, 1).

        Returns:
            torch.Tensor: Logits of shape (Batch_Size, 1).
        """
        # 1. Extract Image Features
        # Pass through ResNet backbone (Conv -> BN -> ReLU -> ... -> GeM -> Flatten)
        # Output shape: (Batch_Size, 512)
        features = self.backbone(x)

        # 2. Prepare Angle Feature
        # Ensure angle has shape (Batch_Size, 1)
        if angle.dim() == 1:
            angle = angle.view(-1, 1)

        # 3. Late Fusion
        # Concatenate image features and angle
        # Output shape: (Batch_Size, 513)
        combined = torch.cat((features, angle), dim=1)

        # 4. Classification Head
        out = self.head_bn(combined)
        out = self.head_dropout(out)
        out = self.head_fc(out)

        return out
