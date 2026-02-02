import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class AngleCalibration(nn.Module):
    """
    Angle-Gated Feature Calibration Module.
    Transforms the scalar incidence angle into a channel-wise attention vector
    using a lightweight MLP.

    Structure: Input(1) -> Linear(64) -> ReLU -> Linear(512) -> Sigmoid
    """

    def __init__(self):
        super(AngleCalibration, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, Config.ANGLE_EMBEDDING_DIM),
            nn.ReLU(),
            nn.Linear(Config.ANGLE_EMBEDDING_DIM, Config.FEATURE_DIM),
            nn.Sigmoid(),
        )

    def forward(self, angle):
        """
        Args:
            angle (torch.Tensor): Shape (Batch_Size,) or (Batch_Size, 1)
        Returns:
            torch.Tensor: Attention vector alpha, Shape (Batch_Size, Feature_Dim)
        """
        # Ensure angle is (Batch_Size, 1)
        if angle.dim() == 1:
            angle = angle.view(-1, 1)

        return self.net(angle)


class AngleGatedResNet(nn.Module):
    """
    ResNet-18 with Angle-Gated Feature Calibration.

    Pipeline:
    1. Visual Features: ResNet18 Backbone -> Global Average Pooling -> 512-dim vector
    2. Angle Attention: Angle -> MLP -> 512-dim attention vector (0-1)
    3. Calibration: Visual Features * Angle Attention (Element-wise)
    4. Classification: BatchNorm -> Dropout -> Linear -> Logits
    """

    def __init__(self):
        super(AngleGatedResNet, self).__init__()

        # Load Pretrained ResNet18
        # We use the weights enum as per modern torchvision standards if available,
        # or fallback to pretrained=True logic implicitly handled by the weights argument.
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if Config.PRETRAINED else None
        base_model = models.resnet18(weights=weights)

        # Extract feature extractor (conv1 up to layer4)
        # We exclude the original avgpool and fc layer
        self.features = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
        )

        # Angle Calibration Module
        self.angle_calibrator = AngleCalibration()

        # Minimalist Classification Head
        # As per design: BN -> Dropout -> Linear
        self.head = nn.Sequential(
            nn.BatchNorm1d(Config.FEATURE_DIM),
            nn.Dropout(p=Config.DROPOUT_RATE),
            nn.Linear(Config.FEATURE_DIM, 1),
        )

    def forward(self, images, angles):
        """
        Args:
            images (torch.Tensor): Shape (B, 3, H, W)
            angles (torch.Tensor): Shape (B,)
        Returns:
            torch.Tensor: Logits, Shape (B, 1)
        """
        # 1. Extract Visual Features
        # Output shape: (B, 512, 7, 7) for 224x224 input
        x = self.features(images)

        # 2. Global Average Pooling
        # Output shape: (B, 512, 1, 1) -> Flatten to (B, 512)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        # 3. Angle-Gated Calibration (FiLM)
        # Get attention vector alpha: (B, 512)
        alpha = self.angle_calibrator(angles)

        # Modulate features: F_calibrated = F_img * alpha
        x = x * alpha

        # 4. Classification Head
        logits = self.head(x)

        return logits
