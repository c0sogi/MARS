import torch
import torch.nn as nn
from torchvision import models
from library import config


class IcebergResNet18(nn.Module):
    """
    ResNet-18 based model for Iceberg classification with Late Fusion of incidence angle.

    Architecture:
    1. Backbone: ResNet-18 (pretrained on ImageNet), layers up to the final pooling removed.
    2. Pooling: Global Average Pooling (AdaptiveAvgPool2d).
    3. Fusion: Concatenation of the 512-dim image feature vector with the scalar incidence angle.
    4. Head: BatchNormalization -> Dropout -> Linear Layer.
    """

    def __init__(self):
        super(IcebergResNet18, self).__init__()

        # Load pretrained ResNet-18
        # Using the weights API compatible with modern torchvision versions
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.backbone = models.resnet18(weights=weights)

        # Remove the original fully connected layer (fc) and average pooling layer (avgpool)
        # We keep the feature extractor part: conv1 -> bn1 -> relu -> maxpool -> layers 1-4
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])

        # Global Average Pooling
        # Ensures output is (Batch, 512, 1, 1) regardless of spatial dimensions (though we use 224x224)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Feature dimension from ResNet-18 is 512
        self.img_feature_dim = 512

        # Head Architecture
        # Input dimension: 512 (Image Features) + 1 (Incidence Angle) = 513
        # Batch Normalization here serves to normalize the concatenated vector,
        # handling the different scales of the image features and the raw incidence angle.
        self.bn_head = nn.BatchNorm1d(self.img_feature_dim + 1)

        # Dropout for regularization as specified in the idea
        self.dropout = nn.Dropout(p=config.DROPOUT_RATE)

        # Final classification layer
        self.fc = nn.Linear(self.img_feature_dim + 1, config.NUM_CLASSES)

    def forward(self, x, angle):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Image tensor of shape (Batch, 3, H, W).
            angle (torch.Tensor): Incidence angle tensor of shape (Batch,).

        Returns:
            torch.Tensor: Logits of shape (Batch, 1).
        """
        # 1. Extract Image Features
        x = self.features(x)  # Shape: (Batch, 512, 7, 7) for 224x224 input
        x = self.global_avg_pool(x)  # Shape: (Batch, 512, 1, 1)
        x = torch.flatten(x, 1)  # Shape: (Batch, 512)

        # 2. Prepare Angle
        # Ensure angle has shape (Batch, 1) to match feature vector dimensions for concatenation
        angle = angle.view(-1, 1)

        # 3. Late Fusion
        # Concatenate image features and angle
        x = torch.cat((x, angle), dim=1)  # Shape: (Batch, 513)

        # 4. Classification Head
        x = self.bn_head(x)  # Normalizes the combined features (including angle)
        x = self.dropout(x)
        x = self.fc(x)  # Shape: (Batch, 1)

        return x
