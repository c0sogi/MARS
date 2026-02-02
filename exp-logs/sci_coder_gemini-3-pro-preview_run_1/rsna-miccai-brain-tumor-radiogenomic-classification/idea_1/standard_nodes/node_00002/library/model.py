import torch
import torch.nn as nn
import timm
from library.config import Config


class MGMTClassifier(nn.Module):
    """
    MGMTClassifier architecture for predicting MGMT promoter methylation.

    This model implements a 2.5D approach by treating stacked MRI slices (FLAIR, T1wCE, T2w)
    as a multi-channel 2D image. It utilizes a pretrained CNN backbone (e.g., EfficientNet)
    to extract spatial features, followed by a custom classification head.
    """

    def __init__(
        self,
        model_name=Config.BACKBONE,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
        dropout_rate=Config.DROPOUT_RATE,
    ):
        """
        Initialize the model.

        Args:
            model_name (str): The name of the backbone architecture (default: efficientnet_b0).
            pretrained (bool): Whether to use ImageNet pretrained weights.
            num_classes (int): Number of output classes (1 for binary classification).
            dropout_rate (float): Dropout probability for the classification head.
        """
        super(MGMTClassifier, self).__init__()

        # Create the backbone using timm.
        # num_classes=0 returns the pooled feature vector (Global Average Pooling).
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )

        # Dynamically determine the number of input features for the linear layer
        if hasattr(self.backbone, "num_features"):
            in_features = self.backbone.num_features
        else:
            # Fallback mechanism if num_features is not present
            # Run a dummy forward pass to infer shape
            with torch.no_grad():
                dummy = torch.zeros(1, 3, Config.IMG_SIZE, Config.IMG_SIZE)
                features = self.backbone(dummy)
                in_features = features.shape[1]

        # Define the custom classification head
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input batch of images with shape (B, 3, H, W).

        Returns:
            torch.Tensor: Raw logits with shape (B, num_classes).
        """
        # Pass input through the backbone to get pooled features
        features = self.backbone(x)

        # Apply dropout for regularization
        features = self.dropout(features)

        # Generate logits using the fully connected layer
        logits = self.fc(features)

        return logits
