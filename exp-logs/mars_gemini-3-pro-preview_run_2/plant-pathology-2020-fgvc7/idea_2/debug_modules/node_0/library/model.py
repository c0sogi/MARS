import torch
import torch.nn as nn
import timm
from library.config import Config


class AppleEfficientNet(nn.Module):
    """
    Apple Disease Detection Model based on EfficientNet-B4.

    Architecture:
    - Backbone: EfficientNet-B4 (pre-trained on ImageNet)
    - Head: Global Average Pooling -> Dropout -> Linear
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAME,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
        dropout_rate=Config.DROPOUT_RATE,
    ):
        """
        Initializes the model.

        Args:
            model_name (str): Name of the timm model to use.
            pretrained (bool): Whether to load ImageNet weights.
            num_classes (int): Number of output classes.
            dropout_rate (float): Dropout probability for the head.
        """
        super(AppleEfficientNet, self).__init__()

        # Create the backbone
        # num_classes=0 removes the original classifier
        # global_pool='' removes the original pooling, returning spatial features (B, C, H, W)
        # This allows us to explicitly construct the head as requested.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )

        # Get the number of input features for the head (e.g., 1792 for EfficientNet-B4)
        in_features = self.backbone.num_features

        # Custom head as specified: Global Average Pooling -> Dropout -> Linear
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, num_classes).
        """
        # Extract spatial features from backbone
        features = self.backbone(x)

        # Pass through custom head
        logits = self.head(features)

        return logits
