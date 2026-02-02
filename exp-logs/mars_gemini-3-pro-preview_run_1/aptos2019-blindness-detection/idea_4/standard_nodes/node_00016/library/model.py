import torch
import torch.nn as nn
import timm


class OrdinalModel(nn.Module):
    """
    Generic timm-based model for Ordinal Regression in Diabetic Retinopathy detection.
    Supports EfficientNet, ConvNeXt, etc.
    """

    def __init__(
        self, model_name="efficientnet_b0", pretrained=True, num_classes=4, in_chans=3
    ):
        """
        Args:
            model_name (str): Name of the backbone model in timm.
            pretrained (bool): Whether to load pretrained ImageNet weights.
            num_classes (int): Number of ordinal output units.
            in_chans (int): Number of input channels.
        """
        super(OrdinalModel, self).__init__()

        # Initialize the backbone.
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, in_chans=in_chans
        )

        # Determine the input feature dimension for the head
        if hasattr(self.backbone, "num_features"):
            self.in_features = self.backbone.num_features
        else:
            # Fallback
            self.in_features = (
                self.backbone.num_features
                if hasattr(self.backbone, "num_features")
                else 768
            )

        # Rank-Consistent Ordinal Regression Head
        self.head = nn.Sequential(
            nn.Linear(self.in_features, num_classes), nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, Channels, Height, Width).

        Returns:
            torch.Tensor: Output probabilities of shape (Batch_Size, num_classes).
                          Each value is in [0, 1].
        """
        # Extract pooled and normalized features from the backbone
        features = self.backbone(x)

        # Generate ordinal probabilities
        probs = self.head(features)

        return probs
