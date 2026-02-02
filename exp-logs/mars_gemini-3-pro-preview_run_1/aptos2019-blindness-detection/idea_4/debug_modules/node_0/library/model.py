import torch
import torch.nn as nn
import timm


class OrdinalConvNeXt(nn.Module):
    """
    ConvNeXt-Tiny based model for Ordinal Regression in Diabetic Retinopathy detection.

    This model uses a ConvNeXt-Tiny backbone, which utilizes Layer Normalization, making it
    suitable for high-resolution inputs where batch sizes might be small. The classification
    head is designed for rank-consistent ordinal regression, outputting probabilities
    for the ordinal thresholds.
    """

    def __init__(
        self, model_name="convnext_tiny", pretrained=True, num_classes=4, in_chans=3
    ):
        """
        Args:
            model_name (str): Name of the backbone model in timm. Defaults to 'convnext_tiny'.
            pretrained (bool): Whether to load pretrained ImageNet weights. Defaults to True.
            num_classes (int): Number of ordinal output units. For 5 DR grades (0-4),
                               we need 4 outputs corresponding to P(y>0), P(y>1), P(y>2), P(y>3).
            in_chans (int): Number of input channels. Defaults to 3 (RGB).
        """
        super(OrdinalConvNeXt, self).__init__()

        # Initialize the backbone.
        # setting num_classes=0 returns the pooled feature vector (Batch, Num_Features).
        # ConvNeXt's forward pass with num_classes=0 includes Global Average Pooling
        # and the final Layer Normalization.
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, in_chans=in_chans
        )

        # Determine the input feature dimension for the head
        if hasattr(self.backbone, "num_features"):
            self.in_features = self.backbone.num_features
        else:
            # Fallback for convnext_tiny if attribute is not found
            self.in_features = 768

        # Rank-Consistent Ordinal Regression Head
        # Maps the feature vector to 'num_classes' outputs with Sigmoid activation.
        # Output values represent the probability that the true label is greater than threshold k.
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
