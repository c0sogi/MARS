import torch
import torch.nn as nn
import timm
from library.config import Config


class CAWIVModel(nn.Module):
    """
    Centroid-Aligned Weight-Inflated Volumetric (CA-WIV) Network.

    This model adapts an EfficientNet-B0 backbone to accept 9-channel volumetric inputs
    (3 modalities x 3 depths) by inflating the weights of the first convolutional layer
    using a Gaussian-like distribution. This preserves ImageNet priors while integrating
    3D spatial context.
    """

    def __init__(self, model_name=Config.BACKBONE, pretrained=True):
        """
        Args:
            model_name (str): Name of the timm backbone (default: efficientnet_b0).
            pretrained (bool): Whether to load ImageNet pretrained weights.
        """
        super().__init__()

        # Load backbone with 1 output class for binary classification
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=1
        )

        # Apply Gaussian Weight Inflation to the first layer
        self._inflate_weights()

        # Add Dropout to the classifier for regularization
        self._modify_classifier()

    def _inflate_weights(self):
        """
        Modifies the first convolutional layer (conv_stem) to accept 9 channels.
        Initializes weights using a Center-Biased (Gaussian) Prior:
        - Channels 0-2 (z-delta): 25% of original weights
        - Channels 3-5 (z-center): 50% of original weights
        - Channels 6-8 (z+delta): 25% of original weights
        """
        # EfficientNet stem is usually named 'conv_stem'
        if not hasattr(self.backbone, "conv_stem"):
            raise AttributeError(
                f"Backbone {Config.BACKBONE} does not have 'conv_stem'."
            )

        old_layer = self.backbone.conv_stem

        # Parameters for the new layer
        in_channels = 9  # 3 modalities * 3 depths
        out_channels = old_layer.out_channels
        kernel_size = old_layer.kernel_size
        stride = old_layer.stride
        padding = old_layer.padding
        bias = old_layer.bias is not None

        # Create new layer
        new_layer = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )

        # Initialize weights with Gaussian Inflation strategy
        with torch.no_grad():
            original_weights = old_layer.weight  # Shape: (Out, 3, K, K)
            new_weights = torch.zeros_like(new_layer.weight)  # Shape: (Out, 9, K, K)

            # Distribute energy to preserve activation magnitude statistics
            # Channels 0-2: Peripheral (z-delta) -> 0.25
            new_weights[:, 0:3, :, :] = original_weights * 0.25

            # Channels 3-5: Center (z) -> 0.50
            new_weights[:, 3:6, :, :] = original_weights * 0.50

            # Channels 6-8: Peripheral (z+delta) -> 0.25
            new_weights[:, 6:9, :, :] = original_weights * 0.25

            new_layer.weight.copy_(new_weights)

            # Copy bias if it exists
            if bias:
                new_layer.bias.copy_(old_layer.bias)

        # Replace the layer in the backbone
        self.backbone.conv_stem = new_layer

    def _modify_classifier(self):
        """
        Injects Dropout into the classifier head.
        """
        if hasattr(self.backbone, "classifier"):
            # EfficientNet classifier is typically a Linear layer
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=Config.DROPOUT_RATE), nn.Linear(in_features, 1)
            )
        elif hasattr(self.backbone, "fc"):
            # ResNet style
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=Config.DROPOUT_RATE), nn.Linear(in_features, 1)
            )
        elif hasattr(self.backbone, "head"):
            # ViT style
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Sequential(
                nn.Dropout(p=Config.DROPOUT_RATE), nn.Linear(in_features, 1)
            )

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (B, 9, H, W).
        Returns:
            torch.Tensor: Logits of shape (B, 1).
        """
        return self.backbone(x)
