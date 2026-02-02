import torch
import torch.nn as nn
import timm
from library.config import (
    BACKBONE,
    NUM_CHANNELS,
    DROPOUT_RATE,
    NUM_CLASSES,
)


class SICAVModel(nn.Module):
    """
    Scale-Invariant Centroid-Aligned Volumetric (SICAV) Network.

    This model uses an EfficientNet-B0 backbone modified to accept 9-channel inputs.
    The input channels correspond to [FLAIR, T1wCE, T2w] stacked at three relative
    depths (40%, 50%, 60%) of the brain ROI.

    The first convolutional layer is initialized using Gaussian Weight Inflation to
    preserve ImageNet priors while integrating volumetric context.
    """

    def __init__(
        self,
        backbone_name=BACKBONE,
        num_channels=NUM_CHANNELS,
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE,
        pretrained=True,
    ):
        super(SICAVModel, self).__init__()

        # 1. Load Backbone
        # num_classes=0 removes the default classifier head, returning pooled features
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=3,  # Initialize with standard 3 channels first, then modify
        )

        # 2. Modify First Layer for 9 Channels
        self._modify_first_layer(num_channels)

        # 3. Define Classifier Head
        # Get the number of features output by the backbone
        num_features = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate), nn.Linear(num_features, num_classes)
        )

    def _modify_first_layer(self, new_in_channels):
        """
        Replaces the first convolutional layer to accept `new_in_channels`
        and initializes it using Gaussian Weight Inflation.
        """
        # Identify the first layer. For EfficientNet in timm, it's usually 'conv_stem'.
        # We check common names just in case, but 'conv_stem' is standard for EffNet.
        if hasattr(self.backbone, "conv_stem"):
            old_layer = self.backbone.conv_stem
            layer_name = "conv_stem"
        elif hasattr(self.backbone, "conv1"):
            old_layer = self.backbone.conv1
            layer_name = "conv1"
        else:
            raise AttributeError(
                "Could not find the first convolutional layer in the backbone."
            )

        # Create new layer with same parameters but different in_channels
        new_layer = nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=old_layer.out_channels,
            kernel_size=old_layer.kernel_size,
            stride=old_layer.stride,
            padding=old_layer.padding,
            bias=old_layer.bias is not None,
        )

        # Apply Gaussian Weight Inflation
        self._gaussian_weight_inflation(old_layer, new_layer)

        # Replace the layer in the backbone
        setattr(self.backbone, layer_name, new_layer)

    def _gaussian_weight_inflation(self, old_layer, new_layer):
        """
        Initializes the new 9-channel weights based on the original 3-channel weights.

        Strategy:
        - Channels 3-5 (Center/50% depth): 50% of original weight energy.
        - Channels 0-2 (Peripheral/40% depth): 25% of original weight energy.
        - Channels 6-8 (Peripheral/60% depth): 25% of original weight energy.

        This sums to 100% of the original signal magnitude, preserving initialization statistics.
        """
        with torch.no_grad():
            old_weights = old_layer.weight  # Shape: (Out, 3, K, K)
            new_weights = new_layer.weight  # Shape: (Out, 9, K, K)

            # Ensure the new weights are zeroed out first
            new_weights.zero_()

            # Copy and scale weights
            # 1. Peripheral (40% depth) -> Channels 0, 1, 2
            new_weights[:, 0:3, :, :] = old_weights * 0.25

            # 2. Center (50% depth) -> Channels 3, 4, 5
            new_weights[:, 3:6, :, :] = old_weights * 0.50

            # 3. Peripheral (60% depth) -> Channels 6, 7, 8
            new_weights[:, 6:9, :, :] = old_weights * 0.25

            # If bias exists, copy it directly
            if old_layer.bias is not None and new_layer.bias is not None:
                new_layer.bias = old_layer.bias

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (B, 9, H, W)
        Returns:
            torch.Tensor: Logits of shape (B, 1)
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
