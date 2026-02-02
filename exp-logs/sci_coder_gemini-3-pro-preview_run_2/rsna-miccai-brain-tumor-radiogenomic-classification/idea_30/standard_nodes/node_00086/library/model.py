import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class AsymmetricEfficientNet(nn.Module):
    """
    Asymmetric Grouped EfficientNet-B0 with Hybrid-Integral Consensus inputs.

    This model modifies the EfficientNet-B0 stem to accept 12 channels (4 modalities * 3 slices)
    using grouped convolutions to isolate modalities initially. It uses asymmetric initialization
    to distribute pre-trained ImageNet filters across these groups.
    """

    def __init__(self, pretrained=True, dropout_rate=0.5):
        super(AsymmetricEfficientNet, self).__init__()

        # Load the backbone with ImageNet weights
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)

        # Modify the first layer (Stem) for 12-channel input with Grouped Convolutions
        self._modify_stem()

        # Reconstruct the classifier head
        # EfficientNet B0 final feature map depth is 1280
        in_features = self.backbone.classifier[1].in_features
        # Increased dropout to 0.5 to combat overfitting
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True), nn.Linear(in_features, 1)
        )

    def _modify_stem(self):
        """
        Replaces the first convolutional layer to handle 12 input channels using
        4 groups (one per modality). Initializes weights using the pre-trained
        filters distributed across groups.
        """
        # The first layer in EfficientNet-B0 is within features[0], which is a Conv2dNormActivation block.
        # features[0][0] is the Conv2d layer.
        original_conv = self.backbone.features[0][0]

        # Define the new stem convolution
        # in_channels=12 (4 modalities * 3 slices)
        # out_channels=32 (Standard for EfficientNet-B0 stem)
        # groups=4 (Isolate FLAIR, T1w, T1wCE, T2w initially)
        # padding=1 (Explicit padding to preserve spatial dimensions 224x224 -> 112x112 with stride 2)
        new_conv = nn.Conv2d(
            in_channels=12,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=1,
            bias=False,
            groups=4,
        )

        # Asymmetric Filter Initialization
        # Original weights shape: (32, 3, 3, 3) -> (Out, In, K, K)
        # New weights shape: (32, 3, 3, 3) -> (Out, In/Groups, K, K)
        # Since the shapes are identical, we can copy the weights directly.
        # This maps:
        #   Filters 0-7   -> Input Channels 0-2 (Modality 1)
        #   Filters 8-15  -> Input Channels 3-5 (Modality 2)
        #   Filters 16-23 -> Input Channels 6-8 (Modality 3)
        #   Filters 24-31 -> Input Channels 9-11 (Modality 4)
        with torch.no_grad():
            new_conv.weight.data = original_conv.weight.data.clone()

        # Replace the layer in the backbone
        self.backbone.features[0][0] = new_conv

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 12, 224, 224).

        Returns:
            torch.Tensor: Logits of shape (Batch, 1).
        """
        return self.backbone(x)
