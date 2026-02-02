import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class MILEfficientNet(nn.Module):
    """
    Asymmetric Grouped EfficientNet-B0 with Multi-Instance Learning (MIL).

    Architecture:
    1. Backbone: EfficientNet-B0 initialized with ImageNet weights.
    2. Stem: Modified to accept 12 channels (4 modalities x 3 slices) using Grouped Convolutions (groups=4).
       Weights are initialized via Direct Block Copy from the original RGB weights.
    3. Head: Regularized with Dropout(p=0.5) and a single linear output.

    Input: (Batch, Candidates, Channels, Height, Width)
    Output: (Batch, Candidates) - Logits per candidate
    """

    def __init__(self):
        super().__init__()

        # Load Pre-trained Backbone
        # Using the string alias for weights which is robust across recent torchvision versions
        self.backbone = models.efficientnet_b0(weights="IMAGENET1K_V1")

        # Modify the stem (first convolution)
        self._modify_stem()

        # Reconstruct the Classifier Head
        # Original: Sequential(Dropout(p=0.2), Linear(1280, 1000))
        # New: Sequential(Dropout(p=0.5), Linear(1280, 1))

        # Get the input features of the final linear layer
        # In EfficientNet implementation, classifier is a Sequential block
        # usually index 1 is the Linear layer
        if (
            isinstance(self.backbone.classifier, nn.Sequential)
            and len(self.backbone.classifier) > 1
        ):
            in_features = self.backbone.classifier[1].in_features
        else:
            # Fallback for safety, though B0 usually has 1280
            in_features = 1280

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(in_features, 1)
        )

    def _modify_stem(self):
        """
        Replaces the first convolutional layer with a Grouped Convolution.
        Performs Direct Asymmetric Initialization.
        """
        # Access the first layer of the features block
        # efficientnet_b0.features[0] is Conv2dNormActivation
        # efficientnet_b0.features[0][0] is the actual Conv2d layer
        old_conv = self.backbone.features[0][0]

        # Define the new Grouped Convolution
        # in_channels=12 (Config.NUM_CHANNELS)
        # groups=4 (Config.NUM_MODALITIES)
        # This isolates FLAIR, T1w, T1wCE, T2w processing in the first layer
        new_conv = nn.Conv2d(
            in_channels=Config.NUM_CHANNELS,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
            groups=Config.NUM_MODALITIES,
        )

        # Direct Block Copy Initialization
        # Old weights shape: (32, 3, 3, 3) [Out, In, K, K]
        # New weights shape: (32, 3, 3, 3) [Out, In/Groups, K, K] where In=12, Groups=4 -> 3
        # Since shapes match, we clone the weights directly.
        # This assigns Filters 0-7 to Modality 0, Filters 8-15 to Modality 1, etc.
        with torch.no_grad():
            new_conv.weight.data = old_conv.weight.data.clone()
            if old_conv.bias is not None:
                new_conv.bias.data = old_conv.bias.data.clone()

        # Replace the layer in the backbone
        self.backbone.features[0][0] = new_conv

    def forward(self, x):
        """
        Forward pass for Multi-Instance Learning.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Candidates, Channels, H, W)

        Returns:
            torch.Tensor: Logits of shape (Batch, Candidates)
        """
        b, n, c, h, w = x.shape

        # Fold the candidate dimension into the batch dimension
        # Shape becomes (Batch * Candidates, Channels, H, W)
        x = x.view(b * n, c, h, w)

        # Pass through the backbone
        # Output shape: (Batch * Candidates, 1)
        x = self.backbone(x)

        # Reshape back to separate batch and candidates
        # Shape: (Batch, Candidates)
        logits = x.view(b, n)

        return logits
