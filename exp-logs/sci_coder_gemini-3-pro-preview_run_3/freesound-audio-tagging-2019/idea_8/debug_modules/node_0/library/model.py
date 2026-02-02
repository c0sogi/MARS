import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from library.config import Config


class ClassWiseEfficientNet(nn.Module):
    """
    EfficientNet-B0 with Class-Wise Dual-Stream Pooling.

    This model adapts a pretrained EfficientNet-B0 for multi-label audio classification.
    It projects the backbone features directly into class-specific activation maps
    before pooling, allowing for better disentanglement of overlapping sound events.
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=True):
        """
        Args:
            num_classes (int): Number of target classes (default: 80).
            pretrained (bool): Whether to initialize backbone with ImageNet weights.
        """
        super().__init__()

        # 1. Load Backbone
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)

        # 2. Input Adaptation: Modify first layer for 1-channel input
        # Access the first Conv2d layer: features -> block 0 -> layer 0
        original_conv = self.backbone.features[0][0]

        # Create new Conv2d with in_channels=1
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )

        # Initialize weights by summing RGB weights along the channel dimension
        # This preserves the magnitude of the filters while adapting to mono input
        with torch.no_grad():
            # original_conv.weight shape: (Out, 3, K, K)
            # Sum over channel dim (dim 1) -> (Out, 1, K, K)
            new_conv.weight.data = original_conv.weight.data.sum(dim=1, keepdim=True)
            if original_conv.bias is not None:
                new_conv.bias.data = original_conv.bias.data

        # Replace the layer in the backbone
        self.backbone.features[0][0] = new_conv

        # Remove unneeded heads to save parameters and memory
        # We only use .features which returns the spatial feature map
        del self.backbone.avgpool
        del self.backbone.classifier

        # 3. Class-Wise Projection
        # EfficientNet-B0 outputs 1280 channels at the final feature map.
        # We project these 1280 abstract features directly to 'num_classes' activation maps.
        self.projection = nn.Conv2d(1280, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input spectrograms of shape (Batch, 1, F, T).

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes).
        """
        # Extract features
        # Shape: (Batch, 1280, F', T')
        x = self.backbone.features(x)

        # Project to Class Activation Maps (CAMs)
        # Shape: (Batch, Num_Classes, F', T')
        x = self.projection(x)

        # 4. Dual-Stream Pooling
        # Stream A: Global Max Pooling (over spatial dims F' and T')
        # Captures the strongest activation of a sound event anywhere in the clip
        x_max = torch.amax(x, dim=(2, 3))

        # Stream B: Global Average Pooling (over spatial dims F' and T')
        # Captures the average presence of a sound event
        x_avg = torch.mean(x, dim=(2, 3))

        # 5. Fusion (Sum)
        # Combine both streams to handle both sparse/loud and continuous/ambient sounds
        # Shape: (Batch, Num_Classes)
        out = x_max + x_avg

        return out
