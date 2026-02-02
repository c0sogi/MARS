import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from library.config import Config


class EfficientNetAudio(nn.Module):
    """
    EfficientNet-B0 based model for audio classification.
    Adapts the first layer for single-channel spectrograms and the classifier for specific classes.
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=True):
        """
        Args:
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to load ImageNet pretrained weights.
        """
        super(EfficientNetAudio, self).__init__()

        # Load EfficientNet-B0 backbone
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = efficientnet_b0(weights=weights)

        # ------------------------------------------------------------------
        # Adapt First Layer: 3-channel RGB -> 1-channel Spectrogram
        # ------------------------------------------------------------------
        # The first layer in EfficientNet-B0 is within the 'features' Sequential block.
        # Structure: features[0] -> Conv2dNormActivation -> [0] is Conv2d
        original_conv = self.model.features[0][0]

        # Create a new Conv2d layer with in_channels=1
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=(original_conv.bias is not None),
        )

        # Initialize weights
        if pretrained:
            # Sum weights across the 3 input channels to preserve texture/edge detection
            # Original shape: (out_channels, 3, k, k)
            # New shape: (out_channels, 1, k, k)
            with torch.no_grad():
                new_conv.weight.copy_(original_conv.weight.sum(dim=1, keepdim=True))

        # Replace the layer in the model
        self.model.features[0][0] = new_conv

        # ------------------------------------------------------------------
        # Adapt Classifier: ImageNet (1000) -> Audio Classes (12)
        # ------------------------------------------------------------------
        # The classifier is a Sequential block:
        # (0): Dropout
        # (1): Linear

        # Get the input features of the final linear layer
        in_features = self.model.classifier[1].in_features

        # Replace the linear layer
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input spectrograms of shape (Batch, 1, n_mels, time).

        Returns:
            torch.Tensor: Logits of shape (Batch, num_classes).
        """
        return self.model(x)
