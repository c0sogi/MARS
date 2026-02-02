import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights
from library.config import Config


class ResNetAudio(nn.Module):
    """
    ResNet34 based model for audio classification.
    Adapts the first layer for single-channel spectrograms and the classifier for specific classes.
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=True):
        """
        Args:
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to load ImageNet pretrained weights.
        """
        super(ResNetAudio, self).__init__()

        # Load ResNet34 backbone
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = resnet34(weights=weights)

        # ------------------------------------------------------------------
        # Adapt First Layer: 3-channel RGB -> 1-channel Spectrogram
        # ------------------------------------------------------------------
        # The first layer in ResNet is 'conv1'
        original_conv = self.model.conv1

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
            # Cite solution_lesson_node_00010
            with torch.no_grad():
                new_conv.weight.copy_(original_conv.weight.sum(dim=1, keepdim=True))

        # Replace the layer in the model
        self.model.conv1 = new_conv

        # ------------------------------------------------------------------
        # Adapt Classifier: ImageNet (1000) -> Audio Classes (12)
        # ------------------------------------------------------------------
        # The classifier is 'fc'
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input spectrograms of shape (Batch, 1, n_mels, time).

        Returns:
            torch.Tensor: Logits of shape (Batch, num_classes).
        """
        return self.model(x)
