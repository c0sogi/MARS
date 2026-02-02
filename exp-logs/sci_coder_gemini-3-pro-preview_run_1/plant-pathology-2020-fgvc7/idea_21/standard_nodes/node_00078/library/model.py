import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class AppleResNet34(nn.Module):
    """
    ResNet34 model for Apple Disease Detection.

    Architecture:
    - Backbone: ResNet34 initialized with ImageNet weights.
    - Head: Global Average Pooling (native to ResNet) + Linear Layer (replaced).
    """

    def __init__(self, pretrained: bool = True):
        """
        Args:
            pretrained (bool): If True, loads ImageNet weights for the backbone.
        """
        super(AppleResNet34, self).__init__()

        # Load ResNet34 backbone
        # Using the modern weights API compatible with torchvision 0.23+
        if pretrained:
            weights = models.ResNet34_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.backbone = models.resnet34(weights=weights)

        # The ResNet architecture ends with:
        # (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
        # (fc): Linear(in_features=512, out_features=1000, bias=True)
        # We replace the 'fc' layer to match our number of classes.

        in_features = self.backbone.fc.in_features

        # Replace head with a new Linear layer
        # Initialization of this new layer is handled by PyTorch's default (Kaiming Uniform)
        self.backbone.fc = nn.Linear(in_features, Config.NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input batch of images.

        Returns:
            torch.Tensor: Raw logits (before Softmax).
        """
        return self.backbone(x)

    def check_initial_weights(self):
        """
        Verifies that the model weights are correctly initialized.

        This serves as a sanity check (Initial Loss Test equivalent for weights) to ensure
        the backbone is not degenerate (all zeros or NaNs) and that the pre-trained
        weights have been loaded effectively.
        """
        print("Running Initial Weight Verification...")

        # 1. Check Backbone (First Convolutional Layer)
        # We expect specific statistics from ImageNet weights, not random ~0 mean
        conv1_weights = self.backbone.conv1.weight.data
        conv1_mean = conv1_weights.mean().item()
        conv1_std = conv1_weights.std().item()

        print(f"  Backbone (Conv1) - Mean: {conv1_mean:.8f}, Std: {conv1_std:.8f}")

        if conv1_std == 0:
            raise ValueError(
                "Backbone weights have zero variance! Pre-training load failed."
            )

        if torch.isnan(conv1_weights).any():
            raise ValueError("Backbone weights contain NaNs!")

        # 2. Check Head (Fully Connected Layer)
        # We expect this to be random, but valid
        fc_weights = self.backbone.fc.weight.data
        fc_std = fc_weights.std().item()

        print(
            f"  Head (FC)        - Mean: {fc_weights.mean().item():.8f}, Std: {fc_std:.8f}"
        )

        if fc_std == 0:
            raise ValueError("Classification head weights have zero variance!")

        print("Weight verification passed successfully.")
