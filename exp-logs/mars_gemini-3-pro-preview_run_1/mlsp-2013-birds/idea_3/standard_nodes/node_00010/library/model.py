import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from library.config import Config


class BirdResNet(nn.Module):
    """
    ResNet-50 based model for Bird Species Classification.
    Adapts a pretrained ResNet-50 backbone for multi-label classification of spectrograms.
    """

    def __init__(self, pretrained=Config.PRETRAINED):
        """
        Initialize the BirdResNet model.

        Args:
            pretrained (bool): If True, load ImageNet pretrained weights.
                               Defaults to Config.PRETRAINED.
        """
        super(BirdResNet, self).__init__()

        # Load ResNet-50 backbone
        if pretrained:
            weights = ResNet50_Weights.DEFAULT
        else:
            weights = None

        self.backbone = resnet50(weights=weights)

        # Input Channel Handling:
        # The provided BirdDataset in library/dataset.py replicates the 1-channel
        # spectrogram to 3 channels (RGB) using .repeat(3, 1, 1).
        # Therefore, we retain the standard ResNet conv1 which accepts 3 channels.
        #
        # Note: If the dataset were to provide 1-channel input directly, we would
        # modify the first layer as follows:
        # self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace the final Fully Connected layer (Classification Head)
        # ResNet-50's final layer is named 'fc' and has 2048 input features.
        # We replace it with a Linear layer mapping to the number of bird species.
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input batch of shape (N, 3, H, W).
                              The dataset provides 3-channel inputs.

        Returns:
            torch.Tensor: Raw logits of shape (N, NUM_CLASSES).
        """
        return self.backbone(x)
