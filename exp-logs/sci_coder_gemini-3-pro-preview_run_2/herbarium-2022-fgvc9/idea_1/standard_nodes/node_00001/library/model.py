import torch
import torch.nn as nn
from torchvision import models
from library.config import MODEL_ARCH


class FeatureExtractor(nn.Module):
    """
    A wrapper class for a pre-trained backbone (EfficientNet-B0) used for feature extraction.
    The classifier head is removed, and weights are frozen to serve as a fixed feature extractor.
    """

    def __init__(self, architecture=MODEL_ARCH):
        """
        Initialize the feature extractor.

        Args:
            architecture (str): The name of the architecture to use. Defaults to 'efficientnet_b0'.
        """
        super(FeatureExtractor, self).__init__()

        if architecture == "efficientnet_b0":
            # Load the pre-trained EfficientNet-B0 model with default ImageNet weights
            weights = models.EfficientNet_B0_Weights.DEFAULT
            self.model = models.efficientnet_b0(weights=weights)
            self.output_dim = 1280
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # Remove the final classification layer (which typically includes Dropout and Linear)
        # by replacing it with an Identity layer.
        # EfficientNet's forward pass is: features -> avgpool -> flatten -> classifier.
        # Replacing classifier with Identity ensures we get the flattened feature vectors.
        self.model.classifier = nn.Identity()

        # Freeze all parameters in the model to prevent updates during 'training'
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Forward pass to extract features from input images.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Height, Width).

        Returns:
            torch.Tensor: Flattened feature vectors of shape (Batch, Feature_Dim).
        """
        return self.model(x)
