import torch
import torch.nn as nn
import timm
from library.config import MODEL_NAME, NUM_CLASSES, PRETRAINED


class DetectorGuidedCNN(nn.Module):
    """
    A Detector-Guided Single-Stream CNN for animal species classification.

    This model uses a pre-trained backbone (e.g., ResNet-18) to extract features
    from an image crop defined by a detector's bounding box. It replaces the
    standard ImageNet classification head with a new linear layer matching the
    dataset's specific number of classes.
    """

    def __init__(
        self, model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=PRETRAINED
    ):
        """
        Args:
            model_name (str): Name of the backbone architecture (default: from config).
            num_classes (int): Number of target classes (default: from config).
            pretrained (bool): Whether to load pre-trained ImageNet weights (default: from config).
        """
        super(DetectorGuidedCNN, self).__init__()

        # Create the model using timm.
        # timm.create_model handles:
        # 1. Loading the specified architecture.
        # 2. Loading pre-trained weights if 'pretrained=True'.
        # 3. Configuring the global pooling layer (default is usually average pooling).
        # 4. Replacing the final fully connected layer to output 'num_classes' logits.
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, Channels, Height, Width).

        Returns:
            torch.Tensor: Logits of shape (Batch_Size, Num_Classes).
        """
        return self.backbone(x)
