import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class AnimalClassifier(nn.Module):
    """
    AnimalClassifier wraps a MobileNetV3-Large backbone for multi-class classification.
    It replaces the final classification head to output logits for the specific animal classes.
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED):
        """
        Args:
            num_classes (int): Number of output classes. Defaults to Config.NUM_CLASSES.
            pretrained (bool): Whether to use ImageNet pre-trained weights. Defaults to Config.PRETRAINED.
        """
        super(AnimalClassifier, self).__init__()

        # Select weights based on pretrained flag
        if pretrained:
            weights = models.MobileNet_V3_Large_Weights.DEFAULT
        else:
            weights = None

        # Initialize backbone
        self.model = models.mobilenet_v3_large(weights=weights)

        # The classifier in MobileNetV3 is a Sequential module.
        # Structure:
        # (0): Linear
        # (1): Hardswish
        # (2): Dropout
        # (3): Linear (This is the final projection layer we need to replace)

        classifier_seq = self.model.classifier
        final_layer_idx = 3

        # Get the input dimension of the final layer
        in_features = classifier_seq[final_layer_idx].in_features

        # Replace the final layer with a new Linear layer for our specific class count
        self.model.classifier[final_layer_idx] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, num_classes).
        """
        return self.model(x)


def get_model(device=Config.DEVICE, weights_path=None):
    """
    Factory function to initialize the model.

    Args:
        device (str): The device to move the model to (e.g., 'cuda', 'cpu').
        weights_path (str, optional): Path to a state dict file to load weights from.

    Returns:
        nn.Module: The initialized AnimalClassifier model.
    """
    model = AnimalClassifier(
        num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED
    )

    if weights_path:
        try:
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading weights from {weights_path}: {e}")
            raise e

    model.to(device)
    return model
