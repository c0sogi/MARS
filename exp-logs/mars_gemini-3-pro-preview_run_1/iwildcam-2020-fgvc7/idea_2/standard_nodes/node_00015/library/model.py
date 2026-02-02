import torch
import torch.nn as nn
import timm
from library import config


class AnimalClassifier(nn.Module):
    """
    Animal Classifier using EfficientNet-B3 backbone.
    Replaces the default head with Global Average Pooling (handled by timm),
    Dropout, and a Dense Linear Layer.
    """

    def __init__(
        self,
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        pretrained=True,
    ):
        """
        Args:
            model_name (str): Name of the timm model to load.
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to load pretrained ImageNet weights.
        """
        super(AnimalClassifier, self).__init__()

        # Create the backbone
        # num_classes=0 removes the final fully connected layer.
        # global_pool='avg' ensures the output is the result of global average pooling.
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        # Retrieve the number of features output by the backbone
        in_features = self.backbone.num_features

        # Define the custom classification head
        self.dropout = nn.Dropout(p=config.DROPOUT_RATE)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, num_classes).
        """
        # Extract features using the backbone (includes global average pooling)
        features = self.backbone(x)

        # Apply dropout for regularization
        features = self.dropout(features)

        # Compute final logits
        logits = self.fc(features)

        return logits


def get_model(device=config.DEVICE, pretrained=True):
    """
    Factory function to initialize the model and move it to the specified device.

    Args:
        device (str): Device to move the model to ('cpu' or 'cuda').
        pretrained (bool): Whether to use pretrained weights.

    Returns:
        nn.Module: The initialized AnimalClassifier model.
    """
    model = AnimalClassifier(pretrained=pretrained)
    model.to(device)
    return model
