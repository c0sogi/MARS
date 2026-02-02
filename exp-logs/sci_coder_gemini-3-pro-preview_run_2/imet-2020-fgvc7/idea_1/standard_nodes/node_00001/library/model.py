import torch
import torch.nn as nn
import timm


class ArtworkClassifier(nn.Module):
    """
    ArtworkClassifier model based on EfficientNetV2-S.

    This class loads a pre-trained backbone from the timm library,
    removes the original classification head, and replaces it with a
    fully connected layer matching the number of artwork attributes.
    """

    def __init__(self, model_name, num_classes, pretrained=True):
        """
        Args:
            model_name (str): Name of the model architecture (e.g., 'tf_efficientnetv2_s').
            num_classes (int): Number of output classes (attributes).
            pretrained (bool): Whether to load pre-trained ImageNet weights.
        """
        super(ArtworkClassifier, self).__init__()

        # Load the pre-trained model
        self.model = timm.create_model(model_name, pretrained=pretrained)

        # Replace the original classification head with a new Linear layer
        # reset_classifier automatically handles finding the correct input features
        # and replacing the head (e.g., 'classifier' or 'fc') with a new Linear layer.
        self.model.reset_classifier(num_classes=num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, num_classes).
        """
        return self.model(x)
