import torch
import torch.nn as nn
import timm
from library.config import Config


class DogModel(nn.Module):
    """
    Dog Breed Classification Model based on ConvNeXt architecture.
    """

    def __init__(self, model_name=Config.MODEL_NAME, pretrained=True):
        """
        Initializes the model.

        Args:
            model_name (str): Name of the model architecture in timm.
            pretrained (bool): Whether to load pretrained weights.
        """
        super().__init__()

        # Load the backbone with the specified number of classes.
        # timm handles the logic of removing the original head and creating a new
        # linear layer with the correct number of outputs (Config.NUM_CLASSES).
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=Config.NUM_CLASSES
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input batch of images.

        Returns:
            torch.Tensor: Logits for each class.
        """
        return self.backbone(x)

    def freeze_backbone(self):
        """
        Freezes the backbone parameters for the warm-up phase.
        Only the classifier head remains trainable.
        """
        # Freeze all parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the classifier head
        # In timm ConvNeXt implementation, the classifier is usually named 'head'
        # We use get_classifier() to ensure compatibility
        classifier = self.backbone.get_classifier()
        for param in classifier.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self):
        """
        Unfreezes all parameters for the fine-tuning phase.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
