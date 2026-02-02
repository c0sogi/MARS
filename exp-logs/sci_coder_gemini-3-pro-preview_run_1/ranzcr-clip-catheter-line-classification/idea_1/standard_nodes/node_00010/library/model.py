import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class CatheterModel(nn.Module):
    """
    CNN based model for catheter detection.
    Replaces the default classifier with a single dense layer for multi-label classification.
    Cite solution_lesson_node_00002
    """

    def __init__(self, pretrained=Config.PRETRAINED):
        """
        Initialize the model.

        Args:
            pretrained (bool): Whether to load ImageNet pre-trained weights.
        """
        super(CatheterModel, self).__init__()

        # Load Backbone dynamically based on Config
        # Cite solution_lesson_node_00003: Decouple model selection from class implementation
        if pretrained:
            weights = "DEFAULT"
        else:
            weights = None

        try:
            model_fn = getattr(models, Config.MODEL_NAME)
        except AttributeError:
            raise ValueError(
                f"Model {Config.MODEL_NAME} not found in torchvision.models"
            )

        self.backbone = model_fn(weights=weights)

        # The torchvision EfficientNet implementation consists of:
        # features -> avgpool -> flatten -> classifier
        # We want to keep features and avgpool (GAP), but replace the classifier.

        # Get the input features of the original classifier.
        # Cite debug_lesson_1: Avoid Hardcoding Layer Indices When Swapping Torchvision Backbones
        in_features = None
        if isinstance(self.backbone.classifier, nn.Sequential):
            for layer in self.backbone.classifier:
                if isinstance(layer, nn.Linear):
                    in_features = layer.in_features
                    break
        elif isinstance(self.backbone.classifier, nn.Linear):
            in_features = self.backbone.classifier.in_features

        if in_features is None:
            raise AttributeError(
                f"Could not find nn.Linear in classifier of {Config.MODEL_NAME}"
            )

        # Replace the classifier with a single Dense layer.
        # This satisfies the requirement: "Global Average Pooling layer followed immediately by a single dense layer"
        self.backbone.classifier = nn.Linear(in_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input batch of images (B, C, H, W)

        Returns:
            torch.Tensor: Raw logits (B, NUM_CLASSES)
        """
        # The torchvision implementation of mobilenet_v3_large.forward does:
        # x = self.features(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        return self.backbone(x)
