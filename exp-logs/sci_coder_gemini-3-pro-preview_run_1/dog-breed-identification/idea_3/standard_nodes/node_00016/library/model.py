import torch
import torch.nn as nn
import timm
from library.config import Config


class DogModel(nn.Module):
    """
    A wrapper class for the ConvNeXt backbone with a custom classification head.
    """

    def __init__(self, pretrained=True):
        super(DogModel, self).__init__()

        # Load the ConvNeXt backbone using timm.
        # num_classes=0 removes the default classification head (Linear) and returns
        # the pooled feature vector (after GlobalAvgPool and LayerNorm).
        self.backbone = timm.create_model(
            Config.MODEL_NAME, pretrained=pretrained, num_classes=0
        )

        # The number of output features from the backbone
        in_features = self.backbone.num_features

        # Define the custom head: Dropout -> Linear
        self.dropout = nn.Dropout(p=Config.HEAD_DROPOUT)
        self.fc = nn.Linear(in_features, Config.NUM_CLASSES)

    def forward(self, x):
        # Forward pass through the backbone to get features
        features = self.backbone(x)

        # Apply Dropout
        x = self.dropout(features)

        # Apply Final Classification Layer
        logits = self.fc(x)

        return logits


def build_model(pretrained=True):
    """
    Builds and returns the Dog Breed Classification model.

    Args:
        pretrained (bool): If True, loads pre-trained ImageNet weights for the backbone.

    Returns:
        nn.Module: The constructed PyTorch model.
    """
    model = DogModel(pretrained=pretrained)
    return model
