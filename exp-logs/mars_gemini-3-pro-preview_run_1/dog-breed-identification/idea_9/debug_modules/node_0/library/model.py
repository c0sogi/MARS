import torch
import torch.nn as nn
import timm
from library.config import Config


class DogModel(nn.Module):
    """
    Dog Breed Classification Model based on ConvNeXt-Base.

    Architecture:
    - Backbone: ConvNeXt-Base (ImageNet-1k weights)
    - Head: Dropout -> Linear Layer
    """

    def __init__(self, config: Config, pretrained: bool = True):
        """
        Args:
            config (Config): Configuration object containing model hyperparameters.
            pretrained (bool): Whether to load pretrained ImageNet-1k weights.
        """
        super(DogModel, self).__init__()

        # Load ConvNeXt-Base backbone
        # num_classes=0 removes the default head
        # global_pool='avg' ensures we get a feature vector
        # drop_path_rate is used for Stochastic Depth regularization
        self.backbone = timm.create_model(
            config.model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            drop_path_rate=config.drop_path_rate,
        )

        # Retrieve the number of output features from the backbone
        in_features = self.backbone.num_features

        # Define the custom classification head
        # Dropout for regularization followed by the final classification layer
        self.head = nn.Sequential(
            nn.Dropout(p=config.head_dropout),
            nn.Linear(in_features, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images (Batch, Channels, Height, Width).

        Returns:
            torch.Tensor: Logits (Batch, Num_Classes).
        """
        # Extract features using the backbone
        features = self.backbone(x)

        # Pass features through the classification head
        logits = self.head(features)

        return logits


def create_model(config: Config, pretrained: bool = True) -> DogModel:
    """
    Factory function to create the DogModel instance.

    Args:
        config (Config): Configuration object.
        pretrained (bool): Whether to use pretrained weights.

    Returns:
        DogModel: Instantiated model.
    """
    model = DogModel(config, pretrained=pretrained)
    return model
