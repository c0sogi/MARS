import torch
import torch.nn as nn
import timm
from library.config import Config
from library.utils import set_seed


class BirdModel(nn.Module):
    """
    A unified model class for the Tri-Backbone Heterogeneous Ensemble.
    Supports ResNet18, DenseNet121, and EfficientNet-B0 via timm.

    Structure:
    1. Backbone (pretrained, no pooling, no head) -> returns feature maps (B, C, H, W)
    2. Custom Global Average Pooling -> (B, C, 1, 1)
    3. Flatten -> (B, C)
    4. Linear Layer -> (B, Num_Classes)
    """

    def __init__(self, model_name, pretrained=True):
        super(BirdModel, self).__init__()

        # Create the backbone using timm
        # global_pool='' ensures we get the spatial feature maps (B, C, H, W)
        # num_classes=0 removes the default fully connected head
        # in_chans ensures compatibility with the Pseudo-RGB input (3 channels)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            in_chans=Config.IN_CHANNELS,
        )

        # Retrieve the number of output channels (features) from the backbone
        # This attribute is standard in timm models
        self.in_features = self.backbone.num_features

        # Custom Head Design
        # 1. Global Average Pooling (Strictly Average, no Max/Dual)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 2. Fully Connected Layer
        # Maps feature dimension to the number of bird species (19)
        self.fc = nn.Linear(self.in_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, Num_Classes).
        """
        # Pass through backbone to get feature maps
        x = self.backbone(x)

        # Apply Global Average Pooling
        x = self.global_pool(x)

        # Flatten the output for the linear layer
        x = torch.flatten(x, 1)

        # Compute logits
        logits = self.fc(x)

        return logits


def get_bird_model(model_name, pretrained=True):
    """
    Factory function to initialize a BirdModel.
    Ensures deterministic initialization by setting the seed.

    Args:
        model_name (str): The name of the architecture (e.g., 'resnet18', 'densenet121', 'efficientnet_b0').
        pretrained (bool): Whether to initialize with pretrained weights (ImageNet).

    Returns:
        BirdModel: The initialized PyTorch model.
    """
    # Set seed for reproducible initialization of the head layer
    set_seed(Config.SEED)

    model = BirdModel(model_name, pretrained=pretrained)
    return model
