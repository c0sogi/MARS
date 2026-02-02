import torch
import torch.nn as nn
import timm
from library.config import Config


class CassavaModel(nn.Module):
    """
    A unified model wrapper for Cassava Leaf Disease Classification.
    Wraps timm models (ViT, BEiT, ConvNeXt) to provide a consistent interface
    and handle specific initialization configurations like Stochastic Depth.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = 5,
        pretrained: bool = True,
        drop_path_rate: float = 0.0,
        drop_rate: float = 0.0,
    ):
        """
        Args:
            model_name (str): Name of the model architecture in timm.
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to load pretrained ImageNet weights.
            drop_path_rate (float): Rate for Stochastic Depth (DropPath).
            drop_rate (float): Dropout rate for the classification head.
        """
        super(CassavaModel, self).__init__()

        # Initialize the backbone using timm
        # timm handles the replacement of the head with num_classes
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate,
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, num_classes).
        """
        return self.backbone(x)


def get_model(model_name: str, pretrained: bool = True, num_classes: int = None):
    """
    Factory function to create a CassavaModel instance based on the project Configuration.

    Args:
        model_name (str): The name of the architecture to instantiate.
        pretrained (bool): Whether to initialize with pretrained weights.
        num_classes (int, optional): Number of classes. Defaults to Config.NUM_CLASSES.

    Returns:
        CassavaModel: The initialized model.
    """
    if num_classes is None:
        num_classes = Config.NUM_CLASSES

    # Instantiate the model with hyperparameters from Config
    model = CassavaModel(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        drop_path_rate=Config.DROP_PATH_RATE,
        drop_rate=0.0,  # Default classifier dropout to 0 unless specified otherwise
    )

    return model
