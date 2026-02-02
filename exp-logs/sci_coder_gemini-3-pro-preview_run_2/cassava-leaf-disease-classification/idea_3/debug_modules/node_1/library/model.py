import torch
import torch.nn as nn
import timm
from library.config import CFG


class CassavaConvNeXt(nn.Module):
    """
    CassavaConvNeXt model class based on the ConvNeXt architecture.

    This class encapsulates a ConvNeXt model created via the `timm` library.
    It supports loading pre-trained weights, configuring the number of output classes,
    and applying Stochastic Depth regularization.
    """

    def __init__(
        self,
        model_name: str = CFG.model_name,
        pretrained: bool = CFG.pretrained,
        num_classes: int = CFG.num_classes,
        drop_path_rate: float = CFG.drop_path_rate,
    ):
        """
        Initialize the CassavaConvNeXt model.

        Args:
            model_name (str): Name of the model architecture in timm (default: CFG.model_name).
            pretrained (bool): Whether to load pre-trained ImageNet weights (default: CFG.pretrained).
            num_classes (int): Number of target classes for the final head (default: CFG.num_classes).
            drop_path_rate (float): Stochastic Depth rate (default: CFG.drop_path_rate).
        """
        super(CassavaConvNeXt, self).__init__()

        # Create the model using timm
        # timm handles the modification of the head (num_classes) and
        # the initialization of weights (pretrained) internally.
        # drop_path_rate is passed as a keyword argument to configure Stochastic Depth.
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Height, Width).

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes).
        """
        return self.model(x)
