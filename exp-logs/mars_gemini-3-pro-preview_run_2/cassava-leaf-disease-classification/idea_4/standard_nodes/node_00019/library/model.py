import torch
import torch.nn as nn
import timm
from library.config import Config


class CassavaConvNext(nn.Module):
    """
    CassavaConvNext model based on the ConvNeXt architecture.

    This implementation uses ConvNeXt-Tiny initialized with ImageNet weights.
    It incorporates Stochastic Depth (Drop Path) regularization to prevent
    overfitting in high-capacity models, as specified in the configuration.
    """

    def __init__(self, cfg: Config):
        """
        Initializes the model.

        Args:
            cfg (Config): Configuration object containing model hyperparameters
                          (model_name, pretrained, num_classes, drop_path_rate).
        """
        super(CassavaConvNext, self).__init__()

        # Initialize the model using timm
        # drop_path_rate enables Stochastic Depth, randomly dropping residual paths
        # during training to act as an ensemble of shallower networks.
        self.model = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            num_classes=cfg.num_classes,
            drop_path_rate=cfg.drop_path_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input batch of images [B, C, H, W]

        Returns:
            torch.Tensor: Raw logits [B, num_classes]
        """
        return self.model(x)
