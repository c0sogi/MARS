import torch
import torch.nn as nn
import timm
from library.config import Config


class CassavaNet(nn.Module):
    """
    Cassava Leaf Disease Classification Model.
    Wraps a timm backbone (ConvNeXt Small) with a classification head.
    """

    def __init__(self, cfg, pretrained=True):
        """
        Initialize the model.

        Args:
            cfg: Configuration object containing model parameters.
            pretrained (bool): Whether to load pretrained ImageNet weights.
        """
        super(CassavaNet, self).__init__()

        # Create the backbone model using timm
        # drop_path_rate controls Stochastic Depth (regularization for deep networks)
        # drop_rate controls Dropout in the classification head
        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=pretrained,
            num_classes=cfg.num_classes,
            drop_path_rate=cfg.drop_path_rate,
            drop_rate=cfg.dropout,
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input batch of images [B, C, H, W].

        Returns:
            torch.Tensor: Logits for the 5 classes [B, num_classes].
        """
        return self.backbone(x)


def get_model(cfg, pretrained=True):
    """
    Factory function to create and return the CassavaNet model.

    Args:
        cfg: Configuration object.
        pretrained (bool): Whether to load pretrained weights.

    Returns:
        nn.Module: The initialized model.
    """
    model = CassavaNet(cfg, pretrained=pretrained)
    return model
