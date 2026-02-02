import torch
import torch.nn as nn
import timm
from library.config import Config


class CassavaModel(nn.Module):
    """
    Model wrapper for Cassava Leaf Disease Classification.
    Supports ViT, BEiT, and ConvNeXt architectures via timm.
    """

    def __init__(
        self,
        model_name,
        num_classes=Config.NUM_CLASSES,
        pretrained=True,
        drop_path_rate=Config.DROP_PATH_RATE,
    ):
        """
        Args:
            model_name (str): Name of the model architecture (e.g., 'vit_base_patch16_384').
            num_classes (int): Number of target classes.
            pretrained (bool): Whether to load pretrained ImageNet weights.
            drop_path_rate (float): Stochastic depth rate for regularization.
        """
        super(CassavaModel, self).__init__()

        self.model_name = model_name

        # Initialize the model using timm
        # timm handles the replacement of the head (num_classes)
        # and the configuration of drop_path_rate automatically.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input batch of images [B, C, H, W].

        Returns:
            torch.Tensor: Logits [B, num_classes].
        """
        return self.backbone(x)
