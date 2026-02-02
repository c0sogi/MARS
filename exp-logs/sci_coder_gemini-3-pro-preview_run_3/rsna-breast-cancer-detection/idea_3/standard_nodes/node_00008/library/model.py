import torch
import torch.nn as nn
import timm
from library.config import Config


class BreastCancerModel(nn.Module):
    """
    Breast Cancer Detection Model based on EfficientNetV2.

    This model uses the timm library to create a backbone (EfficientNetV2-Small)
    and modifies the head for binary classification.
    """

    def __init__(self, pretrained: bool = Config.PRETRAINED):
        """
        Initialize the model.

        Args:
            pretrained (bool): Whether to load pretrained ImageNet weights.
                               Defaults to Config.PRETRAINED.
        """
        super(BreastCancerModel, self).__init__()

        # Initialize the backbone using timm
        # Config.MODEL_NAME is "efficientnet_v2_s"
        # Config.IN_CHANNELS is 3 (RGB)
        # Config.NUM_CLASSES is 1 (Binary Classification: Cancer/No Cancer)
        self.model = timm.create_model(
            model_name=Config.MODEL_NAME,
            pretrained=pretrained,
            in_chans=Config.IN_CHANNELS,
            num_classes=Config.NUM_CLASSES,
            drop_rate=Config.DROPOUT_RATE,
            drop_path_rate=Config.DROP_PATH_RATE,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input batch of images. Shape: (B, C, H, W)

        Returns:
            torch.Tensor: Raw logits. Shape: (B, 1)
        """
        return self.model(x)
