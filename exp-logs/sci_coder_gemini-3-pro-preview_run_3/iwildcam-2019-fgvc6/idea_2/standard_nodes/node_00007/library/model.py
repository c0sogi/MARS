import torch
import torch.nn as nn
import timm
from library.config import Config


class AnimalClassifier(nn.Module):
    """
    AnimalClassifier model based on EfficientNetV2-Small.

    This class utilizes the `timm` library to instantiate a pre-trained
    `tf_efficientnetv2_s` backbone and replaces the final classification head
    to output logits for the target classes defined in Config.
    """

    def __init__(self, pretrained=Config.PRETRAINED):
        """
        Initialize the model.

        Args:
            pretrained (bool): Whether to load pretrained ImageNet weights.
                               Defaults to Config.PRETRAINED.
        """
        super(AnimalClassifier, self).__init__()

        # Create the model using timm
        # - model_name: defined in Config (tf_efficientnetv2_s)
        # - pretrained: loads ImageNet weights if True
        # - num_classes: replaces the head to output 23 classes
        # - drop_rate: dropout rate for the classification head
        # - drop_path_rate: stochastic depth rate for training stability
        self.backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=pretrained,
            num_classes=Config.NUM_CLASSES,
            drop_rate=Config.DROPOUT_RATE,
            drop_path_rate=Config.DROP_PATH_RATE,
        )

    def forward(self, x):
        """
        Forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, Channels, Height, Width).

        Returns:
            torch.Tensor: Output logits of shape (Batch_Size, Num_Classes).
        """
        return self.backbone(x)
