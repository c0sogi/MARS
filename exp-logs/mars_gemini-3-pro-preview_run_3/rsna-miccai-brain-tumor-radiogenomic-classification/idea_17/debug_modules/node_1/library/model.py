import torch
import torch.nn as nn
import timm
from library.config import Config


class SSVEModel(nn.Module):
    """
    Stochastic Strided-View Ensemble (SSVE) Network.

    This model utilizes a 2.5D CNN architecture based on EfficientNet-B0.
    It is designed to handle high-channel inputs (64 channels) corresponding to
    16 spatial slices across 4 MRI modalities.

    Key Features:
    - Backbone: EfficientNet-B0 (via timm).
    - Input Adaptation: Adapts the first convolutional layer (conv_stem) to 64 channels
      using weight recycling (handled by timm's in_chans argument).
    - Regularization: Applies Stochastic Depth (Drop Path) for better generalization.
    - Output: Returns logits for Binary Cross Entropy with Logits Loss.
    """

    def __init__(self):
        super(SSVEModel, self).__init__()

        # Initialize EfficientNet-B0 backbone
        # in_chans=Config.IN_CHANS (64): Triggers timm's weight recycling to adapt
        # the first layer from 3 channels to 64, ensuring stability.
        # drop_path_rate=Config.DROP_PATH_RATE (0.2): Enables Stochastic Depth.
        # num_classes=Config.NUM_CLASSES (1): Single output unit for binary classification.
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=True,
            in_chans=Config.IN_CHANS,
            num_classes=Config.NUM_CLASSES,
            drop_path_rate=Config.DROP_PATH_RATE,
            global_pool="avg",
        )

    def forward(self, x):
        """
        Forward pass of the SSVE Model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, 64, 256, 256).
                              64 channels = 16 slices * 4 modalities.

        Returns:
            torch.Tensor: Output logits of shape (Batch_Size, 1).
        """
        # Pass through the backbone
        # The backbone handles feature extraction, global pooling, and the final linear classifier.
        return self.backbone(x)
