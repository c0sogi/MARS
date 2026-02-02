import torch
import torch.nn as nn
import timm
from library.config import MODEL_NAME, DROP_PATH_RATE, TOTAL_CHANNELS, NUM_CLASSES


class HRVANet(nn.Module):
    """
    High-Resolution View-Adaptive 2.5D Network (HR-VA-Net).

    This model uses an EfficientNet-B0 backbone modified to accept 64 input channels
    (representing 16 spatial slices across 4 MRI modalities). It prioritizes
    spatial resolution (320x320) and uses Stochastic Depth (DropPath) for regularization.
    """

    def __init__(
        self,
        model_name=MODEL_NAME,
        pretrained=True,
        in_chans=TOTAL_CHANNELS,
        num_classes=NUM_CLASSES,
        drop_path_rate=DROP_PATH_RATE,
    ):
        """
        Initialize the HR-VA-Net model.

        Args:
            model_name (str): Name of the backbone model (default: 'efficientnet_b0').
            pretrained (bool): Whether to load pretrained ImageNet weights.
            in_chans (int): Number of input channels. Default is 64 (16 slices * 4 modalities).
                            timm handles weight recycling for the first layer.
            num_classes (int): Number of output classes. Default is 1 for binary classification.
            drop_path_rate (float): Stochastic depth rate for regularization.
        """
        super(HRVANet, self).__init__()

        # Create the model using timm
        # timm automatically handles the adaptation of the first convolutional layer
        # weights when in_chans != 3.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, 64, 320, 320).

        Returns:
            torch.Tensor: Raw logits of shape (Batch_Size, 1).
        """
        # The timm backbone includes the Global Average Pooling and the Linear Classifier head.
        logits = self.backbone(x)
        return logits
