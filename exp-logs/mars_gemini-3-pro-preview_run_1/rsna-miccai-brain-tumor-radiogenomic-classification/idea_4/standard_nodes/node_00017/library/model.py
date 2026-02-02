import torch
import torch.nn as nn
import timm
from library.config import Config


class MGMTNet(nn.Module):
    """
    Neural Network for MGMT Promoter Methylation Prediction.
    Wraps a timm EfficientNet-B0 backbone to process 2D slices composed of
    stacked MRI modalities (FLAIR, T1wCE, T2w).
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAME,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
        in_chans=Config.NUM_CHANNELS,
        drop_rate=Config.DROPOUT_RATE,
    ):
        """
        Initialize the model.

        Args:
            model_name (str): Name of the model architecture (default: 'efficientnet_b0').
            pretrained (bool): Whether to load ImageNet weights (default: True).
            num_classes (int): Number of output classes (default: 1 for binary logit).
            in_chans (int): Number of input channels (default: 3).
            drop_rate (float): Dropout rate for the classifier head (default: 0.2).
        """
        super(MGMTNet, self).__init__()

        # Create the backbone using timm
        # This handles the loading of pretrained weights and modification of the head
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_chans,
            drop_rate=drop_rate,
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, C, H, W).
                              For inference with multiple slices, the calling loop
                              should reshape (B, N, C, H, W) -> (B*N, C, H, W) before passing.

        Returns:
            torch.Tensor: Output logits of shape (Batch, num_classes).
        """
        return self.model(x)
