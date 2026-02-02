import torch
import torch.nn as nn
import timm
from library import config


class MontageEfficientNet(nn.Module):
    """
    Neural network architecture for Glioblastoma subtype prediction using a Spatial Montage.

    This model wraps a timm-based EfficientNet backbone. It is designed to process
    a 'Montage' input: a 2.5D representation where multiple axial slices are tiled
    spatially into a grid, but kept in separate channels corresponding to MRI modalities.

    Input Shape: (Batch, 3, Height, Width)
        - Channels (3): FLAIR, T1wCE, T2w
        - Height/Width: Combined spatial dimensions of the grid (e.g., 448x448)

    Output Shape: (Batch, 1)
        - Logits for binary classification (MGMT promoter methylation presence).
    """

    def __init__(
        self, model_name=None, pretrained=None, num_classes=None, drop_rate=None
    ):
        """
        Initialize the MontageEfficientNet model.

        Args:
            model_name (str, optional): Name of the timm model architecture.
                                        Defaults to config.MODEL_NAME.
            pretrained (bool, optional): Whether to load pretrained ImageNet weights.
                                         Defaults to config.PRETRAINED.
            num_classes (int, optional): Number of output classes.
                                         Defaults to config.NUM_CLASSES.
            drop_rate (float, optional): Dropout rate for the classification head.
                                         Defaults to config.DROPOUT_RATE.
        """
        super(MontageEfficientNet, self).__init__()

        # Resolve parameters from arguments or configuration file
        self.model_name = model_name if model_name is not None else config.MODEL_NAME
        self.pretrained = pretrained if pretrained is not None else config.PRETRAINED
        self.num_classes = (
            num_classes if num_classes is not None else config.NUM_CLASSES
        )
        self.drop_rate = drop_rate if drop_rate is not None else config.DROPOUT_RATE

        # Create the backbone using timm
        # in_chans=3: The model expects 3 input channels corresponding to the 3 selected MRI modalities.
        # Although the spatial dimensions are larger due to the montage (e.g. 448x448),
        # CNNs with global pooling (like EfficientNet) handle variable spatial inputs naturally.
        self.backbone = timm.create_model(
            self.model_name,
            pretrained=self.pretrained,
            num_classes=self.num_classes,
            in_chans=3,
            drop_rate=self.drop_rate,
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 3, H, W).

        Returns:
            torch.Tensor: Output logits of shape (Batch, num_classes).
        """
        return self.backbone(x)
