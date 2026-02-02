import torch
import torch.nn as nn
import torchvision.models as models
from library.config import CNN_PARAMS


class SeismicCNN(nn.Module):
    """
    A 2D Convolutional Neural Network for seismic eruption prediction.
    Uses a ResNet18 backbone modified for 10-channel spectrogram input and scalar regression output.
    """

    def __init__(self):
        super(SeismicCNN, self).__init__()

        # Extract configuration
        model_name = CNN_PARAMS.get("model_name", "resnet18")
        use_pretrained = CNN_PARAMS.get("pretrained", True)
        in_channels = CNN_PARAMS.get("in_channels", 10)
        dropout_p = CNN_PARAMS.get("dropout", 0.2)

        # Load Backbone
        # Using 'weights' parameter as 'pretrained' is deprecated in newer torchvision versions
        if model_name == "resnet18":
            if use_pretrained:
                weights = models.ResNet18_Weights.DEFAULT
            else:
                weights = None

            self.backbone = models.resnet18(weights=weights)
        else:
            raise NotImplementedError(
                f"Model {model_name} is not supported in this implementation."
            )

        # 1. Modify the first convolutional layer
        # Standard ResNet accepts 3 channels (RGB). We need to accept 'in_channels' (10 sensors).
        # Original layer: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        original_conv1 = self.backbone.conv1

        self.backbone.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None,
        )

        # Initialize the new layer using Kaiming Normal (He initialization)
        # This is preferable when training from scratch or adapting input domains significantly
        nn.init.kaiming_normal_(
            self.backbone.conv1.weight, mode="fan_out", nonlinearity="relu"
        )

        # 2. Modify the fully connected layer for Regression
        # The original FC layer outputs 1000 classes. We need 1 scalar output.
        num_ftrs = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_p), nn.Linear(num_ftrs, 1)
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Height, Width).
                              Expected shape: (B, 10, 224, 224).

        Returns:
            torch.Tensor: Output predictions of shape (B, 1).
        """
        return self.backbone(x)
