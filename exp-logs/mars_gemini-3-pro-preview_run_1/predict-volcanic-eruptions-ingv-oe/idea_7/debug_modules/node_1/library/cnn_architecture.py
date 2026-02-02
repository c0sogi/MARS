import torch
import torch.nn as nn
import timm
from library.config import NUM_SENSORS


class VolcanoEfficientNet(nn.Module):
    """
    A wrapper around EfficientNet-B0 modified to accept multi-channel input
    corresponding to seismic sensors for regression tasks.

    This architecture allows the model to process the full sensor array as a
    multi-channel image (Batch, Sensors, Freq, Time).
    """

    def __init__(
        self,
        model_name="efficientnet_b0",
        pretrained=True,
        in_chans=NUM_SENSORS,
        num_classes=1,
    ):
        """
        Initialize the VolcanoEfficientNet model.

        Args:
            model_name (str): Name of the model architecture in timm. Defaults to 'efficientnet_b0'.
            pretrained (bool): Whether to load pretrained ImageNet weights. Defaults to True.
            in_chans (int): Number of input channels (sensors). Defaults to NUM_SENSORS (10).
            num_classes (int): Number of output units. Defaults to 1 for regression.
        """
        super(VolcanoEfficientNet, self).__init__()

        # Create the model using timm.
        # Specifying in_chans causes timm to construct the first convolutional layer
        # with the correct number of input channels. If pretrained=True, it will
        # adapt the original 3-channel weights (typically by recycling/averaging)
        # to initialize the new 10-channel layer.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
            global_pool="avg",
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Freq, Time).
                              Expected Channels = 10.

        Returns:
            torch.Tensor: Output prediction of shape (Batch, 1).
        """
        # Pass input through the backbone
        # Output shape is (Batch, num_classes) -> (Batch, 1)
        output = self.backbone(x)
        return output
