import torch
import torch.nn as nn
import timm
import library.config as config


class MGMTNet(nn.Module):
    """
    A 2.5D Convolutional Neural Network for MGMT promoter methylation prediction.

    This model uses an EfficientNet-B0 backbone. The first convolutional layer is
    adapted to accept 12 input channels (3 slices x 4 modalities), and the
    classifier head is replaced with a single linear layer for binary classification.
    """

    def __init__(self, model_name="efficientnet_b0", pretrained=True):
        """
        Args:
            model_name (str): Name of the timm model to load.
            pretrained (bool): Whether to load pretrained ImageNet weights.
        """
        super(MGMTNet, self).__init__()

        # Instantiate the model using timm
        # in_chans=12: Modifies the first layer to accept 12 channels.
        #              Weights for the new channels are typically initialized by
        #              recycling the original RGB weights (e.g., repeating/averaging).
        # num_classes=1: Replaces the classification head with a Linear(in_features, 1).
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=config.IN_CHANNELS,
            num_classes=1,
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 12, H, W).

        Returns:
            torch.Tensor: Logits of shape (Batch, 1).
        """
        # Pass through the backbone (features -> global pool -> classifier)
        return self.backbone(x)
