import torch
import torch.nn as nn
import timm
from library.config import Config


class WSILModel(nn.Module):
    """
    Wide-Field Stratified Instance Learning (WSIL) Model.

    This architecture uses an EfficientNet-B0 backbone initialized with ImageNet weights.
    It is designed to process 3-channel composite MRI images (FLAIR, T1wCE, T2w) and
    output a single logit for binary classification.
    """

    def __init__(self, model_name=Config.MODEL_NAME, pretrained=Config.PRETRAINED):
        super(WSILModel, self).__init__()

        # Initialize the backbone using timm
        # in_chans=3: Corresponds to the stacked [FLAIR, T1wCE, T2w] slices
        # num_classes=1: Binary classification (MGMT value 0 or 1)
        # drop_rate: Regularization for the classifier head
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=Config.INPUT_CHANNELS,
            num_classes=1,
            drop_rate=Config.DROPOUT_RATE,
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, 3, H, W).

        Returns:
            torch.Tensor: Raw logits of shape (Batch_Size, 1).
                          Sigmoid activation should be applied externally (e.g., via BCEWithLogitsLoss).
        """
        return self.backbone(x)


def build_model(device=Config.DEVICE):
    """
    Factory function to create the WSIL model and move it to the configured device.

    Args:
        device (str): The target device ('cpu' or 'cuda'). Defaults to Config.DEVICE.

    Returns:
        nn.Module: The initialized model moved to the specified device.
    """
    model = WSILModel()
    model.to(device)
    return model
