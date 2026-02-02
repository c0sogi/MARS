import torch
import torch.nn as nn
import timm
from library.config import Config


class StochasticModalityDropout(nn.Module):
    """
    Custom dropout layer that randomly zeros out metadata channels (Age, Implant)
    during training. This prevents the model from engaging in 'shortcut learning'
    where it relies solely on demographic priors (Age) and ignores the complex
    visual signals in the mammogram.
    """

    def __init__(self, p=0.5):
        """
        Args:
            p (float): Probability of dropping the metadata channels.
        """
        super().__init__()
        self.p = p

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W).
                              Channel 0: Image
                              Channel 1: Age
                              Channel 2: Implant
        Returns:
            torch.Tensor: Processed tensor with metadata potentially zeroed out.
        """
        # Only apply dropout during training
        if self.training:
            # Randomly decide to drop metadata for the entire batch
            # We use a per-batch decision to simulate 'missing' metadata steps
            if torch.rand(1).item() < self.p:
                # Create a mask to keep Channel 0 (Image) and zero out Channels 1 & 2
                # Shape: (1, 3, 1, 1) to broadcast over Batch, Height, and Width
                mask = torch.tensor([1.0, 0.0, 0.0], device=x.device).view(1, 3, 1, 1)
                return x * mask

        return x


class EarlyFusionEfficientNet(nn.Module):
    """
    EfficientNet-B2 architecture adapted for Early Fusion of mammography images
    and spatially broadcasted metadata.
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAME,
        pretrained=True,
        dropout_prob=Config.MODALITY_DROPOUT_PROB,
        in_chans=Config.INPUT_CHANNELS,
        num_classes=Config.NUM_CLASSES,
    ):
        """
        Args:
            model_name (str): Name of the timm model to load (e.g., 'efficientnet_b2').
            pretrained (bool): Whether to load pretrained ImageNet weights.
            dropout_prob (float): Probability for the StochasticModalityDropout layer.
            in_chans (int): Number of input channels (3: Image, Age, Implant).
            num_classes (int): Number of output classes (1 for binary classification).
        """
        super().__init__()

        # 1. Stochastic Modality Dropout
        # Applied immediately after input to regularize feature learning
        self.modality_dropout = StochasticModalityDropout(p=dropout_prob)

        # 2. Backbone
        # EfficientNet-B2 provides a good balance of parameter count (receptive field)
        # and computational efficiency.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, 1).
        """
        # Apply Modality Dropout (Training only)
        x = self.modality_dropout(x)

        # Pass through the backbone CNN
        logits = self.backbone(x)

        return logits
