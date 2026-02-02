import torch
import torch.nn as nn
import timm
from library.config import IN_CHANNELS, DROPOUT_RATE


class AGIVEfficientNet(nn.Module):
    """
    Standard EfficientNet-B0 Wrapper.

    Cite solution_lesson_node_00025: Simplified architecture.
    Uses standard 3-channel input (IN_CHANNELS=3) to leverage ImageNet priors directly.
    """

    def __init__(self, model_name="efficientnet_b0", pretrained=True):
        super(AGIVEfficientNet, self).__init__()

        # Initialize the backbone with 1 output class for binary classification (logits)
        # Cite solution_lesson_node_00012: Explicitly pass drop_rate to override defaults
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=1,
            in_chans=IN_CHANNELS,
            drop_rate=DROPOUT_RATE,
        )

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W)
        Returns:
            torch.Tensor: Logits of shape (B, 1)
        """
        return self.backbone(x)


def build_model(device):
    """
    Factory function to build and move the model to the specified device.
    """
    model = AGIVEfficientNet()
    model.to(device)
    return model
