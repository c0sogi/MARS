import torch
import torch.nn as nn
import timm
from library.config import Config


class TechnosignatureModel(nn.Module):
    """
    An EfficientNet-B0 model with Global Max Pooling adapted for Technosignature Detection.
    Uses a vertically stacked input representation to handle signal drift.
    """

    def __init__(self, model_name="efficientnet_b0", pretrained=True):
        super(TechnosignatureModel, self).__init__()

        # Use timm to create an EfficientNet model
        # We modify in_chans to match our input (1 channel)
        # We use Global Max Pooling to detect sparse signals (Cite solution_lesson_node_00007)
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=Config.NUM_CHANNELS,
            num_classes=1,
            global_pool="max",
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 1, 1638, 256)

        Returns:
            torch.Tensor: Raw logits of shape (Batch, 1)
        """
        return self.model(x)
