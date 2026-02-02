import torch
import torch.nn as nn
import timm
from library.config import Config


class BirdModel(nn.Module):
    """
    Deep-Stem ResNet-34d (Non-Anti-Aliased) Model for Bird Species Classification.

    Architecture:
        - Backbone: ResNet-34d (Deep Stem). Replaces the standard 7x7 input convolution
          with stacked 3x3 convolutions to better capture fine-grained temporal morphology.
        - Anti-Aliasing: Explicitly DISABLED (antialiased=False). This avoids the low-pass
          filtering (BlurPool) that was found to degrade performance by smoothing out
          sharp spectral transients required for separating overlapping calls.
        - Head: Standard Linear Layer mapping to 19 species.
    """

    def __init__(self, config: Config):
        """
        Args:
            config (Config): Configuration object containing model parameters.
        """
        super(BirdModel, self).__init__()

        self.config = config

        # Extract model configuration
        model_name = config.MODEL_NAME
        params = config.MODEL_PARAMS

        # Initialize the backbone using timm
        # We pass **params to unpack arguments like:
        # - pretrained=True
        # - num_classes=19 (Creates the specific linear head)
        # - in_chans=3
        # - global_pool='avg'
        # - antialiased=False (Critical for this strategy)
        # - drop_rate=0.0
        # - drop_path_rate=0.0
        self.backbone = timm.create_model(model_name, **params)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 3, Height, Width).
                              Expected to be normalized ImageNet-style RGB spectrograms.

        Returns:
            torch.Tensor: Logits of shape (Batch, NumClasses).
        """
        return self.backbone(x)


def create_model(config: Config):
    """
    Factory function to instantiate the BirdModel.

    Args:
        config (Config): Configuration object.

    Returns:
        BirdModel: An instance of the configured model.
    """
    model = BirdModel(config)
    return model
