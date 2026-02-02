import torch
import torch.nn as nn
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
import numpy as np
import random
from library.config import Config


def set_seed(seed=Config.SEED):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ConvNeXtDualExtractor(nn.Module):
    """
    Wrapper around ConvNeXt Large that extracts and pools features from
    Stage 3 and Stage 4.
    """

    def __init__(self):
        super().__init__()

        # Load the pre-trained model using weights specified in Config
        # We use the string identifier which torchvision resolves to the specific Weights enum
        weights = Config.WEIGHTS
        base_model = torchvision.models.convnext_large(weights=weights)

        # Define the nodes to extract.
        # Based on ConvNeXt architecture in torchvision:
        # features.5 corresponds to the output of Stage 3 (Stride 16)
        # features.7 corresponds to the output of Stage 4 (Stride 32, Final Feature Map)
        return_nodes = {"features.5": "stage3", "features.7": "stage4"}

        # Create the graph wrapper
        self.feature_extractor = create_feature_extractor(
            base_model, return_nodes=return_nodes
        )

        # Define pooling and flattening layers
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

    def forward(self, x):
        """
        Args:
            x (Tensor): Input batch of images (B, C, H, W)

        Returns:
            dict: Dictionary containing pooled feature vectors:
                  {
                      'stage3': Tensor (B, D3),
                      'stage4': Tensor (B, D4)
                  }
        """
        # Extract spatial feature maps
        # outputs will be {'stage3': (B, C3, H3, W3), 'stage4': (B, C4, H4, W4)}
        outputs = self.feature_extractor(x)

        pooled_outputs = {}
        for key, feature_map in outputs.items():
            # Apply Global Average Pooling: (B, C, H, W) -> (B, C, 1, 1)
            pooled = self.pool(feature_map)
            # Flatten: (B, C, 1, 1) -> (B, C)
            flat = self.flatten(pooled)
            pooled_outputs[key] = flat

        return pooled_outputs


def create_backbone():
    """
    Factory function to create the multi-scale backbone model.

    Returns:
        nn.Module: The configured ConvNeXtDualExtractor on the CPU.
    """
    set_seed()
    model = ConvNeXtDualExtractor()
    return model
