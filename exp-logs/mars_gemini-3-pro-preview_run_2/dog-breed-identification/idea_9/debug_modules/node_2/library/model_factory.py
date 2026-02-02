import torch
import torch.nn as nn
from torchvision.models import convnext_large, ConvNeXt_Large_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from library import config


class MultiScaleFeatureExtractor(nn.Module):
    """
    A wrapper around ConvNeXt-Large to extract features from multiple stages
    of the network (Deep Feature Pyramid).

    Extracts:
        - Stage 3 (Intermediate): Texture/Pattern information.
        - Stage 4 (Final): Semantic/Shape information.

    Applies Global Average Pooling to spatial feature maps to produce embeddings.
    """

    def __init__(self):
        super().__init__()

        # Initialize backbone with specified weights
        # Using "IMAGENET1K_V1" as per config (New Recipe)
        weights = ConvNeXt_Large_Weights.IMAGENET1K_V1
        base_model = convnext_large(weights=weights)

        # Define which layers to extract
        # config.FEATURE_LAYERS maps internal layer names to logical output names
        # e.g., {'features.5': 'stage3', 'features.7': 'stage4'}
        return_nodes = config.FEATURE_LAYERS

        # Create the graph node extractor
        self.body = create_feature_extractor(base_model, return_nodes=return_nodes)

        # Pooling layer to convert (B, C, H, W) -> (B, C, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Freeze all parameters as we are using this strictly for feature extraction
        for param in self.body.parameters():
            param.requires_grad = False

        # Ensure model is in eval mode
        self.eval()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image batch of shape (B, 3, H, W)

        Returns:
            dict: Dictionary containing flattened embeddings for each stage.
                  Keys: 'stage3', 'stage4'
                  Values: Tensor of shape (B, Embedding_Dim)
        """
        # Extract spatial features
        # outputs is a dict: {'stage3': Tensor(B, 768, H, W), 'stage4': Tensor(B, 1536, H, W)}
        outputs = self.body(x)

        embeddings = {}
        for key, feature_map in outputs.items():
            # Apply Global Average Pooling
            pooled = self.gap(feature_map)

            # Flatten to vector: (B, C, 1, 1) -> (B, C)
            flat = torch.flatten(pooled, 1)

            embeddings[key] = flat

        return embeddings


def get_feature_extractor(device=config.DEVICE):
    """
    Factory function to instantiate the Multi-Scale Feature Extractor.

    Args:
        device (str): Device to move the model to ('cpu' or 'cuda').

    Returns:
        nn.Module: The configured feature extractor model.
    """
    model = MultiScaleFeatureExtractor()
    model.to(device)
    return model
