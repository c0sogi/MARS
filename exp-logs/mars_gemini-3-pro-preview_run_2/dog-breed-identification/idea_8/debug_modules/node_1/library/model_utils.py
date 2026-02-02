import torch
import torch.nn as nn
from torchvision.models import convnext_large
import library.config as config


class ConvNeXtMultiLayerExtractor(nn.Module):
    """
    A wrapper around ConvNeXt-Large to extract features from specific intermediate stages.
    Specifically targets 'features.5' (Stage 3) and 'features.7' (Stage 4).
    """

    def __init__(self):
        super().__init__()
        # Initialize backbone with specified weights
        # Using the string directly as supported by recent torchvision versions
        # config.MODEL_WEIGHTS is "IMAGENET1K_V1"
        backbone = convnext_large(weights=config.MODEL_WEIGHTS)

        # Access the features sequential container
        # Structure of convnext_large.features:
        # 0: Stem (Conv2dNormActivation)
        # 1: Stage 0 (Sequential)
        # 2: Downsample (Sequential)
        # 3: Stage 1 (Sequential)
        # 4: Downsample (Sequential)
        # 5: Stage 2 (Sequential) -> Corresponds to "Stage 3" (Texture/Pattern) in prompt
        # 6: Downsample (Sequential)
        # 7: Stage 3 (Sequential) -> Corresponds to "Stage 4" (Semantic/Shape) in prompt

        layers = list(backbone.features.children())

        # Branch 1: Input -> Stage 3 Output (features.5)
        # Indices 0 to 5 inclusive
        self.stage3_path = nn.Sequential(*layers[:6])

        # Branch 2: Stage 3 Output -> Stage 4 Output (features.7)
        # Indices 6 to 7 inclusive
        self.stage4_path = nn.Sequential(*layers[6:8])

        # Global Average Pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # Forward pass to Stage 3
        x_s3 = self.stage3_path(x)

        # Pool and flatten Stage 3 features
        # Shape: (B, C_s3, H, W) -> (B, C_s3, 1, 1) -> (B, C_s3)
        feat_s3 = self.pool(x_s3).flatten(1)

        # Forward pass to Stage 4 (continue from x_s3)
        x_s4 = self.stage4_path(x_s3)

        # Pool and flatten Stage 4 features
        # Shape: (B, C_s4, H, W) -> (B, C_s4, 1, 1) -> (B, C_s4)
        feat_s4 = self.pool(x_s4).flatten(1)

        # Return dictionary mapping layer names to features
        return {"stage3": feat_s3, "stage4": feat_s4}


def build_feature_extractor():
    """
    Constructs and returns the multi-layer feature extractor model.
    The model is moved to the configured device and set to eval mode.
    """
    model = ConvNeXtMultiLayerExtractor()
    model.to(config.DEVICE)
    model.eval()  # Set to evaluation mode as we are using it for feature extraction
    return model
