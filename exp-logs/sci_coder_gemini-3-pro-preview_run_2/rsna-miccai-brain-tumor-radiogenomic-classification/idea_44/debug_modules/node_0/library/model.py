import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class SiameseEfficientNet(nn.Module):
    def __init__(self):
        super(SiameseEfficientNet, self).__init__()

        # 1. Load Pre-trained Backbone
        # We use the default weights (IMAGENET1K_V1)
        weights = models.EfficientNet_B0_Weights.DEFAULT
        self.backbone = models.efficientnet_b0(weights=weights)

        # 2. Surgical Stem Replacement
        # The first layer in EfficientNet-B0 is within features[0][0]
        # features[0] is a Conv2dNormActivation block
        original_stem = self.backbone.features[0][0]

        # Create a new Conv2d layer
        # in_channels=12 (4 modalities * 3 slices)
        # groups=4 (Modality isolation)
        new_stem = nn.Conv2d(
            in_channels=Config.NUM_CHANNELS,
            out_channels=original_stem.out_channels,
            kernel_size=original_stem.kernel_size,
            stride=original_stem.stride,
            padding=original_stem.padding,
            bias=original_stem.bias is not None,
            groups=Config.STEM_GROUPS,
        )

        # 3. Direct Asymmetric Initialization
        # Original weights shape: (32, 3, 3, 3)
        # New weights shape: (32, 12/4, 3, 3) -> (32, 3, 3, 3)
        # We copy the weights directly. This assigns the 32 filters in blocks of 8
        # to each of the 4 modality groups (0-7 -> Mod1, 8-15 -> Mod2, etc.)
        with torch.no_grad():
            new_stem.weight.data = original_stem.weight.data.clone()

        # Replace the layer in the backbone
        self.backbone.features[0][0] = new_stem

        # 4. Define Fusion Head
        # EfficientNet-B0 outputs 1280 features after the pooling layer
        # We concatenate two vectors (Texture + Context), so input dim is doubled
        backbone_out_features = self.backbone.classifier[1].in_features
        fusion_input_dim = backbone_out_features * 2

        self.fusion_head = nn.Sequential(
            nn.Dropout(p=Config.DROPOUT_RATE), nn.Linear(fusion_input_dim, 1)
        )

        # Disable the original classifier to avoid unused parameter overhead
        self.backbone.classifier = nn.Identity()

    def _forward_features(self, x):
        """
        Helper to run the backbone feature extraction:
        Conv Features -> Global Avg Pool -> Flatten
        """
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x_texture, x_context):
        """
        Siamese Forward Pass.

        Args:
            x_texture: Tensor of shape (B, 12, 224, 224) - Stride 2 input
            x_context: Tensor of shape (B, 12, 224, 224) - Stride 5 input

        Returns:
            logits: Tensor of shape (B, 1)
        """
        # Shared Backbone Feature Extraction
        feat_texture = self._forward_features(x_texture)
        feat_context = self._forward_features(x_context)

        # Feature Fusion (Concatenation)
        combined = torch.cat([feat_texture, feat_context], dim=1)

        # Classification Head
        logits = self.fusion_head(combined)

        return logits
