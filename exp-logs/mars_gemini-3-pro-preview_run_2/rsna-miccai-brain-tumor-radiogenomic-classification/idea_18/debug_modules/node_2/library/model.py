import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from library.config import Config


class SiameseEfficientNet(nn.Module):
    """
    Siamese Asymmetric EfficientNet-B0 with Dual-Hypothesis Fusion.

    Architecture:
    1. Backbone: EfficientNet-B0 (Pretrained).
    2. Stem: Modified to accept 12 channels (4 modalities * 3 slices) using Grouped Convolutions (groups=4).
       - Weights initialized via Asymmetric Filter Distribution (direct copy of 32 RGB filters).
    3. Siamese Logic: Shared weights process 'Bulk' (FLAIR) and 'Core' (T1wCE) views.
    4. Fusion: Late fusion of feature vectors via concatenation followed by a Dropout + Linear head.
    """

    def __init__(self):
        super(SiameseEfficientNet, self).__init__()

        # 1. Load Pretrained Backbone
        weights = EfficientNet_B0_Weights.DEFAULT
        self.backbone = efficientnet_b0(weights=weights)

        # 2. Modify Stem (First Convolutional Layer)
        # Original: Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # Target:   Conv2d(12, 32, kernel_size=3, stride=2, padding=1, groups=4, bias=False)

        original_stem = self.backbone.features[0][0]

        new_stem = nn.Conv2d(
            in_channels=Config.IN_CHANNELS,  # 12
            out_channels=32,  # 32
            kernel_size=3,
            stride=2,
            padding=1,
            groups=4,  # Enforce modality isolation
            bias=False,
        )

        # 3. Asymmetric Filter Initialization
        # The weight shape for Grouped Conv2d is (out_channels, in_channels/groups, k, k).
        # Here: (32, 12/4, 3, 3) -> (32, 3, 3, 3).
        # This matches the original EfficientNet stem weights exactly.
        # By copying directly, we distribute the 32 diverse ImageNet filters across the 4 groups.
        # Filters 0-7 process Group 0 (FLAIR), Filters 8-15 process Group 1 (T1w), etc.
        with torch.no_grad():
            new_stem.weight.copy_(original_stem.weight)

        # Replace the stem in the backbone
        self.backbone.features[0][0] = new_stem

        # 4. Define Dual-Hypothesis Fusion Head
        # EfficientNet-B0 outputs a 1280-dim vector after pooling.
        # We concatenate two views, so input dim is 1280 * 2 = 2560.
        self.fusion_head = nn.Sequential(
            nn.Dropout(p=Config.DROPOUT_RATE), nn.Linear(1280 * 2, 1)
        )

        # Remove original classifier to prevent unused parameter overhead/warnings
        self.backbone.classifier = nn.Identity()

    def forward_one(self, x):
        """
        Forward pass for a single view through the backbone.
        """
        # x shape: (Batch, 12, 224, 224)

        # Feature extraction
        # Output shape: (Batch, 1280, 7, 7)
        features = self.backbone.features(x)

        # Global Average Pooling
        # Output shape: (Batch, 1280, 1, 1)
        pooled = self.backbone.avgpool(features)

        # Flatten
        # Output shape: (Batch, 1280)
        flattened = torch.flatten(pooled, 1)

        return flattened

    def forward(self, x_bulk, x_core):
        """
        Siamese forward pass.
        Args:
            x_bulk: Tensor (B, 12, 224, 224) - The Anatomical/Edema view.
            x_core: Tensor (B, 12, 224, 224) - The Pathological/Tumor Core view.
        Returns:
            logits: Tensor (B, 1)
        """
        # 1. Process views with shared backbone
        v_bulk = self.forward_one(x_bulk)
        v_core = self.forward_one(x_core)

        # 2. Concatenate features (Dual-Hypothesis Fusion)
        v_fused = torch.cat([v_bulk, v_core], dim=1)

        # 3. Classification Head
        logits = self.fusion_head(v_fused)

        return logits
