import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import (
    MODEL_NAME,
    MODALITY_DROPOUT_PROB,
    HEAD_DROPOUT_PROB,
    NUM_CLASSES,
    SEED,
)

# -----------------------------------------------------------------------------
# Custom Layers
# -----------------------------------------------------------------------------


class ModalityDropout(nn.Module):
    """
    Drops an entire modality group from the feature maps.

    Assumes the input x comes from a grouped convolution stem where:
    - Input channels: 12 (4 modalities * 3 slices)
    - Output channels: 32
    - Groups: 4

    Therefore, the 32 output channels are divided into 4 groups of 8 channels each.
    Group 0 (0-7): Modality 1 (FLAIR)
    Group 1 (8-15): Modality 2 (T1w)
    Group 2 (16-23): Modality 3 (T1wCE)
    Group 3 (24-31): Modality 4 (T2w)
    """

    def __init__(self, p=0.2, num_groups=4):
        super().__init__()
        self.p = p
        self.num_groups = num_groups

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x

        # x shape: (B, C, H, W), where C=32
        B, C, H, W = x.shape
        channels_per_group = C // self.num_groups

        # We want to drop each group with probability p
        # However, the prompt implies dropping *one* modality group or potentially multiple?
        # "randomly zeroes out an entire modality group" usually implies independent Bernoulli per group
        # or selecting one to drop. Standard dropout is Bernoulli.

        # Create a mask of shape (B, num_groups)
        # Keep probability is 1-p
        mask_groups = torch.bernoulli(
            torch.full((B, self.num_groups), 1 - self.p, device=x.device)
        )

        # Expand mask to channels: (B, num_groups) -> (B, num_groups, 1) -> (B, num_groups, channels_per_group)
        mask_expanded = mask_groups.unsqueeze(-1).expand(-1, -1, channels_per_group)

        # Flatten to (B, C)
        mask_final = mask_expanded.reshape(B, C)

        # Expand to spatial dims: (B, C, 1, 1)
        mask_final = mask_final.unsqueeze(-1).unsqueeze(-1)

        # Apply mask. We scale by 1/(1-p) to maintain expected value (inverted dropout)
        # to ensure the next layers receive consistent magnitude.
        scale = 1.0 / (1 - self.p)
        return x * mask_final * scale


# -----------------------------------------------------------------------------
# Model Architecture
# -----------------------------------------------------------------------------


class AsymmetricEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. Load Backbone
        # Use pretrained=True to get ImageNet weights
        self.backbone = timm.create_model(
            MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES
        )

        # 2. Modify Stem for 12 Channels (Grouped Conv)
        # Original stem: Conv2d(3, 32, k=3, s=2, p=1, bias=False)
        old_stem = self.backbone.conv_stem

        # New stem: Conv2d(12, 32, k=3, s=2, p=1, groups=4, bias=False)
        # We assume 4 modalities * 3 channels = 12 input channels.
        # Groups=4 ensures Modality 0 connects only to Output Filters 0-7, etc.
        self.backbone.conv_stem = nn.Conv2d(
            in_channels=12,
            out_channels=old_stem.out_channels,
            kernel_size=old_stem.kernel_size,
            stride=old_stem.stride,
            padding=old_stem.padding,
            groups=4,
            bias=(old_stem.bias is not None),
        )

        # 3. Asymmetric Initialization
        # The weight shape for Grouped Conv is (Out, In/Groups, k, k).
        # For original: (32, 3, 3, 3).
        # For new (groups=4): (32, 12/4, 3, 3) -> (32, 3, 3, 3).
        # The shapes are identical. We copy the weights directly.
        # This maps Pretrained Filters 0-7 to Modality 0, Filters 8-15 to Modality 1, etc.
        self.backbone.conv_stem.weight.data = old_stem.weight.data.clone()

        # 4. Insert Modality Dropout
        self.modality_dropout = ModalityDropout(p=MODALITY_DROPOUT_PROB, num_groups=4)

        # 5. Regularized Head
        # Replace the final classifier with Dropout + Linear
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=HEAD_DROPOUT_PROB), nn.Linear(in_features, NUM_CLASSES)
        )

    def forward(self, x):
        # We need to manually run the stem to insert Modality Dropout
        # EfficientNet B0 structure in timm:
        # x -> conv_stem -> bn1 -> act1 -> blocks -> conv_head -> bn2 -> act2 -> global_pool -> classifier

        # 1. Stem
        x = self.backbone.conv_stem(x)
        x = self.backbone.bn1(x)
        x = F.silu(x, inplace=True)

        # 2. Modality Dropout (Structural Innovation)
        x = self.modality_dropout(x)

        # 3. Blocks
        x = self.backbone.blocks(x)

        # 4. Head Features
        x = self.backbone.conv_head(x)
        x = self.backbone.bn2(x)
        x = F.silu(x, inplace=True)

        # 5. Pooling & Classification
        x = self.backbone.global_pool(x)

        # Note: timm's global_pool usually returns (B, C).
        # If it returns (B, C, 1, 1), we flatten.
        if x.dim() == 4:
            x = x.flatten(1)

        # The classifier contains the dropout
        x = self.backbone.classifier(x)

        return x


def get_model():
    """
    Factory function to create the model.
    """
    model = AsymmetricEfficientNet()
    return model
