import torch
import torch.nn as nn
import timm
from library.config import Config


class PyramidSiameseEfficientNet(nn.Module):
    """
    Pyramid Symmetry-Difference Siamese Network using EfficientNet-B2 backbone.

    Architecture:
    1. Shared Backbone (Siamese): EfficientNet-B2
    2. Inputs: Target Image, Contralateral Image (3 channels: Image + Age + Implant)
    3. Multi-Scale Feature Extraction: P3 (Stride 8), P4 (Stride 16), P5 (Stride 32)
    4. Difference Module: Computes (Target - Contralateral) at each scale.
    5. Fusion: Concatenates GAP(Target) and GAP(Difference) for all scales.
    6. Head: Single Linear Layer.
    """

    def __init__(self):
        super(PyramidSiameseEfficientNet, self).__init__()

        # Initialize backbone
        # features_only=True allows extraction of intermediate layers
        # out_indices=(2, 3, 4) corresponds to blocks with strides 8, 16, 32 (P3, P4, P5)
        # in_chans=3 matches the (Image, Age, Implant) input strategy
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=True,
            features_only=True,
            out_indices=(2, 3, 4),
            in_chans=Config.IN_CHANNELS,
        )

        # Retrieve channel counts for the selected indices
        # feature_info returns info for the selected out_indices
        self.feature_channels = [info["num_chs"] for info in self.backbone.feature_info]

        # Calculate the input dimension for the classifier
        # For each scale, we concatenate GAP(Target) and GAP(Difference)
        # Total Dim = Sum(Channel_i * 2) for i in [P3, P4, P5]
        total_features = sum([c * 2 for c in self.feature_channels])

        # Classification Head
        self.classifier = nn.Linear(total_features, Config.NUM_CLASSES)

    def forward_features(self, x):
        """
        Passes a single image batch through the backbone.
        Returns a list of feature maps [P3, P4, P5].
        """
        return self.backbone(x)

    def forward(self, x_target, x_contra):
        """
        Forward pass for the Siamese Network.

        Args:
            x_target (Tensor): Batch of target images (B, 3, H, W)
            x_contra (Tensor): Batch of contralateral images (B, 3, H, W)

        Returns:
            Tensor: Logits (B, 1)
        """
        # 1. Extract features for both branches (Shared Weights)
        # feats_target and feats_contra are lists of tensors: [P3, P4, P5]
        feats_target = self.forward_features(x_target)
        feats_contra = self.forward_features(x_contra)

        global_descriptors = []

        # 2. Multi-Scale Difference & Pooling
        for f_t, f_c in zip(feats_target, feats_contra):
            # Compute Signed Feature Difference
            # This captures asymmetry while suppressing symmetric background (like age signals)
            diff = f_t - f_c

            # Global Average Pooling (GAP)
            # f_t shape: (B, C, H, W) -> (B, C)
            gap_target = torch.mean(f_t, dim=(2, 3))
            gap_diff = torch.mean(diff, dim=(2, 3))

            # Collect vectors
            global_descriptors.append(gap_target)
            global_descriptors.append(gap_diff)

        # 3. Concatenation
        # Combine all scale descriptors into one global vector
        final_embedding = torch.cat(global_descriptors, dim=1)

        # 4. Classification
        logits = self.classifier(final_embedding)

        return logits
