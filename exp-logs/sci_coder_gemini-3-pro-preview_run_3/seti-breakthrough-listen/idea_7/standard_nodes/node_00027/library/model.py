import torch
import torch.nn as nn
import timm
from library.config import Config


class SiameseMultiScaleDiffNet(nn.Module):
    """
    Siamese EfficientNet-B0 with Pyramidal Spatial Difference.

    Architecture:
    1. Shared Backbone: EfficientNet-B0 (pretrained).
    2. Multi-Scale Extraction: Features from Stride 16 (Level 1) and Stride 32 (Level 2).
    3. Spatial Difference: Explicit subtraction (On - Off) at each scale.
    4. Hybrid Pooling: GAP and GMP on On-Target, Off-Target, and Difference maps.
    5. Aggregation: Concatenation of all pooled vectors.
    6. Classifier: Single Linear Layer.
    """

    def __init__(self):
        super(SiameseMultiScaleDiffNet, self).__init__()

        # Initialize the backbone with features_only=True to extract intermediate maps
        # out_indices=(4,) targets only the deepest stage (Stride 32) for EfficientNet-B0.
        # Cite solution_lesson_node_00026: Prefer Deepest-Layer Feature Comparison.
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            features_only=True,
            out_indices=(4,),
            in_chans=Config.IN_CHANNELS,
        )

        # Determine the input dimension for the final linear layer
        # We perform a dummy pass to dynamically calculate the channel counts
        # This ensures robustness if the backbone or input size changes
        dummy_input = torch.zeros(
            1, Config.IN_CHANNELS, Config.IMG_HEIGHT, Config.IMG_WIDTH
        )
        with torch.no_grad():
            # features is a list of tensors [Level1_Map, Level2_Map]
            features = self.backbone(dummy_input)

        # Calculate total embedding dimension
        # For each scale, we extract 6 vectors:
        # 1. GAP(On)
        # 2. GMP(On)
        # 3. GAP(Off)
        # 4. GMP(Off)
        # 5. GAP(Diff)
        # 6. GMP(Diff)
        self.embed_dim = 0
        for f in features:
            # f.shape is (B, C, H, W)
            # We pool to (B, C), so we add C * 6 for this scale
            self.embed_dim += f.shape[1] * 6

        # Single linear classification layer
        self.fc = nn.Linear(self.embed_dim, 1)

    def forward_features(self, x):
        """
        Passes input through the backbone and returns a list of feature maps.
        """
        return self.backbone(x)

    def forward(self, stream_a, stream_b):
        """
        Forward pass for the Siamese Multi-Scale Spatial-Difference Network.

        Args:
            stream_a (torch.Tensor): On-Target input batch (B, C, H, W)
            stream_b (torch.Tensor): Off-Target input batch (B, C, H, W)

        Returns:
            torch.Tensor: Logits (B, 1)
        """
        # Extract features for both streams
        # feats_a and feats_b are lists of tensors [Level1_Map, Level2_Map]
        feats_a = self.forward_features(stream_a)
        feats_b = self.forward_features(stream_b)

        pooled_vectors = []

        # Iterate over each scale (Level 1 and Level 2)
        for fa, fb in zip(feats_a, feats_b):
            # Compute Explicit Spatial Difference
            # F_diff = F_on - F_off
            f_diff = fa - fb

            # Apply Global Average Pooling (GAP)
            # mean over spatial dimensions (H, W) -> (B, C)
            gap_a = fa.mean(dim=(2, 3))
            gap_b = fb.mean(dim=(2, 3))
            gap_diff = f_diff.mean(dim=(2, 3))

            # Apply Global Max Pooling (GMP)
            # amax over spatial dimensions (H, W) -> (B, C)
            # GMP is crucial for detecting sparse "needle" signals
            gmp_a = fa.amax(dim=(2, 3))
            gmp_b = fb.amax(dim=(2, 3))
            gmp_diff = f_diff.amax(dim=(2, 3))

            # Collect all statistics for this scale
            pooled_vectors.extend([gap_a, gmp_a, gap_b, gmp_b, gap_diff, gmp_diff])

        # Concatenate all vectors from all scales into a single dense representation
        # Shape: (B, Total_Channels * 6)
        concat_features = torch.cat(pooled_vectors, dim=1)

        # Final classification
        logits = self.fc(concat_features)

        return logits
