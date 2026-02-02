import torch
import torch.nn as nn
import timm
from library.config import Config


class PyramidDiffSiameseNet(nn.Module):
    """
    Pyramid Symmetry-Difference Siamese Network.

    This architecture leverages the bilateral nature of mammograms to detect cancer.
    It uses a shared EfficientNet-B2 backbone to extract features from both the
    target breast and the contralateral (opposite) breast.

    Key Innovation:
    Instead of a simple subtraction at the end, it computes feature differences
    at multiple scales (P3, P4, P5). This allows the network to detect asymmetry
    at the texture level (calcifications) and the structural level (masses),
    while suppressing symmetric background tissue which is strongly correlated
    with demographic priors (Age bias).
    """

    def __init__(self):
        super(PyramidDiffSiameseNet, self).__init__()

        # ---------------------------------------------------------------------
        # 1. Siamese Backbone
        # ---------------------------------------------------------------------
        # We use 'features_only=True' to extract intermediate feature maps.
        # out_indices=(2, 3, 4) corresponds to:
        #   - P3 (Stride 8):  Fine-grained details (e.g., microcalcifications)
        #   - P4 (Stride 16): Mid-level patterns
        #   - P5 (Stride 32): Global semantic structure (e.g., masses, architecture)
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            features_only=True,
            out_indices=(2, 3, 4),
            in_chans=Config.IN_CHANNELS,
            drop_rate=Config.DROP_RATE,
            drop_path_rate=Config.DROP_PATH_RATE,
        )

        # ---------------------------------------------------------------------
        # 2. Dynamic Feature Dimension Calculation
        # ---------------------------------------------------------------------
        # We run a dummy pass to dynamically determine the number of channels
        # at each stage. This makes the code robust to backbone changes.
        dummy_input = torch.zeros(1, Config.IN_CHANNELS, 256, 256)
        with torch.no_grad():
            # features is a list of tensors: [P3, P4, P5]
            features = self.backbone(dummy_input)

        # Store channel counts [C3, C4, C5]
        self.feature_channels = [f.shape[1] for f in features]

        # ---------------------------------------------------------------------
        # 3. Pooling & Classification Head
        # ---------------------------------------------------------------------
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # We concatenate:
        #   1. GAP(Target Feature) -> Context
        #   2. GAP(Target - Contra) -> Asymmetry Signal
        # So the input dim per scale is Channels * 2.
        # Total dim is sum over all scales.
        total_embedding_dim = sum(c * 2 for c in self.feature_channels)

        self.fc = nn.Linear(total_embedding_dim, Config.NUM_CLASSES)

    def forward(self, image, image_contra):
        """
        Forward pass for the Siamese Network.

        Args:
            image (torch.Tensor): Target image batch (B, 3, H, W).
            image_contra (torch.Tensor): Contralateral image batch (B, 3, H, W).

        Returns:
            torch.Tensor: Logits (B, 1).
        """
        # ---------------------------------------------------------------------
        # 1. Shared Feature Extraction
        # ---------------------------------------------------------------------
        # Extract multi-scale features for both images using the shared backbone.
        # feats_t and feats_c are lists of tensors: [P3, P4, P5]
        feats_t = self.backbone(image)
        feats_c = self.backbone(image_contra)

        pooled_embeddings = []

        # ---------------------------------------------------------------------
        # 2. Pyramid Difference & Fusion
        # ---------------------------------------------------------------------
        # Iterate through each scale (P3, P4, P5)
        for ft, fc in zip(feats_t, feats_c):
            # A. Compute Signed Feature Difference
            # D_i = F_target - F_contra
            # This operation highlights differences (lesions) and zeros out
            # similarities (healthy tissue, age/implant metadata).
            diff = ft - fc

            # B. Global Average Pooling
            # We retain the target features to provide context to the classifier
            # (e.g., "is this a dense breast?") and the difference features
            # to indicate abnormality.
            gap_t = self.global_pool(ft).flatten(1)
            gap_d = self.global_pool(diff).flatten(1)

            pooled_embeddings.append(gap_t)
            pooled_embeddings.append(gap_d)

        # ---------------------------------------------------------------------
        # 3. Classification
        # ---------------------------------------------------------------------
        # Concatenate all vectors from all scales
        # Shape: (B, Total_Embedding_Dim)
        concat_features = torch.cat(pooled_embeddings, dim=1)

        # Predict Logits
        logits = self.fc(concat_features)

        return logits
