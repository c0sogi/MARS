import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library.config import Config


class MultiLevelResNet(nn.Module):
    """
    Multi-Level ResNet-50 with Feature Fusion and Deep Hierarchical Supervision.

    Architecture:
    - Backbone: ResNet-50 (Pretrained on ImageNet)
    - Inputs: Batch of products, each with N=1..4 images.
    - Feature Extraction:
        - Stage 3 (Layer 3): Mid-level features (1024 channels). Captures texture and parts.
        - Stage 4 (Layer 4): High-level features (2048 channels). Captures semantic shape.
    - Aggregation: Global Max Pooling across views (N images) to retain the strongest features.
    - Fusion: Concatenation of Stage 3 and Stage 4 aggregated features.
    - Heads:
        - Coarse (Level 1): Connected to Stage 3 features.
        - Intermediate (Level 2): Connected to Stage 4 features.
        - Fine (Level 3 - Target): Connected to Fused features (Stage 3 + Stage 4).
    """

    def __init__(self):
        super(MultiLevelResNet, self).__init__()

        # Load Pretrained ResNet50
        # Using the modern weights API
        weights = models.ResNet50_Weights.DEFAULT
        base_model = models.resnet50(weights=weights)

        # Deconstruct ResNet to access intermediate layers
        # We keep the stem (conv1, bn1, relu, maxpool) and first two layers as is
        self.stem = nn.Sequential(
            base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool
        )

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2

        # We need to access outputs of layer3 and layer4 separately for feature fusion
        self.layer3 = base_model.layer3  # Output: 1024 channels
        self.layer4 = base_model.layer4  # Output: 2048 channels

        # Feature Dimensions
        self.dim_stage3 = 1024
        self.dim_stage4 = 2048
        self.dim_fused = self.dim_stage3 + self.dim_stage4

        # Classification Heads
        # Level 1: Coarse categories (Auxiliary) - Uses mid-level features
        self.head_l1 = nn.Linear(self.dim_stage3, Config.NUM_CLASSES_L1)

        # Level 2: Intermediate categories (Auxiliary) - Uses high-level features
        self.head_l2 = nn.Linear(self.dim_stage4, Config.NUM_CLASSES_L2)

        # Level 3: Fine-grained categories (Target) - Uses fused features
        # Input is concatenation of Stage 3 and Stage 4 features
        self.head_l3 = nn.Linear(self.dim_fused, Config.NUM_CLASSES_L3)

        # Initialize new layers
        self._init_weights(self.head_l1)
        self._init_weights(self.head_l2)
        self._init_weights(self.head_l3)

    def _init_weights(self, module):
        """Xavier initialization for linear layers."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(self, x):
        """
        Passes images through the backbone and extracts multi-level features.

        Args:
            x: Tensor of shape (B*N, 3, H, W)

        Returns:
            tuple: (feat3, feat4)
                feat3: (B*N, 1024, H/16, W/16)
                feat4: (B*N, 2048, H/32, W/32)
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)

        feat3 = self.layer3(x)
        feat4 = self.layer4(feat3)

        return feat3, feat4

    def aggregate_views(self, features, batch_size, num_views, mask=None):
        """
        Aggregates features across the variable number of views (images) per product.
        Strategy: Spatial Average Pooling -> View Max Pooling.

        Args:
            features: Tensor of shape (B*N, C, H, W)
            batch_size: B
            num_views: N (usually 4)
            mask: Tensor of shape (B, N) indicating valid images (1.0) vs padding (0.0)

        Returns:
            aggregated: Tensor of shape (B, C)
        """
        # 1. Spatial Pooling: Global Average Pooling per image
        # (B*N, C, H, W) -> (B*N, C, 1, 1) -> (B*N, C)
        spatial_pooled = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)

        _, C = spatial_pooled.shape

        # 2. Reshape to separate views
        # (B, N, C)
        view_features = spatial_pooled.view(batch_size, num_views, C)

        # 3. Apply Masking (if provided)
        if mask is not None:
            # mask shape: (B, N)
            # Expand to (B, N, C)
            mask_expanded = mask.unsqueeze(-1).expand_as(view_features)

            # We want to ignore padded images in Max Pooling.
            # We set their values to a very small number (-1e9).
            # We clone to avoid in-place modification errors.
            view_features = view_features.clone()
            view_features[mask_expanded == 0] = -1e9

        # 4. View Pooling: Global Max Pooling across views
        # (B, N, C) -> (B, C)
        # We use max pooling to capture the most prominent features across all available images
        aggregated = torch.max(view_features, dim=1)[0]

        return aggregated

    def forward(self, images, mask=None):
        """
        Forward pass.

        Args:
            images: Tensor of shape (Batch, 4, 3, H, W)
            mask: Tensor of shape (Batch, 4) indicating valid images

        Returns:
            logits_l3: Predictions for target fine-grained categories
            logits_l2: Predictions for Level 2 (Auxiliary)
            logits_l1: Predictions for Level 1 (Auxiliary)
        """
        B, N, C, H, W = images.shape

        # Flatten batch and view dimensions for backbone processing
        # The backbone processes all images independently first
        x = images.view(B * N, C, H, W)

        # Extract multi-level features
        feat3_map, feat4_map = self.forward_features(x)

        # Aggregate views
        # feat3_agg: (B, 1024)
        feat3_agg = self.aggregate_views(feat3_map, B, N, mask)
        # feat4_agg: (B, 2048)
        feat4_agg = self.aggregate_views(feat4_map, B, N, mask)

        # --- Heads ---

        # Level 1 (Coarse) Prediction
        logits_l1 = self.head_l1(feat3_agg)

        # Level 2 (Intermediate) Prediction
        logits_l2 = self.head_l2(feat4_agg)

        # Level 3 (Fine) Prediction - Fused Features
        # Concatenate Stage 3 and Stage 4 features to combine mid-level texture/parts
        # with high-level semantic shape information.
        fused_features = torch.cat([feat3_agg, feat4_agg], dim=1)  # (B, 3072)
        logits_l3 = self.head_l3(fused_features)

        return logits_l3, logits_l2, logits_l1
