import torch
import torch.nn as nn
import torchvision.models as models


class DeepSupervisedResNet50(nn.Module):
    """
    ResNet-50 with Deep Hierarchical Supervision and Multi-View Aggregation.

    Architecture:
    - Backbone: ResNet-50 (pretrained on ImageNet).
    - Input: (Batch, 4, 3, H, W) - 4 images per product.
    - Aggregation: Global Max Pooling across the 4 views.
    - Deep Supervision:
        - Level 1 (Coarse): Attached to Stage 3 (layer3, 1024 dim).
        - Level 2 (Intermediate) & Level 3 (Fine): Attached to Stage 4 (layer4, 2048 dim).
    """

    def __init__(self, num_classes_l1, num_classes_l2, num_classes_l3, pretrained=True):
        """
        Args:
            num_classes_l1 (int): Number of classes for Level 1 (Coarse).
            num_classes_l2 (int): Number of classes for Level 2 (Intermediate).
            num_classes_l3 (int): Number of classes for Level 3 (Fine/Target).
            pretrained (bool): Whether to load ImageNet weights for the backbone.
        """
        super(DeepSupervisedResNet50, self).__init__()

        # Load ResNet-50 backbone
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)

        # ---------------------------------------------------------
        # Feature Extractor Stages
        # ---------------------------------------------------------
        # Initial: Conv1 -> MaxPool -> Layer1 -> Layer2
        self.initial = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
        )

        # Stage 3: Output channels = 1024
        self.layer3 = backbone.layer3

        # Stage 4: Output channels = 2048
        self.layer4 = backbone.layer4

        # ---------------------------------------------------------
        # Pooling & Heads
        # ---------------------------------------------------------
        # Spatial Pooling: Reduces (C, H, W) -> (C, 1, 1)
        # We use AdaptiveAvgPool2d to handle variable spatial dimensions if input size changes
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Level 1 Head (Coarse)
        # Input: 1024 channels (from layer3)
        self.head_l1 = nn.Linear(1024, num_classes_l1)

        # Level 2 Head (Intermediate)
        # Input: 2048 channels (from layer4)
        self.head_l2 = nn.Linear(2048, num_classes_l2)

        # Level 3 Head (Fine / Target)
        # Input: 2048 channels (from layer4)
        self.head_l3 = nn.Linear(2048, num_classes_l3)

    def forward_features(self, x):
        """
        Passes input through the backbone and returns intermediate features.
        x shape: (Batch * Views, 3, H, W)
        """
        x = self.initial(x)
        feat_l3 = self.layer3(x)  # Shape: (B*V, 1024, H/16, W/16)
        feat_l4 = self.layer4(feat_l3)  # Shape: (B*V, 2048, H/32, W/32)
        return feat_l3, feat_l4

    def aggregate_and_classify(self, features, head, batch_size, num_views=4):
        """
        Applies spatial pooling, aggregates views via Max Pooling, and classifies.

        Args:
            features: Tensor of shape (Batch * Views, Channels, H, W)
            head: The linear classification layer
            batch_size: Int
            num_views: Int (default 4)

        Returns:
            logits: (Batch, Num_Classes)
        """
        # 1. Spatial Pooling: (B*V, C, H, W) -> (B*V, C, 1, 1)
        x = self.spatial_pool(features)

        # 2. Flatten: (B*V, C)
        x = x.flatten(1)

        # 3. Reshape for Aggregation: (B, V, C)
        _, c = x.size()
        x = x.view(batch_size, num_views, c)

        # 4. View Aggregation (Global Max Pooling across images)
        # This aggregates the features from the 4 images into a single product representation
        # Shape: (B, C)
        x_agg, _ = torch.max(x, dim=1)

        # 5. Classification
        logits = head(x_agg)
        return logits

    def forward(self, images):
        """
        Args:
            images: Tensor of shape (Batch, 4, 3, H, W)

        Returns:
            (logits_l1, logits_l2, logits_l3)
        """
        b, n, c, h, w = images.size()

        # Flatten batch and view dimensions for backbone processing
        # Shape: (Batch * 4, 3, H, W)
        x = images.view(b * n, c, h, w)

        # Extract features
        feat_l3, feat_l4 = self.forward_features(x)

        # ---------------------------------------------------------
        # Level 1 Prediction (Deep Supervision)
        # ---------------------------------------------------------
        # Uses layer3 features (1024 dim)
        logits_l1 = self.aggregate_and_classify(feat_l3, self.head_l1, b, n)

        # ---------------------------------------------------------
        # Level 2 & 3 Predictions
        # ---------------------------------------------------------
        # Uses layer4 features (2048 dim)
        logits_l2 = self.aggregate_and_classify(feat_l4, self.head_l2, b, n)
        logits_l3 = self.aggregate_and_classify(feat_l4, self.head_l3, b, n)

        return logits_l1, logits_l2, logits_l3
