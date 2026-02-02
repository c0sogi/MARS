import torch
import torch.nn as nn
import torchvision.models as models
from torch_scatter import scatter_softmax, scatter_add
from library.config import Config


class AttentionModule(nn.Module):
    """
    Computes attention scores for a set of instance features.
    Architecture: Linear -> Tanh -> Linear
    """

    def __init__(self, input_dim, hidden_dim=512):
        super(AttentionModule, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: (N_instances, input_dim)
        # output: (N_instances, 1)
        return self.network(x)


class HierarchicalAttentionResNet(nn.Module):
    """
    ResNet-50 with Attention-Gated Multi-Instance Aggregation and Hierarchical Supervision.

    This model processes a batch of products, where each product contains a variable number of images.
    It aggregates image features using a learnable attention mechanism and predicts categories
    at three hierarchical levels.
    """

    def __init__(self):
        super(HierarchicalAttentionResNet, self).__init__()

        # 1. Backbone: ResNet50
        # Use pretrained ImageNet weights for better feature extraction initialization
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        resnet = models.resnet50(weights=weights)

        # Remove the final FC layer to keep the feature extractor
        # ResNet50 structure ends with: ... -> AdaptiveAvgPool2d -> Linear
        # We keep everything up to the pooling layer.
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = resnet.fc.in_features  # 2048 for ResNet50

        # 2. Attention Module
        # Learns to weight images within a bag (product)
        self.attention = AttentionModule(self.feature_dim)

        # 3. Hierarchical Classification Heads
        # Simple Linear layers as per design strategy
        self.head_l1 = nn.Linear(self.feature_dim, Config.NUM_CLASSES_L1)
        self.head_l2 = nn.Linear(self.feature_dim, Config.NUM_CLASSES_L2)
        self.head_l3 = nn.Linear(self.feature_dim, Config.NUM_CLASSES_L3)

    def forward(self, images, batch_index, **kwargs):
        """
        Forward pass of the MIL model.

        Args:
            images (torch.Tensor): Flattened batch of images. Shape: (N_total, 3, H, W).
            batch_index (torch.Tensor): Indices mapping images to products. Shape: (N_total,).
                                        Example: [0, 0, 1, 2, 2, 2] for 3 products with 2, 1, 3 images.
            **kwargs: Additional arguments (e.g., sample_ids, targets) are ignored.

        Returns:
            dict: Dictionary containing logits for all three hierarchy levels.
                  {
                      "logits_l1": (Batch_Size, Num_L1),
                      "logits_l2": (Batch_Size, Num_L2),
                      "logits_l3": (Batch_Size, Num_L3)
                  }
        """
        # 1. Feature Extraction
        # Pass all images through the backbone
        # Output shape: (N_total, 2048, 1, 1)
        x = self.backbone(images)

        # Flatten features: (N_total, 2048)
        features = x.view(x.size(0), -1)

        # 2. Attention Scoring
        # Compute unnormalized attention scores for each image
        # Output shape: (N_total, 1)
        attn_logits = self.attention(features)

        # 3. Attention Softmax (Per Product/Bag)
        # Normalize scores such that they sum to 1 for each product group defined by batch_index
        # scatter_softmax handles the variable number of images per bag
        attn_weights = scatter_softmax(attn_logits, batch_index, dim=0)

        # 4. Weighted Aggregation
        # Weight the features by the attention scores
        # (N_total, 2048) * (N_total, 1) -> (N_total, 2048)
        weighted_features = features * attn_weights

        # Sum the weighted features for each product
        # We determine the batch size dynamically from the indices
        batch_size = batch_index.max().item() + 1

        # product_features: (Batch_Size, 2048)
        product_features = scatter_add(
            weighted_features, batch_index, dim=0, dim_size=batch_size
        )

        # 5. Hierarchical Classification
        # Pass the aggregated product representation to the classification heads
        out_l1 = self.head_l1(product_features)
        out_l2 = self.head_l2(product_features)
        out_l3 = self.head_l3(product_features)

        return {"logits_l1": out_l1, "logits_l2": out_l2, "logits_l3": out_l3}
