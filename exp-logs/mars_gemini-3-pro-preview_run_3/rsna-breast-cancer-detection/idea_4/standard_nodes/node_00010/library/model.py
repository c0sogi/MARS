import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from library.config import Config


class GatedAttention(nn.Module):
    """
    Gated Attention Mechanism for Multiple Instance Learning.
    Reference: Ilse et al., "Attention-based Deep Multiple Instance Learning", ICML 2018.

    Computes attention weights for a bag of instances to determine which instances
    contribute most to the bag label.
    """

    def __init__(self, input_dim, hidden_dim=128):
        super(GatedAttention, self).__init__()

        self.attention_V = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())

        self.attention_U = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Sigmoid())

        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Bag of instance embeddings. Shape (N, input_dim).

        Returns:
            torch.Tensor: Attention weights. Shape (N, 1).
        """
        # Gated mechanism: a = w^T ( tanh(V h) . sigmoid(U h) )
        v_out = self.attention_V(x)  # (N, D)
        u_out = self.attention_U(x)  # (N, D)

        # Element-wise multiplication
        gated = v_out * u_out

        # Compute scores
        scores = self.attention_weights(gated)  # (N, 1)

        # Softmax over instances in the bag ensures weights sum to 1
        weights = torch.softmax(scores, dim=0)

        return weights


class BreastMILModel(nn.Module):
    """
    MIL Model using EfficientNetV2 backbone and Gated Attention aggregation.
    Processes a bag of images (views) to predict cancer likelihood for the breast.
    """

    def __init__(self, backbone_name=Config.BACKBONE_NAME, pretrained=True):
        super(BreastMILModel, self).__init__()

        # 1. Backbone
        # We use EfficientNetV2-S as defined in Config
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_v2_s(weights=weights)

        # Get feature dimension (1280 for EfficientNetV2-S)
        # We access the input features of the original classifier head
        self.feature_dim = self.backbone.classifier[1].in_features

        # Global Average Pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

        # 2. Attention Mechanism
        # Hidden dim 128 is a standard choice for this embedding size
        self.attention = GatedAttention(input_dim=self.feature_dim, hidden_dim=128)

        # 3. Classifier
        # Maps the aggregated breast representation to a single logit
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 1)
            # No Sigmoid here; we output logits for BCEWithLogitsLoss
        )

    def forward_features(self, x):
        """
        Extract features from a batch of images using the backbone.

        Args:
            x (torch.Tensor): Batch of images (Sum_N, 3, H, W)

        Returns:
            torch.Tensor: Feature embeddings (Sum_N, Feature_Dim)
        """
        # EfficientNet features() returns (B, C, H, W)
        x = self.backbone.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, images):
        """
        Forward pass for a batch of bags.

        Args:
            images (List[torch.Tensor]): A list of length B, where each element is a
                                         tensor of shape (N_i, 3, H, W) representing
                                         the bag of views for a specific breast.

        Returns:
            torch.Tensor: Logits of shape (B, 1).
        """
        # Optimization: Concatenate all images into a single tensor to maximize GPU utilization
        # for the backbone pass, rather than looping through the backbone B times.

        # Track the split sizes to reconstruct bags later
        split_sizes = [bag.shape[0] for bag in images]

        # Flatten list of bags into one big batch of images
        # Shape: (Sum(N_i), 3, H, W)
        flat_images = torch.cat(images, dim=0)

        # Extract features for all images at once
        # Shape: (Sum(N_i), Feature_Dim)
        flat_features = self.forward_features(flat_images)

        # Split features back into bags
        # List of tensors, each shape (N_i, Feature_Dim)
        bag_features_list = torch.split(flat_features, split_sizes, dim=0)

        bag_embeddings = []

        for features in bag_features_list:
            # features: (N_i, Feature_Dim)

            # Calculate attention weights
            # weights: (N_i, 1)
            weights = self.attention(features)

            # Aggregate features: Weighted Sum
            # embedding: (1, Feature_Dim)
            # sum( (N_i, D) * (N_i, 1) ) -> sum over N_i -> (D)
            bag_embedding = torch.sum(features * weights, dim=0, keepdim=True)

            bag_embeddings.append(bag_embedding)

        # Stack bag embeddings
        # Shape: (B, Feature_Dim)
        bag_embeddings = torch.cat(bag_embeddings, dim=0)

        # Classification
        # Shape: (B, 1)
        logits = self.classifier(bag_embeddings)

        return logits
