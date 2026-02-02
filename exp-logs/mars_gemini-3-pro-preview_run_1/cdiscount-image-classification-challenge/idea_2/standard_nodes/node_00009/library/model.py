import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class HierarchicalResNet50(nn.Module):
    """
    Hierarchical Multi-Task ResNet-50 Model.

    This model processes a variable number of images per product, aggregates their features
    via Global Max Pooling, and predicts categories at three hierarchical levels
    (Coarse, Intermediate, Fine).
    """

    def __init__(self):
        super(HierarchicalResNet50, self).__init__()

        # 1. Load Backbone
        # Use weights parameter for torchvision >= 0.13
        weights = models.ResNet50_Weights.DEFAULT if Config.PRETRAINED else None
        resnet = models.resnet50(weights=weights)

        # Remove the final fully connected layer
        # We keep everything up to the Global Average Pooling layer (avgpool)
        # ResNet50 structure: [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc]
        # We want the output of avgpool which is (B, 2048, 1, 1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Feature dimension for ResNet50 is 2048
        self.feature_dim = resnet.fc.in_features

        # 2. Define Hierarchical Heads
        # Level 1: Coarse Category
        self.head_l1 = nn.Linear(self.feature_dim, Config.NUM_CLASSES_L1)

        # Level 2: Intermediate Category
        self.head_l2 = nn.Linear(self.feature_dim, Config.NUM_CLASSES_L2)

        # Level 3: Fine-grained Target Category
        self.head_l3 = nn.Linear(self.feature_dim, Config.NUM_CLASSES_L3)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images of shape (Batch, Num_Images, C, H, W)

        Returns:
            dict: Dictionary containing logits for each hierarchy level:
                  {
                      'l1': (Batch, NUM_CLASSES_L1),
                      'l2': (Batch, NUM_CLASSES_L2),
                      'target': (Batch, NUM_CLASSES_L3)
                  }
        """
        batch_size, num_imgs, c, h, w = x.shape

        # 1. Flatten Batch and Image dimensions to pass through backbone
        # Shape: (Batch * Num_Images, C, H, W)
        x_flat = x.view(batch_size * num_imgs, c, h, w)

        # 2. Extract Features
        # Backbone output shape: (Batch * Num_Images, 2048, 1, 1)
        features = self.backbone(x_flat)

        # Flatten feature spatial dims: (Batch * Num_Images, 2048)
        features = features.view(features.size(0), -1)

        # 3. Reshape back to separate images per product
        # Shape: (Batch, Num_Images, 2048)
        features = features.view(batch_size, num_imgs, self.feature_dim)

        # 4. Multi-View Aggregation (Global Max Pooling)
        # We take the max value across the Num_Images dimension (dim=1)
        # This aggregates the most salient features from all images of the product.
        # Since ResNet features (post-ReLU/AvgPool) are non-negative,
        # zero-padding in the input (for products with < 4 images) does not affect the max.
        aggregated_features, _ = torch.max(features, dim=1)  # Shape: (Batch, 2048)

        # 5. Prediction Heads
        logits_l1 = self.head_l1(aggregated_features)
        logits_l2 = self.head_l2(aggregated_features)
        logits_l3 = self.head_l3(aggregated_features)

        return {"l1": logits_l1, "l2": logits_l2, "target": logits_l3}
