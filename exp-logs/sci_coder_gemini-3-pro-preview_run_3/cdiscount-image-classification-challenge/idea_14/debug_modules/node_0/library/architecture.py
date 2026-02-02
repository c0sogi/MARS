import torch
import torch.nn as nn
import torchvision.models as models
from library.configuration import Config


class DualBackboneExtractor(nn.Module):
    """
    A feature extraction module that utilizes two frozen backbones (ResNet50 and EfficientNet-B0)
    to generate robust visual representations. It handles the aggregation of features from
    multiple images belonging to the same product.
    """

    def __init__(self):
        super(DualBackboneExtractor, self).__init__()

        # 1. Initialize Backbones
        # Using default weights (pretrained on ImageNet)
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.efficientnet = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        # 2. Freeze Parameters
        # We set requires_grad to False to ensure we are only doing inference/extraction
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.efficientnet.parameters():
            param.requires_grad = False

        # 3. Modify Architectures to output features instead of logits
        # ResNet50: Remove fc layer (2048 -> num_classes)
        self.resnet.fc = nn.Identity()

        # EfficientNet-B0: Remove classifier (1280 -> num_classes)
        # The classifier in EfficientNet is usually a Sequential(Dropout, Linear).
        # Replacing it with Identity gives us the pooled features.
        self.efficientnet.classifier = nn.Identity()

    def forward(self, images, batch_sizes=None):
        """
        Args:
            images: Tensor of shape (Sum_K, C, H, W) containing all images in the batch.
            batch_sizes: Optional Tensor of shape (B,) indicating how many images belong to each product.
                         Used for mean pooling aggregation.

        Returns:
            features: Tensor of shape (B, 3328) if batch_sizes is provided,
                      else (Sum_K, 3328).
        """
        # Extract features from ResNet50 -> (N, 2048)
        # Note: resnet.fc is Identity, so output is the flattened pooled feature
        res_features = self.resnet(images)

        # Extract features from EfficientNet-B0 -> (N, 1280)
        eff_features = self.efficientnet(images)

        # Concatenate features -> (N, 3328)
        combined_features = torch.cat([res_features, eff_features], dim=1)

        # If batch_sizes is provided, perform aggregation (Mean Pooling per product)
        if batch_sizes is not None:
            # We split the combined features tensor into chunks based on the number of images per product
            # batch_sizes is a tensor, convert to list for split
            splits = torch.split(combined_features, batch_sizes.tolist())

            # Compute mean for each split and stack them
            # This results in one feature vector per product
            aggregated_features = torch.stack([s.mean(dim=0) for s in splits])

            return aggregated_features

        return combined_features


class ConditionalCascadeMLP(nn.Module):
    """
    A hierarchical classifier that predicts categories at Level 1, Level 2, and Level 3 sequentially.
    Lower levels are conditioned on the predictions of higher levels via concatenation.
    """

    def __init__(self):
        super(ConditionalCascadeMLP, self).__init__()

        # Dimensions from Config
        self.feat_dim = Config.TOTAL_FEATURE_DIM  # 3328
        self.l1_classes = Config.NUM_CLASSES_L1  # 49
        self.l2_classes = Config.NUM_CLASSES_L2  # 483
        self.l3_classes = Config.NUM_CLASSES_L3  # 5270

        self.hidden_dim_l1 = Config.HIDDEN_DIM_L1
        self.hidden_dim_l2 = Config.HIDDEN_DIM_L2
        self.dropout_rate = Config.DROPOUT

        # ==========================================
        # Stage 1: Level 1 Prediction
        # Input: Base Features
        # Output: L1 Logits
        # ==========================================
        self.stage1_block = nn.Sequential(
            nn.Linear(self.feat_dim, self.hidden_dim_l1),
            nn.BatchNorm1d(self.hidden_dim_l1),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim_l1, self.l1_classes),
        )

        # ==========================================
        # Stage 2: Level 2 Prediction
        # Input: Base Features + L1 Logits
        # Output: L2 Logits
        # ==========================================
        input_dim_l2 = self.feat_dim + self.l1_classes
        self.stage2_block = nn.Sequential(
            nn.Linear(input_dim_l2, self.hidden_dim_l2),
            nn.BatchNorm1d(self.hidden_dim_l2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim_l2, self.l2_classes),
        )

        # ==========================================
        # Stage 3: Level 3 Prediction (Target)
        # Input: Base Features + L2 Logits
        # Output: L3 Logits
        # ==========================================
        # Note: We skip concatenating L1 logits here to keep dimensionality reasonable,
        # assuming L2 logits capture sufficient hierarchical information.
        # Alternatively, one could concat both, but the prompt specified "concatenation of product embedding and Stage 2 logits".
        input_dim_l3 = self.feat_dim + self.l2_classes
        # We use a slightly deeper or wider head for the final massive classification if needed,
        # but sticking to a similar structure for consistency.
        self.stage3_block = nn.Sequential(
            nn.Linear(input_dim_l3, self.hidden_dim_l2),
            nn.BatchNorm1d(self.hidden_dim_l2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim_l2, self.l3_classes),
        )

    def forward(self, x):
        """
        Args:
            x: Input features of shape (B, 3328)

        Returns:
            l1_logits: (B, 49)
            l2_logits: (B, 483)
            l3_logits: (B, 5270)
        """
        # Stage 1
        l1_logits = self.stage1_block(x)

        # Stage 2
        # Condition on L1 logits
        l2_input = torch.cat([x, l1_logits], dim=1)
        l2_logits = self.stage2_block(l2_input)

        # Stage 3
        # Condition on L2 logits
        l3_input = torch.cat([x, l2_logits], dim=1)
        l3_logits = self.stage3_block(l3_input)

        return l1_logits, l2_logits, l3_logits
