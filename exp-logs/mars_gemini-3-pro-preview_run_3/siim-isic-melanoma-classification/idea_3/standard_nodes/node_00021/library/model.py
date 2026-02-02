import torch
import torch.nn as nn
import timm
from library.config import Config


class HybridEfficientNet(nn.Module):
    """
    A Hybrid Neural Network that fuses EfficientNet-B4 image features with
    tabular metadata embeddings for Skin Lesion Classification.
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAME,
        pretrained=True,
        num_classes=Config.NUM_CLASSES,
        num_tabular_features=0,
        tabular_hidden_dim=Config.TABULAR_HIDDEN_DIM,
        final_dropout=Config.FINAL_DROPOUT,
    ):
        """
        Args:
            model_name (str): Name of the timm model backbone (default: tf_efficientnet_b4_ns).
            pretrained (bool): Whether to load pretrained ImageNet weights.
            num_classes (int): Number of output classes (1 for binary classification).
            num_tabular_features (int): Number of input tabular features.
            tabular_hidden_dim (int): Dimension of the tabular feature embedding.
            final_dropout (float): Dropout probability for the final classification head.
        """
        super(HybridEfficientNet, self).__init__()

        # 1. Image Backbone
        # Setting num_classes=0 removes the classifier and applies Global Average Pooling,
        # resulting in a feature vector output.
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )
        self.img_feature_dim = self.backbone.num_features

        # 2. Tabular Branch
        # A lightweight MLP to process clinical metadata (age, sex, site)
        self.num_tabular_features = num_tabular_features
        if num_tabular_features > 0:
            self.tabular_mlp = nn.Sequential(
                nn.Linear(num_tabular_features, tabular_hidden_dim), nn.ReLU()
            )
            fusion_dim = self.img_feature_dim + tabular_hidden_dim
        else:
            self.tabular_mlp = None
            fusion_dim = self.img_feature_dim

        # 3. Classification Head
        # Concatenated features -> Dropout -> Logit
        self.head = nn.Sequential(
            nn.Dropout(p=final_dropout), nn.Linear(fusion_dim, num_classes)
        )

    def forward(self, image, tabular):
        """
        Forward pass of the hybrid model.

        Args:
            image (torch.Tensor): Batch of images [Batch_Size, Channels, Height, Width].
            tabular (torch.Tensor): Batch of tabular data [Batch_Size, Num_Tabular_Features].

        Returns:
            torch.Tensor: Logits [Batch_Size, Num_Classes].
        """
        # Extract visual features
        # Output shape: [Batch_Size, img_feature_dim]
        img_features = self.backbone(image)

        # Extract and fuse tabular features
        if self.tabular_mlp is not None:
            # Process tabular data
            tab_features = self.tabular_mlp(tabular)
            # Concatenate along the feature dimension
            combined_features = torch.cat([img_features, tab_features], dim=1)
        else:
            combined_features = img_features

        # Generate predictions
        logits = self.head(combined_features)

        return logits
