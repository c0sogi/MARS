import torch
import torch.nn as nn
import timm


class HybridEfficientNet(nn.Module):
    """
    A hybrid neural network that fuses mammogram image features (via EfficientNet-B0)
    with clinical tabular metadata (via an MLP) to predict breast cancer.
    """

    def __init__(
        self,
        num_tabular_features: int,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
    ):
        """
        Args:
            num_tabular_features (int): Number of input features in the tabular data.
            backbone_name (str): Name of the timm model to use as backbone.
            pretrained (bool): Whether to load pretrained ImageNet weights.
        """
        super(HybridEfficientNet, self).__init__()

        # ==========================================
        # Image Branch
        # ==========================================
        # Create EfficientNet backbone.
        # num_classes=0 removes the classification head and returns pooled features.
        # global_pool='avg' ensures we get a feature vector (e.g., 1280 dim for B0).
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        self.img_feature_dim = self.backbone.num_features

        # ==========================================
        # Tabular Branch
        # ==========================================
        # MLP to process clinical metadata (age, site, view, etc.)
        # Architecture: Linear -> BN -> ReLU -> Dropout -> Linear -> BN -> ReLU
        self.tabular_mlp = nn.Sequential(
            nn.Linear(num_tabular_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.tabular_feature_dim = 32

        # ==========================================
        # Fusion Head
        # ==========================================
        # Concatenates image and tabular embeddings and predicts cancer likelihood.
        self.fusion_head = nn.Sequential(
            nn.Linear(self.img_feature_dim + self.tabular_feature_dim, 1)
        )

    def forward(self, x):
        """
        Forward pass of the hybrid model.

        Args:
            x (tuple): A tuple containing:
                - images (torch.Tensor): Batch of images (B, C, H, W).
                - tabular (torch.Tensor): Batch of tabular data (B, num_tabular_features).

        Returns:
            torch.Tensor: Logits of shape (B, 1).
                          Note: Sigmoid activation is omitted here to allow the use
                          of BCEWithLogitsLoss for numerical stability.
        """
        images, tabular = x

        # 1. Extract Image Features
        img_emb = self.backbone(images)  # Output shape: (B, 1280)

        # 2. Extract Tabular Features
        tab_emb = self.tabular_mlp(tabular)  # Output shape: (B, 32)

        # 3. Feature Fusion
        combined = torch.cat((img_emb, tab_emb), dim=1)

        # 4. Final Prediction
        logits = self.fusion_head(combined)

        return logits
