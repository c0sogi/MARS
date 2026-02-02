import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    Computes the generalized mean of each channel in the feature map.
    f(X) = (1/|X| * sum(x^p))^(1/p)
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x: (B, C, H, W)
        # Clamp to avoid NaN gradients with pow (and effectively act as ReLU for negative activations)
        x = x.clamp(min=eps)
        # Average pooling on x^p
        # F.avg_pool2d calculates (1/HW) * sum(x^p)
        x = F.avg_pool2d(x.pow(p), (x.size(-2), x.size(-1)))
        # Raise to 1/p
        x = x.pow(1.0 / p)
        return x

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class HybridEfficientNet(nn.Module):
    """
    Hybrid Vision-Tabular Network based on EfficientNet-B2 and GeM Pooling.
    Fuses visual features from mammograms with clinical metadata.
    """

    def __init__(self, tabular_input_dim):
        """
        Args:
            tabular_input_dim (int): The number of input features for the tabular branch.
        """
        super(HybridEfficientNet, self).__init__()

        # ==========================
        # Visual Backbone
        # ==========================
        # Load EfficientNet-B2, pretrained on ImageNet
        # global_pool='' ensures we get the spatial feature map (B, C, H, W)
        # drop_path_rate applies Stochastic Depth for regularization
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="",
            drop_path_rate=Config.DROP_PATH_RATE,
        )

        # Determine the number of output channels from the backbone (typically 1408 for B2)
        self.visual_dim = self.backbone.num_features

        # Generalized Mean Pooling to aggregate spatial features
        self.pooling = GeM()

        # ==========================
        # Tabular Branch (MLP)
        # ==========================
        # Processes clinical metadata (Age, View, etc.)
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, Config.TABULAR_EMBED_DIM),
            nn.BatchNorm1d(Config.TABULAR_EMBED_DIM),
            nn.ReLU(),
        )

        # ==========================
        # Fusion Head
        # ==========================
        # Concatenate Visual (visual_dim) + Tabular (TABULAR_EMBED_DIM)
        fusion_dim = self.visual_dim + Config.TABULAR_EMBED_DIM

        # Final classifier
        self.head = nn.Sequential(
            nn.Dropout(Config.DROP_RATE), nn.Linear(fusion_dim, Config.NUM_CLASSES)
        )

    def forward(self, images, tabular_features):
        """
        Args:
            images: Tensor of shape (B, 3, H, W)
            tabular_features: Tensor of shape (B, tabular_input_dim)
        Returns:
            logits: Tensor of shape (B, 1)
        """
        # 1. Visual Branch
        x_vis = self.backbone(images)  # (B, C, H, W)
        x_vis = self.pooling(x_vis)  # (B, C, 1, 1)
        x_vis = x_vis.flatten(1)  # (B, C)

        # 2. Tabular Branch
        x_tab = self.tabular_mlp(tabular_features)  # (B, Emb)

        # 3. Fusion
        x_combined = torch.cat([x_vis, x_tab], dim=1)  # (B, C + Emb)

        # 4. Classification
        logits = self.head(x_combined)  # (B, 1)

        return logits
