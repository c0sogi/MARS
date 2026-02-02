import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class VisualBackbone(nn.Module):
    """
    Extracts features from a CT slab using EfficientNet-B0.
    """

    def __init__(self, pretrained=True):
        super(VisualBackbone, self).__init__()
        # Create EfficientNet B0
        # num_classes=0 with global_pool='' removes the classifier but keeps the spatial features
        # However, timm's global_pool='avg' with num_classes=0 returns the pooled feature vector directly.
        self.backbone = timm.create_model(
            Config.BACKBONE, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        # EfficientNet-B0 outputs 1280-dim features
        self.in_features = 1280

        # Projection to shared embedding dimension
        self.projection = nn.Linear(self.in_features, Config.EMBED_DIM)
        self.bn = nn.BatchNorm1d(Config.EMBED_DIM)
        self.act = nn.SiLU()  # Swish activation (common with EfficientNet)

    def forward(self, x):
        # x: (B, 3, H, W)
        features = self.backbone(x)  # (B, 1280)
        out = self.projection(features)
        out = self.bn(out)
        out = self.act(out)
        return out  # (B, Embed_Dim)


class TabularMLP(nn.Module):
    """
    Encodes clinical metadata into feature space.
    """

    def __init__(self, input_dim=5):
        super(TabularMLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.act1 = nn.ReLU()

        self.layer2 = nn.Linear(128, Config.EMBED_DIM)
        self.bn2 = nn.BatchNorm1d(Config.EMBED_DIM)
        self.act2 = nn.ReLU()

    def forward(self, x):
        # x: (B, 5)
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.layer2(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x  # (B, Embed_Dim)


class PriorPreservingDualAxisNet(nn.Module):
    """
    Prior-Preserving Symmetric Dual-Axis Network.
    Fuses Axial and Coronal CT views with Clinical Metadata using Symmetric Attention
    and a Prior-Preserving Skip Connection.
    """

    def __init__(self):
        super(PriorPreservingDualAxisNet, self).__init__()

        # 1. Independent Visual Backbones
        self.axial_backbone = VisualBackbone(pretrained=Config.PRETRAINED)
        self.coronal_backbone = VisualBackbone(pretrained=Config.PRETRAINED)

        # 2. Tabular Embedding
        # Input dim is 5: Age, Sex, Smoking, Percent, Baseline_FVC (all normalized)
        self.tabular_mlp = TabularMLP(input_dim=5)

        # 3. Symmetric Attention Fusion
        # Embed dim must match the projection output of backbones and MLP
        self.attention = nn.MultiheadAttention(
            embed_dim=Config.EMBED_DIM, num_heads=8, batch_first=True, dropout=0.1
        )

        # 4. Regression Head
        # Input: Contextualized Tabular (D) + Original Tabular (D) = 2*D
        self.head_input_dim = Config.EMBED_DIM * 2

        self.head = nn.Sequential(
            nn.Linear(self.head_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(
                512, Config.OUTPUT_DIM
            ),  # Output: 3 (Alpha, Sigma_Base, Sigma_Growth)
        )

    def forward(self, image_axial, image_coronal, tabular):
        """
        Args:
            image_axial: (B, 3, 224, 224)
            image_coronal: (B, 3, 224, 224)
            tabular: (B, 5)

        Returns:
            preds: (B, 3) -> [Alpha, Sigma_Base, Sigma_Growth]
        """
        # --- Feature Extraction ---
        # (B, D)
        feat_axial = self.axial_backbone(image_axial)
        feat_coronal = self.coronal_backbone(image_coronal)
        feat_tabular = self.tabular_mlp(tabular)

        # --- Symmetric Attention ---
        # Stack tokens: [Axial, Coronal, Tabular] -> (B, 3, D)
        tokens = torch.stack([feat_axial, feat_coronal, feat_tabular], dim=1)

        # Self-Attention: allows visual views to contextualize each other and the tabular data
        attn_out, _ = self.attention(tokens, tokens, tokens)

        # Extract the contextualized tabular token (Index 2)
        # We focus on refining the clinical state with visual context
        contextualized_tabular = attn_out[:, 2, :]  # (B, D)

        # --- Prior-Preserving Skip Connection ---
        # Concatenate contextualized vector with original raw tabular embedding
        # This ensures the strong scalar priors are directly available to the head
        fused = torch.cat([contextualized_tabular, feat_tabular], dim=1)  # (B, 2*D)

        # --- Prediction ---
        raw_preds = self.head(fused)  # (B, 3)

        # Apply constraints
        # Column 0: Alpha (Slope) - No activation (can be negative)
        # Column 1: Sigma_Base - Softplus (must be positive)
        # Column 2: Sigma_Growth - Softplus (must be positive)

        alpha = raw_preds[:, 0].unsqueeze(1)
        sigma_base = F.softplus(raw_preds[:, 1]).unsqueeze(1)
        sigma_growth = F.softplus(raw_preds[:, 2]).unsqueeze(1)

        return torch.cat([alpha, sigma_base, sigma_growth], dim=1)
