import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from library.config import Config


class VisualBackbone(nn.Module):
    """
    EfficientNet-B0 backbone for feature extraction.
    Returns the Global Average Pooled feature vector (B, 1280).
    """

    def __init__(self, pretrained=True):
        super(VisualBackbone, self).__init__()
        # Load EfficientNet-B0
        try:
            weights = "DEFAULT" if pretrained else None
            self.backbone = torchvision.models.efficientnet_b0(weights=weights)
        except TypeError:
            self.backbone = torchvision.models.efficientnet_b0(pretrained=pretrained)

        # Feature extractor
        self.features = self.backbone.features
        # Global Average Pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.flatten(1)


class TabularEncoder(nn.Module):
    """
    Encodes clinical metadata into a latent vector.
    """

    def __init__(self, input_dim, hidden_dim):
        super(TabularEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.mlp(x)


class TQSAN(nn.Module):
    """
    Global Fusion Dual-Axis Network with Skip Connections.
    Replaces spatial attention with global descriptor fusion to prevent overfitting.
    Cite solution_lesson_node_00020
    """

    def __init__(self):
        super(TQSAN, self).__init__()

        # Hyperparameters
        self.visual_dim = Config.visual_feature_dim  # 1280
        self.tab_dim = Config.tabular_hidden_dim  # 128

        # 1. Independent Visual Backbones (Axial + Coronal)
        self.backbone_axial = VisualBackbone(pretrained=Config.pretrained)
        self.backbone_coronal = VisualBackbone(pretrained=Config.pretrained)

        # 2. Tabular Encoder
        self.tab_encoder = TabularEncoder(
            input_dim=Config.n_tabular_features, hidden_dim=self.tab_dim
        )

        # 3. Projection for Tabular Features (to match visual dim for attention)
        self.tab_proj = nn.Linear(self.tab_dim, self.visual_dim)

        # 4. Self-Attention Fusion
        # Fuses [Axial, Coronal, Tabular_Proj]
        self.fusion = nn.MultiheadAttention(
            embed_dim=self.visual_dim, num_heads=4, batch_first=True
        )

        # 5. Prediction Head with Skip Connection
        # Input: Context (1280) + Skip Tabular (128) = 1408
        self.head = nn.Sequential(
            nn.Linear(self.visual_dim + self.tab_dim, 128),
            nn.ReLU(),
            nn.Dropout(Config.dropout_rate),
            nn.Linear(128, 3),
        )
        self._init_weights()

    def _init_weights(self):
        final_layer = self.head[-1]
        nn.init.xavier_uniform_(final_layer.weight)
        # Initialize biases: [alpha=0, sigma_base=100, sigma_growth=0]
        bias_init = torch.tensor([0.0, 100.0, 0.0])
        final_layer.bias.data = bias_init

    def forward(self, axial_img, coronal_img, tabular_features):
        # 1. Extract Global Visual Descriptors (B, 1280)
        v_axial = self.backbone_axial(axial_img)
        v_coronal = self.backbone_coronal(coronal_img)

        # 2. Encode Tabular Data (B, 128)
        t_emb = self.tab_encoder(tabular_features)

        # 3. Project Tabular to Visual Dim (B, 1280)
        t_proj = self.tab_proj(t_emb)

        # 4. Create Token Sequence (B, 3, 1280)
        tokens = torch.stack([v_axial, v_coronal, t_proj], dim=1)

        # 5. Self-Attention Fusion
        # Output: (B, 3, 1280)
        attn_out, _ = self.fusion(tokens, tokens, tokens)

        # 6. Global Pooling of Context (B, 1280)
        context = torch.mean(attn_out, dim=1)

        # 7. Skip Connection (Cite solution_lesson_node_00018)
        # Concatenate fused context with raw tabular embedding
        combined = torch.cat([context, t_emb], dim=1)

        # 8. Predict
        return self.head(combined)
