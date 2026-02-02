import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from library.config import Config


class VisualBackbone(nn.Module):
    """
    EfficientNet-B0 backbone for spatial feature extraction.
    Returns the final global average pooled feature vector.
    Cite solution_lesson_node_00020: Prefer fusing global descriptors over spatial attention maps when training data is scarce.
    """

    def __init__(self, pretrained=True):
        super(VisualBackbone, self).__init__()
        # Load EfficientNet-B0
        try:
            weights = "DEFAULT" if pretrained else None
            self.backbone = torchvision.models.efficientnet_b0(weights=weights)
        except TypeError:
            self.backbone = torchvision.models.efficientnet_b0(pretrained=pretrained)

        # We need features + pooling
        self.features = self.backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


class TabularEncoder(nn.Module):
    """
    Encodes clinical metadata (Age, Sex, Smoking, Percent) into a latent vector.
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


class ParametricHead(nn.Module):
    """
    Regression head predicting trajectory parameters.
    Input: Concatenation of [Visual_Axial, Visual_Coronal, Tabular_Embed]
    Output: [alpha, sigma_base, sigma_growth]

    Cite solution_lesson_node_00021: Enforce physical constraints via architectural activation functions.
    """

    def __init__(self, input_dim, hidden_dim=128, dropout=0.2):
        super(ParametricHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self._init_weights()

    def _init_weights(self):
        final_layer = self.head[-1]
        nn.init.xavier_uniform_(final_layer.weight)

        # Initialize biases
        # alpha ~ 0
        # sigma_base ~ 100 (inverse softplus of 100 is approx 100)
        # sigma_growth ~ 0
        bias_init = torch.tensor(
            [0.0, 100.0, -5.0]
        )  # -5 for sigma_growth makes it small positive after softplus
        final_layer.bias.data = bias_init

    def forward(self, x):
        raw = self.head(x)

        # Unpack
        alpha = raw[:, 0:1]
        raw_sigma_base = raw[:, 1:2]
        raw_sigma_growth = raw[:, 2:3]

        # Apply Softplus to ensure positivity
        sigma_base = F.softplus(raw_sigma_base)
        sigma_growth = F.softplus(raw_sigma_growth)

        return torch.cat([alpha, sigma_base, sigma_growth], dim=1)


class TQSAN(nn.Module):
    """
    Dual-Axis Network with Global Fusion.
    Replaces Attention with Global Pooling + Concatenation.
    """

    def __init__(self):
        super(TQSAN, self).__init__()

        # Hyperparameters
        self.visual_dim = Config.visual_feature_dim
        self.tab_dim = Config.tabular_hidden_dim

        # 1. Independent Visual Backbones
        self.backbone_axial = VisualBackbone(pretrained=Config.pretrained)
        self.backbone_coronal = VisualBackbone(pretrained=Config.pretrained)

        # 2. Tabular Encoder
        self.tab_encoder = TabularEncoder(
            input_dim=Config.n_tabular_features, hidden_dim=self.tab_dim
        )

        # 3. Parametric Head
        # Input size = Visual_Axial (1280) + Visual_Coronal (1280) + Tabular_Embed (128)
        head_input_dim = (self.visual_dim * 2) + self.tab_dim
        self.head = ParametricHead(
            input_dim=head_input_dim, dropout=Config.dropout_rate
        )

    def forward(self, axial_img, coronal_img, tabular_features):
        # 1. Extract Global Spatial Features
        # (B, 1280)
        feat_axial = self.backbone_axial(axial_img)
        feat_coronal = self.backbone_coronal(coronal_img)

        # 2. Encode Tabular Data
        # (B, 128)
        tab_embed = self.tab_encoder(tabular_features)

        # 3. Concatenate (Fusion)
        # Cite solution_lesson_node_00018: Concatenate structured features directly to the fused representation.
        combined = torch.cat([feat_axial, feat_coronal, tab_embed], dim=1)

        # 4. Prediction
        preds = self.head(combined)

        return preds
