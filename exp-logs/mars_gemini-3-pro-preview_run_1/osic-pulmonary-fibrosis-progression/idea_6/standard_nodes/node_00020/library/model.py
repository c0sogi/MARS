import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from library.config import Config


class VisualBackbone(nn.Module):
    """
    EfficientNet-B0 backbone for spatial feature extraction.
    Returns the final convolutional feature map (C x H x W) without pooling.
    """

    def __init__(self, pretrained=True):
        super(VisualBackbone, self).__init__()
        # Load EfficientNet-B0
        # We use 'DEFAULT' weights if available or pretrained=True for compatibility
        try:
            weights = "DEFAULT" if pretrained else None
            self.backbone = torchvision.models.efficientnet_b0(weights=weights)
        except TypeError:
            # Fallback for older torchvision versions
            self.backbone = torchvision.models.efficientnet_b0(pretrained=pretrained)

        # We only need the feature extraction part
        # Input: (B, 3, 224, 224) -> Output: (B, 1280, 7, 7)
        self.features = self.backbone.features

    def forward(self, x):
        return self.features(x)


class TabularEncoder(nn.Module):
    """
    Encodes clinical metadata (Age, Sex, Smoking, Percent) into a latent Query vector.
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


class CrossModalAttention(nn.Module):
    """
    Cross-Attention mechanism: Tabular Vector (Query) attends to Spatial Visual Grid (Key/Value).
    """

    def __init__(self, visual_dim, query_dim, projection_dim):
        super(CrossModalAttention, self).__init__()
        self.visual_dim = visual_dim
        self.query_dim = query_dim
        self.projection_dim = projection_dim
        # Register scale as buffer for device safety
        self.register_buffer(
            "scale", torch.sqrt(torch.tensor(projection_dim, dtype=torch.float32))
        )

        # Projections
        self.W_q = nn.Linear(query_dim, projection_dim)
        self.W_k = nn.Linear(visual_dim, projection_dim)
        # We keep Value dimension same as input visual dimension to preserve feature richness
        self.W_v = nn.Linear(visual_dim, visual_dim)

    def forward(self, visual_feats, tabular_query):
        """
        Args:
            visual_feats: (B, C, H, W) -> Spatial Grid
            tabular_query: (B, D_tab) -> Context
        Returns:
            context_vector: (B, C)
        """
        b, c, h, w = visual_feats.size()
        n = h * w

        # Flatten visual features: (B, C, H, W) -> (B, N, C)
        # Permute to (B, H, W, C) then flatten spatial dims
        visual_flat = visual_feats.permute(0, 2, 3, 1).view(b, n, c)

        # 1. Project Query
        # (B, D_tab) -> (B, 1, D_proj)
        query = self.W_q(tabular_query).unsqueeze(1)

        # 2. Project Key
        # (B, N, C) -> (B, N, D_proj)
        key = self.W_k(visual_flat)

        # 3. Project Value
        # (B, N, C) -> (B, N, C)
        value = self.W_v(visual_flat)

        # 4. Compute Attention Scores
        # (B, 1, D_proj) @ (B, D_proj, N) -> (B, 1, N)
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)

        # 5. Weighted Sum
        # (B, 1, N) @ (B, N, C) -> (B, 1, C)
        context = torch.matmul(attn_weights, value)

        # Remove singleton dimension -> (B, C)
        return context.squeeze(1)


class ParametricHead(nn.Module):
    """
    Regression head predicting trajectory parameters.
    Input: Concatenation of [Visual_Axial, Visual_Coronal, Tabular_Query]
    Output: [alpha, sigma_base, sigma_growth]
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
        """
        Initialize the final layer to output reasonable starting values.
        alpha ~ 0
        sigma_base ~ 100 (to be safely above the clip threshold of 70)
        sigma_growth ~ 0
        """
        final_layer = self.head[-1]
        nn.init.xavier_uniform_(final_layer.weight)

        # Set biases: [alpha, sigma_base, sigma_growth]
        # We want sigma_base to start > 70 to ensure gradients flow through the max(sigma, 70) op.
        # 100 is a safe starting point.
        bias_init = torch.tensor([0.0, 100.0, 0.0])
        final_layer.bias.data = bias_init

    def forward(self, x):
        return self.head(x)


class TQSAN(nn.Module):
    """
    Tabular-Query Spatial-Attention Network.
    """

    def __init__(self):
        super(TQSAN, self).__init__()

        # Hyperparameters
        self.visual_dim = Config.visual_feature_dim
        self.tab_dim = Config.tabular_hidden_dim
        self.proj_dim = Config.projection_dim

        # 1. Independent Visual Backbones
        self.backbone_axial = VisualBackbone(pretrained=Config.pretrained)
        self.backbone_coronal = VisualBackbone(pretrained=Config.pretrained)

        # 2. Tabular Encoder
        self.tab_encoder = TabularEncoder(
            input_dim=Config.n_tabular_features, hidden_dim=self.tab_dim
        )

        # 3. Cross-Modal Attention Modules (One per view)
        self.attn_axial = CrossModalAttention(
            visual_dim=self.visual_dim,
            query_dim=self.tab_dim,
            projection_dim=self.proj_dim,
        )
        self.attn_coronal = CrossModalAttention(
            visual_dim=self.visual_dim,
            query_dim=self.tab_dim,
            projection_dim=self.proj_dim,
        )

        # 4. Parametric Head
        # Input size = Visual_Axial (1280) + Visual_Coronal (1280) + Tabular_Skip (128)
        head_input_dim = (self.visual_dim * 2) + self.tab_dim
        self.head = ParametricHead(
            input_dim=head_input_dim, dropout=Config.dropout_rate
        )

    def forward(self, axial_img, coronal_img, tabular_features):
        """
        Args:
            axial_img: (B, 3, 224, 224)
            coronal_img: (B, 3, 224, 224)
            tabular_features: (B, 4)
        Returns:
            preds: (B, 3) -> [alpha, sigma_base, sigma_growth]
        """
        # 1. Extract Spatial Features
        # (B, 1280, 7, 7)
        feat_axial = self.backbone_axial(axial_img)
        feat_coronal = self.backbone_coronal(coronal_img)

        # 2. Encode Tabular Query
        # (B, 128)
        query = self.tab_encoder(tabular_features)

        # 3. Apply Cross-Modal Attention
        # (B, 1280)
        ctx_axial = self.attn_axial(feat_axial, query)
        ctx_coronal = self.attn_coronal(feat_coronal, query)

        # 4. Feature Fusion (Skip Connection)
        # Concatenate: [Axial_Ctx, Coronal_Ctx, Tabular_Query]
        combined = torch.cat([ctx_axial, ctx_coronal, query], dim=1)

        # 5. Prediction
        preds = self.head(combined)

        return preds
