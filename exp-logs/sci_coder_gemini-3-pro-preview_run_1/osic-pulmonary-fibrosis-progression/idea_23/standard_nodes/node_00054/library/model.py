import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class EfficientNetEncoder(nn.Module):
    """
    Extracts features from an image using EfficientNet-B0.
    Returns the global average pooled feature vector (1280-dim).
    """

    def __init__(self, model_name="efficientnet_b0", pretrained=True):
        super().__init__()
        # Create model with num_classes=0 to get the feature vector after pooling
        # global_pool='avg' ensures we get the (Batch, 1280) vector directly
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

    def forward(self, x):
        # x: (Batch, 3, H, W)
        # Output: (Batch, 1280)
        return self.backbone(x)


class TabularEncoder(nn.Module):
    """
    Projects low-dimensional tabular data up to the visual feature dimension.
    """

    def __init__(self, input_dim=6, output_dim=1280):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim // 2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        # x: (Batch, 6)
        # Output: (Batch, 1280)
        return self.net(x)


class ChannelAdaptiveReadout(nn.Module):
    """
    Implements Softmax Channel-Wise Competition.
    Dynamically weights channels from different views based on global context.
    """

    def __init__(self, feature_dim=1280, num_views=3):
        super().__init__()
        self.num_views = num_views
        self.feature_dim = feature_dim

        # Project global context to weights for each view and channel
        # Projects 1280 -> 3 * 1280
        self.weight_generator = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, num_views * feature_dim),
        )

    def forward(self, features):
        """
        Args:
            features: (Batch, Num_Views, Feature_Dim)
        Returns:
            weighted_features: (Batch, Feature_Dim)
        """
        # 1. Compute Global Context G (Mean across views)
        # Shape: (Batch, Feature_Dim)
        global_context = features.mean(dim=1)

        # 2. Generate Weights W
        # Shape: (Batch, Num_Views * Feature_Dim)
        weights_flat = self.weight_generator(global_context)

        # Reshape to (Batch, Num_Views, Feature_Dim)
        weights = weights_flat.view(-1, self.num_views, self.feature_dim)

        # 3. Softmax across views (dim=1)
        # This creates competition: for each channel index c,
        # sum(weights[b, :, c]) = 1.
        weights_norm = F.softmax(weights, dim=1)

        # 4. Aggregate
        # Element-wise multiplication and sum across views
        # (Batch, Views, Dim) * (Batch, Views, Dim) -> Sum over Views -> (Batch, Dim)
        weighted_features = (features * weights_norm).sum(dim=1)

        return weighted_features


class ChannelAdaptiveDualAxisNet(nn.Module):
    """
    Main Architecture: Channel-Adaptive Symmetric Dual-Axis Network.
    Integrates Axial, Coronal, and Tabular data via symmetric attention and
    channel-wise competitive readout.
    """

    def __init__(self):
        super().__init__()

        # 1. Independent Visual Backbones
        # One for Axial, one for Coronal to preserve view-specific semantics
        self.encoder_ax = EfficientNetEncoder(Config.BACKBONE_NAME)
        self.encoder_cor = EfficientNetEncoder(Config.BACKBONE_NAME)

        # 2. Tabular Encoder
        # Input: Age, Sex, Smoking(3), Percent -> 6 dims
        self.encoder_tab = TabularEncoder(input_dim=6, output_dim=Config.BACKBONE_DIM)

        # 3. Modality Embeddings
        # Learnable vectors added to features: [Axial, Coronal, Tabular]
        self.modality_embed = nn.Parameter(torch.zeros(1, 3, Config.BACKBONE_DIM))
        nn.init.normal_(self.modality_embed, std=0.02)

        # 4. Symmetric Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=Config.BACKBONE_DIM, num_heads=8, batch_first=True
        )

        # 5. Channel-Adaptive Readout
        self.readout = ChannelAdaptiveReadout(
            feature_dim=Config.BACKBONE_DIM, num_views=3
        )

        # 6. Prediction Head
        # Concatenates aggregated visual features (1280) + raw tabular (6)
        self.head = nn.Sequential(
            nn.Linear(Config.BACKBONE_DIM + 6, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 3),  # Outputs: Alpha, Sigma_Base, Sigma_Growth
        )

    def forward(self, axial_img, coronal_img, tabular):
        """
        Args:
            axial_img: (Batch, 3, 224, 224)
            coronal_img: (Batch, 3, 224, 224)
            tabular: (Batch, 6)
        Returns:
            alpha, sigma_base, sigma_growth
        """
        batch_size = axial_img.size(0)

        # --- Feature Extraction ---
        # 1. Extract Visual Features
        feat_ax = self.encoder_ax(axial_img)  # (B, 1280)
        feat_cor = self.encoder_cor(coronal_img)  # (B, 1280)

        # 2. Extract Tabular Features (Up-projection)
        feat_tab = self.encoder_tab(tabular)  # (B, 1280)

        # --- Sequence Construction ---
        # Stack features: (B, 3, 1280)
        # Order: [Axial, Coronal, Tabular]
        sequence = torch.stack([feat_ax, feat_cor, feat_tab], dim=1)

        # Add Modality Embeddings
        # modality_embed broadcasts across batch
        sequence = sequence + self.modality_embed

        # --- Symmetric Attention ---
        # Self-attention over the 3 views
        # Returns (Batch, Seq_Len, Dim)
        attn_out, _ = self.attention(sequence, sequence, sequence)

        # --- Channel-Adaptive Readout ---
        # Aggregates the 3 views into one vector based on channel-wise competition
        # Input: (B, 3, 1280) -> Output: (B, 1280)
        global_feat = self.readout(attn_out)

        # --- Prior-Anchored Head ---
        # Concatenate with raw tabular features for direct access to clinical priors
        combined = torch.cat([global_feat, tabular], dim=1)

        # Predict parameters
        out = self.head(combined)

        # Split outputs
        # alpha: slope (unbounded, can be negative for decline)
        # sigma_base: uncertainty at t=0 (must be positive)
        # sigma_growth: uncertainty growth rate (must be positive)
        alpha = out[:, 0]
        sigma_base = F.softplus(out[:, 1])
        sigma_growth = F.softplus(out[:, 2])

        return alpha, sigma_base, sigma_growth
