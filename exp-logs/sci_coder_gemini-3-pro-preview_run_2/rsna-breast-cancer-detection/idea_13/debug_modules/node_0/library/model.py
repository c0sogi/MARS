import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    Computes the generalized mean of the spatial dimensions.
    f(X) = (1/|X| * sum(x^p))^(1/p)
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)
        # Clamp to avoid NaN with power
        x = x.clamp(min=self.eps)
        # Average pooling on x^p
        x_pow = x.pow(self.p)
        # Global Average Pooling over spatial dimensions (H, W)
        avg = F.avg_pool2d(x_pow, (x.size(-2), x.size(-1)))
        # Root p
        return avg.pow(1.0 / self.p)


class TriSpectralHybridModel(nn.Module):
    """
    Tri-Spectral Hybrid Network (TS-HN).

    Architecture:
    1. Visual Branch: EfficientNetV2-Small (Tri-Spectral Input) -> GeM Pooling
    2. Tabular Branch: MLP (Clinical Metadata)
    3. Fusion: Concatenation -> Dense Head -> Logit
    """

    def __init__(self):
        super(TriSpectralHybridModel, self).__init__()

        # =====================================================================
        # 1. Visual Backbone
        # =====================================================================
        # Load pretrained EfficientNetV2-Small
        # num_classes=0 removes the final linear layer
        # global_pool='' removes the default pooling, keeping spatial features
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="",
            in_chans=Config.NUM_CHANNELS,
            drop_path_rate=Config.DROP_PATH_RATE,
        )

        # Determine the number of output channels from the backbone
        # We run a dummy forward pass to dynamically get the shape
        with torch.no_grad():
            dummy_input = torch.zeros(1, Config.NUM_CHANNELS, 256, 256)
            dummy_features = self.backbone(dummy_input)
            # Shape is (1, C, H, W)
            self.visual_embedding_dim = dummy_features.shape[1]

        # Generalized Mean Pooling
        self.gem_pool = GeM()
        self.flatten = nn.Flatten()

        # =====================================================================
        # 2. Tabular Branch
        # =====================================================================
        # Input dimension is 10 (Age, Implant, Lat(2), View(6))
        TABULAR_INPUT_DIM = 10

        self.tabular_mlp = nn.Sequential(
            nn.Linear(TABULAR_INPUT_DIM, Config.TABULAR_HIDDEN_DIM),
            nn.BatchNorm1d(Config.TABULAR_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROP_RATE),
        )

        # =====================================================================
        # 3. Fusion Head
        # =====================================================================
        fusion_input_dim = self.visual_embedding_dim + Config.TABULAR_HIDDEN_DIM

        self.head = nn.Sequential(
            nn.Linear(fusion_input_dim, Config.FUSION_HIDDEN_DIM),
            nn.BatchNorm1d(Config.FUSION_HIDDEN_DIM),
            nn.SiLU(),  # Swish activation matches EfficientNet design
            nn.Dropout(Config.DROP_RATE),
            nn.Linear(Config.FUSION_HIDDEN_DIM, 1),
        )

    def forward(self, images, tabular_features):
        """
        Args:
            images: Tensor of shape (Batch, 3, H, W)
            tabular_features: Tensor of shape (Batch, 10)

        Returns:
            logits: Tensor of shape (Batch, 1)
        """
        # --- Visual Branch ---
        # Extract features: (B, C, H, W)
        features = self.backbone(images)

        # Pool features: (B, C, 1, 1)
        pooled_features = self.gem_pool(features)

        # Flatten: (B, C)
        visual_embedding = self.flatten(pooled_features)

        # --- Tabular Branch ---
        # Process metadata: (B, Tabular_Hidden)
        tabular_embedding = self.tabular_mlp(tabular_features)

        # --- Fusion ---
        # Concatenate: (B, C + Tabular_Hidden)
        combined_embedding = torch.cat([visual_embedding, tabular_embedding], dim=1)

        # Classification
        logits = self.head(combined_embedding)

        return logits
