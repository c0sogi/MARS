import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    Computes the generalized mean of each channel in the feature map.
    f(X) = (1/N * sum(x^p))^(1/p)
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Clamp to avoid NaN gradients for negative values (though ReLU usually prevents this)
        # AvgPool2d effectively calculates 1/N * sum()
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class MTSIN(nn.Module):
    """
    Multi-Task Single-Instance Network (MT-SIN).

    Architecture:
    1. Backbone: EfficientNetV2-Small (Unfrozen, initialized with ImageNet weights).
    2. Pooling: GeM Pooling on feature maps.
    3. Fusion: Concatenates visual features with processed metadata embeddings.
    4. Heads:
       - Cancer Head (Binary Classification)
       - BIRADS Head (Auxiliary Multi-class Classification)
       - Density Head (Auxiliary Multi-class Classification)
    """

    def __init__(self):
        super(MTSIN, self).__init__()

        # 1. Visual Backbone
        # num_classes=0 removes the classifier, global_pool='' keeps spatial feature maps
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="",
            in_chans=Config.IN_CHANNELS,
        )

        # Determine the number of output channels from the backbone
        # We run a dummy forward pass to get the shape
        with torch.no_grad():
            dummy_input = torch.randn(1, Config.IN_CHANNELS, 256, 256)
            features = self.backbone(dummy_input)
            self.n_features = features.shape[1]

        # 2. Pooling
        self.gem = GeM()

        # 3. Metadata Processing
        # Input dim is the number of meta features defined in Config (age, implant, lat, view, site)
        self.meta_input_dim = len(Config.META_FEATURES)
        self.meta_embed_dim = Config.META_EMBED_DIM

        self.meta_mlp = nn.Sequential(
            nn.Linear(self.meta_input_dim, self.meta_embed_dim),
            nn.BatchNorm1d(self.meta_embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.meta_embed_dim, self.meta_embed_dim),
            nn.ReLU(),
        )

        # 4. Heads
        # Combined dimension = Visual features + Meta embedding
        self.combined_dim = self.n_features + self.meta_embed_dim

        # Primary Task: Cancer (Binary)
        self.cancer_head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(self.combined_dim, Config.NUM_CLASSES)
        )

        # Aux Task: BIRADS (Multi-class)
        self.birads_head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(self.combined_dim, Config.NUM_BIRADS_CLASSES)
        )

        # Aux Task: Density (Multi-class)
        self.density_head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(self.combined_dim, Config.NUM_DENSITY_CLASSES)
        )

    def forward(self, image, meta):
        """
        Forward pass.

        Args:
            image (torch.Tensor): Image tensor of shape (B, C, H, W).
            meta (torch.Tensor): Metadata tensor of shape (B, meta_dim).

        Returns:
            dict: Dictionary containing logits for 'cancer', 'birads', and 'density'.
        """
        # --- Visual Branch ---
        # Extract features: (B, n_features, H_feat, W_feat)
        x_visual = self.backbone(image)

        # Pooling: (B, n_features, 1, 1)
        x_visual = self.gem(x_visual)

        # Flatten: (B, n_features)
        x_visual = x_visual.view(x_visual.size(0), -1)

        # --- Meta Branch ---
        # Process metadata: (B, meta_embed_dim)
        x_meta = self.meta_mlp(meta)

        # --- Fusion ---
        # Concatenate: (B, n_features + meta_embed_dim)
        x_combined = torch.cat([x_visual, x_meta], dim=1)

        # --- Heads ---
        cancer_logits = self.cancer_head(x_combined)
        birads_logits = self.birads_head(x_combined)
        density_logits = self.density_head(x_combined)

        return {
            "cancer": cancer_logits,
            "birads": birads_logits,
            "density": density_logits,
        }
