import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    Computes the generalized mean of the feature map, which allows the model
    to focus on high-activation regions (like lesions) more effectively than
    standard average pooling.

    Formula: f(X) = (1/|X| * sum(x^p))^(1/p)
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Clamp to avoid numerical instability with pow
        # Average pooling over the spatial dimensions (H, W)
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

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


class SHR_MTN(nn.Module):
    """
    Stabilized High-Resolution Multi-Task Network.

    Architecture:
    1. Backbone: EfficientNetV2-Small (Unfrozen, Pretrained)
    2. Pooling: GeM
    3. Fusion: Concatenation of Visual Embedding + Processed Metadata
    4. Heads:
       - Cancer (Binary Classification)
       - BIRADS (Regression)
       - Density (Multi-class Classification)
    """

    def __init__(self, num_aux_features):
        """
        Args:
            num_aux_features (int): Dimensionality of the auxiliary metadata input.
        """
        super(SHR_MTN, self).__init__()

        # 1. Visual Backbone
        # num_classes=0 removes the classifier, global_pool='' removes the default pooling
        self.backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="",
            in_chans=Config.NUM_CHANNELS,
        )

        # Get feature dimension (e.g., 1280 for efficientnetv2_s)
        self.n_features = self.backbone.num_features

        # 2. Pooling Layer
        self.gem = GeM()

        # 3. Metadata Processing (MLP)
        # Projects metadata into a latent space before fusion
        self.meta_embedding_dim = 64
        self.meta_mlp = nn.Sequential(
            nn.Linear(num_aux_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.meta_embedding_dim),
            nn.BatchNorm1d(self.meta_embedding_dim),
            nn.ReLU(),
        )

        # 4. Heads
        # Input dimension is Visual Features + Metadata Embedding
        combined_dim = self.n_features + self.meta_embedding_dim

        # Primary Head: Cancer Detection
        self.cancer_head = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(combined_dim, Config.NUM_CANCER_CLASSES)
        )

        # Auxiliary Head 1: BIRADS (Regression)
        self.birads_head = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(combined_dim, Config.NUM_BIRADS_CLASSES)
        )

        # Auxiliary Head 2: Density (Classification)
        self.density_head = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(combined_dim, Config.NUM_DENSITY_CLASSES)
        )

        # 5. Initialization
        self._init_weights()

    def _init_weights(self):
        """
        Custom initialization for stability.
        """
        # Initialize Cancer Head Bias to handle class imbalance
        # Bias ~ -3.9 corresponds to a probability of ~0.02 (sigmoid(-3.9) approx 0.02)
        if (
            hasattr(self.cancer_head[1], "bias")
            and self.cancer_head[1].bias is not None
        ):
            self.cancer_head[1].bias.data.fill_(Config.INIT_BIAS)

    def forward(self, image, aux_features):
        """
        Forward pass of the network.

        Args:
            image (torch.Tensor): Input images (B, C, H, W)
            aux_features (torch.Tensor): Metadata features (B, N_meta)

        Returns:
            dict: Dictionary containing outputs for 'cancer', 'birads', and 'density'.
        """
        # 1. Visual Feature Extraction
        # Output: (B, C, H, W)
        features = self.backbone(image)

        # 2. Pooling
        # Output: (B, C, 1, 1) -> Flatten to (B, C)
        features = self.gem(features).flatten(1)

        # 3. Metadata Processing
        # Output: (B, meta_dim)
        meta_emb = self.meta_mlp(aux_features)

        # 4. Fusion
        # Output: (B, C + meta_dim)
        combined = torch.cat([features, meta_emb], dim=1)

        # 5. Heads
        cancer_logits = self.cancer_head(combined)
        birads_logits = self.birads_head(combined)
        density_logits = self.density_head(combined)

        return {
            "cancer": cancer_logits,
            "birads": birads_logits,
            "density": density_logits,
        }
