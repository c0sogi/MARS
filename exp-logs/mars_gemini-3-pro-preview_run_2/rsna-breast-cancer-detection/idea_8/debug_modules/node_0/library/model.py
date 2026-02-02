import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    Computes the p-th power mean of the input feature map.
    f(X) = (1/N * sum(x^p))^(1/p)
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: (B, C, H, W)
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Clamp to avoid numerical instability
        x = x.clamp(min=eps).pow(p)
        # Average pooling over spatial dimensions (H, W)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        # Root p
        return x.pow(1.0 / p)


class MultiTaskEfficientNet(nn.Module):
    """
    High-Resolution Multi-Task Network.
    Backbone: EfficientNetV2-Small (Fine-tuned)
    Pooling: GeM
    Fusion: Concatenation of visual and metadata embeddings
    Heads: Cancer (Primary), BIRADS (Aux), Density (Aux)
    """

    def __init__(self, pretrained=True):
        super(MultiTaskEfficientNet, self).__init__()

        # --- 1. Visual Backbone ---
        # Load EfficientNetV2-Small from timm
        # num_classes=0 and global_pool='' removes the head and pooling,
        # returning spatial feature maps (B, C, H, W)
        self.backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )

        # Retrieve the number of output channels from the backbone
        self.in_features = self.backbone.num_features

        # --- 2. Pooling Layer ---
        self.gem = GeM()

        # --- 3. Metadata Processing ---
        # Input vector size from dataset.py is 4: [age_norm, implant, view_enc, machine_enc]
        self.meta_input_dim = 4
        self.meta_embed_dim = Config.META_EMBED_DIM

        self.meta_mlp = nn.Sequential(
            nn.Linear(self.meta_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.meta_embed_dim),
            nn.BatchNorm1d(self.meta_embed_dim),
            nn.ReLU(),
        )

        # --- 4. Prediction Heads ---
        # Dimension after concatenating visual and metadata features
        self.combined_dim = self.in_features + self.meta_embed_dim

        # Primary Head: Cancer Detection (Binary)
        self.cancer_head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(self.combined_dim, Config.NUM_CLASSES)
        )

        # Auxiliary Heads (Multi-Task Learning)
        if Config.USE_AUX_HEADS:
            # BIRADS: 3 classes (0, 1, 2)
            self.birads_head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(self.combined_dim, Config.NUM_BIRADS_CLASSES),
            )

            # Density: 4 classes (A, B, C, D)
            self.density_head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(self.combined_dim, Config.NUM_DENSITY_CLASSES),
            )

    def forward(self, images, meta_vec):
        """
        Forward pass of the network.

        Args:
            images (torch.Tensor): Image tensor of shape (B, 3, H, W).
            meta_vec (torch.Tensor): Metadata vector of shape (B, 4).

        Returns:
            dict: Dictionary containing logits for 'cancer', 'birads', and 'density'.
        """
        # --- 1. Extract Visual Features ---
        # (B, C, H, W)
        features = self.backbone(images)

        # Apply GeM Pooling -> (B, C, 1, 1)
        features = self.gem(features)

        # Flatten -> (B, C)
        features = features.flatten(1)

        # --- 2. Extract Metadata Features ---
        # (B, meta_embed_dim)
        meta_features = self.meta_mlp(meta_vec)

        # --- 3. Feature Fusion ---
        # Concatenate visual and metadata embeddings
        combined = torch.cat([features, meta_features], dim=1)

        # --- 4. Heads ---
        outputs = {}

        # Primary Task
        outputs["cancer"] = self.cancer_head(combined)

        # Auxiliary Tasks
        if Config.USE_AUX_HEADS:
            outputs["birads"] = self.birads_head(combined)
            outputs["density"] = self.density_head(combined)

        return outputs
