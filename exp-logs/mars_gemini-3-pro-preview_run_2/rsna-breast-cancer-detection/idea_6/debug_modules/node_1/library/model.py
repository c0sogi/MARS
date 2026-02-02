import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes (1/N * sum(x^p))^(1/p).
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x shape: (B, C, H, W)
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class MultiTaskEfficientNet(nn.Module):
    """
    Multi-Task EfficientNetV2 with Tabular Fusion.

    Backbone: EfficientNetV2-Small (Unfrozen)
    Pooling: GeM
    Heads:
        - Cancer (Binary)
        - BIRADS (Classification)
        - Density (Classification)
    """

    def __init__(self, num_machine_ids=32, pretrained=True):
        super(MultiTaskEfficientNet, self).__init__()

        # 1. Visual Backbone
        # Load pretrained EfficientNetV2, remove classification head and pooling
        self.backbone = timm.create_model(
            Config.MODEL_NAME, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Get feature dimension (e.g., 1280 for efficientnetv2_s)
        self.num_features = self.backbone.num_features

        # Learnable Pooling
        self.global_pool = GeM()

        # 2. Tabular Processing
        # Machine ID Embedding
        embedding_dim = 16
        self.machine_embedding = nn.Embedding(num_machine_ids, embedding_dim)

        # Calculate input dimension for tabular MLP
        # Age (1) + Implant (1) + View (6, one-hot) + Machine Embedding (16)
        tabular_input_dim = 1 + 1 + len(Config.VIEW_MAPPING) + embedding_dim

        self.tabular_projector = nn.Sequential(
            nn.Linear(tabular_input_dim, Config.NUM_TABULAR_FEATURES),
            nn.BatchNorm1d(Config.NUM_TABULAR_FEATURES),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # 3. Fusion & Heads
        fusion_dim = self.num_features + Config.NUM_TABULAR_FEATURES

        # Primary Head: Cancer (Binary)
        self.cancer_head = nn.Linear(fusion_dim, 1)

        # Auxiliary Head 1: BIRADS (Classification)
        self.birads_head = nn.Linear(fusion_dim, Config.NUM_BIRADS_CLASSES)

        # Auxiliary Head 2: Density (Classification)
        self.density_head = nn.Linear(fusion_dim, Config.NUM_DENSITY_CLASSES)

    def forward(self, image, tabular):
        """
        Args:
            image (torch.Tensor): Image tensor of shape (B, C, H, W)
            tabular (dict): Dictionary containing tabular features
                - 'age': (B,) float
                - 'implant': (B,) float
                - 'view': (B, 6) float (one-hot)
                - 'machine_id': (B,) long (indices)

        Returns:
            dict: Dictionary containing logits for 'cancer', 'birads', 'density'
        """
        # --- Visual Path ---
        # Extract features: (B, C, H, W)
        x = self.backbone(image)
        # Pool: (B, C, 1, 1)
        x = self.global_pool(x)
        # Flatten: (B, C)
        x = x.flatten(1)

        # --- Tabular Path ---
        # Ensure dimensions are correct for concatenation
        age = tabular["age"].unsqueeze(1)  # (B, 1)
        implant = tabular["implant"].unsqueeze(1)  # (B, 1)
        view = tabular["view"]  # (B, 6)
        machine_id = tabular["machine_id"]  # (B,)

        # Embedding lookup
        machine_emb = self.machine_embedding(machine_id)  # (B, 16)

        # Concatenate raw tabular features
        tab_feat = torch.cat([age, implant, view, machine_emb], dim=1)

        # Project to latent space
        tab_feat = self.tabular_projector(tab_feat)  # (B, 64)

        # --- Fusion ---
        fused = torch.cat([x, tab_feat], dim=1)

        # --- Heads ---
        cancer_logits = self.cancer_head(fused)
        birads_logits = self.birads_head(fused)
        density_logits = self.density_head(fused)

        return {
            "cancer": cancer_logits,
            "birads": birads_logits,
            "density": density_logits,
        }
