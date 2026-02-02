import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes (1/N * sum(x^p))^(1/p).
    p is a learnable parameter.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x: (B, C, H, W)
        # Ensure p is effectively > 1 and handle numerical stability
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3.0, eps=1e-6):
        # Clamp x to avoid NaN with power
        # Global Average Pooling logic generalized
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class CMTSINModel(nn.Module):
    """
    Calibrated Multi-Task Single-Instance Network (CMT-SIN).

    Architecture:
    1. Backbone: EfficientNetV2-S (in_chans=1, pretrained)
    2. Pooling: GeM
    3. Metadata Branch: Embeddings for categorical, MLP for fusion
    4. Heads:
       - Cancer (Binary)
       - BIRADS (Auxiliary, 3 classes)
       - Density (Auxiliary, 4 classes)
    """

    def __init__(self):
        super(CMTSINModel, self).__init__()

        # 1. Visual Backbone
        # in_chans=1 adapts the first layer weights (summing RGB) for grayscale
        self.backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=Config.PRETRAINED,
            in_chans=1,
            num_classes=0,  # No classifier, just features
            global_pool="",  # We implement custom pooling
            drop_rate=Config.DROP_RATE,
            drop_path_rate=Config.DROP_PATH_RATE,
        )

        # Determine backbone output channels
        # EfficientNetV2-S usually outputs 1280 features
        dummy_input = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            features = self.backbone(dummy_input)
            self.n_features = features.shape[1]

        # 2. Pooling
        self.pooling = GeM()
        self.bn_visual = nn.BatchNorm1d(self.n_features)

        # 3. Metadata Branch
        # Metadata tensor structure from Dataset: [Age, Implant, View, Lat, Machine]
        # View: 6 classes, Machine: ~10 classes
        self.view_embed = nn.Embedding(6, 4)
        self.machine_embed = nn.Embedding(10, 4)

        # Continuous/Binary inputs: Age (1) + Implant (1) + Laterality (1) = 3
        # Total metadata input dimension = 4 (View) + 4 (Machine) + 3 (Dense) = 11
        self.meta_input_dim = 11
        self.meta_hidden_dim = 32

        self.meta_mlp = nn.Sequential(
            nn.Linear(self.meta_input_dim, self.meta_hidden_dim),
            nn.BatchNorm1d(self.meta_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # 4. Fusion & Heads
        self.fusion_dim = self.n_features + self.meta_hidden_dim

        # Primary Head: Cancer
        self.cancer_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.fusion_dim, 1))

        # Auxiliary Heads
        if Config.USE_AUX_TASKS:
            # BIRADS (3 classes: 0, 1, 2)
            self.birads_head = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.fusion_dim, Config.AUX_TASKS["BIRADS"]["num_classes"]),
            )
            # Density (4 classes: A, B, C, D)
            self.density_head = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.fusion_dim, Config.AUX_TASKS["density"]["num_classes"]),
            )

    def forward(self, image, meta):
        """
        Args:
            image: (B, 1, H, W)
            meta: (B, 5) -> [Age, Implant, View_Idx, Lat_Idx, Machine_Idx]
        """
        # --- Visual Path ---
        # Extract features: (B, C, H_feat, W_feat)
        x = self.backbone(image)

        # Pool: (B, C, 1, 1) -> (B, C)
        x = self.pooling(x)
        x = x.flatten(1)
        x = self.bn_visual(x)

        # --- Metadata Path ---
        # Parse metadata tensor
        # meta[:, 0] -> Age (float)
        # meta[:, 1] -> Implant (0/1)
        # meta[:, 2] -> View Idx (0-5)
        # meta[:, 3] -> Lat Idx (0/1)
        # meta[:, 4] -> Machine Idx (0-9)

        age = meta[:, 0].unsqueeze(1)  # (B, 1)
        implant = meta[:, 1].unsqueeze(1)  # (B, 1)
        view_idx = meta[:, 2].long()  # (B,)
        lat = meta[:, 3].unsqueeze(1)  # (B, 1)
        machine_idx = meta[:, 4].long()  # (B,)

        # Embeddings
        view_emb = self.view_embed(view_idx)  # (B, 4)
        machine_emb = self.machine_embed(machine_idx)  # (B, 4)

        # Concatenate metadata features
        meta_features = torch.cat([age, implant, lat, view_emb, machine_emb], dim=1)

        # Process via MLP
        meta_out = self.meta_mlp(meta_features)

        # --- Fusion ---
        fused = torch.cat([x, meta_out], dim=1)

        # --- Heads ---
        outputs = {}

        # Primary Cancer Prediction (Logits)
        outputs["cancer"] = self.cancer_head(fused).squeeze(1)  # (B,)

        # Auxiliary Predictions
        if Config.USE_AUX_TASKS:
            outputs["BIRADS"] = self.birads_head(fused)  # (B, 3)
            outputs["density"] = self.density_head(fused)  # (B, 4)

        return outputs
