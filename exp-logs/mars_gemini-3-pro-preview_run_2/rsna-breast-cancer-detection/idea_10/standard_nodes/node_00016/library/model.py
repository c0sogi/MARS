import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    Computes the generalized mean of the input tensor.
    Formula: f(X) = (1/|X| * sum(x^p))^(1/p)
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Clamp to avoid numerical instability with power function
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


class SMTSINModel(nn.Module):
    """
    Stabilized Multi-Task Single-Instance Network (SMT-SIN).

    Architecture:
    1. Backbone: EfficientNetV2-Small (Fine-tuned, 3-channel input).
    2. Pooling: GeM Pooling.
    3. Metadata: Parallel MLP processing Age, Implant, View, and Machine ID.
    4. Heads:
       - Cancer (Binary Classification)
       - BIRADS (Regression)
       - Density (Multi-class Classification)
    """

    def __init__(self):
        super(SMTSINModel, self).__init__()

        # ==========================
        # 1. Visual Backbone
        # ==========================
        # Load EfficientNetV2-Small from timm
        # global_pool='' ensures we get the spatial feature map (B, C, H, W)
        self.backbone = timm.create_model(
            Config.MODEL_BACKBONE,
            pretrained=Config.PRETRAINED,
            in_chans=Config.IN_CHANNELS,
            num_classes=0,  # Remove default classifier
            global_pool="",  # Remove default pooling
            drop_rate=Config.DROP_RATE,
            drop_path_rate=Config.DROP_PATH_RATE,
        )

        # Determine the number of output channels from the backbone dynamically
        with torch.no_grad():
            dummy_input = torch.randn(1, Config.IN_CHANNELS, 256, 256)
            features = self.backbone(dummy_input)
            self.n_features = features.shape[1]

        # ==========================
        # 2. Pooling Layer
        # ==========================
        self.pool = GeM()

        # ==========================
        # 3. Metadata Branch
        # ==========================
        # Metadata inputs: [Age (1), Implant (1), View (Idx), Machine (Idx)]
        # We use embeddings for categorical variables (View and Machine)

        # View: 6 unique values -> Embedding Dim 4
        self.view_embed = nn.Embedding(6, 4)

        # Machine: 10 unique values -> Embedding Dim 4
        self.machine_embed = nn.Embedding(10, 4)

        # Total input dimension for MLP:
        # Age(1) + Implant(1) + View_Emb(4) + Machine_Emb(4) = 10
        self.meta_input_dim = 1 + 1 + 4 + 4

        self.meta_mlp = nn.Sequential(
            nn.Linear(self.meta_input_dim, Config.META_EMBED_DIM),
            nn.BatchNorm1d(Config.META_EMBED_DIM),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(Config.META_EMBED_DIM, Config.META_EMBED_DIM),
            nn.ReLU(),
        )

        # ==========================
        # 4. Fusion & Heads
        # ==========================
        self.fusion_dim = self.n_features + Config.META_EMBED_DIM

        # Primary Head: Cancer Detection (Binary)
        self.cancer_head = nn.Linear(self.fusion_dim, Config.NUM_CLASSES)

        # Auxiliary Head 1: BIRADS Score (Regression)
        self.birads_head = nn.Linear(self.fusion_dim, Config.AUX_BIRADS_CLASSES)

        # Auxiliary Head 2: Density Rating (Classification)
        self.density_head = nn.Linear(self.fusion_dim, Config.AUX_DENSITY_CLASSES)

    def forward(self, images, metadata):
        """
        Forward pass of the network.

        Args:
            images (torch.Tensor): Input images of shape (B, C, H, W).
            metadata (torch.Tensor): Metadata tensor of shape (B, 4).
                                     [age_norm, implant, view_idx, machine_idx]

        Returns:
            dict: Dictionary containing outputs for 'cancer', 'birads', and 'density'.
        """
        # --- Visual Feature Extraction ---
        # Get spatial features: (B, C, H, W)
        features = self.backbone(images)

        # Apply GeM pooling and flatten: (B, C, 1, 1) -> (B, C)
        global_features = self.pool(features).flatten(1)

        # --- Metadata Processing ---
        # Unpack metadata (input is float tensor)
        age = metadata[:, 0].unsqueeze(1)  # (B, 1)
        implant = metadata[:, 1].unsqueeze(1)  # (B, 1)

        # Cast indices to long for embedding lookup
        view_idx = metadata[:, 2].long()  # (B,)
        machine_idx = metadata[:, 3].long()  # (B,)

        # Get embeddings
        view_emb = self.view_embed(view_idx)  # (B, 4)
        machine_emb = self.machine_embed(machine_idx)  # (B, 4)

        # Concatenate all metadata features
        meta_features = torch.cat([age, implant, view_emb, machine_emb], dim=1)

        # Pass through MLP
        meta_embedding = self.meta_mlp(meta_features)  # (B, META_EMBED_DIM)

        # --- Fusion ---
        # Concatenate visual and metadata embeddings
        combined = torch.cat([global_features, meta_embedding], dim=1)

        # --- Output Heads ---
        cancer_logits = self.cancer_head(combined)
        birads_pred = self.birads_head(combined)
        density_logits = self.density_head(combined)

        return {
            "cancer": cancer_logits,
            "birads": birads_pred,
            "density": density_logits,
        }
