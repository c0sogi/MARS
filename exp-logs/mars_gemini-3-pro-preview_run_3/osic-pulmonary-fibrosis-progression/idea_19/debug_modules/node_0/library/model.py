import torch
import torch.nn as nn
import timm
from library.config import Config


class ClinicalStream(nn.Module):
    """
    Stream A: The Clinical Anchor.
    Processes tabular clinical features to learn the expected trajectory.
    Architecture: Over-Parameterized MLP (Input -> 128 -> ReLU -> 64).
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ClinicalStream, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class VisualStream(nn.Module):
    """
    Stream B: The Visual Interaction Stream.
    Processes CT scans and clinical features to capture non-linear cross-modal interactions.

    Components:
    1. Backbone: EfficientNet-B2 (Top 2 stages unfrozen).
    2. Projection: Linear map of image features.
    3. Interaction MLP: Processes [Image_Emb, Tabular] -> Latent.
    """

    def __init__(self, backbone_name, img_embed_dim, tab_dim, hidden_dim, output_dim):
        super(VisualStream, self).__init__()

        # 1. Backbone
        # num_classes=0 returns the global pool features directly
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)

        # Freezing Logic: Freeze all, then unfreeze top 2 stages
        # EfficientNet B2 typically has blocks 0-6. We unfreeze 5, 6 and head.
        for param in self.backbone.parameters():
            param.requires_grad = False

        for name, child in self.backbone.named_children():
            if name == "blocks":
                for i, block in enumerate(child):
                    if i >= 5:  # Unfreeze last 2 block stages
                        for param in block.parameters():
                            param.requires_grad = True
            elif name in ["conv_head", "bn2"]:
                for param in child.parameters():
                    param.requires_grad = True

        # 2. Image Projection
        self.n_features = self.backbone.num_features
        self.img_projector = nn.Linear(self.n_features, img_embed_dim)

        # 3. Interaction MLP
        # Input: Image Embedding (64) + Clinical Features (5)
        interaction_input_dim = img_embed_dim + tab_dim
        self.interaction_mlp = nn.Sequential(
            nn.Linear(interaction_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, img, tab):
        # Extract features from backbone
        # Shape: (B, n_features)
        feat = self.backbone(img)

        # Project to embedding dimension
        # Shape: (B, img_embed_dim)
        img_emb = self.img_projector(feat)

        # Concatenate with tabular features
        # Shape: (B, img_embed_dim + tab_dim)
        combined = torch.cat([img_emb, tab], dim=1)

        # Compute interaction latent vector
        # Shape: (B, output_dim)
        out = self.interaction_mlp(combined)
        return out


class MACRNet(nn.Module):
    """
    Metric-Aligned Clinical-Residual Fusion Network (MACR-Net).

    Implements a Latent Residual Summation strategy:
    H_final = H_clin + H_vis

    Where:
    - H_clin is the robust clinical anchor.
    - H_vis is the visual correction term.
    """

    def __init__(self):
        super(MACRNet, self).__init__()

        # Dimensions
        self.tab_dim = 5  # [BaseFVC_norm, t_rel, Age_norm, Sex_Code, Smoking_Code]
        self.hidden_dim = Config.HIDDEN_DIM
        self.latent_dim = 64  # Constrained latent dimension
        self.img_embed_dim = Config.IMG_EMBED_DIM

        # Stream A: Clinical Stream
        self.clinical_stream = ClinicalStream(
            input_dim=self.tab_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.latent_dim,
        )

        # Stream B: Visual Stream
        self.visual_stream = VisualStream(
            backbone_name=Config.BACKBONE_NAME,
            img_embed_dim=self.img_embed_dim,
            tab_dim=self.tab_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.latent_dim,
        )

        # Shared Head
        # Projects fused latent representation to mu (FVC) and raw sigma (Confidence)
        self.head = nn.Linear(self.latent_dim, 2)

    def forward(self, img, tab):
        """
        Args:
            img (torch.Tensor): (B, 3, H, W)
            tab (torch.Tensor): (B, 5)
        Returns:
            torch.Tensor: (B, 2) -> [mu, raw_sigma]
        """
        # 1. Compute Clinical Anchor
        h_clin = self.clinical_stream(tab)

        # 2. Compute Visual Correction
        h_vis = self.visual_stream(img, tab)

        # 3. Latent Residual Summation
        # The visual stream acts as a residual correction to the clinical anchor
        h_final = h_clin + h_vis

        # 4. Prediction
        out = self.head(h_final)

        return out
