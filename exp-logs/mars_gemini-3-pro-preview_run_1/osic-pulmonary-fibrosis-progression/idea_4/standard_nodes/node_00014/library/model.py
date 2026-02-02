import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class SiameseDualAxisNet(nn.Module):
    """
    Siamese Dual-Axis Tri-Slab Network.

    Architecture:
    1. Shared Backbone (EfficientNet-B0): Processes Axial and Coronal views independently
       but shares weights to learn isotropic texture features.
    2. Tabular Encoder: Encodes clinical metadata.
    3. Fusion Layer: Concatenates visual and tabular features.
    4. Multi-Task Heads:
       - Trajectory Head: Predicts alpha (slope), sigma_base, and sigma_growth.
       - Auxiliary Head: Predicts Baseline Percent for regularization.
    """

    def __init__(self):
        super(SiameseDualAxisNet, self).__init__()

        # 1. Visual Backbone (Siamese)
        # We use num_classes=0 to get the pooled feature vector directly
        # Config.BACKBONE is 'efficientnet_b0'
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            in_chans=3,
            drop_rate=Config.DROP_RATE,
        )
        self.backbone_dim = self.backbone.num_features  # 1280 for EfficientNet-B0

        # 2. Tabular Encoder
        # Input features: Age, Base_Percent, Sex, Smoking (4 total)
        self.tabular_input_dim = 4
        self.tabular_embed_dim = 128

        self.tabular_net = nn.Sequential(
            nn.Linear(self.tabular_input_dim, self.tabular_embed_dim),
            nn.LayerNorm(self.tabular_embed_dim),
            nn.ReLU(),
            nn.Dropout(Config.DROP_RATE),
            nn.Linear(self.tabular_embed_dim, self.tabular_embed_dim),
            nn.LayerNorm(self.tabular_embed_dim),
            nn.ReLU(),
        )

        # 3. Fusion Layer
        # Concatenates: Axial Features + Coronal Features + Tabular Embeddings
        self.fusion_input_dim = (self.backbone_dim * 2) + self.tabular_embed_dim
        self.fusion_dim = Config.FUSION_DIM

        self.fusion_net = nn.Sequential(
            nn.Linear(self.fusion_input_dim, self.fusion_dim),
            nn.BatchNorm1d(self.fusion_dim),
            nn.ReLU(),
            nn.Dropout(Config.DROP_RATE),
        )

        # 4. Heads

        # Trajectory Head: Outputs [alpha, sigma_base, sigma_growth]
        self.trajectory_head = nn.Linear(self.fusion_dim, 3)

        # Auxiliary Head: Outputs [pred_percent]
        self.aux_head = nn.Linear(self.fusion_dim, 1)

        # Weight Initialization for heads
        self._init_weights(self.trajectory_head)
        self._init_weights(self.aux_head)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, axial_img, coronal_img, tabular):
        """
        Args:
            axial_img: (B, 3, H, W) - Axial Tri-Slab
            coronal_img: (B, 3, H, W) - Coronal Tri-Slab
            tabular: (B, 4) - Normalized tabular features

        Returns:
            dict: {
                'alpha': (B,),
                'sigma_base': (B,),
                'sigma_growth': (B,),
                'pred_percent': (B,)
            }
        """
        # --- Visual Pathway (Siamese) ---
        # Pass both views through the same backbone
        feat_axial = self.backbone(axial_img)  # (B, backbone_dim)
        feat_coronal = self.backbone(coronal_img)  # (B, backbone_dim)

        # --- Tabular Pathway ---
        feat_tabular = self.tabular_net(tabular)  # (B, tabular_embed_dim)

        # --- Fusion ---
        # Concatenate all feature vectors
        concat_features = torch.cat([feat_axial, feat_coronal, feat_tabular], dim=1)
        fused_features = self.fusion_net(concat_features)  # (B, fusion_dim)

        # --- Heads ---
        traj_output = self.trajectory_head(fused_features)  # (B, 3)
        aux_output = self.aux_head(fused_features)  # (B, 1)

        # Extract trajectory components
        # alpha: Slope of decline (can be negative or positive)
        # sigma_base: Uncertainty at t=0 (must be positive)
        # sigma_growth: Uncertainty growth over time (must be positive)

        alpha = traj_output[:, 0]

        # Use Softplus to enforce positivity for sigma values
        # Add small epsilon to prevent numerical instability
        sigma_base = F.softplus(traj_output[:, 1]) + 1e-6
        sigma_growth = F.softplus(traj_output[:, 2]) + 1e-6

        pred_percent = aux_output.squeeze(1)

        return {
            "alpha": alpha,
            "sigma_base": sigma_base,
            "sigma_growth": sigma_growth,
            "pred_percent": pred_percent,
        }
