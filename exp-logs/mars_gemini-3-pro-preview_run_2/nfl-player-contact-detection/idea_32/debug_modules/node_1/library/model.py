import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DenseFeatureEncoder(nn.Module):
    """
    A simple dense encoder block (Linear -> BN -> ReLU) used to process
    independent kinematic feature groups.
    """

    def __init__(self, input_dim, output_dim):
        super(DenseFeatureEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim), nn.BatchNorm1d(output_dim), nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    """
    Deep Residual MLP Block: Linear -> BN -> ReLU -> Dropout -> Linear -> Add.
    """

    def __init__(self, dim, dropout_rate):
        super(ResidualBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return self.net(x) + x


class HPIRVN(nn.Module):
    """
    Hierarchical Physics-Informed Residual-Visual Network (HPI-RVN).

    Architecture:
    1. Kinematic Stream:
       - Splits inputs into Geometry, Motion, and Dynamics groups.
       - Processes each group with independent DenseFeatureEncoders.
       - Fuses encodings via concatenation and a Residual MLP backbone.
    2. Visual Stream:
       - Processes helmet box metrics via a shallow MLP.
    3. Fusion:
       - Combines streams via residual connection: L_final = L_kin + lambda * L_vis.
    """

    def __init__(self):
        super(HPIRVN, self).__init__()

        # --- Input Dimensions ---
        # Calculate flattened dimensions: num_features * total_frames
        self.dim_geo = len(Config.FEAT_GROUP_A_GEO) * Config.TOTAL_FRAMES
        self.dim_motion = len(Config.FEAT_GROUP_B_MOTION) * Config.TOTAL_FRAMES
        self.dim_dynamics = len(Config.FEAT_GROUP_C_DYNAMICS) * Config.TOTAL_FRAMES

        # Visual features are current-step only
        self.dim_visual = len(Config.FEAT_VISUAL)

        # --- Kinematic Stream Components ---
        # Independent Encoders
        self.encoder_geo = DenseFeatureEncoder(self.dim_geo, Config.HIDDEN_DIM_GEO)
        self.encoder_motion = DenseFeatureEncoder(
            self.dim_motion, Config.HIDDEN_DIM_MOTION
        )
        self.encoder_dynamics = DenseFeatureEncoder(
            self.dim_dynamics, Config.HIDDEN_DIM_DYNAMICS
        )

        # Fusion Backbone
        # 1. Projection from concatenated dim to fusion dim
        concat_dim = (
            Config.HIDDEN_DIM_GEO
            + Config.HIDDEN_DIM_MOTION
            + Config.HIDDEN_DIM_DYNAMICS
        )
        self.kinematic_proj = nn.Sequential(
            nn.Linear(concat_dim, Config.FUSION_DIM),
            nn.BatchNorm1d(Config.FUSION_DIM),
            nn.ReLU(),
        )

        # 2. Residual Block
        self.kinematic_res = ResidualBlock(Config.FUSION_DIM, Config.DROPOUT)

        # 3. Output Head
        self.kinematic_head = nn.Linear(Config.FUSION_DIM, 1)

        # --- Visual Stream Components ---
        # Shallow MLP for visual correction
        self.visual_net = nn.Sequential(
            nn.Linear(self.dim_visual, Config.HIDDEN_DIM_VISUAL),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM_VISUAL, 1),
        )

        # Fusion Hyperparameter
        self.visual_lambda = Config.VISUAL_LAMBDA

    def forward(self, geometry, motion, dynamics, visual):
        """
        Args:
            geometry (Tensor): Batch of flattened geometry features.
            motion (Tensor): Batch of flattened motion features.
            dynamics (Tensor): Batch of flattened dynamics features.
            visual (Tensor): Batch of visual features.

        Returns:
            Tensor: Final logits (B, 1).
        """
        # 1. Kinematic Encoding
        e_geo = self.encoder_geo(geometry)
        e_mot = self.encoder_motion(motion)
        e_dyn = self.encoder_dynamics(dynamics)

        # 2. Kinematic Fusion
        concat = torch.cat([e_geo, e_mot, e_dyn], dim=1)
        x = self.kinematic_proj(concat)
        x = self.kinematic_res(x)
        logit_kin = self.kinematic_head(x)

        # 3. Visual Encoding
        logit_vis = self.visual_net(visual)

        # 4. Residual Fusion
        # Logit_final = L_kin + lambda * L_vis
        logit_final = logit_kin + self.visual_lambda * logit_vis

        return logit_final


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for binary classification.
    Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha=Config.ALPHA, gamma=Config.GAMMA, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        # Use BCEWithLogitsLoss for numerical stability
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, inputs, targets):
        # Calculate BCE loss per element
        bce_loss = self.bce(inputs, targets)

        # Calculate probabilities (p_t)
        # p = sigmoid(inputs)
        # if y=1, pt = p; if y=0, pt = 1-p
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)

        # Calculate Focal term
        loss = bce_loss * ((1 - p_t) ** self.gamma)

        # Apply Alpha weighting
        # alpha_t = alpha if y=1 else (1-alpha)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
