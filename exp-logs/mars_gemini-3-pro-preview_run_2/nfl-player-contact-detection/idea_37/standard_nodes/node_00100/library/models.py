import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class StochasticInputLayer(nn.Module):
    """
    Injects Gaussian noise N(0, sigma) into the input tensor during training.
    Acts as a continuous data augmentation strategy for kinematic features.
    """

    def __init__(self, sigma=0.05):
        super(StochasticInputLayer, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training and self.sigma > 0:
            noise = torch.randn_like(x) * self.sigma
            return x + noise
        return x


class PyramidalResBlock(nn.Module):
    """
    Residual Block for the Kinematic Backbone.
    Structure: Linear -> BN -> ReLU -> Dropout -> Linear -> Add
    """

    def __init__(self, in_features, dropout_rate=0.3):
        super(PyramidalResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class SPIRVNet(nn.Module):
    """
    Stochastic Pyramidal Invariant Residual-Visual Network (SPIRV-Net).

    Dual-Stream Architecture:
    1. Kinematic Stream: Stochastic inputs -> Pyramidal Residual Backbone.
    2. Visual Stream: Shallow MLP.
    3. Fusion: Logit_final = Logit_kin + lambda * Logit_vis.
    """

    def __init__(self, input_dim_kin, input_dim_vis):
        super(SPIRVNet, self).__init__()

        # --- Kinematic Stream ---
        self.kin_stochastic = StochasticInputLayer(sigma=Config.INPUT_NOISE_SIGMA)

        # Build Pyramidal Backbone
        # Structure: Project -> ResBlock -> Project -> ResBlock ...

        # Layer 1: Input -> 512
        self.kin_proj1 = nn.Sequential(
            nn.Linear(input_dim_kin, Config.KINEMATIC_HIDDEN_DIMS[0]),
            nn.BatchNorm1d(Config.KINEMATIC_HIDDEN_DIMS[0]),
            nn.ReLU(),
            nn.Dropout(Config.KINEMATIC_DROPOUT),
        )
        self.kin_res1 = PyramidalResBlock(
            Config.KINEMATIC_HIDDEN_DIMS[0], Config.KINEMATIC_DROPOUT
        )

        # Layer 2: 512 -> 256
        self.kin_proj2 = nn.Sequential(
            nn.Linear(Config.KINEMATIC_HIDDEN_DIMS[0], Config.KINEMATIC_HIDDEN_DIMS[1]),
            nn.BatchNorm1d(Config.KINEMATIC_HIDDEN_DIMS[1]),
            nn.ReLU(),
            nn.Dropout(Config.KINEMATIC_DROPOUT),
        )
        self.kin_res2 = PyramidalResBlock(
            Config.KINEMATIC_HIDDEN_DIMS[1], Config.KINEMATIC_DROPOUT
        )

        # Layer 3: 256 -> 128
        self.kin_proj3 = nn.Sequential(
            nn.Linear(Config.KINEMATIC_HIDDEN_DIMS[1], Config.KINEMATIC_HIDDEN_DIMS[2]),
            nn.BatchNorm1d(Config.KINEMATIC_HIDDEN_DIMS[2]),
            nn.ReLU(),
            nn.Dropout(Config.KINEMATIC_DROPOUT),
        )
        self.kin_res3 = PyramidalResBlock(
            Config.KINEMATIC_HIDDEN_DIMS[2], Config.KINEMATIC_DROPOUT
        )

        # Kinematic Head: 128 -> 1
        self.kin_head = nn.Linear(Config.KINEMATIC_HIDDEN_DIMS[2], 1)

        # --- Visual Stream ---
        # Shallow MLP: Input -> 64 -> 1
        self.vis_mlp = nn.Sequential(
            nn.Linear(input_dim_vis, Config.VISUAL_HIDDEN_DIMS[0]),
            nn.ReLU(),
            nn.Linear(Config.VISUAL_HIDDEN_DIMS[0], 1),
        )

        # Fusion Parameter
        self.residual_lambda = Config.RESIDUAL_LAMBDA

    def forward(self, x_kin, x_vis):
        # --- Kinematic Forward ---
        k = self.kin_stochastic(x_kin)

        k = self.kin_proj1(k)
        k = self.kin_res1(k)

        k = self.kin_proj2(k)
        k = self.kin_res2(k)

        k = self.kin_proj3(k)
        k = self.kin_res3(k)

        logit_kin = self.kin_head(k)

        # --- Visual Forward ---
        logit_vis = self.vis_mlp(x_vis)

        # --- Residual Fusion ---
        # Additive residual: Kinematic + lambda * Visual
        logit_final = logit_kin + self.residual_lambda * logit_vis

        return logit_final


class FocalLoss(nn.Module):
    """
    Binary Focal Loss implementation.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits
        # targets: binary labels (0 or 1)

        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)  # pt is the probability of the true class

        # Calculate alpha_t
        # if target=1, alpha_t = alpha
        # if target=0, alpha_t = 1 - alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
