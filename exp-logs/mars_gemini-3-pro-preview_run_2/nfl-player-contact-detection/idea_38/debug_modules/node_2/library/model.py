import torch
import torch.nn as nn
import numpy as np
from library.config import (
    KINEMATIC_BASE_FEATURES,
    VISUAL_BASE_FEATURES,
    WINDOW_SIZE,
    NOISE_SIGMA,
    RESIDUAL_LAMBDA,
    DROPOUT_RATE,
    HIDDEN_DIMS,
    SEED,
)


class HardClamp(nn.Module):
    """
    Clamps input tensor to a fixed range for numerical stability.
    Suggested range in idea: [-50, 50] for normalized inputs.
    """

    def __init__(self, min_val=-50.0, max_val=50.0):
        super(HardClamp, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return torch.clamp(x, min=self.min_val, max=self.max_val)


class GaussianNoise(nn.Module):
    """
    Injects Gaussian noise to the input during training only.
    """

    def __init__(self, sigma=0.1):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training and self.sigma > 0:
            noise = torch.randn_like(x) * self.sigma
            return x + noise
        return x


class ResBlock(nn.Module):
    """
    Standard Residual Block: x + f(x)
    f(x) = Linear -> BN -> ReLU -> Dropout -> Linear
    """

    def __init__(self, dim, dropout_rate=0.2):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.block(x)


class KinematicStream(nn.Module):
    """
    Pyramidal Invariant Backbone with Noise Regularization.
    Structure:
    Input -> Clamp -> Noise ->
    [Project -> ResBlock] x N -> Output Logit
    """

    def __init__(self, input_dim, hidden_dims, dropout_rate, noise_sigma):
        super(KinematicStream, self).__init__()

        # 1. Stability & Regularization
        self.clamp = HardClamp(min_val=-50.0, max_val=50.0)
        self.noise = GaussianNoise(sigma=noise_sigma)

        # 2. Pyramidal Backbone
        layers = []
        current_dim = input_dim

        for h_dim in hidden_dims:
            # Projection Layer
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

            # Residual Block at new dimension
            layers.append(ResBlock(h_dim, dropout_rate))

            current_dim = h_dim

        self.backbone = nn.Sequential(*layers)

        # 3. Output Head
        self.head = nn.Linear(current_dim, 1)

    def forward(self, x):
        x = self.clamp(x)
        x = self.noise(x)
        features = self.backbone(x)
        return self.head(features)


class VisualStream(nn.Module):
    """
    Shallow MLP for Visual Correction.
    Prevents overfitting to noisy bounding box proxies.
    """

    def __init__(self, input_dim):
        super(VisualStream, self).__init__()
        # Simple Shallow Architecture
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x)


class PIRVNoiseModel(nn.Module):
    """
    Pyramidal Invariant Residual-Visual Network with Noise Regularization.

    Fuses a robust kinematic stream with a shallow visual stream via
    additive residual connection.
    """

    def __init__(self):
        super(PIRVNoiseModel, self).__init__()

        # Calculate Input Dimensions
        # (2 * WINDOW + 1) lags * number of base features
        num_lags = 2 * WINDOW_SIZE + 1
        self.kin_input_dim = len(KINEMATIC_BASE_FEATURES) * num_lags
        self.vis_input_dim = len(VISUAL_BASE_FEATURES) * num_lags

        # Hyperparameters
        self.residual_lambda = RESIDUAL_LAMBDA

        # Streams
        self.kinematic_stream = KinematicStream(
            input_dim=self.kin_input_dim,
            hidden_dims=HIDDEN_DIMS,
            dropout_rate=DROPOUT_RATE,
            noise_sigma=NOISE_SIGMA,
        )

        self.visual_stream = VisualStream(input_dim=self.vis_input_dim)

    def forward(self, x_kin, x_vis):
        """
        Args:
            x_kin: Tensor of shape (Batch, Kinematic_Feats)
            x_vis: Tensor of shape (Batch, Visual_Feats)
        Returns:
            logits: Tensor of shape (Batch, 1)
        """
        # Kinematic Logit
        l_kin = self.kinematic_stream(x_kin)

        # Visual Logit
        l_vis = self.visual_stream(x_vis)

        # Additive Residual Fusion
        # Logit_final = L_kin + lambda * L_vis
        logits = l_kin + (self.residual_lambda * l_vis)

        return logits
