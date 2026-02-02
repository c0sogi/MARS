import torch
import torch.nn as nn
from library import config


class InputClampingLayer(nn.Module):
    """
    A fixed, non-trainable layer that strictly clamps inputs to a specific range
    to ensure numerical stability.

    Since inputs are normalized (Z-scores), we use a wide symmetric range
    (e.g., [-50, 50]) to filter numerical outliers while preserving the
    distribution of valid data.
    """

    def __init__(self, min_val=-50.0, max_val=50.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return torch.clamp(x, self.min_val, self.max_val)


class ResidualBlock(nn.Module):
    """
    A Residual MLP Block: x + (Linear -> BN -> ReLU -> Dropout -> Linear)
    """

    def __init__(self, hidden_dim, dropout_rate):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return x + self.block(x)


class KinematicStream(nn.Module):
    """
    Deep Residual MLP for the Invariant Kinematic Backbone.
    Strictly invariant: processes only physical features, no entity embeddings.
    """

    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super().__init__()
        # Initial projection
        self.project = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Deep Residual Blocks
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim, dropout_rate),
            ResidualBlock(hidden_dim, dropout_rate),
        )

        # Output Logit
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.project(x)
        x = self.res_blocks(x)
        return self.head(x)


class VisualStream(nn.Module):
    """
    Shallow MLP for the Visual Correction Stream.
    Processes bounding box metrics to provide a geometric correction logit.
    """

    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class IPRVN(nn.Module):
    """
    Invariant-Physical Residual-Visual Network (IP-RVN).

    Architecture:
    1. Input Clamping (Numerical Stability)
    2. Stream Splitting (Kinematic vs Visual)
    3. Kinematic Stream (Deep ResNet) -> Logit_Kin
    4. Visual Stream (Shallow MLP) -> Logit_Vis
    5. Residual Fusion: Logit_Kin + lambda * Logit_Vis
    """

    def __init__(self, feature_names):
        super().__init__()

        # 1. Identify Feature Indices for Splitting
        self.kin_indices = []
        self.vis_indices = []

        # Visual features are identified by keywords from the config/schema
        # Added "visual" to capture visual_iou and visual_centroid_dist
        vis_keywords = set(["left", "top", "width", "height", "area", "visual"])

        for idx, name in enumerate(feature_names):
            # Check if feature name contains any visual keyword (e.g. "left_1_lag_0")
            parts = name.split("_")
            is_visual = any(p in vis_keywords for p in parts)

            if is_visual:
                self.vis_indices.append(idx)
            else:
                self.kin_indices.append(idx)

        # Register buffers so they move to device with the model
        self.register_buffer(
            "kin_idx", torch.tensor(self.kin_indices, dtype=torch.long)
        )
        self.register_buffer(
            "vis_idx", torch.tensor(self.vis_indices, dtype=torch.long)
        )

        kin_dim = len(self.kin_indices)
        vis_dim = len(self.vis_indices)

        # 2. Initialize Layers
        self.clamping = InputClampingLayer(min_val=-50.0, max_val=50.0)

        self.kin_stream = KinematicStream(
            input_dim=kin_dim,
            hidden_dim=config.HIDDEN_DIM_KIN,
            dropout_rate=config.DROPOUT_RATE,
        )

        self.vis_stream = VisualStream(
            input_dim=vis_dim,
            hidden_dim=config.HIDDEN_DIM_VIS,
            dropout_rate=config.DROPOUT_RATE,
        )

        self.vis_weight = config.VISUAL_LOSS_WEIGHT

    def forward(self, x):
        # 1. Clamp Inputs for Stability
        x = self.clamping(x)

        # 2. Split Streams
        x_kin = torch.index_select(x, 1, self.kin_idx)
        x_vis = torch.index_select(x, 1, self.vis_idx)

        # 3. Forward Pass
        logit_kin = self.kin_stream(x_kin)
        logit_vis = self.vis_stream(x_vis)

        # 4. Residual Fusion
        return logit_kin + (self.vis_weight * logit_vis)
