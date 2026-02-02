import torch
import torch.nn as nn
from library.config import Config


class CrossLayer(nn.Module):
    """
    Explicit Feature Crossing Layer for DCNv2.
    Formula: x_{l+1} = x_0 * (x_l^T . w_l) + b_l + x_l
    """

    def __init__(self, input_dim):
        super(CrossLayer, self).__init__()
        self.input_dim = input_dim

        # Weights and Biases
        self.weight = nn.Parameter(torch.Tensor(input_dim))
        self.bias = nn.Parameter(torch.Tensor(input_dim))

        # Initialization
        nn.init.xavier_uniform_(self.weight.unsqueeze(0))
        nn.init.zeros_(self.bias)

    def forward(self, x0, xl):
        """
        Args:
            x0: Original input features (Batch, Dim)
            xl: Output from previous layer (Batch, Dim)
        """
        # Compute scalar (x_l^T . w_l) per sample
        # (Batch, Dim) * (Dim) -> (Batch, Dim) -> sum dim 1 -> (Batch, 1)
        scalar = torch.sum(xl * self.weight, dim=1, keepdim=True)

        # Apply formula: x0 * scalar + bias + xl
        out = x0 * scalar + self.bias + xl
        return out


class DCNv2(nn.Module):
    """
    Deep Cross Network v2 Backbone.
    Contains parallel Cross Network and Deep Dense Network branches.
    """

    def __init__(self, input_dim, cross_layers, deep_layers, dropout):
        super(DCNv2, self).__init__()

        # --- Cross Network Branch ---
        self.cross_layers = nn.ModuleList(
            [CrossLayer(input_dim) for _ in range(cross_layers)]
        )

        # --- Deep Dense Branch ---
        deep_modules = []
        in_dim = input_dim
        for h_dim in deep_layers:
            deep_modules.append(nn.Linear(in_dim, h_dim))
            deep_modules.append(nn.BatchNorm1d(h_dim))
            deep_modules.append(nn.ReLU())
            deep_modules.append(nn.Dropout(dropout))
            in_dim = h_dim
        self.deep_network = nn.Sequential(*deep_modules)

        # --- Fusion ---
        # Concatenate output of Cross (input_dim) and Deep (last h_dim)
        final_dim = input_dim + deep_layers[-1]
        self.final_linear = nn.Linear(final_dim, 1)

    def forward(self, x):
        # Cross Branch
        x0 = x
        xl = x
        for layer in self.cross_layers:
            xl = layer(x0, xl)

        # Deep Branch
        xd = self.deep_network(x)

        # Combine: Concatenate and Project
        combined = torch.cat([xl, xd], dim=1)
        return self.final_linear(combined)


class VisualMLP(nn.Module):
    """
    Shallow MLP for Visual Correction Stream.
    """

    def __init__(self, input_dim, hidden_layers, dropout):
        super(VisualMLP, self).__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        # Output scalar logit
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class KCVRNet(nn.Module):
    """
    Kinematic Cross-Visual Residual Network.
    Fuses Physics-Aware Kinematics with Visual Correction.
    """

    def __init__(self):
        super(KCVRNet, self).__init__()

        # Calculate Input Dimensions based on Config
        # Flattened wide format: features * (2 * window + 1)
        num_frames = 2 * Config.WINDOW_SIZE + 1
        self.kin_input_dim = len(Config.KINEMATIC_FEATURES) * num_frames
        self.vis_input_dim = len(Config.VISUAL_FEATURES) * num_frames

        # Kinematic Stream (Backbone)
        self.kinematic_stream = DCNv2(
            input_dim=self.kin_input_dim,
            cross_layers=Config.DCN_NUM_CROSS_LAYERS,
            deep_layers=Config.DCN_DEEP_LAYERS,
            dropout=Config.DCN_DROPOUT,
        )

        # Visual Stream (Correction)
        self.visual_stream = VisualMLP(
            input_dim=self.vis_input_dim,
            hidden_layers=Config.VISUAL_HIDDEN_LAYERS,
            dropout=Config.VISUAL_DROPOUT,
        )

        # Residual Fusion Parameter (Learnable)
        # Initialized to a small value (0.1) to start with kinematic dominance
        self.fusion_lambda = nn.Parameter(torch.tensor(0.1))

    def forward(self, x_kin, x_vis):
        """
        Args:
            x_kin: Kinematic features (Batch, Kin_Dim)
            x_vis: Visual features (Batch, Vis_Dim)
        Returns:
            logit_final: Combined logits (Batch, 1)
        """
        logit_kin = self.kinematic_stream(x_kin)
        logit_vis = self.visual_stream(x_vis)

        # Residual Fusion: Kinematic + lambda * Visual
        logit_final = logit_kin + self.fusion_lambda * logit_vis

        return logit_final
