import torch
import torch.nn as nn
from library import config


class GatedResidualBlock(nn.Module):
    """
    Implements a Gated Residual Block with GLU mechanism.
    Operation: Output = (W1 * x) * sigmoid(W2 * x) + x
    Includes LayerNorm and Dropout for stability.
    """

    def __init__(self, dim, dropout=0.0):
        super(GatedResidualBlock, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.w1 = nn.Linear(dim, dim)
        self.w2 = nn.Linear(dim, dim)
        self.act = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-LayerNorm for stability in deep residual networks
        residual = x
        x_norm = self.norm(x)

        # Gated Linear Unit mechanism
        content = self.w1(x_norm)
        gate = self.act(self.w2(x_norm))

        # Element-wise multiplication
        out = content * gate

        # Residual connection
        return residual + self.dropout(out)


class GRVNet(nn.Module):
    """
    Gated Residual-Visual Network (GRV-Net).
    A Dual-Stream Network fusing a high-capacity Kinematic stream with a
    lightweight Visual stream via residual correction.
    """

    def __init__(self):
        super(GRVNet, self).__init__()

        # Hyperparameters from config
        kin_in = config.MODEL_PARAMS["input_dim_kinematic"]
        vis_in = config.MODEL_PARAMS["input_dim_visual"]
        hidden_dim = config.MODEL_PARAMS["hidden_dim"]
        num_blocks = config.MODEL_PARAMS["num_blocks"]
        dropout = config.MODEL_PARAMS["dropout"]
        vis_hidden = config.MODEL_PARAMS["visual_hidden_dim"]
        init_lambda = config.MODEL_PARAMS["fusion_lambda"]

        # --- Kinematic Stream (Gated Backbone) ---
        # Projects flattened tracking window to hidden dimension
        self.kin_proj = nn.Sequential(
            nn.Linear(kin_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Stack of Gated Residual Blocks
        self.kin_blocks = nn.Sequential(
            *[GatedResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )

        # Final Kinematic Head (outputs scalar logit)
        self.kin_head = nn.Linear(hidden_dim, 1)

        # --- Visual Stream (Correction Branch) ---
        # Shallow MLP for visual context
        self.vis_mlp = nn.Sequential(
            nn.Linear(vis_in, vis_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(vis_hidden, 1),
        )

        # --- Fusion Parameter ---
        # Learnable weight for visual correction
        self.fusion_lambda = nn.Parameter(torch.tensor(float(init_lambda)))

    def forward(self, x_kin, x_vis):
        """
        Args:
            x_kin: Tensor of shape (batch, input_dim_kinematic)
            x_vis: Tensor of shape (batch, input_dim_visual)
        Returns:
            logit_final: Tensor of shape (batch, 1)
        """
        # 1. Kinematic Stream
        k = self.kin_proj(x_kin)
        k = self.kin_blocks(k)
        logit_kin = self.kin_head(k)

        # 2. Visual Stream
        logit_vis = self.vis_mlp(x_vis)

        # 3. Residual Fusion
        # Logit_final = Logit_kinematic + lambda * Logit_visual
        # No final activation (logits only)
        logit_final = logit_kin + self.fusion_lambda * logit_vis

        return logit_final
