import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        # Calculate BCE with logits
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return loss.mean()


class GatedResidualBlock(nn.Module):
    """
    Gated Residual Block using Gated Linear Units (GLU).
    Output = LayerNorm(Dropout((W1 x) * sigmoid(W2 x)) + x)
    """

    def __init__(self, dim, dropout=0.1):
        super(GatedResidualBlock, self).__init__()
        self.linear_feat = nn.Linear(dim, dim)
        self.linear_gate = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # GLU Mechanism
        feat = self.linear_feat(x)
        gate = torch.sigmoid(self.linear_gate(x))
        out = feat * gate

        # Dropout
        out = self.dropout(out)

        # Residual Connection + Normalization
        return self.norm(out + x)


class KinematicStream(nn.Module):
    """
    Kinematic Stream: Gated Backbone for temporal/kinematic features.
    Projects input -> Stack of Gated Blocks -> Logit.
    """

    def __init__(self, input_dim, hidden_dim, dropout, num_blocks=2):
        super(KinematicStream, self).__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [GatedResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.projection(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class VisualStream(nn.Module):
    """
    Visual Stream: Shallow MLP for visual correction.
    """

    def __init__(self, input_dim, hidden_dim):
        super(VisualStream, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class GRVCNet(nn.Module):
    """
    Gated Residual-Visual Corrected Network.
    Fuses Kinematic and Visual streams via residual correction.
    Logit = KinematicLogit + lambda * VisualLogit
    """

    def __init__(self, kin_input_dim, vis_input_dim, config):
        super(GRVCNet, self).__init__()

        # Kinematic Stream
        self.kin_stream = KinematicStream(
            kin_input_dim,
            config.HIDDEN_DIM,
            config.DROPOUT,
            num_blocks=2,  # Standard depth for this architecture
        )

        # Visual Stream
        self.vis_stream = VisualStream(vis_input_dim, config.VISUAL_HIDDEN_DIM)

        # Learnable Fusion Parameter (initialized to 0.1)
        self.fusion_lambda = nn.Parameter(torch.tensor(0.1))

    def forward(self, x_kin, x_vis):
        kin_logit = self.kin_stream(x_kin)
        vis_logit = self.vis_stream(x_vis)

        # Ensure broadcasting works if shapes differ slightly (e.g. [B, 1] vs [B])
        if kin_logit.shape != vis_logit.shape:
            vis_logit = vis_logit.view_as(kin_logit)

        # Residual Fusion
        final_logit = kin_logit + self.fusion_lambda * vis_logit

        # Return squeezed logit for compatibility with loss functions expecting [B]
        return final_logit.squeeze()
