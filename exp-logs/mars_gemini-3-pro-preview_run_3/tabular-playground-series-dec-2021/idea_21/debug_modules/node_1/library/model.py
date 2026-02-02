import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import INPUT_DIM, HIDDEN_DIM, NUM_CLASSES


class VectorCrossLayer(nn.Module):
    """
    Vector-based (Rank-1) Cross Layer using Dot-Product Mixing.
    Formula: x_{l+1} = x_0 * (x_l . w) + b + x_l
    """

    def __init__(self, input_dim):
        super(VectorCrossLayer, self).__init__()
        # Weight vector w: (D, 1)
        self.w = nn.Parameter(torch.empty(input_dim, 1))
        # Bias vector b: (D,)
        self.b = nn.Parameter(torch.empty(input_dim))
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.w)
        nn.init.constant_(self.b, 0.0)

    def forward(self, x0, xl):
        """
        Args:
            x0: Initial input features (B, D)
            xl: Output from previous layer (B, D)
        Returns:
            x_next: (B, D)
        """
        # Compute scalar mixing coefficient per sample: (B, D) @ (D, 1) -> (B, 1)
        mixing = torch.matmul(xl, self.w)

        # Apply mixing to x0, add bias and residual
        # x0 * mixing broadcasts (B, D) * (B, 1) -> (B, D)
        x_next = x0 * mixing + self.b + xl
        return x_next


class ResNetBlock(nn.Module):
    """
    Standard Deep ResNet Block for Tabular Data.
    Structure: Input -> [Linear->BN->ReLU->Dropout->Linear->BN] + Input -> ReLU
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super(ResNetBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)

        out += residual
        out = F.relu(out)
        return out


class DeepVectorDCNResNet(nn.Module):
    """
    Deep-Scaled Parallel Vector-DCN-ResNet Architecture.
    Combines a Vector-based Deep & Cross Network with a Deep ResNet backbone.
    """

    def __init__(
        self,
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        num_cross_layers=3,
        num_res_blocks=4,
        dropout=0.1,
    ):
        super(DeepVectorDCNResNet, self).__init__()

        # Branch 1: Vector-Based Deep & Cross Network
        # Maintains original input dimension D throughout
        self.cross_layers = nn.ModuleList(
            [VectorCrossLayer(input_dim) for _ in range(num_cross_layers)]
        )

        # Branch 2: Deep ResNet Backbone
        # Projects D -> Hidden -> ResBlocks
        self.res_embedding = nn.Linear(input_dim, hidden_dim)
        self.res_bn_input = nn.BatchNorm1d(hidden_dim)
        self.res_blocks = nn.ModuleList(
            [ResNetBlock(hidden_dim, dropout=dropout) for _ in range(num_res_blocks)]
        )

        # Combination Head
        concat_dim = input_dim + hidden_dim
        self.final_bn = nn.BatchNorm1d(concat_dim)
        self.classifier = nn.Linear(concat_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: (B, D)

        # --- Branch 1: DCN ---
        x_dcn = x
        for layer in self.cross_layers:
            x_dcn = layer(x, x_dcn)
        # x_dcn shape: (B, D)

        # --- Branch 2: ResNet ---
        x_res = self.res_embedding(x)
        x_res = self.res_bn_input(x_res)
        x_res = F.relu(x_res)

        for block in self.res_blocks:
            x_res = block(x_res)
        # x_res shape: (B, Hidden)

        # --- Combination ---
        combined = torch.cat([x_dcn, x_res], dim=1)
        combined = self.final_bn(combined)

        logits = self.classifier(combined)
        return logits
