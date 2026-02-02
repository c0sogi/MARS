import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block adapted for 1D feature vectors (MLP).
    Dynamically recalibrates channel-wise feature responses.
    """

    def __init__(self, input_dim, reduction_ratio=Config.SE_REDUCTION_RATIO):
        super(SEBlock, self).__init__()
        reduced_dim = max(1, input_dim // reduction_ratio)

        self.fc1 = nn.Linear(input_dim, reduced_dim)
        self.fc2 = nn.Linear(reduced_dim, input_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        # Squeeze: Compress feature dimension
        out = self.relu(self.fc1(x))
        # Excitation: Expand and apply sigmoid to get weights
        out = self.sigmoid(self.fc2(out))
        # Scale: Reweight original features
        return x * out


class ResidualBlock(nn.Module):
    """
    Residual Block with Squeeze-and-Excitation.
    Structure: Linear -> BN -> ReLU -> Dropout -> Linear -> SE -> Add
    """

    def __init__(self, hidden_dim, dropout=Config.KIN_DROPOUT):
        super(ResidualBlock, self).__init__()

        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.se = SEBlock(hidden_dim)

    def forward(self, x):
        residual = x

        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = self.se(out)

        return out + residual


class SERVN(nn.Module):
    """
    Squeeze-and-Excitation Residual-Visual Network (SE-RVN).
    Dual-stream architecture fusing kinematic and visual data with reliability gating.
    """

    def __init__(self, kin_input_dim, vis_input_dim, gate_input_dim, num_pos, num_team):
        """
        Args:
            kin_input_dim (int): Dimension of flattened continuous kinematic features.
            vis_input_dim (int): Dimension of flattened visual features.
            gate_input_dim (int): Dimension of visual metadata for gating.
            num_pos (int): Vocabulary size for position embeddings.
            num_team (int): Vocabulary size for team embeddings.
        """
        super(SERVN, self).__init__()

        # --- Entity Embeddings ---
        self.pos_embedding = nn.Embedding(num_pos, Config.EMBEDDING_DIM_POS)
        self.team_embedding = nn.Embedding(num_team, Config.EMBEDDING_DIM_TEAM)

        # Total input dimension for kinematic stream
        # (Continuous features + Position Embedding + Team Embedding)
        total_kin_dim = (
            kin_input_dim + Config.EMBEDDING_DIM_POS + Config.EMBEDDING_DIM_TEAM
        )

        # --- Kinematic Stream (Deep SE-Residual Backbone) ---
        # Projection layer to hidden dimension
        self.kin_projection = nn.Sequential(
            nn.Linear(total_kin_dim, Config.KIN_HIDDEN_DIM),
            nn.BatchNorm1d(Config.KIN_HIDDEN_DIM),
            nn.ReLU(),
        )

        # Stack of Residual Blocks
        self.kin_backbone = nn.ModuleList(
            [
                ResidualBlock(Config.KIN_HIDDEN_DIM, Config.KIN_DROPOUT)
                for _ in range(Config.NUM_RES_BLOCKS)
            ]
        )

        # Kinematic Head
        self.kin_head = nn.Linear(Config.KIN_HIDDEN_DIM, 1)

        # --- Visual Stream (Shallow Correction) ---
        # Shallow MLP to prevent overfitting to noisy visual proxies
        self.vis_backbone = nn.Sequential(
            nn.Linear(vis_input_dim, Config.VIS_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.VIS_DROPOUT),
            nn.Linear(Config.VIS_HIDDEN_DIM, 1),
        )

        # --- Reliability Gating ---
        # Computes a scalar weight based on visual metadata (e.g., box area/confidence)
        self.gate_network = nn.Sequential(
            nn.Linear(gate_input_dim, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, x_kin, x_pos, x_team, x_vis, x_gate):
        """
        Args:
            x_kin: Continuous kinematic features (Batch, kin_input_dim)
            x_pos: Position indices (Batch, 1)
            x_team: Team indices (Batch, 1)
            x_vis: Visual features (Batch, vis_input_dim)
            x_gate: Visual metadata for gating (Batch, gate_input_dim)

        Returns:
            logits: (Batch, 1)
        """

        # 1. Process Embeddings
        # Squeeze to remove extra dimension if present (Batch, 1) -> (Batch)
        if x_pos.dim() > 1:
            x_pos = x_pos.squeeze(-1)
        if x_team.dim() > 1:
            x_team = x_team.squeeze(-1)

        emb_pos = self.pos_embedding(x_pos)
        emb_team = self.team_embedding(x_team)

        # 2. Kinematic Stream
        # Concatenate continuous features with embeddings
        kin_input = torch.cat([x_kin, emb_pos, emb_team], dim=1)

        # Project and pass through backbone
        k = self.kin_projection(kin_input)
        for block in self.kin_backbone:
            k = block(k)

        # Kinematic Logit
        l_kin = self.kin_head(k)

        # 3. Visual Stream
        # Visual Logit
        l_vis = self.vis_backbone(x_vis)

        # 4. Reliability Gating
        # Compute gate weight: sigma(W * metadata)
        gate_logit = self.gate_network(x_gate)
        gate_weight = torch.sigmoid(gate_logit)

        # 5. Fusion
        # L_final = L_kin + (L_vis * Gate)
        # We add the visual correction only if the gate is open (high confidence)
        l_final = l_kin + (l_vis * gate_weight)

        return l_final
