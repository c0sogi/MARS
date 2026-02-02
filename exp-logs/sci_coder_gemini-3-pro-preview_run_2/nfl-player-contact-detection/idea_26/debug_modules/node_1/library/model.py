import torch
import torch.nn as nn
from library.config import Config


class InputClampingLayer(nn.Module):
    """
    A fixed, non-trainable layer that strictly clamps inputs to a pre-defined physical range.
    Used to prevent outliers in derivative features from destabilizing gradients.
    """

    def __init__(self, min_val=Config.CLAMP_MIN, max_val=Config.CLAMP_MAX):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return torch.clamp(x, min=self.min_val, max=self.max_val)


class ResidualBlock(nn.Module):
    """
    A standard residual block for the kinematic backbone.
    Structure: Input -> [Linear -> BN -> ReLU -> Dropout -> Linear] + Input -> ReLU
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
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return self.activation(out)


class KinematicStream(nn.Module):
    """
    Context-Aware Kinematic Backbone.
    Fuses continuous kinematic data with categorical entity embeddings.
    """

    def __init__(self, input_dim, hidden_dim, dropout_rate, embedding_dims):
        super().__init__()

        # 1. Input Clamping
        self.clamping = InputClampingLayer()

        # 2. Entity Embeddings
        # Position Embedding
        self.pos_embedding = nn.Embedding(
            num_embeddings=embedding_dims["position"][0],
            embedding_dim=embedding_dims["position"][1],
        )
        # Team Embedding
        self.team_embedding = nn.Embedding(
            num_embeddings=embedding_dims["team"][0],
            embedding_dim=embedding_dims["team"][1],
        )

        # Calculate total input dimension after concatenation
        # Input: continuous_dim + (pos_emb_dim + team_emb_dim) * 2 (for Player 1 and Player 2)
        total_emb_dim = (embedding_dims["position"][1] + embedding_dims["team"][1]) * 2
        self.total_input_dim = input_dim + total_emb_dim

        # 3. Backbone
        # Projection Layer
        self.project = nn.Sequential(
            nn.Linear(self.total_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Deep Residual Blocks (3 blocks)
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim, dropout_rate),
            ResidualBlock(hidden_dim, dropout_rate),
            ResidualBlock(hidden_dim, dropout_rate),
        )

        # 4. Output Head
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x_kin, x_cat):
        """
        Args:
            x_kin: Continuous kinematic features (Batch, Dim)
            x_cat: Categorical indices (Batch, 4) -> [pos1, team1, pos2, team2]
        """
        # Clamp continuous inputs
        x_kin = self.clamping(x_kin)

        # Lookup Embeddings
        # x_cat is expected to be LongTensor
        p1_pos = self.pos_embedding(x_cat[:, 0])
        p1_team = self.team_embedding(x_cat[:, 1])
        p2_pos = self.pos_embedding(x_cat[:, 2])
        p2_team = self.team_embedding(x_cat[:, 3])

        # Concatenate: [Kinematics, P1_Pos, P1_Team, P2_Pos, P2_Team]
        x = torch.cat([x_kin, p1_pos, p1_team, p2_pos, p2_team], dim=1)

        # Forward pass through backbone
        x = self.project(x)
        x = self.res_blocks(x)

        # Output Logit
        return self.head(x)


class VisualStream(nn.Module):
    """
    Shallow Visual Correction Network.
    """

    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_vis):
        return self.net(x_vis)


class SEARVN(nn.Module):
    """
    Stabilized Entity-Aware Residual-Visual Network.
    Fuses Kinematic and Visual streams via residual addition.
    """

    def __init__(self, kin_input_dim, vis_input_dim):
        super().__init__()

        # Initialize Kinematic Stream
        self.kinematic_stream = KinematicStream(
            input_dim=kin_input_dim,
            hidden_dim=Config.KINEMATIC_HIDDEN_DIM,
            dropout_rate=Config.DROPOUT_RATE,
            embedding_dims=Config.EMBEDDING_DIMS,
        )

        # Initialize Visual Stream
        self.visual_stream = VisualStream(
            input_dim=vis_input_dim,
            hidden_dim=Config.VISUAL_HIDDEN_DIM,
            dropout_rate=Config.DROPOUT_RATE,
        )

        # Learnable weight for visual correction
        # Initialized to 0.1 to start with kinematic dominance
        self.visual_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x_kin, x_vis, x_cat):
        """
        Args:
            x_kin: Kinematic features
            x_vis: Visual features
            x_cat: Categorical features
        Returns:
            logit: Fused output logit
        """
        l_kin = self.kinematic_stream(x_kin, x_cat)
        l_vis = self.visual_stream(x_vis)

        # Residual Fusion: L_kin + lambda * L_vis
        return l_kin + self.visual_weight * l_vis
