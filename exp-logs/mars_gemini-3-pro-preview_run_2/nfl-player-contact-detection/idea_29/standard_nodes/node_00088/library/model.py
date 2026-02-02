import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ClampingLayer(nn.Module):
    """
    A fixed, non-trainable layer that strictly clamps inputs to a pre-defined range.
    Ensures numerical stability by preventing outliers in derivative features
    from destabilizing gradients.
    """

    def __init__(self, min_val, max_val):
        super(ClampingLayer, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return torch.clamp(x, self.min_val, self.max_val)


class ResidualBlock(nn.Module):
    """
    Deep Residual MLP Block: Linear -> BN -> ReLU -> Dropout -> Linear -> Add.
    Provides capacity to learn non-linear interactions while preserving signal flow.
    """

    def __init__(self, hidden_dim, dropout_rate):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.bn2(out)
        # Residual connection
        return self.relu(out + residual)


class KinematicStream(nn.Module):
    """
    Context-Aware Backbone.
    Fuses clamped continuous kinematic features with Entity Embeddings.
    """

    def __init__(self, input_cont_dim):
        super(KinematicStream, self).__init__()

        # 1. Clamping Layer for Stability
        self.clamping = ClampingLayer(Config.CLAMP_MIN, Config.CLAMP_MAX)

        # 2. Entity Embeddings
        # We have 4 categorical inputs: pos1, team1, pos2, team2
        # Config.EMBEDDING_DIMS = {'position': (N, D), 'team': (N, D)}
        self.emb_pos = nn.Embedding(*Config.EMBEDDING_DIMS["position"])
        self.emb_team = nn.Embedding(*Config.EMBEDDING_DIMS["team"])

        # Calculate total embedding dimension
        # pos1 + team1 + pos2 + team2
        self.total_emb_dim = (
            Config.EMBEDDING_DIMS["position"][1] * 2
            + Config.EMBEDDING_DIMS["team"][1] * 2
        )

        # 3. Backbone MLP
        in_dim = input_cont_dim + self.total_emb_dim
        hidden_dims = Config.KIN_HIDDEN_DIMS

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dims[0])
        self.input_bn = nn.BatchNorm1d(hidden_dims[0])
        self.input_act = nn.ReLU()

        # Residual Blocks
        self.blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            # Ensure dimensions match for residual connection (dim -> dim)
            # If we wanted changing dims, we'd need a projection on the residual path.
            # Here we assume blocks keep dim constant or we project between them.
            # To strictly follow "Residual MLP", we usually keep dim constant or project.
            # Let's implement blocks that maintain dimension, and project between stages if needed.
            # Simplified: Project input to hidden[0], then ResBlocks of hidden[0], then project to hidden[1]...
            # Current Config: [512, 256, 128].
            # Strategy: Linear(in, 512) -> ResBlock(512) -> Linear(512, 256) -> ResBlock(256) ...

            h_in = hidden_dims[i]
            h_out = hidden_dims[i + 1]

            block = nn.Sequential(
                ResidualBlock(h_in, Config.DROPOUT),
                nn.Linear(h_in, h_out),
                nn.BatchNorm1d(h_out),
                nn.ReLU(),
            )
            self.blocks.append(block)

        # Final Output Logit
        self.head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x_cont, x_cat):
        # x_cat shape: (Batch, 4) -> [pos1, team1, pos2, team2]

        # 1. Clamp Continuous
        x_cont = self.clamping(x_cont)

        # 2. Embeddings
        # x_cat columns: 0=pos1, 1=team1, 2=pos2, 3=team2
        e_p1 = self.emb_pos(x_cat[:, 0])
        e_t1 = self.emb_team(x_cat[:, 1])
        e_p2 = self.emb_pos(x_cat[:, 2])
        e_t2 = self.emb_team(x_cat[:, 3])

        # Concatenate embeddings
        x_emb = torch.cat([e_p1, e_t1, e_p2, e_t2], dim=1)

        # 3. Fuse
        x = torch.cat([x_cont, x_emb], dim=1)

        # 4. Forward Pass
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = self.input_act(x)

        for block in self.blocks:
            x = block(x)

        logit = self.head(x)
        return logit


class VisualStream(nn.Module):
    """
    Shallow Visual Correction Stream.
    Outputs a logit correction and a reliability gate.
    """

    def __init__(self, input_dim):
        super(VisualStream, self).__init__()

        hidden_dims = Config.VIS_HIDDEN_DIMS
        layers = []

        # Input Layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(Config.DROPOUT))

        # Hidden Layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(Config.DROPOUT))

        self.backbone = nn.Sequential(*layers)

        # Heads
        self.logit_head = nn.Linear(hidden_dims[-1], 1)
        self.gate_head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        feat = self.backbone(x)

        vis_logit = self.logit_head(feat)
        vis_gate = torch.sigmoid(self.gate_head(feat))

        return vis_logit, vis_gate


class SEARVN(nn.Module):
    """
    Stabilized Entity-Aware Residual-Visual Network.
    """

    def __init__(self):
        super(SEARVN, self).__init__()

        # Calculate Input Dimensions based on Config
        num_timesteps = 2 * Config.WINDOW_SIZE + 1

        # Kinematic Continuous Dim
        # KINEMATIC_CONT_FEATURES contains the base names.
        # Data processing generates lags for Player 1 and Player 2 for each feature.
        # Plus derived features (distance, closing_speed, relative_angle) which are single per timestep?
        # Let's check data_processing.py logic:
        # It generates lags for all raw_feats for P1 and P2.
        # It generates distance, closing_speed, relative_angle (derived).
        # In get_datasets, it constructs `kin_cont_cols`.
        #   track_base (7 feats) * 2 players * num_timesteps
        #   + 3 derived feats (distance, closing_speed, relative_angle).
        #   Wait, derived features in `get_derived_features` are calculated on lag_0 only?
        #   Looking at `process_features`:
        #   "Derived Features ... df = get_derived_features(df)"
        #   `get_derived_features` computes `distance`, `closing_speed`, `relative_angle` based on `lag_0`.
        #   So these are 1D per sample (scalar), not windowed.
        #   Let's re-verify `kin_cont_cols` construction in `get_datasets`:
        #   It loops track_base over shifts for P1 and P2.
        #   Then `kin_cont_cols.extend(["distance", "closing_speed", "relative_angle"])`.
        #   So yes, 3 derived features total.

        num_base_kin = len(
            [
                "x_position",
                "y_position",
                "speed",
                "acceleration",
                "orientation",
                "direction",
                "sa",
            ]
        )
        kin_cont_dim = (num_base_kin * 2 * num_timesteps) + 3

        # Visual Dim
        # VISUAL_FEATURES * 2 players * num_timesteps
        num_base_vis = len(Config.VISUAL_FEATURES)
        vis_dim = num_base_vis * 2 * num_timesteps

        # Initialize Streams
        self.kinematic_stream = KinematicStream(kin_cont_dim)
        self.visual_stream = VisualStream(vis_dim)

    def forward(self, x_kin_cont, x_kin_cat, x_vis):
        """
        Args:
            x_kin_cont: (Batch, kin_cont_dim)
            x_kin_cat: (Batch, 4) -> [pos1, team1, pos2, team2]
            x_vis: (Batch, vis_dim)
        """
        # 1. Kinematic Path
        l_kin = self.kinematic_stream(x_kin_cont, x_kin_cat)

        # 2. Visual Path
        l_vis, g_vis = self.visual_stream(x_vis)

        # 3. Reliability-Gated Fusion
        # Logit_final = L_kin + (L_vis * G_vis)
        # Note: We do not apply sigmoid here, as Loss function expects raw logits.
        out = l_kin + (l_vis * g_vis)

        return out
