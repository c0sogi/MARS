import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ClampingLayer(nn.Module):
    """
    A fixed, non-trainable layer that strictly clamps inputs to a safe numerical range.
    Since inputs are standardized (mean=0, std=1), we clamp to +/- 10.0 sigma
    to prevent extreme outliers from destabilizing the network gradients.
    """

    def __init__(self, min_val=-10.0, max_val=10.0):
        super(ClampingLayer, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return torch.clamp(x, self.min_val, self.max_val)


class TimeDistributedEncoder(nn.Module):
    """
    Applies a shared encoder layer to each time step independently.
    Input: (Batch, Window, Features)
    Output: (Batch, Window, Hidden)
    """

    def __init__(self, input_dim, hidden_dim, dropout):
        super(TimeDistributedEncoder, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (Batch, Window, Input_Dim)
        b, w, f = x.size()

        # Collapse batch and window for linear layer
        x_flat = x.view(b * w, f)

        # Encode
        out = self.linear(x_flat)

        # BatchNorm (1D expects Batch x Channels, but here we have (B*W) x Hidden)
        # We treat (B*W) as the batch dimension
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)

        # Reshape back to (Batch, Window, Hidden)
        out = out.view(b, w, -1)
        return out


class TDSRVNet(nn.Module):
    def __init__(self, num_positions=35, num_teams=5):
        super(TDSRVNet, self).__init__()

        # --- Configuration & Dimensions ---
        self.window_size = Config.WINDOW_SIZE

        # Calculate Kinematic Features per Step
        # Structure per lag in Dataset: [P1_Kin, P2_Kin, Derived]
        n_kin_raw = len(Config.KINEMATIC_FEATURES)
        n_derived = 4  # rel_dist, rel_speed, rel_accel, closing_speed
        self.kin_feats_per_step = (n_kin_raw * 2) + n_derived

        # Calculate Visual Features per Step
        # Structure per lag in Dataset: [P1_Vis, P2_Vis]
        n_vis_raw = len(Config.VISUAL_FEATURES)
        self.vis_feats_per_step = n_vis_raw * 2

        # Total flattened input dimensions
        self.kin_input_dim = self.kin_feats_per_step * self.window_size
        self.vis_input_dim = self.vis_feats_per_step * self.window_size

        # --- 1. Stability Layer ---
        self.clamping = ClampingLayer()

        # --- 2. Kinematic Stream (Time-Distributed) ---
        self.kin_td_encoder = TimeDistributedEncoder(
            input_dim=self.kin_feats_per_step,
            hidden_dim=Config.HIDDEN_DIM,
            dropout=Config.DROPOUT_RATE,
        )

        # Entity Embeddings
        # 4 categorical inputs: pos1, team1, pos2, team2
        self.emb_dim = Config.EMBEDDING_DIM
        self.pos_embedding = nn.Embedding(num_positions, self.emb_dim)
        self.team_embedding = nn.Embedding(num_teams, self.emb_dim)

        # Aggregator Input Dim: (Window * Hidden) + (4 * Emb_Dim)
        agg_input_dim = (self.window_size * Config.HIDDEN_DIM) + (4 * self.emb_dim)

        # Deep Residual MLP for Temporal Aggregation
        self.kin_aggregator = nn.Sequential(
            nn.Linear(agg_input_dim, Config.HIDDEN_DIM),
            nn.BatchNorm1d(Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM // 2),
            nn.BatchNorm1d(Config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM // 2, 1),
        )

        # --- 3. Visual Stream (Shallow Correction) ---
        self.vis_mlp = nn.Sequential(
            nn.Linear(self.vis_input_dim, Config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(Config.HIDDEN_DIM // 2, 1),
        )

        # --- 4. Fusion Parameter ---
        self.vis_lambda = Config.VISUAL_FUSION_LAMBDA

    def forward(self, x_kin, x_vis, x_cat):
        """
        Args:
            x_kin: (Batch, Window * Kin_Feats_Per_Step)
            x_vis: (Batch, Window * Vis_Feats_Per_Step)
            x_cat: (Batch, 4) -> [pos1, team1, pos2, team2]
        """
        batch_size = x_kin.size(0)

        # 1. Numerical Stability
        x_kin = self.clamping(x_kin)
        x_vis = self.clamping(x_vis)

        # 2. Kinematic Stream
        # Reshape for Time-Distributed Encoding
        x_kin_reshaped = x_kin.view(
            batch_size, self.window_size, self.kin_feats_per_step
        )

        # Encode frames
        kin_encoded = self.kin_td_encoder(x_kin_reshaped)  # (B, W, Hidden)

        # Flatten temporal dimension
        kin_flat = kin_encoded.view(batch_size, -1)  # (B, W*Hidden)

        # Entity Embeddings
        pos1 = self.pos_embedding(x_cat[:, 0])
        team1 = self.team_embedding(x_cat[:, 1])
        pos2 = self.pos_embedding(x_cat[:, 2])
        team2 = self.team_embedding(x_cat[:, 3])

        embeddings = torch.cat([pos1, team1, pos2, team2], dim=1)

        # Fuse and Aggregate
        kin_fused = torch.cat([kin_flat, embeddings], dim=1)
        logit_kin = self.kin_aggregator(kin_fused)

        # 3. Visual Stream
        logit_vis = self.vis_mlp(x_vis)

        # 4. Residual Fusion
        # Logit_final = L_kin + lambda * L_vis
        output = logit_kin + (self.vis_lambda * logit_vis)

        return output.squeeze(-1)
