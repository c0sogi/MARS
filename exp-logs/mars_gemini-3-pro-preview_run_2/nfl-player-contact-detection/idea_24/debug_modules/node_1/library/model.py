import torch
import torch.nn as nn
import torch.nn.functional as F
from library.utils import seed_everything
from library.config import SEED

# Ensure reproducibility for model initialization
seed_everything(SEED)


class ResidualBlock(nn.Module):
    """
    A standard Residual Block with Batch Normalization and Dropout.
    Structure: Input -> Linear -> BN -> ReLU -> Dropout -> Linear -> Add(Input) -> ReLU
    """

    def __init__(self, hidden_dim, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


class KinematicStream(nn.Module):
    """
    Context-Aware Kinematic Backbone.
    Fuses continuous kinematic features with Entity Embeddings (Position, Team)
    and processes them through a Deep Residual MLP.
    """

    def __init__(
        self,
        input_dim_cont,
        vocab_sizes,
        embed_dim_pos=8,
        embed_dim_team=4,
        hidden_dims=[256, 128, 64],
        dropout_rate=0.2,
    ):
        super(KinematicStream, self).__init__()

        # Entity Embeddings
        # vocab_sizes is a dict: {'position': int, 'team': int}
        self.pos_embedding = nn.Embedding(vocab_sizes["position"], embed_dim_pos)
        self.team_embedding = nn.Embedding(vocab_sizes["team"], embed_dim_team)

        # Calculate total input dimension after concatenation
        # We have 2 players, so 2 * (pos_dim + team_dim)
        total_embed_dim = 2 * (embed_dim_pos + embed_dim_team)
        self.input_dim_total = input_dim_cont + total_embed_dim

        # Build MLP Backbone
        layers = []
        in_dim = self.input_dim_total

        # Input projection to first hidden dimension
        layers.append(nn.Linear(in_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        # Residual Blocks
        for i in range(len(hidden_dims) - 1):
            # Add residual block keeping dimension same, or project if dims change
            # For simplicity, we project between blocks and use ResidualBlock for same-dim processing
            # Here we implement a structure: Linear(h[i]->h[i+1]) -> ResBlock(h[i+1])

            # Projection to next layer size
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

            # Residual Block at current size
            layers.append(ResidualBlock(hidden_dims[i + 1], dropout_rate))

        self.backbone = nn.Sequential(*layers)

        # Final Output Head (Scalar Logit)
        self.head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x_cont, x_cat):
        """
        Args:
            x_cont: Continuous features (Batch, Kinematic_Dim)
            x_cat: Categorical features (Batch, 4) -> [pos1, team1, pos2, team2]
        """
        # 1. Embed Entities
        # x_cat columns: 0=pos1, 1=team1, 2=pos2, 3=team2
        p1_pos = self.pos_embedding(x_cat[:, 0])
        p1_team = self.team_embedding(x_cat[:, 1])
        p2_pos = self.pos_embedding(x_cat[:, 2])
        p2_team = self.team_embedding(x_cat[:, 3])

        # Concatenate embeddings
        embeddings = torch.cat([p1_pos, p1_team, p2_pos, p2_team], dim=1)

        # 2. Fuse with Continuous Kinematics
        x = torch.cat([x_cont, embeddings], dim=1)

        # 3. Backbone
        features = self.backbone(x)

        # 4. Output Logit
        logit = self.head(features)
        return logit


class VisualStream(nn.Module):
    """
    Shallow Visual Correction Stream.
    Processes geometric helmet features to provide a correction signal.
    Kept shallow to prevent overfitting to noisy visual proxies.
    """

    def __init__(self, input_dim, hidden_dim=32, dropout_rate=0.1):
        super(VisualStream, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),  # Output scalar logit
        )

    def forward(self, x):
        return self.net(x)


class EARVN(nn.Module):
    """
    Entity-Augmented Residual-Visual Network (EA-RVN).

    Architecture:
    1. Kinematic Stream: Deep context-aware physics model.
    2. Visual Stream: Shallow geometric correction model.
    3. Reliability Gating: Dynamic fusion based on visual metadata quality.

    Output:
        L_final = L_kin + (L_vis * Sigmoid(Gate(M_vis)))
    """

    def __init__(self, input_dims, hidden_dims_kin=[256, 128, 64], hidden_dim_vis=32):
        """
        Args:
            input_dims (dict): Dictionary containing dimension sizes for 'kinematic',
                               'categorical', 'visual', 'gating', and 'vocab_sizes'.
        """
        super(EARVN, self).__init__()

        # 1. Kinematic Stream
        self.kinematic_stream = KinematicStream(
            input_dim_cont=input_dims["kinematic"],
            vocab_sizes=input_dims["vocab_sizes"],
            hidden_dims=hidden_dims_kin,
        )

        # 2. Visual Stream
        self.visual_stream = VisualStream(
            input_dim=input_dims["visual"], hidden_dim=hidden_dim_vis
        )

        # 3. Reliability Gating Network
        # Simple linear projection of gating metadata to a single scalar
        self.gating_network = nn.Linear(input_dims["gating"], 1)

    def forward(self, x_kin, x_cat, x_vis, x_gate):
        """
        Args:
            x_kin: Continuous kinematic features
            x_cat: Categorical entity features
            x_vis: Visual bounding box features
            x_gate: Visual metadata/quality features
        """
        # Compute Stream Logits
        l_kin = self.kinematic_stream(x_kin, x_cat)
        l_vis = self.visual_stream(x_vis)

        # Compute Reliability Gate
        # Sigmoid ensures gate is between 0 (ignore visual) and 1 (fully use visual)
        gate_logit = self.gating_network(x_gate)
        gate_score = torch.sigmoid(gate_logit)

        # Fused Output
        # We add the gated visual correction to the kinematic baseline
        l_final = l_kin + (l_vis * gate_score)

        return l_final
