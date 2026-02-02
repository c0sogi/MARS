import torch
import torch.nn as nn
from library.config import (
    EMBEDDING_DIM,
    HIDDEN_DIM_KIN,
    NUM_LAYERS_KIN,
    HIDDEN_DIM_VIS,
    NUM_LAYERS_VIS,
    DROPOUT_RATE,
    CLAMP_MIN,
    CLAMP_MAX,
    KINEMATIC_COLS,
    VISUAL_COLS,
    WINDOW_SIZE,
    CAT_COLS,
)


class ClampingLayer(nn.Module):
    """
    A fixed, non-trainable layer that strictly clamps inputs to a pre-defined physical range.
    """

    def __init__(self, min_val=CLAMP_MIN, max_val=CLAMP_MAX):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return torch.clamp(x, min=self.min_val, max=self.max_val)


class ResidualBlock(nn.Module):
    """
    Deep Residual MLP Block: Linear -> BN -> ReLU -> Dropout -> Linear -> Add
    """

    def __init__(self, hidden_dim, dropout_rate):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out += residual
        return out


class KinematicStream(nn.Module):
    """
    Context-Aware Backbone processing windowed tracking data and entity embeddings.
    """

    def __init__(self, vocab_sizes):
        super().__init__()

        # 1. Input Dimension Calculation
        # Continuous: (Features * Window_Len * 2_Players)
        self.num_continuous = len(KINEMATIC_COLS) * (2 * WINDOW_SIZE + 1) * 2

        # Embeddings
        self.embeddings = nn.ModuleDict()
        total_embedding_dim = 0

        # We expect vocab_sizes to be a dict: {'position': N, 'team': M}
        # Input X_cat is [P1_Pos, P2_Pos, P1_Team, P2_Team]
        # We need embeddings for Position and Team.
        for col in CAT_COLS:
            num_embeddings = vocab_sizes[col]
            # Use same embedding layer for P1 and P2 of the same category type
            self.embeddings[col] = nn.Embedding(num_embeddings, EMBEDDING_DIM)
            # We have P1 and P2 for each category, so we add dim * 2
            total_embedding_dim += EMBEDDING_DIM * 2

        input_dim = self.num_continuous + total_embedding_dim

        # 2. Layers
        self.clamping = ClampingLayer()
        self.input_proj = nn.Linear(input_dim, HIDDEN_DIM_KIN)

        self.blocks = nn.ModuleList(
            [ResidualBlock(HIDDEN_DIM_KIN, DROPOUT_RATE) for _ in range(NUM_LAYERS_KIN)]
        )

        # Final projection to scalar logit
        self.output_head = nn.Linear(HIDDEN_DIM_KIN, 1)

    def forward(self, x_kin, x_cat):
        # x_kin: [Batch, num_continuous]
        # x_cat: [Batch, 4] -> P1_Pos, P2_Pos, P1_Team, P2_Team

        # 1. Clamp Continuous Features
        x_kin = self.clamping(x_kin)

        # 2. Process Embeddings
        # CAT_COLS order is ['position', 'team']
        # x_cat indices: 0=P1_Pos, 1=P2_Pos, 2=P1_Team, 3=P2_Team

        emb_list = []

        # Position Embeddings
        p1_pos_emb = self.embeddings["position"](x_cat[:, 0])
        p2_pos_emb = self.embeddings["position"](x_cat[:, 1])
        emb_list.extend([p1_pos_emb, p2_pos_emb])

        # Team Embeddings
        p1_team_emb = self.embeddings["team"](x_cat[:, 2])
        p2_team_emb = self.embeddings["team"](x_cat[:, 3])
        emb_list.extend([p1_team_emb, p2_team_emb])

        # Concatenate all features
        x = torch.cat([x_kin] + emb_list, dim=1)

        # 3. Backbone
        x = self.input_proj(x)
        # Usually a non-linearity after projection before residual blocks helps,
        # but adhering to "Linear -> Residual Blocks" structure implies projection enters blocks.
        # We'll apply a ReLU/Dropout after projection to align with typical MLP starts if needed,
        # but strictly following the description: "Backbone: A Deep Residual MLP".
        # We will apply ReLU here to ensure non-linearity before the first residual addition path.
        x = torch.relu(x)

        for block in self.blocks:
            x = block(x)

        # 4. Output
        logit = self.output_head(x)
        return logit


class VisualStream(nn.Module):
    """
    Shallow MLP processing max-pooled visual features.
    """

    def __init__(self):
        super().__init__()

        # Input: Visual Features * 2 Players
        input_dim = len(VISUAL_COLS) * 2

        layers = []
        # Input Projection
        layers.append(nn.Linear(input_dim, HIDDEN_DIM_VIS))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(DROPOUT_RATE))

        # Hidden Layers (Shallow)
        # NUM_LAYERS_VIS defines depth. We already added one projection.
        # If NUM_LAYERS_VIS=2, we add one more hidden or just output?
        # "Backbone: A Shallow MLP" usually implies Input -> Hidden -> ... -> Output.
        # We'll implement a simple sequence of Linear->ReLU blocks.

        for _ in range(NUM_LAYERS_VIS - 1):
            layers.append(nn.Linear(HIDDEN_DIM_VIS, HIDDEN_DIM_VIS))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(DROPOUT_RATE))

        self.mlp = nn.Sequential(*layers)
        self.output_head = nn.Linear(HIDDEN_DIM_VIS, 1)

    def forward(self, x_vis):
        x = self.mlp(x_vis)
        logit = self.output_head(x)
        return logit


class SEARVN(nn.Module):
    """
    Stabilized Entity-Aware Residual-Visual Network.
    Fuses Kinematic and Visual streams via residual connection.
    """

    def __init__(self, vocab_sizes, lambda_vis=1.0):
        """
        Args:
            vocab_sizes (dict): Dictionary mapping category names ('position', 'team') to vocabulary sizes.
            lambda_vis (float): Weighting factor for the visual stream residual.
        """
        super().__init__()
        self.lambda_vis = lambda_vis

        self.kinematic_stream = KinematicStream(vocab_sizes)
        self.visual_stream = VisualStream()

    def forward(self, x_kin, x_vis, x_cat):
        """
        Args:
            x_kin: Continuous kinematic features [Batch, F_kin]
            x_vis: Visual features [Batch, F_vis]
            x_cat: Categorical indices [Batch, 4]

        Returns:
            logit: Final fused logit [Batch, 1]
        """
        logit_kin = self.kinematic_stream(x_kin, x_cat)
        logit_vis = self.visual_stream(x_vis)

        # Residual Fusion
        # Logit_final = L_kin + lambda * L_vis
        logit_final = logit_kin + self.lambda_vis * logit_vis

        return logit_final
