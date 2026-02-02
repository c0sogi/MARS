import torch
import torch.nn as nn
from library.config import Config


class InputClampingLayer(nn.Module):
    """
    A fixed layer that strictly clamps inputs to a pre-defined range.
    Used to prevent outliers in derivative features from destabilizing gradients.
    """

    def __init__(self, min_val, max_val):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return torch.clamp(x, self.min_val, self.max_val)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block adapted for 1D feature vectors.
    Dynamically recalibrates feature importance.
    """

    def __init__(self, channel, reduction=Config.SE_REDUCTION):
        super().__init__()
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (Batch, Channel)
        # Squeeze path
        y = self.fc1(x)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        # Excitation (Scale)
        return x * y


class ResSEBlock(nn.Module):
    """
    Residual Block with Squeeze-and-Excitation.
    Structure: Linear -> BN -> ReLU -> Dropout -> Linear -> SE -> Add
    """

    def __init__(self, hidden_dim, dropout=Config.DROPOUT):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.se = SEBlock(hidden_dim)

    def forward(self, x):
        identity = x

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.se(out)

        out += identity
        return out


class SSERVN(nn.Module):
    """
    Stabilized Squeeze-and-Excitation Residual-Visual Network.
    Fuses a context-aware kinematic backbone with a lightweight visual correction stream.
    """

    def __init__(self, kin_input_dim, vis_input_dim, cat_cardinalities):
        """
        Args:
            kin_input_dim (int): Dimensionality of the continuous kinematic features.
            vis_input_dim (int): Dimensionality of the continuous visual features.
            cat_cardinalities (list[int]): List of vocabulary sizes for the 4 categorical inputs
                                           [team_1, pos_1, team_2, pos_2].
        """
        super().__init__()

        # ==========================
        # Kinematic Stream
        # ==========================

        # 1. Input Clamping
        self.clamping = InputClampingLayer(Config.CLAMP_MIN, Config.CLAMP_MAX)

        # 2. Entity Embeddings
        # We create an embedding layer for each categorical feature
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=c, embedding_dim=Config.EMBEDDING_DIM)
                for c in cat_cardinalities
            ]
        )

        # Calculate total dimension after concatenation
        # Kinematic features + (4 * Embedding Dim)
        total_kin_dim = kin_input_dim + (len(cat_cardinalities) * Config.EMBEDDING_DIM)

        # 3. Projection to Hidden Dimension
        self.kin_proj = nn.Linear(total_kin_dim, Config.HIDDEN_DIM)
        self.kin_bn_proj = nn.BatchNorm1d(Config.HIDDEN_DIM)
        self.kin_relu_proj = nn.ReLU()

        # 4. Residual Backbone (Stack of ResSEBlocks)
        # Using 2 blocks to form the backbone
        self.kin_backbone = nn.Sequential(
            ResSEBlock(Config.HIDDEN_DIM), ResSEBlock(Config.HIDDEN_DIM)
        )

        # 5. Kinematic Head
        self.kin_head = nn.Linear(Config.HIDDEN_DIM, 1)

        # ==========================
        # Visual Stream
        # ==========================

        # Shallow MLP for visual correction
        self.vis_mlp = nn.Sequential(
            nn.Linear(vis_input_dim, Config.VISUAL_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.VISUAL_HIDDEN_DIM, 1),
        )

    def forward(self, x_kin, x_vis, x_cat):
        """
        Args:
            x_kin (Tensor): Continuous kinematic features (Batch, kin_dim).
            x_vis (Tensor): Continuous visual features (Batch, vis_dim).
            x_cat (Tensor): Categorical indices (Batch, 4).

        Returns:
            Tensor: Final logits (Batch, 1).
        """

        # --- Process Kinematic Stream ---

        # 1. Clamp continuous features
        x_k = self.clamping(x_kin)

        # 2. Lookup and concatenate embeddings
        embs = []
        for i, emb_layer in enumerate(self.embeddings):
            # Select the i-th column of categorical inputs
            embs.append(emb_layer(x_cat[:, i]))

        x_emb = torch.cat(embs, dim=1)  # (Batch, 4 * Emb_Dim)

        # 3. Combine and Project
        x_combined = torch.cat([x_k, x_emb], dim=1)

        x_feat = self.kin_proj(x_combined)
        x_feat = self.kin_bn_proj(x_feat)
        x_feat = self.kin_relu_proj(x_feat)

        # 4. Pass through Backbone
        x_feat = self.kin_backbone(x_feat)

        # 5. Generate Kinematic Logit
        kin_logit = self.kin_head(x_feat)

        # --- Process Visual Stream ---

        vis_logit = self.vis_mlp(x_vis)

        # --- Fusion ---

        # Additive fusion of logits
        return kin_logit + vis_logit
