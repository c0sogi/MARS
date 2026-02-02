import torch
import torch.nn as nn
from library.config import Config


class ProjectionBlock(nn.Module):
    """
    A block to project backbone features to a common latent space.
    Structure: Linear -> BatchNorm -> ReLU
    """

    def __init__(self, in_features, out_features):
        super(ProjectionBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features, bias=False),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DualStreamProjectedNetwork(nn.Module):
    """
    Dual-Stream Projected Multi-Task Network.

    Architecture:
    1. Splits input into ResNet and EfficientNet feature vectors.
    2. Projects each vector to a shared dimension using independent ProjectionBlocks.
    3. Fuses (concatenates) the projected vectors.
    4. Passes fused vector through a shared MLP trunk.
    5. Branches into three independent heads for hierarchical classification (L1, L2, L3).
    """

    def __init__(self):
        super(DualStreamProjectedNetwork, self).__init__()

        # Dimensions from Config
        resnet_dim = Config.RESNET_DIM
        effnet_dim = Config.EFFNET_DIM
        proj_dim = Config.PROJECTION_DIM
        dropout_rate = Config.DROPOUT_RATE

        # Stream 1: ResNet Projection
        self.proj_resnet = ProjectionBlock(resnet_dim, proj_dim)

        # Stream 2: EfficientNet Projection
        self.proj_effnet = ProjectionBlock(effnet_dim, proj_dim)

        # Fusion Dimension (Concatenation of two projected vectors)
        fusion_dim = proj_dim * 2

        # Shared MLP Trunk
        # Mixes features after fusion before branching
        self.shared_trunk = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim, bias=False),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
        )

        # Multi-Task Heads
        # Head 1: Level 1 Categories (Coarse)
        self.head_l1 = nn.Linear(fusion_dim, Config.NUM_CLASSES_L1)

        # Head 2: Level 2 Categories (Sub-category)
        self.head_l2 = nn.Linear(fusion_dim, Config.NUM_CLASSES_L2)

        # Head 3: Level 3 Categories (Fine-grained Target)
        self.head_l3 = nn.Linear(fusion_dim, Config.NUM_CLASSES_L3)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Concatenated feature vector of shape (Batch, RESNET_DIM + EFFNET_DIM).
                              ResNet features are first, followed by EfficientNet features.
        Returns:
            tuple: (logits_l1, logits_l2, logits_l3)
        """
        # 1. Split Input
        # ResNet features are at indices [0, RESNET_DIM)
        x_resnet = x[:, : Config.RESNET_DIM]
        # EfficientNet features are at indices [RESNET_DIM, END)
        x_effnet = x[:, Config.RESNET_DIM :]

        # 2. Independent Projections
        p1 = self.proj_resnet(x_resnet)
        p2 = self.proj_effnet(x_effnet)

        # 3. Fusion
        fused = torch.cat([p1, p2], dim=1)

        # 4. Shared Trunk
        features = self.shared_trunk(fused)

        # 5. Multi-Task Heads
        out_l1 = self.head_l1(features)
        out_l2 = self.head_l2(features)
        out_l3 = self.head_l3(features)

        return out_l1, out_l2, out_l3
