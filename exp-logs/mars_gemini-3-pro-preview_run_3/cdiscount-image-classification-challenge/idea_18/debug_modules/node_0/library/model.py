import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ProjectedMultiTaskMLP(nn.Module):
    """
    Projected Multi-Task Learning (PMTL) Network.

    Architecture:
    1. Input: Concatenated features from frozen backbones (Dim: 3328).
    2. Projection: Learnable linear bottleneck to fuse sensors and compress dims (3328 -> 1024).
    3. Shared Trunk: MLP layers shared across all hierarchy levels.
    4. Parallel Heads: Independent classifiers for Level 1, Level 2, and Level 3 categories.
    """

    def __init__(self):
        super(ProjectedMultiTaskMLP, self).__init__()

        # Hyperparameters from Config
        input_dim = Config.INPUT_DIM  # 3328
        proj_dim = Config.PROJECTION_DIM  # 1024
        dropout_rate = Config.DROPOUT_RATE

        num_classes_l1 = Config.NUM_CLASSES_L1
        num_classes_l2 = Config.NUM_CLASSES_L2
        num_classes_l3 = Config.NUM_CLASSES_L3

        # 1. Projection Layer (Bottleneck & Fusion)
        # Reduces dimensionality and forces fusion of ResNet/EffNet features
        self.projection = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # 2. Shared Trunk
        # Deepens the representation before branching
        self.shared_trunk = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # 3. Independent Classification Heads
        # Flat Multi-Task structure: Gradient flows directly from each head to the trunk

        # Head 1: Coarse Categories (49 classes)
        self.head_l1 = nn.Linear(proj_dim, num_classes_l1)

        # Head 2: Sub-Categories (483 classes)
        self.head_l2 = nn.Linear(proj_dim, num_classes_l2)

        # Head 3: Fine-Grained Target (5270 classes)
        self.head_l3 = nn.Linear(proj_dim, num_classes_l3)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Kaiming initialization for Linear layers and Constant init for BatchNorm.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input features of shape (Batch_Size, 3328)

        Returns:
            tuple: (logits_l1, logits_l2, logits_l3)
        """
        # Project to latent space
        x_proj = self.projection(x)

        # Pass through shared trunk
        x_shared = self.shared_trunk(x_proj)

        # Compute logits for each hierarchy level independently
        logits_l1 = self.head_l1(x_shared)
        logits_l2 = self.head_l2(x_shared)
        logits_l3 = self.head_l3(x_shared)

        return logits_l1, logits_l2, logits_l3
