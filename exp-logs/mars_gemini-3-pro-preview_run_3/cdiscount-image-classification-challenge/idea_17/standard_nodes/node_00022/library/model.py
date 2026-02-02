import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ProjectionBlock(nn.Module):
    """
    Compresses the high-dimensional concatenated backbone features into a
    compact latent product embedding.
    """

    def __init__(self, input_dim, output_dim, dropout_rate=0.5):
        super(ProjectionBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class CascadeBlock(nn.Module):
    """
    A hierarchical processing block that outputs both a hidden state (for the next level)
    and classification logits (for the current level).
    """

    def __init__(self, input_dim, hidden_dim, num_classes, dropout_rate=0.5):
        super(CascadeBlock, self).__init__()
        # The hidden state generator: learns semantic concepts for this level
        self.hidden_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
        )
        # The classifier head: maps semantic concepts to class probabilities
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Generate dense hidden representation
        h = self.hidden_layer(x)
        # Generate logits
        logits = self.classifier(h)
        return h, logits


class PDFCNet(nn.Module):
    """
    Projected Deep Feature Cascading Network.

    Architecture:
    1. Projects raw features (3328 dim) -> Latent Embedding (1024 dim).
    2. Level 1 Cascade: Predicts L1 categories from Latent Embedding.
    3. Level 2 Cascade: Predicts L2 categories from [Latent Embedding + L1 Hidden State].
    4. Level 3 Cascade: Predicts L3 categories from [Latent Embedding + L2 Hidden State].
    """

    def __init__(self):
        super(PDFCNet, self).__init__()

        # Dimensions from Config
        input_dim = Config.INPUT_DIM  # 3328 (ResNet + EfficientNet)
        proj_dim = Config.PROJECTION_DIM  # 1024
        hidden_dim = Config.HIDDEN_DIM  # 1024

        # 1. Projection Layer
        self.projection = ProjectionBlock(input_dim, proj_dim)

        # 2. Level 1: Coarse Categories (49 classes)
        # Input: Projected Embedding
        self.block1 = CascadeBlock(
            input_dim=proj_dim, hidden_dim=hidden_dim, num_classes=Config.NUM_CLASSES_L1
        )

        # 3. Level 2: Sub-categories (483 classes)
        # Input: Concat(Projected Embedding, L1 Hidden State)
        # Input Dim: 1024 + 1024 = 2048
        self.block2 = CascadeBlock(
            input_dim=proj_dim + hidden_dim,
            hidden_dim=hidden_dim,
            num_classes=Config.NUM_CLASSES_L2,
        )

        # 4. Level 3: Fine-grained Categories (5270 classes)
        # Input: Concat(Projected Embedding, L2 Hidden State)
        # Input Dim: 1024 + 1024 = 2048
        self.block3 = CascadeBlock(
            input_dim=proj_dim + hidden_dim,
            hidden_dim=hidden_dim,
            num_classes=Config.NUM_CLASSES_L3,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, 3328)
        Returns:
            logits1, logits2, logits3
        """
        # 1. Projection
        # x: (B, 3328) -> emb: (B, 1024)
        emb = self.projection(x)

        # 2. Level 1 Prediction
        # h1: (B, 1024), logits1: (B, 49)
        h1, logits1 = self.block1(emb)

        # 3. Level 2 Prediction
        # Condition on embedding and L1 concepts
        in2 = torch.cat([emb, h1], dim=1)  # (B, 2048)
        # h2: (B, 1024), logits2: (B, 483)
        h2, logits2 = self.block2(in2)

        # 4. Level 3 Prediction
        # Condition on embedding and L2 concepts
        in3 = torch.cat([emb, h2], dim=1)  # (B, 2048)
        # h3: (B, 1024), logits3: (B, 5270)
        h3, logits3 = self.block3(in3)

        return logits1, logits2, logits3
