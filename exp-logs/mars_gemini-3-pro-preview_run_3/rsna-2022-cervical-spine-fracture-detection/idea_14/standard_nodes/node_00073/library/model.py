import torch
import torch.nn as nn
import timm
from library.config import Config


class RSNAModel(nn.Module):
    """
    Simplified ResNet18 MIL Network.
    Cite Lesson 70: Avoid naive multi-stage fusion.
    Cite Lesson 20: Use 1D Convolution for sequence context.
    Cite Lesson 24: Instance-Level Aggregation (Classify then Pool).
    """

    def __init__(self):
        super().__init__()

        # 1. Backbone
        # ResNet18 with num_classes=0 returns pooled features (B, 512)
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=True,
            num_classes=0,
            in_chans=Config.IN_CHANNELS,
        )
        self.feature_dim = self.backbone.num_features

        # 2. Context Module (Cite Lesson 27: Non-linear context)
        self.context = nn.Sequential(
            nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(),
        )

        # 3. Head
        self.head = nn.Linear(self.feature_dim, Config.NUM_CLASSES)

    def forward(self, x):
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)

        # Backbone -> (B*S, 512)
        x = self.backbone(x)

        # Reshape for Context: (B, 512, S)
        x = x.view(b, s, -1).permute(0, 2, 1)

        # Context Smoothing
        x = self.context(x)

        # Back to (B, S, 512)
        x = x.permute(0, 2, 1)

        # Instance Classification -> (B, S, 8)
        logits = self.head(x)

        # Global Max Pooling -> (B, 8)
        logits, _ = torch.max(logits, dim=1)

        return logits
