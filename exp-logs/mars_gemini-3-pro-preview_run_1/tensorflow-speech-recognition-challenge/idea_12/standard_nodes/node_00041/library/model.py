import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class AttentivePooling(nn.Module):
    """
    Standard Attentive Pooling layer.
    """

    def __init__(self, feat_dim, hidden_dim=128):
        super().__init__()
        self.w = nn.Conv1d(feat_dim, hidden_dim, kernel_size=1)
        self.v = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Feature map. Shape (B, C, T).
        Returns:
            torch.Tensor: Global feature representation. Shape (B, C).
        """
        # Attention Score
        attn = torch.tanh(self.w(x))
        scores = self.v(attn)
        alpha = F.softmax(scores, dim=2)

        # Weighted Pooling
        context = torch.sum(x * alpha, dim=2)
        return context


class DilatedEfficientNet(nn.Module):
    """
    Dilated EfficientNet-B2 with Attentive Pooling.
    """

    def __init__(self, num_classes):
        super().__init__()

        # 1. Backbone: EfficientNet-B2
        # - output_stride=16: Preserves temporal resolution via dilation
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            in_chans=Config.IN_CHANNELS,
            output_stride=16,
            num_classes=0,
            global_pool="",
        )

        self.feat_dim = self.backbone.num_features

        # 2. Attentive Pooling
        self.att_pool = AttentivePooling(self.feat_dim)

        # 3. Classification Head
        self.classifier = nn.Linear(self.feat_dim, num_classes)
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)

    def forward(self, x):
        # x: (B, 1, F, T)
        features = self.backbone(x)  # (B, C, F', T')
        features = torch.mean(features, dim=2)  # (B, C, T')
        embedding = self.att_pool(features)
        embedding = self.dropout(embedding)
        logits = self.classifier(embedding)
        return logits
