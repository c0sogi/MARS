import torch
import torch.nn as nn
import timm
from library.config import Config


class HybridEfficientNet(nn.Module):
    """
    Hybrid EfficientNet with MLP Fusion Head.

    Combines visual features from EfficientNet-B0 with metadata embeddings using
    concatenation and a non-linear MLP classifier.
    Replaces Context Gating with additive fusion (Cite Lesson 00013).
    """

    def __init__(self, meta_dim, model_name=None, pretrained=True):
        """
        Args:
            meta_dim (int): Dimension of the input metadata vector.
            model_name (str): Backbone model name.
            pretrained (bool): Whether to load ImageNet weights.
        """
        super(HybridEfficientNet, self).__init__()

        self.model_name = model_name if model_name else Config.MODEL_NAME

        # 1. Visual Backbone
        self.backbone = timm.create_model(
            self.model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        self.visual_dim = self.backbone.num_features

        # 2. Metadata Embedding
        self.meta_hidden_dim = Config.META_HIDDEN_DIM
        self.meta_embedding = nn.Sequential(
            nn.Linear(meta_dim, self.meta_hidden_dim),
            nn.BatchNorm1d(self.meta_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

        # 3. Fusion & Classification Head (MLP)
        # Cite Lesson 00010: Use MLP with non-linear activations for fusion
        fusion_dim = self.visual_dim + self.meta_hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, Config.NUM_CLASSES),
        )

    def forward(self, images, meta):
        # Visual Features
        visual_features = self.backbone(images)

        # Metadata Features
        meta_emb = self.meta_embedding(meta)

        # Concatenation (Additive Fusion)
        combined_features = torch.cat((visual_features, meta_emb), dim=1)

        # Classification
        logits = self.classifier(combined_features)

        return logits
