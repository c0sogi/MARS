import torch
import torch.nn as nn
import timm
from library.config import Config


class HybridEfficientNet(nn.Module):
    """
    Hybrid EfficientNet with Late Fusion and MLP Head.

    Replaces Context Gating with simple concatenation (Cite solution_lesson_node_00013).
    Uses an MLP head to model non-linear interactions (Cite solution_lesson_node_00010).
    """

    def __init__(self, meta_dim, model_name=None, pretrained=True):
        """
        Args:
            meta_dim (int): Dimension of the input metadata vector.
            model_name (str, optional): Name of the timm model.
            pretrained (bool, optional): Whether to load ImageNet weights.
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

        # 3. Fusion & MLP Classification Head
        # Concatenation of Visual + Metadata
        combined_dim = self.visual_dim + self.meta_hidden_dim

        # MLP: Linear -> ReLU -> Dropout -> Linear
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, Config.NUM_CLASSES),
        )

    def forward(self, images, meta):
        # Visual Branch
        visual_features = self.backbone(images)

        # Metadata Branch
        meta_emb = self.meta_embedding(meta)

        # Late Fusion (Concatenation)
        # Cite solution_lesson_node_00013: Additive fusion is more robust than multiplicative gating
        combined_features = torch.cat((visual_features, meta_emb), dim=1)

        # Classification
        logits = self.classifier(combined_features)

        return logits
