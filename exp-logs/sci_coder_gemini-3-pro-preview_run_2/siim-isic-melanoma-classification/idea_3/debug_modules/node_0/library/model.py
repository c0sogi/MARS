import torch
import torch.nn as nn
import timm
from library.config import Config


class ContextGatedEfficientNet(nn.Module):
    """
    Context-Gated Deep Hybrid Network.

    Combines an EfficientNet-B0 visual backbone with a metadata processing branch.
    Uses a Context Gating mechanism where patient metadata explicitly reweights
    visual features before classification.
    """

    def __init__(self, meta_dim, model_name=None, pretrained=True):
        """
        Args:
            meta_dim (int): Dimension of the input metadata vector (after preprocessing).
            model_name (str, optional): Name of the timm model to use. Defaults to Config.MODEL_NAME.
            pretrained (bool, optional): Whether to load ImageNet weights. Defaults to True.
        """
        super(ContextGatedEfficientNet, self).__init__()

        self.model_name = model_name if model_name else Config.MODEL_NAME

        # 1. Visual Backbone
        # num_classes=0 returns the pooled feature vector
        # global_pool='avg' ensures we get a 1D vector per image
        self.backbone = timm.create_model(
            self.model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        # Retrieve the feature dimension (e.g., 1280 for EfficientNet-B0)
        self.visual_dim = self.backbone.num_features

        # 2. Metadata Embedding
        # Projects raw metadata to a latent representation
        self.meta_hidden_dim = Config.META_HIDDEN_DIM
        self.meta_embedding = nn.Sequential(
            nn.Linear(meta_dim, self.meta_hidden_dim),
            nn.BatchNorm1d(self.meta_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

        # 3. Context Gating Unit
        # Projects metadata embedding to the visual feature dimension
        # Sigmoid ensures the gate values are between 0 and 1
        self.gate_projection = nn.Sequential(
            nn.Linear(self.meta_hidden_dim, self.visual_dim), nn.Sigmoid()
        )

        # 4. Classification Head
        # Concatenates (Gated Visual Features + Metadata Embedding)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.visual_dim + self.meta_hidden_dim, Config.NUM_CLASSES),
        )

    def forward(self, images, meta):
        """
        Args:
            images (torch.Tensor): Image tensor of shape (B, C, H, W)
            meta (torch.Tensor): Metadata tensor of shape (B, meta_dim)

        Returns:
            torch.Tensor: Logits of shape (B, 1)
        """
        # --- Visual Branch ---
        # Shape: (B, visual_dim)
        visual_features = self.backbone(images)

        # --- Metadata Branch ---
        # Shape: (B, meta_hidden_dim)
        meta_emb = self.meta_embedding(meta)

        # --- Context Gating ---
        # Generate Gate: Shape (B, visual_dim)
        gate = self.gate_projection(meta_emb)

        # Apply Gate: Element-wise multiplication
        # This reweights visual features based on patient context
        gated_visual_features = visual_features * gate

        # --- Fusion & Classification ---
        # Concatenate gated visual features with the metadata embedding
        # Shape: (B, visual_dim + meta_hidden_dim)
        combined_features = torch.cat((gated_visual_features, meta_emb), dim=1)

        # Final prediction
        logits = self.classifier(combined_features)

        return logits
