import torch
import torch.nn as nn
import timm
from library.config import Config


class HybridEfficientNet(nn.Module):
    """
    Multi-Task Hybrid EfficientNet-B1.
    Combines image features (EfficientNet) with metadata features (MLP)
    to predict malignancy (Binary) and diagnosis (Multi-class Auxiliary).
    """

    def __init__(self, meta_dim, pretrained=True):
        """
        Args:
            meta_dim (int): Dimension of the input metadata vector.
            pretrained (bool): Whether to load pretrained ImageNet weights.
        """
        super(HybridEfficientNet, self).__init__()

        # =====================================================================
        # 1. Visual Backbone (EfficientNet-B1)
        # =====================================================================
        # num_classes=0 removes the classifier, returning the pooled feature vector.
        # global_pool='avg' ensures we get a vector (B, Num_Features)
        self.backbone = timm.create_model(
            Config.MODEL_NAME, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        # EfficientNet-B1 typically has 1280 features at the final layer
        self.visual_dim = self.backbone.num_features

        # =====================================================================
        # 2. Metadata Processing Branch (MLP)
        # =====================================================================
        # Projects high-dimensional sparse metadata (OHE) into a dense embedding
        self.meta_embedding_dim = 64

        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.meta_embedding_dim),
            nn.ReLU(),
        )

        # =====================================================================
        # 3. Fusion Module
        # =====================================================================
        # Concatenate Visual features and Metadata Embedding
        fusion_input_dim = self.visual_dim + self.meta_embedding_dim
        fusion_output_dim = 512

        # Non-linear projection block as described in the idea
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_output_dim), nn.ReLU(), nn.Dropout(0.4)
        )

        # =====================================================================
        # 4. Multi-Task Heads
        # =====================================================================
        # Primary Head: Malignancy (Binary Classification)
        # Output: 1 logit (to be passed to BCEWithLogitsLoss)
        self.head_malignancy = nn.Linear(fusion_output_dim, 1)

        # Auxiliary Head: Diagnosis (Multi-class Classification)
        # Output: N logits (to be passed to CrossEntropyLoss)
        self.head_diagnosis = nn.Linear(fusion_output_dim, Config.NUM_AUX_CLASSES)

    def forward(self, images, meta):
        """
        Forward pass of the network.

        Args:
            images (torch.Tensor): Image tensor of shape (Batch, 3, H, W).
            meta (torch.Tensor): Metadata tensor of shape (Batch, Meta_Dim).

        Returns:
            tuple: (logits_malignancy, logits_diagnosis)
        """
        # 1. Extract Visual Features
        # Shape: (Batch, 1280)
        visual_features = self.backbone(images)

        # 2. Extract Metadata Embeddings
        # Shape: (Batch, 64)
        meta_features = self.meta_mlp(meta)

        # 3. Feature Fusion
        # Shape: (Batch, 1280 + 64)
        combined = torch.cat((visual_features, meta_features), dim=1)

        # Shape: (Batch, 512)
        fused_features = self.fusion(combined)

        # 4. Prediction Heads
        logits_malignancy = self.head_malignancy(fused_features)
        logits_diagnosis = self.head_diagnosis(fused_features)

        return logits_malignancy, logits_diagnosis
