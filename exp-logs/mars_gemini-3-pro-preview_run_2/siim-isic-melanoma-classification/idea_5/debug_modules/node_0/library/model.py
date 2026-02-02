import torch
import torch.nn as nn
import timm


class HierarchicalEfficientNet(nn.Module):
    """
    Hierarchical Multi-Task Hybrid EfficientNet-B2.

    Architecture:
    1. Visual Backbone: EfficientNet-B2 (Global Average Pooled).
    2. Fusion: Concatenates Visual Features + Metadata -> Dense -> ReLU -> Dropout.
    3. Auxiliary Head: Predicts Diagnosis from Fused Features.
    4. Primary Head: Predicts Malignancy from [Fused Features + Diagnosis Logits].
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        num_diag_classes: int,
        num_meta_features: int,
        pretrained: bool = True,
    ):
        """
        Args:
            model_name (str): Name of the timm model (e.g., 'tf_efficientnet_b2_ns').
            num_classes (int): Number of output classes for the primary task (1 for binary).
            num_diag_classes (int): Number of output classes for the auxiliary diagnosis task.
            num_meta_features (int): Number of input metadata features.
            pretrained (bool): Whether to load pretrained ImageNet weights.
        """
        super(HierarchicalEfficientNet, self).__init__()

        # 1. Visual Backbone
        # global_pool='avg' ensures we get a flat vector (Batch, Num_Features)
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        # Get the feature dimension of the backbone (e.g., 1408 for B2)
        self.visual_dim = self.backbone.num_features

        # 2. Fusion Module
        # Projects concatenated features to a hidden dimension
        self.fusion_dim = 512
        self.fusion = nn.Sequential(
            nn.Linear(self.visual_dim + num_meta_features, self.fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # 3. Auxiliary Head (Diagnosis)
        # Predicts diagnosis directly from the fused representation
        self.aux_head = nn.Linear(self.fusion_dim, num_diag_classes)

        # 4. Primary Head (Malignancy)
        # Input: Fused Features + Auxiliary Logits (Causal Link)
        # This allows the malignancy classifier to explicitly use the diagnosis prediction
        self.primary_head = nn.Linear(self.fusion_dim + num_diag_classes, num_classes)

    def forward(self, image: torch.Tensor, meta: torch.Tensor):
        """
        Args:
            image (torch.Tensor): Image tensor of shape (B, 3, H, W).
            meta (torch.Tensor): Metadata tensor of shape (B, num_meta_features).

        Returns:
            primary_logits (torch.Tensor): Logits for malignancy (B, 1).
            aux_logits (torch.Tensor): Logits for diagnosis (B, num_diag_classes).
        """
        # 1. Extract Visual Features
        visual_features = self.backbone(image)  # Shape: (B, visual_dim)

        # 2. Concatenate Visual + Metadata
        combined_features = torch.cat([visual_features, meta], dim=1)

        # 3. Apply Fusion Block
        fused_features = self.fusion(combined_features)  # Shape: (B, fusion_dim)

        # 4. Auxiliary Prediction (Diagnosis)
        aux_logits = self.aux_head(fused_features)  # Shape: (B, num_diag_classes)

        # 5. Hierarchical Combination
        # Concatenate the latent representation with the explicit diagnosis signal
        primary_input = torch.cat([fused_features, aux_logits], dim=1)

        # 6. Primary Prediction (Malignancy)
        primary_logits = self.primary_head(primary_input)  # Shape: (B, 1)

        return primary_logits, aux_logits
