import torch
import torch.nn as nn
import timm
from library.config import BACKBONE, DROP_PATH_RATE, TABULAR_HIDDEN_DIM, NUM_CLASSES


class SwinTransformerGLU(nn.Module):
    """
    A Hierarchical Vision Transformer architecture (Swin V2) fused with tabular data
    using a Gated Linear Unit (GLU) mechanism.

    The tabular data is processed to generate a gating vector that modulates the
    global image features extracted by the backbone before classification.
    """

    def __init__(self, tabular_input_dim, model_name=BACKBONE, pretrained=True):
        """
        Args:
            tabular_input_dim (int): The number of input features in the tabular data.
            model_name (str): The name of the timm backbone model.
            pretrained (bool): Whether to load pretrained ImageNet weights.
        """
        super(SwinTransformerGLU, self).__init__()

        # 1. Vision Backbone (Swin Transformer V2)
        # num_classes=0 removes the default head, global_pool='avg' returns the pooled feature vector
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            drop_path_rate=DROP_PATH_RATE,
        )

        # Determine the dimensionality of the image features
        # Swin Tiny usually outputs 768 features
        self.img_feature_dim = self.backbone.num_features

        # 2. Tabular Gating Branch
        # This MLP projects the tabular data to the same dimension as the image features.
        # The Sigmoid activation ensures the output acts as a gate (0 to 1).
        self.tabular_gate = nn.Sequential(
            nn.Linear(tabular_input_dim, TABULAR_HIDDEN_DIM),
            nn.BatchNorm1d(TABULAR_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(TABULAR_HIDDEN_DIM, self.img_feature_dim),
            nn.Sigmoid(),
        )

        # 3. Classification Head
        # Takes the gated (fused) features and predicts the target logit
        self.head = nn.Linear(self.img_feature_dim, NUM_CLASSES)

    def forward(self, images, tabular_data):
        """
        Args:
            images (torch.Tensor): Batch of images [B, C, H, W]
            tabular_data (torch.Tensor): Batch of tabular features [B, tabular_input_dim]

        Returns:
            logits (torch.Tensor): Raw output scores [B, NUM_CLASSES]
        """
        # Extract global image features
        # Shape: [Batch_Size, img_feature_dim]
        img_feats = self.backbone(images)

        # Generate gating vector from tabular data
        # Shape: [Batch_Size, img_feature_dim]
        gate = self.tabular_gate(tabular_data)

        # Apply Gated Linear Unit (GLU) fusion
        # Re-weight image features based on patient metadata
        fused_feats = img_feats * gate

        # Final Classification
        logits = self.head(fused_feats)

        return logits
