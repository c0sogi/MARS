import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class AnatomicallyGuidedResNet(nn.Module):
    """
    Anatomically-Guided 2.5D MIL Network.

    This model processes a bag of 2.5D slice stacks (z-1, z, z+1) from a CT scan.
    It uses a ResNet18 backbone for feature extraction and injects slice-level
    positional information (normalized depth) before the classification head.
    Predictions are aggregated via Max Pooling across the bag.
    """

    def __init__(
        self,
        backbone_name=Config.BACKBONE,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
    ):
        """
        Args:
            backbone_name (str): Name of the backbone architecture (default: 'resnet18').
            pretrained (bool): Whether to use ImageNet pretrained weights.
            num_classes (int): Number of specific fracture targets (C1-C7).
        """
        super(AnatomicallyGuidedResNet, self).__init__()

        # 1. Load Backbone
        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            in_features = self.backbone.fc.in_features
        else:
            raise ValueError(f"Backbone '{backbone_name}' is not supported.")

        # Remove the original classification head to extract features
        self.backbone.fc = nn.Identity()

        # 2. Define Feature Dimension with Positional Injection
        # We concatenate the scalar position d_z to the feature vector
        self.feature_dim = in_features + 1

        # 3. Classification Head
        # Maps [Feature_Vector, Position] -> [C1, C2, ..., C7]
        # No dropout is used in the head as per instructions.
        self.fc = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x, positions):
        """
        Forward pass of the MIL network.

        Args:
            x (torch.Tensor): Input images of shape (Batch, Bag_Size, 3, H, W).
            positions (torch.Tensor): Normalized slice depth of shape (Batch, Bag_Size, 1).

        Returns:
            torch.Tensor: Logits of shape (Batch, 8).
                          Columns 0-6: C1-C7 study-level logits.
                          Column 7: Patient overall logit.
        """
        batch_size, bag_size, c, h, w = x.shape

        # --- Feature Extraction ---
        # Flatten Batch and Bag dimensions to process slices in parallel
        # Shape: (Batch * Bag_Size, 3, H, W)
        x_flat = x.view(batch_size * bag_size, c, h, w)

        # Extract features using backbone
        # Shape: (Batch * Bag_Size, 512)
        features = self.backbone(x_flat)

        # --- Positional Injection ---
        # Flatten positions to match features
        # Shape: (Batch * Bag_Size, 1)
        positions_flat = positions.view(batch_size * bag_size, 1)

        # Concatenate features and positional encoding
        # Shape: (Batch * Bag_Size, 513)
        features_aug = torch.cat([features, positions_flat], dim=1)

        # --- Slice-Level Classification ---
        # Predict C1-C7 logits for every slice
        # Shape: (Batch * Bag_Size, 7)
        slice_logits = self.fc(features_aug)

        # Reshape back to (Batch, Bag_Size, 7)
        slice_logits = slice_logits.view(batch_size, bag_size, Config.NUM_CLASSES)

        # --- MIL Aggregation ---
        # Global Max Pooling: The study-level logit for a vertebra is the max
        # probability assigned to that vertebra across all slices in the scan.
        # Shape: (Batch, 7)
        study_logits, _ = torch.max(slice_logits, dim=1)

        # --- Patient Overall Prediction ---
        # The patient is fractured if ANY vertebra is fractured.
        # In logit space, we approximate this by taking the maximum of the vertebral logits.
        # Shape: (Batch, 1)
        patient_logit, _ = torch.max(study_logits, dim=1, keepdim=True)

        # --- Final Output ---
        # Concatenate specific vertebrae logits with the overall patient logit
        # Shape: (Batch, 8)
        output = torch.cat([study_logits, patient_logit], dim=1)

        return output
