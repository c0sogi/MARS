import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class MultiScaleInception1D(nn.Module):
    """
    Multi-Scale Inception-1D Module for contextualizing slice sequences.

    Processes the feature sequence through parallel 1D convolutional branches
    with varying kernel sizes and dilations to capture both local and
    long-range dependencies in the Z-axis.
    """

    def __init__(self, in_channels, out_channels):
        super(MultiScaleInception1D, self).__init__()

        # We split the output channels among 3 branches.
        # Ensure the sum equals out_channels.
        branch_channels = out_channels // 3
        remainder = out_channels - (branch_channels * 2)

        # Branch 1: 1x1 Conv (Local/Pointwise context)
        self.branch1 = nn.Conv1d(
            in_channels, branch_channels, kernel_size=1, padding=0, bias=False
        )

        # Branch 2: 3x3 Conv (Immediate neighbor context)
        self.branch2 = nn.Conv1d(
            in_channels, branch_channels, kernel_size=3, padding=1, bias=False
        )

        # Branch 3: 3x3 Conv with Dilation 2 (Wider anatomical context)
        # Effective receptive field increases without increasing parameters
        self.branch3 = nn.Conv1d(
            in_channels, remainder, kernel_size=3, padding=2, dilation=2, bias=False
        )

        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x shape: (Batch, Channels, Sequence_Length)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        out = torch.cat([b1, b2, b3], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        return out


class CervicalFractureModel(nn.Module):
    """
    2.5D ResNet18 with Multi-Scale Inception-1D Context and Instance-MIL.

    Architecture:
    1. Backbone: ResNet18 (pretrained) extracts features from each slice independently.
    2. Context: MultiScaleInception1D mixes features across the Z-axis (sequence).
    3. Classification: Instance-level classifier predicts C1-C7 for each slice.
    4. Aggregation: Global Max Pooling aggregates slice predictions to study predictions.
    """

    def __init__(self, n_classes=7, pretrained=True):
        super(CervicalFractureModel, self).__init__()

        # --- Backbone ---
        # Use ResNet18 as specified for stability with batch size >= 8
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # Remove the FC layer to get features
        # ResNet18 feature dim is 512
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # --- Context Module ---
        # Input: (B, 512, N_Slices) -> Output: (B, 512, N_Slices)
        self.context_module = MultiScaleInception1D(
            in_channels=self.feature_dim, out_channels=self.feature_dim
        )

        # --- Instance Classifier ---
        # Predicts 7 classes per slice
        self.classifier = nn.Linear(self.feature_dim, n_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, N_Slices, 3, H, W)

        Returns:
            torch.Tensor: Logits of shape (Batch, 8)
                          [patient_overall, C1, C2, C3, C4, C5, C6, C7]
        """
        b, n, c, h, w = x.shape

        # 1. Feature Extraction (Time-Distributed)
        # Collapse Batch and Sequence dimensions to process all slices in parallel
        x = x.view(b * n, c, h, w)

        # Extract features: (B*N, 512)
        # ResNet forward passes through convs -> avgpool -> flatten -> fc(Identity)
        features = self.backbone(x)

        # Reshape back to sequence format: (B, 512, N)
        # Note: Conv1d expects (Batch, Channels, Length)
        features = features.view(b, n, self.feature_dim)
        features = features.permute(0, 2, 1)

        # 2. Contextualization
        # Apply Multi-Scale Inception-1D
        context_features = self.context_module(features)

        # Permute back to (B, N, 512) for Linear layer
        context_features = context_features.permute(0, 2, 1)

        # 3. Instance-Level Classification
        # (B, N, 7)
        instance_logits = self.classifier(context_features)

        # 4. Aggregation (Global Max Pooling)
        # We take the max logit across all slices for each class
        # This corresponds to "if a fracture exists in ANY slice, the study is positive"
        # Shape: (B, 7)
        study_logits, _ = torch.max(instance_logits, dim=1)

        # 5. Patient Overall Prediction
        # "patient_overall" is positive if ANY vertebra is fractured.
        # In logit space, max(logits) approximates the logit of the max probability.
        # Shape: (B, 1)
        patient_overall_logit, _ = torch.max(study_logits, dim=1, keepdim=True)

        # Concatenate to form the final 8 outputs
        # Order: [patient_overall, C1, C2, C3, C4, C5, C6, C7]
        # This matches the target column order in the dataset/loss function
        final_logits = torch.cat([patient_overall_logit, study_logits], dim=1)

        return final_logits
