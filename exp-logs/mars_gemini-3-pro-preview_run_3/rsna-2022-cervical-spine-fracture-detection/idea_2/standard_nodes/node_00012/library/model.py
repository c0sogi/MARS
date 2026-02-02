import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class SequenceSmoothedMIL(nn.Module):
    """
    Sequence-Smoothed 2.5D MIL Network for Cervical Spine Fracture Detection.

    This architecture implements a Weakly Supervised 2.5D CNN-RNN Hybrid approach (using 1D Conv as RNN).
    It processes a sequence of CT slices to detect fractures at the vertebral level and aggregates
    them to the patient level.

    Components:
    1. Backbone: ResNet18 (2.5D input: 3 channels).
    2. Inter-Slice Context: 1D Convolution over sequence dimension to smooth features.
    3. Classification Head: Linear projection to 7 classes (C1-C7).
    4. Aggregation: Global Max Pooling over sequence.
    5. Patient Output: Max of vertebral logits.
    """

    def __init__(self, backbone_name="resnet18", pretrained=True, num_classes=7):
        """
        Args:
            backbone_name (str): Name of the backbone architecture (default: resnet18).
            pretrained (bool): Whether to load pretrained ImageNet weights.
            num_classes (int): Number of specific fracture targets (C1-C7).
        """
        super(SequenceSmoothedMIL, self).__init__()

        # 1. Backbone Setup
        if backbone_name == "resnet18":
            try:
                # Modern torchvision API
                weights = models.ResNet18_Weights.DEFAULT if pretrained else None
                self.backbone = models.resnet18(weights=weights)
            except AttributeError:
                # Legacy torchvision API
                self.backbone = models.resnet18(pretrained=pretrained)

            self.feature_dim = self.backbone.fc.in_features
        else:
            raise NotImplementedError(
                f"Backbone {backbone_name} is not currently supported."
            )

        # Remove the original classification head (fc)
        # We keep the pooling layer to get a (Batch, 512) vector per image
        self.backbone.fc = nn.Identity()

        # 2. Inter-Slice Context Module
        # A 1D Convolution acts as a lightweight sequence model to smooth features
        # across the z-axis (sequence dimension).
        # Input shape to this layer: (Batch, Feature_Dim, Seq_Len)
        self.context_conv = nn.Conv1d(
            in_channels=self.feature_dim,
            out_channels=self.feature_dim,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn_context = nn.BatchNorm1d(self.feature_dim)
        self.relu = nn.ReLU(inplace=True)

        # 3. Classification Head
        # Projects smoothed features to C1-C7 logits
        # No Dropout is used here to maintain gradient stability for Max Pooling
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq_Len, 3, H, W)
                              Represents a batch of exams, each with a sequence of 2.5D slices.
        Returns:
            torch.Tensor: Logits of shape (Batch, 8).
                          Columns: [C1, C2, C3, C4, C5, C6, C7, patient_overall]
        """
        b, s, c, h, w = x.shape

        # Flatten batch and sequence dimensions to process all slices through the backbone
        # Shape: (Batch * Seq_Len, 3, H, W)
        x = x.view(b * s, c, h, w)

        # Extract features using the backbone
        # Output Shape: (Batch * Seq_Len, Feature_Dim)
        features = self.backbone(x)

        # Reshape for the Context Module (1D Conv expects N, C, L)
        # Shape: (Batch, Feature_Dim, Seq_Len)
        features = features.view(b, s, self.feature_dim).permute(0, 2, 1)

        # Apply Inter-Slice Smoothing (Context Module)
        features = self.context_conv(features)
        features = self.bn_context(features)
        features = self.relu(features)

        # Reshape back to (Batch, Seq_Len, Feature_Dim) for classification
        features = features.permute(0, 2, 1)

        # Classify each slice independently (after smoothing)
        # Output Shape: (Batch, Seq_Len, 7)
        logits_seq = self.classifier(features)

        # 4. Aggregation: Global Max Pooling
        # We assume that if a fracture exists in the exam, it will be detected with high confidence
        # in at least one slice. Max pooling captures this signal.
        # Output Shape: (Batch, 7)
        logits_vertebrae, _ = torch.max(logits_seq, dim=1)

        # 5. Patient Output Calculation
        # Logical consistency: A patient is fractured if ANY of the C1-C7 vertebrae are fractured.
        # In logit space, max(logits) is a soft approximation of the OR operator.
        # Output Shape: (Batch, 1)
        logits_overall, _ = torch.max(logits_vertebrae, dim=1, keepdim=True)

        # Concatenate vertebral logits and patient overall logit
        # Final Output Shape: (Batch, 8)
        logits = torch.cat([logits_vertebrae, logits_overall], dim=1)

        return logits
