import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class ResNet18MIL(nn.Module):
    """
    ResNet18 Multiple Instance Learning (MIL) Network for Cervical Spine Fracture Detection.

    This architecture treats a CT scan as a 'bag' of independent 2.5D slices.
    It processes each slice independently and aggregates predictions using Global Max Pooling.

    Cite solution_lesson_node_00012: Avoid Sequence Smoothing on Aggressively Subsampled Volumetric Data.
    We remove the 1D Conv context layer to prevent signal dilution and rely on permutation-invariant
    Max Pooling.

    Components:
    1. Backbone: ResNet18 (2.5D input: 3 channels).
    2. Classification Head: Linear projection to 7 classes (C1-C7).
    3. Aggregation: Global Max Pooling over the bag (sequence).
    4. Patient Output: Max of vertebral logits.
    """

    def __init__(self, backbone_name="resnet18", pretrained=True, num_classes=7):
        """
        Args:
            backbone_name (str): Name of the backbone architecture (default: resnet18).
            pretrained (bool): Whether to load pretrained ImageNet weights.
            num_classes (int): Number of specific fracture targets (C1-C7).
        """
        super(ResNet18MIL, self).__init__()

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

        # 2. Classification Head
        # Projects features directly to C1-C7 logits
        # No Dropout is used here to maintain gradient stability for Max Pooling (Cite solution_lesson_node_00003)
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

        # Classify each slice independently
        # Output Shape: (Batch * Seq_Len, 7)
        logits_flat = self.classifier(features)

        # Reshape back to (Batch, Seq_Len, 7)
        logits_seq = logits_flat.view(b, s, -1)

        # 3. Aggregation: Global Max Pooling
        # We assume that if a fracture exists in the exam, it will be detected with high confidence
        # in at least one slice. Max pooling captures this signal.
        # Output Shape: (Batch, 7)
        logits_vertebrae, _ = torch.max(logits_seq, dim=1)

        # 4. Patient Output Calculation
        # Logical consistency: A patient is fractured if ANY of the C1-C7 vertebrae are fractured.
        # In logit space, max(logits) is a soft approximation of the OR operator.
        # Output Shape: (Batch, 1)
        logits_overall, _ = torch.max(logits_vertebrae, dim=1, keepdim=True)

        # Concatenate vertebral logits and patient overall logit
        # Final Output Shape: (Batch, 8)
        logits = torch.cat([logits_vertebrae, logits_overall], dim=1)

        return logits
