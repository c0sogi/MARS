import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class FractureModel(nn.Module):
    """
    2.5D Stacked-Slice CNN with Max-Pooling Aggregation (MIL).

    This model treats a CT scan as a bag of 2.5D slice stacks. It processes each stack
    independently using a ResNet18 backbone and aggregates the results using Global Max Pooling.
    This allows the model to be trained on study-level labels while learning to identify
    fractures at the slice level.
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
            pretrained (bool): Whether to load ImageNet pretrained weights.
            num_classes (int): Number of target classes (C1-C7).
        """
        super(FractureModel, self).__init__()

        # Initialize Backbone
        if backbone_name == "resnet18":
            # Handle torchvision version compatibility for weights
            try:
                weights = models.ResNet18_Weights.DEFAULT if pretrained else None
                self.backbone = models.resnet18(weights=weights)
            except (AttributeError, TypeError):
                # Fallback for older torchvision versions
                self.backbone = models.resnet18(pretrained=pretrained)
        else:
            raise NotImplementedError(
                f"Backbone {backbone_name} is not currently supported."
            )

        # Modify the final fully connected layer to match the number of classes
        # ResNet18 stores the classifier in 'fc'
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        """
        Forward pass for Multiple Instance Learning.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Slices, Channels, Height, Width).
                              Represents a batch of exams, each with multiple 2.5D slice stacks.

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes).
                          Aggregated study-level predictions.
        """
        b, s, c, h, w = x.shape

        # 1. Collapse Batch and Slice dimensions to process all slices in parallel
        # We treat every slice stack as an independent image for the feature extractor
        # Shape: (Batch * Slices, Channels, Height, Width)
        x = x.view(b * s, c, h, w)

        # 2. Extract features and classify each slice stack
        # Shape: (Batch * Slices, Num_Classes)
        x = self.backbone(x)

        # 3. Reshape back to separate Batch and Slice dimensions
        # Shape: (Batch, Slices, Num_Classes)
        x = x.view(b, s, -1)

        # 4. Global Max Pooling (MIL Aggregation)
        # We take the maximum logit across all slices for each class.
        # Logic: If the model is confident that *any* slice contains a C1 fracture,
        # the study should be classified as having a C1 fracture.
        # Since sigmoid is monotonic, max(logits) corresponds to max(probability).
        # Shape: (Batch, Num_Classes)
        logits, _ = torch.max(x, dim=1)

        return logits
