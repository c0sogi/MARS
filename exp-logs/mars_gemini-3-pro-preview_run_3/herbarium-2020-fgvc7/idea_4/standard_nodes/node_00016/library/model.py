import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ArcFaceClassifier(nn.Module):
    """
    ArcFace layer for angular margin loss.
    """

    def __init__(self, in_features, num_classes, s=30.0, m=0.50):
        super(ArcFaceClassifier, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, features, label=None):
        # Normalize features and weights
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))

        if label is not None:
            # Calculate cos(theta + m)
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
            phi = cosine * self.cos_m - sine * self.sin_m

            # Handle stability issues where theta + m > pi
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

            # Create one-hot encoding for targets
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)

            # Apply margin only to ground truth classes
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output *= self.s
        else:
            # During inference, just scale the cosine similarity
            output = cosine * self.s

        return output


class HierarchicalResNet(nn.Module):
    """
    Hierarchical Multi-Task Model with ResNet-50 Backbone.
    Heads:
        1. Species: ArcFace Head
        2. Genus: Linear Head (Auxiliary)
    """

    def __init__(
        self, num_species, num_genera, backbone_name="resnet50", pretrained=True
    ):
        super(HierarchicalResNet, self).__init__()

        # Load Backbone
        if backbone_name == "resnet50":
            try:
                # Attempt to use the modern weights argument
                weights = "DEFAULT" if pretrained else None
                self.backbone = models.resnet50(weights=weights)
            except TypeError:
                # Fallback for older torchvision versions
                self.backbone = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError("Only resnet50 is supported in this implementation.")

        # ResNet50 output feature dimension
        self.feature_dim = self.backbone.fc.in_features

        # Remove the original fully connected layer
        self.backbone.fc = nn.Identity()

        # Bottleneck / Embedding processing
        self.bn1 = nn.BatchNorm1d(self.feature_dim)
        self.dropout = nn.Dropout(p=0.5)

        # Primary Head: Species (ArcFace)
        self.species_head = ArcFaceClassifier(self.feature_dim, num_species)

        # Auxiliary Head: Genus (Linear/CrossEntropy)
        self.genus_head = nn.Linear(self.feature_dim, num_genera)

    def forward(self, x, species_label=None):
        """
        Args:
            x: Input images
            species_label: Ground truth species labels (required for ArcFace training)
        Returns:
            species_logits: Scaled cosine similarities (with margin if training)
            genus_logits: Raw logits for genus classification
        """
        # Extract features from backbone
        # Output shape: (Batch, 2048)
        features = self.backbone(x)

        # Apply BN and Dropout
        features = self.bn1(features)
        features = self.dropout(features)

        # Forward pass through heads
        species_logits = self.species_head(features, species_label)
        genus_logits = self.genus_head(features)

        return species_logits, genus_logits
