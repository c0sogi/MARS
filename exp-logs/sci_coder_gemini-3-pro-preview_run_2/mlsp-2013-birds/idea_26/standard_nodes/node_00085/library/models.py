import torch
import torch.nn as nn
import timm
from library.config import CFG


class MultiSampleDropout(nn.Module):
    """
    Implements Multi-Sample Dropout (MSD).
    Instead of a single dropout layer, this module uses multiple dropout layers
    with different masks in parallel, followed by a shared linear layer.
    The outputs are averaged to produce the final prediction.
    This technique acts as an ensemble within a single model, accelerating convergence
    and improving generalization.
    """

    def __init__(self, in_features, out_features, num_samples=5, p=0.5):
        super(MultiSampleDropout, self).__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(p) for _ in range(num_samples)])
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        # x shape: (Batch, InFeatures)
        # Apply each dropout mask, pass through FC, stack results
        # Stack shape: (NumSamples, Batch, OutFeatures)
        logits_stack = torch.stack(
            [self.fc(dropout(x)) for dropout in self.dropouts], dim=0
        )

        # Average across the samples dimension
        # Output shape: (Batch, OutFeatures)
        return torch.mean(logits_stack, dim=0)


class BirdModel(nn.Module):
    """
    Main model class for Bird Species Classification.
    Wraps a timm backbone with a Multi-Sample Dropout head.
    """

    def __init__(self, model_name, pretrained=True, num_classes=None):
        super(BirdModel, self).__init__()

        if num_classes is None:
            num_classes = CFG.num_classes

        # Create the backbone using timm
        # num_classes=0 removes the default classifier and returns the pooled feature vector
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",  # Ensure we get a pooled vector (B, Features)
        )

        # Get the number of input features for the head
        in_features = self.backbone.num_features

        # Replace the classifier with Multi-Sample Dropout
        self.head = MultiSampleDropout(
            in_features=in_features, out_features=num_classes, num_samples=5, p=0.5
        )

    def forward(self, x):
        # Extract features (B, C, H, W) -> Pooling -> (B, Features)
        features = self.backbone(x)

        # Pass through custom head
        logits = self.head(features)

        return logits


def get_model(model_name, pretrained=True):
    """
    Factory function to create a BirdModel instance.

    Args:
        model_name (str): Name of the timm backbone (e.g., 'resnet18', 'densenet121').
        pretrained (bool): Whether to load pretrained ImageNet weights.

    Returns:
        BirdModel: The initialized model.
    """
    model = BirdModel(model_name, pretrained=pretrained)
    return model
