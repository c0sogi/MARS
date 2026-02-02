import torch
import torch.nn as nn
import timm
from library.config import Config


class MultiSampleDropout(nn.Module):
    """
    Implements Multi-Sample Dropout (MSD).

    Instead of a single dropout layer followed by a linear layer, MSD uses multiple
    dropout layers with the same probability, passes the outputs through the same
    linear layer, and averages the results. This technique accelerates convergence
    and improves generalization.
    """

    def __init__(self, in_features, out_features, num_samples=5, p=0.5):
        super(MultiSampleDropout, self).__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(p) for _ in range(num_samples)])
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        # Apply each dropout, pass through FC, stack results, and compute mean
        # x shape: (batch_size, in_features)
        # output shape: (batch_size, out_features)
        return torch.mean(
            torch.stack([self.fc(dropout(x)) for dropout in self.dropouts], dim=0),
            dim=0,
        )


class BirdModel(nn.Module):
    """
    Wrapper for timm models with a custom Multi-Sample Dropout head.
    Supports ResNet, DenseNet, and EfficientNet architectures as defined in Config.
    """

    def __init__(self, model_name, pretrained=True):
        super(BirdModel, self).__init__()

        # Create the backbone model
        # num_classes=0 removes the classification head and global pooling (if specified),
        # but usually returns the pooled features if global_pool is left default.
        # For most timm models, num_classes=0 returns the pooled feature vector (N, EmbedDim).
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=Config.IN_CHANNELS,
        )

        # Determine the input features for the head
        if hasattr(self.backbone, "num_features"):
            in_features = self.backbone.num_features
        else:
            # Fallback for models where num_features might not be directly exposed
            # Pass a dummy input to find the shape
            with torch.no_grad():
                dummy_input = torch.randn(1, Config.IN_CHANNELS, 224, 224)
                features = self.backbone(dummy_input)
                in_features = features.shape[1]

        # Custom Head
        self.head = MultiSampleDropout(
            in_features=in_features,
            out_features=Config.NUM_CLASSES,
            num_samples=5,
            p=0.5,
        )

    def forward(self, x):
        # Extract features using the backbone
        # Shape: (batch_size, num_features)
        features = self.backbone(x)

        # Pass through the custom head
        # Shape: (batch_size, num_classes)
        logits = self.head(features)

        return logits
