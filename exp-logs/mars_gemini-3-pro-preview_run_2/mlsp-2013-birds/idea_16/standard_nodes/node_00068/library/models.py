import torch
import torch.nn as nn
import timm
from library.utils import seed_everything


class MultiSampleDropout(nn.Module):
    """
    Implements Multi-Sample Dropout: an ensemble-like technique within a single model.
    The input features are passed through multiple dropout layers with different masks,
    then through a shared fully connected layer. The outputs are averaged.

    Reference: https://arxiv.org/abs/1905.09788
    """

    def __init__(self, in_features, out_features, num_dropouts=5, dropout_rate=0.5):
        super().__init__()
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(num_dropouts)]
        )
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        # x shape: (batch_size, in_features)

        # Apply each dropout mask and then the shared linear layer
        # We collect the logits from each path
        logits_list = []
        for dropout in self.dropouts:
            logits_list.append(self.fc(dropout(x)))

        # Stack logits: (num_dropouts, batch_size, out_features)
        logits = torch.stack(logits_list)

        # Average across the dropout dimension (dim=0)
        return torch.mean(logits, dim=0)


class BirdModel(nn.Module):
    """
    Wrapper model for Bird Species Classification.
    Supports ResNet18, EfficientNet-B0, and DenseNet121 backbones.
    Replaces the default head with Multi-Sample Dropout.
    """

    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()

        # Validate model name
        valid_models = ["resnet18", "efficientnet_b0", "densenet121"]
        if model_name not in valid_models:
            raise ValueError(
                f"Model name must be one of {valid_models}, got {model_name}"
            )

        # Create backbone using timm
        # num_classes=0 removes the classification head and returns pooled features
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",  # Ensure we get a feature vector (B, C)
        )

        # Determine input features for the head
        if hasattr(self.backbone, "num_features"):
            in_features = self.backbone.num_features
        else:
            # Fallback for some models if num_features isn't exposed directly
            # Run a dummy pass to check output shape
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                features = self.backbone(dummy_input)
                in_features = features.shape[1]

        # Custom Head
        self.head = MultiSampleDropout(
            in_features=in_features,
            out_features=num_classes,
            num_dropouts=5,
            dropout_rate=0.5,
        )

    def forward(self, x):
        # Extract features from backbone
        # Shape: (Batch, Features)
        features = self.backbone(x)

        # Pass through Multi-Sample Dropout head
        # Shape: (Batch, Num_Classes)
        logits = self.head(features)

        return logits
