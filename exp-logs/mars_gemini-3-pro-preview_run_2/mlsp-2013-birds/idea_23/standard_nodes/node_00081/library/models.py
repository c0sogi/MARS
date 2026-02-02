import torch
import torch.nn as nn
import timm
from library.config import CFG


class MultiSampleDropout(nn.Module):
    """
    Multi-Sample Dropout Head.

    Applies multiple dropout masks to the input features, passes them through a shared
    linear layer, and averages the results. This technique helps in accelerating
    convergence and improving generalization.
    """

    def __init__(self, in_features, out_features, num_dropouts=5, p=0.5):
        """
        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output classes.
            num_dropouts (int): Number of parallel dropout layers.
            p (float): Dropout probability.
        """
        super().__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(p) for _ in range(num_dropouts)])
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, In_Features).
        Returns:
            torch.Tensor: Averaged logits of shape (Batch, Out_Features).
        """
        # Accumulate outputs from different dropout masks passed through the shared linear layer
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.linear(dropout(x))
            else:
                out += self.linear(dropout(x))

        # Average the results
        return out / len(self.dropouts)


class BirdModel(nn.Module):
    """
    Bird Species Classification Model.

    Wraps a timm backbone with a Multi-Sample Dropout head.
    Supports heterogeneous backbones (ResNet18, EfficientNet-B0, DenseNet121) as defined in CFG.
    """

    def __init__(self, backbone_name, pretrained=True):
        """
        Args:
            backbone_name (str): Name of the timm backbone (e.g., 'resnet18').
            pretrained (bool): Whether to load pretrained ImageNet weights.
        """
        super().__init__()

        # Create the backbone using timm
        # num_classes=0 and global_pool='avg' ensures we get the pooled feature vector
        # without the default classifier.
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=CFG.in_channels,
            num_classes=0,
            global_pool="avg",
        )

        # Determine the input feature dimension for the head
        # Most timm models have a num_features attribute
        if hasattr(self.backbone, "num_features"):
            in_features = self.backbone.num_features
        else:
            # Fallback: Forward pass with dummy input to determine shape
            # This handles cases where num_features might not be directly exposed
            with torch.no_grad():
                dummy = torch.zeros(1, CFG.in_channels, CFG.img_height, CFG.img_width)
                features = self.backbone(dummy)
                in_features = features.shape[1]

        # Initialize the Multi-Sample Dropout Head
        self.head = MultiSampleDropout(
            in_features=in_features,
            out_features=CFG.num_classes,
            num_dropouts=5,
            p=0.5,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images of shape (Batch, C, H, W).
        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes).
        """
        # Extract features from the backbone
        # Shape: (Batch, In_Features)
        features = self.backbone(x)

        # Pass through the Multi-Sample Dropout head
        logits = self.head(features)

        return logits
