import torch
import torch.nn as nn
import timm
from library.config import Config


class MultiSampleDropout(nn.Module):
    """
    Implements Multi-Sample Dropout.
    Applies multiple dropout masks to the same features and averages the predictions
    from the shared fully connected layer. This acts as an ensemble within the model,
    accelerating convergence and improving generalization.
    """

    def __init__(self, in_features, out_features, num_samples=5, p=0.5):
        """
        Args:
            in_features (int): Number of input features from the backbone.
            out_features (int): Number of output classes.
            num_samples (int): Number of dropout masks to apply.
            p (float): Dropout probability.
        """
        super(MultiSampleDropout, self).__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(p) for _ in range(num_samples)])
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, In_Features).

        Returns:
            torch.Tensor: Averaged logits of shape (Batch, Out_Features).
        """
        logits_list = []
        for dropout in self.dropouts:
            # Apply dropout then linear projection
            out = self.fc(dropout(x))
            logits_list.append(out)

        # Stack along new dimension and average
        # Shape: [Num_Samples, Batch, Out_Features] -> [Batch, Out_Features]
        return torch.mean(torch.stack(logits_list, dim=0), dim=0)


class BirdClassifier(nn.Module):
    """
    Bird Species Classifier using a backbone from timm and a Multi-Sample Dropout head.
    Supports ResNet18, EfficientNet-B0, and DenseNet121 as defined in the strategy.
    """

    def __init__(self, backbone_name, pretrained=True, num_classes=None):
        """
        Args:
            backbone_name (str): Name of the backbone architecture (e.g., 'resnet18').
            pretrained (bool): Whether to load ImageNet pretrained weights.
            num_classes (int, optional): Number of target classes. Defaults to Config.NUM_CLASSES.
        """
        super(BirdClassifier, self).__init__()

        if num_classes is None:
            num_classes = Config.NUM_CLASSES

        # Create backbone using timm
        # Setting num_classes=0 and global_pool='avg' removes the classification head
        # and returns the pooled feature vector directly.
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        # Determine input features for the head dynamically
        in_features = self.backbone.num_features

        # Custom Head: Multi-Sample Dropout
        # We use 5 samples as per the ensemble strategy
        self.head = MultiSampleDropout(
            in_features=in_features,
            out_features=num_classes,
            num_samples=5,
            p=0.5,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images of shape (Batch, Channels, Height, Width).

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes).
        """
        # Extract features (pooled)
        # Shape: (Batch, Num_Features)
        features = self.backbone(x)

        # Pass through multi-sample dropout head
        logits = self.head(features)

        return logits
