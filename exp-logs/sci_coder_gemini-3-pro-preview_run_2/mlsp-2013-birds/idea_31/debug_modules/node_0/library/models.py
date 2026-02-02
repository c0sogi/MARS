import torch
import torch.nn as nn
import timm
from library.config import Config


class MultiSampleDropout(nn.Module):
    """
    Implements Multi-Sample Dropout.

    The input features are passed through multiple dropout layers (with different masks)
    in parallel, then through a shared fully connected layer. The resulting logits
    are averaged. This acts as an internal ensemble, improving generalization and
    convergence speed.
    """

    def __init__(self, in_features, out_features, n_samples=5, p=0.5):
        """
        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output classes.
            n_samples (int): Number of dropout samples to average.
            p (float): Dropout probability.
        """
        super().__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(p) for _ in range(n_samples)])
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, In_Features).

        Returns:
            torch.Tensor: Averaged logits of shape (Batch, Out_Features).
        """
        out = None
        for i, dropout in enumerate(self.dropouts):
            # Apply dropout and then the shared linear layer
            logits = self.fc(dropout(x))

            if i == 0:
                out = logits
            else:
                out += logits

        # Average the results
        return out / len(self.dropouts)


class BirdClassifier(nn.Module):
    """
    Main classifier class for the Bird Species Detection task.

    Supports creating ResNet, EfficientNet, and DenseNet backbones using timm.
    Replaces the standard classification head with a MultiSampleDropout head.
    """

    def __init__(
        self,
        model_name,
        num_classes=None,
        pretrained=True,
        drop_rate=0.5,
        drop_samples=5,
    ):
        """
        Args:
            model_name (str): Name of the timm model (e.g., 'resnet18', 'efficientnet_b0').
            num_classes (int, optional): Number of target classes. Defaults to Config.NUM_CLASSES.
            pretrained (bool): Whether to load ImageNet pretrained weights.
            drop_rate (float): Dropout probability for the head.
            drop_samples (int): Number of dropout samples for the head.
        """
        super().__init__()

        if num_classes is None:
            num_classes = Config.NUM_CLASSES

        # Create the backbone.
        # num_classes=0 tells timm to return the global pool features (feature vector)
        # instead of the final logits, and removes the default classifier.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",  # Ensure we get GAP features
        )

        # Determine the number of input features for the head
        # timm models expose this via num_features
        in_features = self.backbone.num_features

        # Define the custom head
        self.head = MultiSampleDropout(
            in_features=in_features,
            out_features=num_classes,
            n_samples=drop_samples,
            p=drop_rate,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images (Batch, Channels, Height, Width).

        Returns:
            torch.Tensor: Logits (Batch, Num_Classes).
        """
        # Extract features using the backbone
        # Shape: (Batch, Num_Features)
        features = self.backbone(x)

        # Pass through the Multi-Sample Dropout head
        # Shape: (Batch, Num_Classes)
        logits = self.head(features)

        return logits
