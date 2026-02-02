import torch
import torch.nn as nn
import timm
from library.config import Config


class MultiSampleDropout(nn.Module):
    """
    Multi-Sample Dropout Head.

    This module applies multiple dropout masks with different random seeds to the
    same input features, passes each dropped-out version through a shared fully
    connected layer, and averages the resulting logits. This technique acts as
    an ensemble within the network, accelerating convergence and improving
    generalization.
    """

    def __init__(self, in_features, out_features, num_samples=5, dropout_rate=0.5):
        """
        Args:
            in_features (int): Number of input features from the backbone.
            out_features (int): Number of output classes.
            num_samples (int): Number of dropout samples to average.
            dropout_rate (float): Probability of an element to be zeroed.
        """
        super(MultiSampleDropout, self).__init__()

        # Create a list of Dropout layers. Each call to a dropout layer
        # during forward pass generates a unique mask.
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(num_samples)]
        )

        # Shared Fully Connected Layer
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        logits = []
        for dropout in self.dropouts:
            # Apply dropout, then the shared linear layer
            logits.append(self.fc(dropout(x)))

        # Stack logits along a new dimension (num_samples, batch_size, num_classes)
        stacked_logits = torch.stack(logits, dim=0)

        # Average across the samples dimension
        return torch.mean(stacked_logits, dim=0)


class BirdModel(nn.Module):
    """
    Bird Species Classification Model.

    Wraps a `timm` backbone with a Multi-Sample Dropout head.
    Supports ResNet18, ResNet34, and EfficientNet-B0 as specified in the configuration.
    """

    def __init__(self, model_name, pretrained=True, num_classes=None):
        """
        Args:
            model_name (str): Name of the backbone model (e.g., 'resnet18').
            pretrained (bool): Whether to load pretrained ImageNet weights.
            num_classes (int, optional): Number of target classes. Defaults to Config.NUM_SPECIES.
        """
        super(BirdModel, self).__init__()

        if num_classes is None:
            num_classes = Config.NUM_SPECIES

        # Initialize the backbone using timm
        # num_classes=0 removes the default classification head
        # global_pool='avg' ensures the output is a pooled feature vector (Batch, Num_Features)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=Config.CHANNELS,
            global_pool="avg",
        )

        # Retrieve the number of output features from the backbone
        # timm models standardize this attribute as `num_features`
        self.in_features = self.backbone.num_features

        # Initialize the custom Multi-Sample Dropout head
        self.head = MultiSampleDropout(
            in_features=self.in_features,
            out_features=num_classes,
            num_samples=5,
            dropout_rate=0.5,
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images of shape (Batch, Channels, Height, Width).

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes).
        """
        # Extract features using the backbone
        # Output shape: (Batch, Num_Features)
        features = self.backbone(x)

        # Pass features through the custom head
        # Output shape: (Batch, Num_Classes)
        logits = self.head(features)

        return logits
