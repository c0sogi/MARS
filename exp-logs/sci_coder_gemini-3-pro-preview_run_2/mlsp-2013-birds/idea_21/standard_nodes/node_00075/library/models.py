import torch
import torch.nn as nn
import timm
from library.config import Config


class MultiSampleDropout(nn.Module):
    """
    Multi-Sample Dropout Head.

    Applies multiple dropout layers with different probabilities to the input features,
    passes each through the same linear layer, and averages the outputs.
    This technique accelerates convergence and improves generalization.
    """

    def __init__(self, in_features, out_features, dropout_rates):
        """
        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output classes.
            dropout_rates (list of float): List of dropout probabilities to apply.
        """
        super(MultiSampleDropout, self).__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropout_rates])
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input feature tensor of shape (Batch, In_Features).

        Returns:
            torch.Tensor: Averaged logits of shape (Batch, Out_Features).
        """
        logits_list = []
        for dropout in self.dropouts:
            # Apply dropout, then linear layer
            logits_list.append(self.linear(dropout(x)))

        # Stack along a new dimension and compute the mean
        # Stack shape: (Batch, Num_Dropouts, Out_Features)
        logits = torch.stack(logits_list, dim=1).mean(dim=1)
        return logits


class BirdModel(nn.Module):
    """
    Main model class for Bird Species Classification.

    Wraps a timm backbone with a Multi-Sample Dropout head.
    Supports ResNet, EfficientNet, and DenseNet architectures.
    """

    def __init__(self, model_name, pretrained=True):
        """
        Args:
            model_name (str): Name of the backbone model (e.g., 'resnet18').
            pretrained (bool): Whether to load pretrained ImageNet weights.
        """
        super(BirdModel, self).__init__()

        # Create the backbone model
        # num_classes=0 removes the default classification head
        # global_pool='avg' ensures the output is a pooled feature vector
        # in_chans=Config.IN_CHANNELS handles the input channel adaptation (Pseudo-RGB)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            in_chans=Config.IN_CHANNELS,
        )

        # Retrieve the number of output features from the backbone
        in_features = self.backbone.num_features

        # Initialize the custom Multi-Sample Dropout head
        self.head = MultiSampleDropout(
            in_features=in_features,
            out_features=Config.NUM_CLASSES,
            dropout_rates=Config.DROPOUT_RATES,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images of shape (Batch, Channels, Height, Width).

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes).
        """
        # Extract features using the backbone
        # Output shape: (Batch, Num_Features)
        features = self.backbone(x)

        # Pass features through the custom head
        logits = self.head(features)

        return logits
