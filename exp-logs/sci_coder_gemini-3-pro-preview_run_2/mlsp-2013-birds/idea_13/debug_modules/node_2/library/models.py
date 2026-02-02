import torch
import torch.nn as nn
import timm
from library.config import Config


class MultiSampleDropout(nn.Module):
    """
    Implements Multi-Sample Dropout.

    Instead of a single dropout layer, this module applies multiple dropout masks
    to the same input features, passes them through a shared linear layer,
    and averages the outputs. This acts as a form of internal ensembling.
    """

    def __init__(self, in_features, out_features, dropout_rate=0.5, num_samples=5):
        """
        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output classes.
            dropout_rate (float): Probability of an element to be zeroed.
            num_samples (int): Number of dropout samples to average.
        """
        super(MultiSampleDropout, self).__init__()
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(num_samples)]
        )
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
            # Apply specific dropout mask then pass through shared FC
            logits_list.append(self.fc(dropout(x)))

        # Stack results along a new dimension and compute the mean
        return torch.mean(torch.stack(logits_list), dim=0)


class BirdClassifier(nn.Module):
    """
    Bird Species Classifier using a CNN backbone and Multi-Sample Dropout head.
    Supports ResNet18, EfficientNet-B0, and DenseNet121 via timm.
    """

    def __init__(self, model_name, num_classes, pretrained=True):
        """
        Args:
            model_name (str): Key from Config.MODEL_SPECS (e.g., 'resnet18').
            num_classes (int): Number of target species.
            pretrained (bool): Whether to load ImageNet weights.
        """
        super(BirdClassifier, self).__init__()

        if model_name not in Config.MODEL_SPECS:
            raise ValueError(f"Model {model_name} is not defined in Config.MODEL_SPECS")

        backbone_name = Config.MODEL_SPECS[model_name]["backbone"]

        # Create the backbone using timm
        # num_classes=0 removes the default classification head
        # global_pool='avg' ensures the output is a flattened feature vector
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            in_chans=Config.IN_CHANNELS,
        )

        # Retrieve the number of features output by the backbone
        in_features = self.backbone.num_features

        # Replace the head with Multi-Sample Dropout
        # Using a standard dropout rate of 0.5 and 5 samples as per strategy
        self.head = MultiSampleDropout(
            in_features=in_features,
            out_features=num_classes,
            dropout_rate=0.5,
            num_samples=5,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images (Batch, Channels, Height, Width).

        Returns:
            torch.Tensor: Logits (Batch, Num_Classes).
        """
        # Extract features using the backbone
        features = self.backbone(x)

        # Pass features through the multi-sample dropout head
        logits = self.head(features)

        return logits
