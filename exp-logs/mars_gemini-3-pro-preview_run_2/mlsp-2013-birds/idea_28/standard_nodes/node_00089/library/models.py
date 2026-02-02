import torch
import torch.nn as nn
import timm
from library.config import Config


class MultiSampleDropout(nn.Module):
    """
    Implements Multi-Sample Dropout (MSD).

    Instead of a single dropout layer followed by a linear layer, this module
    uses multiple dropout layers with different rates (or masks), passes each
    through the same linear layer, and averages the outputs. This acts as a
    form of internal ensemble/regularization, accelerating convergence and
    improving generalization.
    """

    def __init__(self, in_features, out_features, dropout_rates):
        """
        Args:
            in_features (int): Number of input features (from backbone).
            out_features (int): Number of output classes.
            dropout_rates (list of float): List of dropout probabilities to use.
        """
        super(MultiSampleDropout, self).__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropout_rates])
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input feature tensor of shape [batch_size, in_features].

        Returns:
            torch.Tensor: Averaged logits of shape [batch_size, out_features].
        """
        # Generate predictions for each dropout mask
        logits_list = []
        for dropout in self.dropouts:
            # Apply dropout then linear layer
            logits_list.append(self.fc(dropout(x)))

        # Stack and average the predictions
        # Shape: [num_dropouts, batch_size, out_features] -> [batch_size, out_features]
        return torch.mean(torch.stack(logits_list), dim=0)


class BirdModel(nn.Module):
    """
    Main model class for Bird Species Classification.

    Wraps a backbone from the `timm` library and attaches a Multi-Sample Dropout head.
    The model expects 3-channel inputs (Pseudo-RGB) as per the strategy.
    """

    def __init__(self, model_name, num_classes=None, pretrained=True):
        """
        Args:
            model_name (str): Name of the backbone architecture (e.g., 'resnet18').
            num_classes (int, optional): Number of target classes. Defaults to Config.NUM_CLASSES.
            pretrained (bool): Whether to load ImageNet pretrained weights.
        """
        super(BirdModel, self).__init__()

        if num_classes is None:
            num_classes = Config.NUM_CLASSES

        # Create the backbone using timm
        # num_classes=0 removes the original classification head and returns the
        # pooled feature vector (embedding) when global_pool='avg' is set.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            in_chans=3,  # Expecting Pseudo-RGB inputs
        )

        # Retrieve the number of output features from the backbone
        in_features = self.backbone.num_features

        # Attach the Multi-Sample Dropout head
        # Uses dropout rates defined in Config
        self.head = MultiSampleDropout(
            in_features=in_features,
            out_features=num_classes,
            dropout_rates=Config.DROPOUT_RATES,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images of shape [batch_size, 3, H, W].

        Returns:
            torch.Tensor: Logits of shape [batch_size, num_classes].
        """
        # Extract features from backbone
        # Shape: [batch_size, num_features]
        features = self.backbone(x)

        # Pass through Multi-Sample Dropout head
        logits = self.head(features)

        return logits
