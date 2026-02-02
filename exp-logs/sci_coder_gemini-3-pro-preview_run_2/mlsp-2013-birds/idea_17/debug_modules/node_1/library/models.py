import torch
import torch.nn as nn
import timm
from library import config


class MultiSampleDropout(nn.Module):
    """
    Implements Multi-Sample Dropout.

    This module applies multiple dropout masks to the input features and passes
    each result through a shared linear layer. The final output is the average
    of the logits. This technique acts as a regularizer and can accelerate training.
    """

    def __init__(self, in_features, out_features, num_samples=5, dropout_rate=0.5):
        """
        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output classes.
            num_samples (int): Number of dropout samples to average.
            dropout_rate (float): Probability of an element to be zeroed.
        """
        super(MultiSampleDropout, self).__init__()
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(num_samples)]
        )
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        # x shape: [Batch, In_Features]
        # Generate logits for each dropout mask
        logits_list = [self.fc(dropout(x)) for dropout in self.dropouts]

        # Stack and average along the new dimension
        # Shape: [Num_Samples, Batch, Out_Features] -> [Batch, Out_Features]
        return torch.mean(torch.stack(logits_list, dim=0), dim=0)


class BirdClassifier(nn.Module):
    """
    Main classifier class for Bird Species Detection.

    Wraps a timm backbone with a Multi-Sample Dropout head.
    """

    def __init__(self, model_name, num_classes=config.NUM_SPECIES, pretrained=True):
        """
        Args:
            model_name (str): Name of the backbone (e.g., 'resnet18', 'densenet121').
            num_classes (int): Number of target species.
            pretrained (bool): Whether to load ImageNet pretrained weights.
        """
        super(BirdClassifier, self).__init__()

        # Create the backbone
        # num_classes=0 removes the top classification layer
        # global_pool='avg' ensures we get a feature vector (e.g., [B, 512]) not a map
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        # Determine the number of input features for the head
        # We can inspect the num_features attribute which timm provides
        if hasattr(self.backbone, "num_features"):
            in_features = self.backbone.num_features
        else:
            # Fallback: pass a dummy variable to find shape
            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224)
                features = self.backbone(dummy)
                in_features = features.shape[1]

        # Replace the head with Multi-Sample Dropout
        self.head = MultiSampleDropout(
            in_features=in_features,
            out_features=num_classes,
            num_samples=5,
            dropout_rate=0.5,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images of shape [B, 3, H, W]

        Returns:
            torch.Tensor: Logits of shape [B, Num_Classes]
        """
        # Extract features using the backbone
        features = self.backbone(x)

        # Pass through the custom head
        logits = self.head(features)

        return logits
