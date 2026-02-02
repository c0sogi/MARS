import torch
import torch.nn as nn
import timm
from library.config import Config


class MultiSampleDropout(nn.Module):
    """
    Multi-Sample Dropout Head.

    Applies multiple dropout masks to the input features, passes them through
    a shared fully connected layer, and averages the results. This technique
    acts as an internal ensemble, accelerating convergence and improving
    generalization.
    """

    def __init__(self, in_features, out_features, num_samples=5, dropout_rate=0.5):
        """
        Args:
            in_features (int): Dimension of input feature vector.
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
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, In_Features).

        Returns:
            torch.Tensor: Averaged logits of shape (Batch, Out_Features).
        """
        logits_list = []
        for dropout in self.dropouts:
            # Apply dropout, then the shared linear layer
            logits_list.append(self.fc(dropout(x)))

        # Stack results and compute the mean across the sample dimension
        return torch.stack(logits_list, dim=0).mean(dim=0)


class BirdClassifier(nn.Module):
    """
    Bird Species Classifier.

    Wraps a timm backbone with a Multi-Sample Dropout head.
    Supports 'resnet18', 'efficientnet_b0', and 'densenet121'.
    """

    def __init__(self, backbone_name):
        """
        Args:
            backbone_name (str): The name of the backbone architecture to create.
                                 Must be supported by timm (e.g., 'resnet18').
        """
        super(BirdClassifier, self).__init__()

        # Initialize the backbone using timm
        # pretrained=True: Downloads and loads ImageNet weights
        # num_classes=0: Removes the default classification layer
        # global_pool='avg': Applies Global Average Pooling to output a 1D vector
        # in_chans=Config.IN_CHANNELS: Sets input channels (3 for Pseudo-RGB)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
            in_chans=Config.IN_CHANNELS,
        )

        # Determine the output feature dimension of the backbone
        if hasattr(self.backbone, "num_features"):
            in_features = self.backbone.num_features
        else:
            # Fallback: Forward pass with dummy input to determine shape
            with torch.no_grad():
                dummy_input = torch.randn(
                    1, Config.IN_CHANNELS, Config.IMG_HEIGHT, Config.IMG_WIDTH
                )
                features = self.backbone(dummy_input)
                in_features = features.shape[1]

        # Initialize the custom head
        # Uses 5 dropout samples as specified in the strategy
        self.head = MultiSampleDropout(
            in_features=in_features,
            out_features=Config.NUM_CLASSES,
            num_samples=5,
            dropout_rate=0.5,
        )

    def forward(self, x):
        """
        Forward pass of the classifier.

        Args:
            x (torch.Tensor): Input images of shape (Batch, C, H, W).

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes).
        """
        # Extract features from the backbone (Batch, Num_Features)
        features = self.backbone(x)

        # Pass features through the Multi-Sample Dropout head
        logits = self.head(features)

        return logits
