import torch
import torch.nn as nn
import timm
from library.config import Config


class MultiSampleDropout(nn.Module):
    """
    Multi-Sample Dropout Head.

    Applies multiple dropout masks to the input features and passes them
    through a shared linear layer. The resulting logits are averaged.
    This technique accelerates convergence and improves generalization.
    """

    def __init__(self, in_features, out_features, p=0.5, num_samples=5):
        """
        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output classes.
            p (float): Dropout probability.
            num_samples (int): Number of dropout samples to average.
        """
        super(MultiSampleDropout, self).__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(p) for _ in range(num_samples)])
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        # x shape: (Batch, In_Features)
        # Apply each dropout mask, pass through FC, stack results, and average
        return torch.mean(
            torch.stack([self.fc(drop(x)) for drop in self.dropouts]), dim=0
        )


class BirdModel(nn.Module):
    """
    Main model class for Bird Species Classification.

    Features:
    - Supports ResNet18, EfficientNet-B0, and DenseNet121 backbones via timm.
    - Implements Pseudo-RGB adaptation (1-channel -> 3-channel expansion).
    - Uses a Multi-Sample Dropout head for classification.
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        num_classes: int = Config.NUM_CLASSES,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        """
        Args:
            model_name (str): Name of the backbone (e.g., 'resnet18', 'densenet121').
            pretrained (bool): Whether to load ImageNet pre-trained weights.
            num_classes (int): Number of target classes.
            drop_rate (float): Dropout rate for the backbone (if supported).
            drop_path_rate (float): Stochastic depth rate (for efficientnet/resnet).
        """
        super(BirdModel, self).__init__()
        self.model_name = model_name

        # Load the backbone from timm
        # We set num_classes=0 and global_pool='avg' to get the pooled feature vector
        # We set in_chans=3 because we will perform Pseudo-RGB adaptation
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=3,
            global_pool="avg",
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

        # Determine the input feature dimension for the head
        self.in_features = self.backbone.num_features

        # Multi-Sample Dropout Head
        # Using p=0.5 as standard for the final classification layer
        self.head = MultiSampleDropout(
            in_features=self.in_features, out_features=num_classes, p=0.5, num_samples=5
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Height, Width).
                              Channels can be 1 (Grayscale) or 3 (RGB).

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes).
        """
        # Ensure input is 4D
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Pseudo-RGB Adaptation
        # If input is 1-channel (spectrogram), expand to 3 channels for ImageNet weights
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)

        # Extract features using the backbone
        # timm with global_pool='avg' returns (Batch, Num_Features)
        features = self.backbone(x)

        # Pass features through the Multi-Sample Dropout head
        logits = self.head(features)

        return logits
