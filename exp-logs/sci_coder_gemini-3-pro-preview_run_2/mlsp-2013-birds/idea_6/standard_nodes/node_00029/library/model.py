import torch
import torch.nn as nn
from torchvision import models
from library import config


class MultiSampleDropout(nn.Module):
    """
    Implements Multi-Sample Dropout (MSD).

    This module applies multiple dropout masks with different rates to the same input features,
    then passes each dropped-out version through a shared fully connected layer.

    During training, it returns the logits for all branches (to compute average loss).
    During inference, it returns the average of the logits (ensemble prediction).
    """

    def __init__(self, in_features, out_features, dropout_rates):
        super(MultiSampleDropout, self).__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropout_rates])
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, In_Features).

        Returns:
            torch.Tensor:
                - If training: (Batch, Num_Drops, Out_Features)
                - If eval: (Batch, Out_Features)
        """
        # Apply each dropout mask and pass through the shared linear layer
        logits_list = [self.linear(d(x)) for d in self.dropouts]

        # Stack logits along a new dimension: (Batch, Num_Drops, Out_Features)
        logits = torch.stack(logits_list, dim=1)

        if self.training:
            return logits
        else:
            # During inference, average the predictions across all dropout masks
            return torch.mean(logits, dim=1)


class BirdResNet(nn.Module):
    """
    ResNet18-based model for Bird Species Classification.

    Features:
    - Backbone: ResNet18 (Pretrained on ImageNet)
    - Input: 3-Channel (Replicated Spectrogram)
    - Head: Simple Dropout + Linear (Cite solution_lesson_node_00024)
    """

    def __init__(self, pretrained=config.PRETRAINED):
        super(BirdResNet, self).__init__()

        # Load Pretrained ResNet18 (Cite solution_lesson_node_00012)
        weights = "DEFAULT" if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # Feature Extractor: Keep layers up to avgpool
        layers = list(self.backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # Determine input features for the head (512 for ResNet18)
        in_features = self.backbone.fc.in_features

        # Classification Head
        self.head = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(in_features, config.NUM_CLASSES)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images of shape (Batch, 3, Height, Width).

        Returns:
            torch.Tensor: Logits (Batch, Num_Classes).
        """
        # Extract features
        x = self.feature_extractor(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Classification Head
        x = self.head(x)

        return x
