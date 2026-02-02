import torch
import torch.nn as nn
import torchvision.models as models
from library.config import CFG


class AttentionPool(nn.Module):
    """
    Learnable Attention Pooling layer.
    Computes a weighted sum of temporal features, where weights are learned
    dynamically based on the content of each time step.
    """

    def __init__(self, in_channels, hidden_dim=128):
        super(AttentionPool, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, Channels, Time).

        Returns:
            torch.Tensor: Aggregated features of shape (Batch, Channels).
        """
        # Permute to (Batch, Time, Channels) for Linear layers
        x_perm = x.permute(0, 2, 1)

        # Compute attention weights: (Batch, Time, 1)
        # Softmax is applied over the Time dimension (dim=1)
        weights = self.attention(x_perm)

        # Apply weights: (Batch, Time, 1) * (Batch, Time, Channels) -> (Batch, Time, Channels)
        weighted = weights * x_perm

        # Sum over time to get global representation: (Batch, Channels)
        return weighted.sum(dim=1)


class BirdResNet(nn.Module):
    """
    ResNet-34 based model with Attention Pooling for Bird Species Classification.
    """

    def __init__(self, pretrained=True, num_classes=CFG.num_classes):
        super(BirdResNet, self).__init__()

        # Load Backbone based on CFG
        if CFG.model_name == "resnet18":
            if pretrained:
                weights = models.ResNet18_Weights.IMAGENET1K_V1
            else:
                weights = None
            self.backbone = models.resnet18(weights=weights)
        else:
            # Default to ResNet34
            if pretrained:
                weights = models.ResNet34_Weights.IMAGENET1K_V1
            else:
                weights = None
            self.backbone = models.resnet34(weights=weights)

        # Extract the feature extractor part (layers conv1 through layer4)
        # We remove the final avgpool and fc layers to keep the feature map
        self.features = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
        )

        # ResNet18 and ResNet34 both have 512 output channels at layer4
        self.in_features = 512

        # Attention Pooling layer
        self.pool = AttentionPool(self.in_features)

        # Final Classification Head
        self.fc = nn.Linear(self.in_features, num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input spectrograms of shape (Batch, 3, Freq, Time).

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes).
        """
        # Extract spatial/temporal features
        # Output shape: (Batch, 512, F_dim, T_dim)
        x = self.features(x)

        # Pool over the Frequency dimension (dim=2)
        # We use mean pooling here to reduce frequency information while preserving time.
        # Result shape: (Batch, 512, T_dim)
        x = x.mean(dim=2)

        # Apply Attention Pooling over the Time dimension
        # Result shape: (Batch, 512)
        x = self.pool(x)

        # Classification
        logits = self.fc(x)

        return logits
