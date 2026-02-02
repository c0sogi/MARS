import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Learnable Attention Pooling layer.
    Computes a weight for each time step and aggregates features via weighted sum.

    Args:
        in_channels (int): Number of input feature channels.
        hidden_dim (int): Hidden dimension for the attention mechanism.
    """

    def __init__(self, in_channels, hidden_dim=128):
        super(AttentionPooling, self).__init__()
        self.att_conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features of shape (Batch, Channels, Time).

        Returns:
            torch.Tensor: Aggregated features of shape (Batch, Channels).
        """
        # Calculate attention weights
        # w shape: (Batch, 1, Time)
        w = self.att_conv(x)

        # Apply weighted sum
        # x * w broadcasts to (Batch, Channels, Time)
        # Sum over Time dimension
        x = torch.sum(x * w, dim=2)

        return x


class BirdResNet(nn.Module):
    """
    ResNet-34 based model for Bird Species Classification.
    Uses Attention Pooling to handle temporal sparsity of bird calls.
    """

    def __init__(self, pretrained=Config.PRETRAINED, num_classes=Config.NUM_CLASSES):
        super(BirdResNet, self).__init__()

        # Load Pretrained ResNet34
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet34(weights=weights)

        # Identify feature dimension (512 for ResNet34)
        self.in_features = self.backbone.fc.in_features

        # Remove original pooling and fully connected layers
        # We manually implement the forward pass for the backbone features
        del self.backbone.avgpool
        del self.backbone.fc

        # Attention Pooling Layer
        # Replaces Global Average Pooling
        self.att_pool = AttentionPooling(self.in_features)

        # Final Classifier
        self.classifier = nn.Linear(self.in_features, num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input spectrograms of shape (Batch, 3, Freq, Time).

        Returns:
            torch.Tensor: Logits of shape (Batch, NumClasses).
        """
        # Pass through ResNet backbone layers
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # Feature map shape: (Batch, 512, H, W)
        # H corresponds to Frequency, W corresponds to Time.

        # Pool the Frequency dimension (H)
        # We assume bird calls can happen at any frequency range captured by filters,
        # but we want to preserve Time (W) for attention.
        x = torch.mean(x, dim=2)  # Result: (Batch, 512, W)

        # Apply Attention Pooling over the Time dimension
        x = self.att_pool(x)  # Result: (Batch, 512)

        # Classification
        logits = self.classifier(x)  # Result: (Batch, NumClasses)

        return logits
