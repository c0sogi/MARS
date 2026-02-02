import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class AttentionPool(nn.Module):
    """
    Attention Pooling Layer.
    Learns to weight the spatial/temporal features (H x W) adaptively before aggregation.
    """

    def __init__(self, in_channels, hidden_dim):
        super(AttentionPool, self).__init__()
        # 1x1 convolution acts as a linear projection for each spatial location
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Feature map of shape (Batch, Channels, Height, Width)

        Returns:
            torch.Tensor: Pooled feature vector of shape (Batch, Channels)
        """
        # x: (B, C, H, W)

        # Calculate attention scores
        # Project to hidden dimension and apply non-linearity
        attn = torch.tanh(self.conv1(x))  # (B, hidden_dim, H, W)
        # Project to scalar score
        attn = self.conv2(attn)  # (B, 1, H, W)

        # Flatten spatial dimensions to apply Softmax over the entire map
        b, c, h, w = x.size()
        attn = attn.view(b, 1, -1)  # (B, 1, H*W)
        attn = F.softmax(attn, dim=-1)  # Softmax over H*W

        # Apply attention weights to the input features
        x_flat = x.view(b, c, -1)  # (B, C, H*W)

        # Weighted sum: (B, C, N) @ (B, N, 1) -> (B, C, 1)
        # We transpose attn to (B, H*W, 1)
        out = torch.bmm(x_flat, attn.transpose(1, 2))

        # Remove the last dimension
        out = out.squeeze(-1)  # (B, C)

        return out


class BirdResNet(nn.Module):
    """
    ResNet-34 based model for Bird Species Classification.
    Uses Attention Pooling to handle temporal dynamics in spectrograms.
    """

    def __init__(self, pretrained=True, num_classes=Config.NUM_CLASSES):
        super(BirdResNet, self).__init__()

        # Load Pretrained ResNet-34
        # We use the updated weights enum if available, or boolean for older versions
        try:
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            base_model = models.resnet34(weights=weights)
        except AttributeError:
            base_model = models.resnet34(pretrained=pretrained)

        # Remove the original pooling (avgpool) and fully connected (fc) layers
        # ResNet components: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4
        layers = list(base_model.children())[:-2]
        self.backbone = nn.Sequential(*layers)

        # ResNet-34 final feature map depth is 512
        self.in_features = base_model.fc.in_features

        # Custom Attention Pooling
        self.pool = AttentionPool(self.in_features, Config.ATTENTION_HIDDEN_DIM)

        # Classification Head
        # Includes BatchNorm and Dropout for regularization
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.in_features),
            nn.Dropout(0.5),
            nn.Linear(self.in_features, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input spectrograms (Batch, 3, Freq, Time)

        Returns:
            torch.Tensor: Logits (Batch, NumClasses)
        """
        # Extract features using backbone
        # Output shape: (Batch, 512, F', T')
        x = self.backbone(x)

        # Apply Attention Pooling
        # Output shape: (Batch, 512)
        x = self.pool(x)

        # Classification
        # Output shape: (Batch, NumClasses)
        logits = self.classifier(x)

        return logits
