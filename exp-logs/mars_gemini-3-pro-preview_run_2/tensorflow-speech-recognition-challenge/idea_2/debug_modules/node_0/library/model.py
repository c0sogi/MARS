import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from library.config import Config


class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        """
        Learnable Attention Pooling Module.

        Computes a weighted average of features across spatial/temporal dimensions,
        allowing the model to focus on relevant parts of the spectrogram (e.g., the spoken word)
        while suppressing silence or background noise.

        Args:
            in_dim (int): The number of input channels (feature dimension).
        """
        super(AttentionPooling, self).__init__()
        # A small MLP to compute attention scores from features
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2), nn.Tanh(), nn.Linear(in_dim // 2, 1)
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input feature map of shape (Batch, Channels, Height, Width).

        Returns:
            Tensor: Pooled feature vector of shape (Batch, Channels).
        """
        B, C, H, W = x.size()

        # Flatten spatial dimensions: (B, C, H, W) -> (B, C, H*W)
        # Permute to (B, H*W, C) so the Linear layer applies to the channel dimension
        x_flat = x.view(B, C, -1).permute(0, 2, 1)

        # Compute attention scores for each spatial location: (B, H*W, 1)
        scores = self.attention(x_flat)

        # Normalize scores using Softmax to get probability distribution over spatial locations
        weights = torch.softmax(scores, dim=1)

        # Compute weighted sum of features:
        # (B, H*W, C) * (B, H*W, 1) -> (B, H*W, C)
        # Sum over the spatial dimension (dim=1) -> (B, C)
        context = torch.sum(x_flat * weights, dim=1)

        return context


class AudioEfficientNet(nn.Module):
    def __init__(self, num_classes=Config.NUM_CLASSES):
        """
        Audio Classification Model using EfficientNet-B0 backbone and Attention Pooling.

        Args:
            num_classes (int): Number of output classes.
        """
        super(AudioEfficientNet, self).__init__()

        # Load EfficientNet-B0 with ImageNet weights
        # We use the string identifier for compatibility with recent torchvision versions
        self.backbone = efficientnet_b0(weights="IMAGENET1K_V1")

        # 1. Modify the first convolutional layer
        # Standard EfficientNet takes 3-channel RGB images. Spectrograms are 1-channel.
        # We replace the first Conv2d layer with one that accepts 1 input channel.
        original_conv = self.backbone.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )

        # Initialize the new weights by summing the original weights across the channel dimension.
        # This preserves the magnitude and structure of the learned filters.
        with torch.no_grad():
            new_conv.weight.copy_(original_conv.weight.sum(dim=1, keepdim=True))

        self.backbone.features[0][0] = new_conv

        # 2. Determine Feature Dimension
        # EfficientNet-B0 outputs 1280 channels at the final feature map
        self.feature_dim = 1280

        # 3. Attention Pooling Head
        # Replaces the standard Global Average Pooling
        self.attention_pooling = AttentionPooling(self.feature_dim)

        # 4. Classifier
        # Standard linear projection to class logits
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2), nn.Linear(self.feature_dim, num_classes)
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (Tensor): Input tensor of shape (Batch, 1, Freq, Time).

        Returns:
            Tensor: Logits of shape (Batch, Num_Classes).
        """
        # Extract features using the backbone: (Batch, 1280, H, W)
        x = self.backbone.features(x)

        # Apply Attention Pooling to aggregate features: (Batch, 1280)
        x = self.attention_pooling(x)

        # Predict classes: (Batch, Num_Classes)
        x = self.classifier(x)

        return x
