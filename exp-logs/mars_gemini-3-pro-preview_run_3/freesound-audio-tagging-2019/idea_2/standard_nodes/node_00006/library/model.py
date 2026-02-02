import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class AttentionPooling(nn.Module):
    """
    Learnable Attention Pooling module.
    Aggregates features over time/frequency dimensions using a weighted sum,
    where weights are computed via a small learnable network.
    """

    def __init__(self, in_dim, hidden_dim=None):
        super(AttentionPooling, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim // 2

        self.attention_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # Input x shape: (Batch, Channels, Freq, Time)
        B, C, H, W = x.size()

        # Flatten spatial dimensions to create a sequence of features
        # (B, C, H, W) -> (B, C, H*W)
        x = x.view(B, C, -1)

        # Transpose to (B, Sequence_Length, Channels) for the Linear layer
        x_t = x.permute(0, 2, 1)

        # Calculate attention scores
        # (B, Seq, C) -> (B, Seq, 1)
        attn_scores = self.attention_net(x_t)

        # Apply Softmax over the sequence dimension to get probability weights
        attn_weights = torch.softmax(attn_scores, dim=1)

        # Weighted sum of features
        # Permute weights to (B, 1, Seq) to broadcast against (B, C, Seq)
        attn_weights = attn_weights.permute(0, 2, 1)

        # (B, C, Seq) * (B, 1, Seq) -> (B, C, Seq) -> Sum over Seq -> (B, C)
        pooled_features = torch.sum(x * attn_weights, dim=2)

        return pooled_features


class AudioClassifier(nn.Module):
    """
    Audio Classification Model using EfficientNet-B0 backbone and Attention Pooling.
    """

    def __init__(self, num_classes=Config.num_classes, pretrained=Config.pretrained):
        super(AudioClassifier, self).__init__()

        # 1. Load Backbone
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        # 2. Modify Input Layer
        # EfficientNet expects 3 channels (RGB), but spectrograms are 1 channel.
        # We replace the first Conv2d layer.
        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=(original_conv.bias is not None),
        )

        # Initialize the new 1-channel weights by summing the original 3-channel weights.
        # This helps preserve the distribution of activations from the pretrained model.
        if pretrained:
            with torch.no_grad():
                self.backbone.features[0][0].weight.data = (
                    original_conv.weight.data.sum(dim=1, keepdim=True)
                )

        # 3. Define Architecture Components
        # EfficientNet-B0 outputs 1280 channels at the final feature map
        self.feature_dim = 1280

        # Replace standard pooling with Attention Pooling
        self.pooling = AttentionPooling(self.feature_dim)

        # Final Classification Head
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        # Input: (Batch, 1, Freq, Time)

        # Extract features using the backbone (excluding original pool/fc)
        # Output: (Batch, 1280, F', T')
        x = self.backbone.features(x)

        # Apply Attention Pooling to aggregate temporal/spectral features
        # Output: (Batch, 1280)
        x = self.pooling(x)

        # Predict classes
        # Output: (Batch, num_classes)
        logits = self.classifier(x)

        return logits
