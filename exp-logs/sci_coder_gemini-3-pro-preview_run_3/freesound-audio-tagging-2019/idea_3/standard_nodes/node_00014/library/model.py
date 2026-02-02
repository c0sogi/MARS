import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
from library.config import Config


class MultiHeadAttentionPooling(nn.Module):
    """
    Multi-Head Attention Pooling layer.
    Aggregates a sequence of features into a fixed-size vector using multiple attention heads.
    """

    def __init__(self, in_dim, num_heads):
        super(MultiHeadAttentionPooling, self).__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads

        # Learnable attention weights for each head: (num_heads, in_dim)
        # We use a linear layer to compute attention scores for all heads at once
        self.attn_linear = nn.Linear(in_dim, num_heads)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, in_dim)
        Returns:
            Pooled tensor of shape (batch_size, num_heads * in_dim)
        """
        # Compute attention scores: (batch_size, seq_len, num_heads)
        attn_scores = self.attn_linear(x)

        # Apply softmax over the sequence dimension (dim=1)
        attn_weights = F.softmax(attn_scores, dim=1)

        # Transpose to (batch_size, num_heads, seq_len) for matrix multiplication
        attn_weights = attn_weights.transpose(1, 2)

        # Compute weighted sum:
        # (batch_size, num_heads, seq_len) @ (batch_size, seq_len, in_dim)
        # -> (batch_size, num_heads, in_dim)
        weighted_sum = torch.bmm(attn_weights, x)

        # Flatten the heads: (batch_size, num_heads * in_dim)
        out = weighted_sum.view(x.size(0), -1)

        return out


class AudioClassifier(nn.Module):
    """
    CNN + Attention Pooling Architecture for Audio Tagging.
    Uses EfficientNet-B0 backbone and Attention Pooling.
    Cite Lesson 00010: Avoid Transformers on CNNs for this data size.
    """

    def __init__(self):
        super(AudioClassifier, self).__init__()

        # 1. Backbone: EfficientNet-B0
        weights = models.EfficientNet_B0_Weights.DEFAULT if Config.pretrained else None
        effnet = models.efficientnet_b0(weights=weights)

        # Modify the first convolution to accept 1 channel (spectrogram) instead of 3
        original_conv = effnet.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=Config.in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

        # Initialize new conv weights by summing the original RGB weights
        with torch.no_grad():
            new_conv.weight.data = original_conv.weight.data.sum(dim=1, keepdim=True)

        effnet.features[0][0] = new_conv
        self.backbone = effnet.features

        # Determine backbone output channels dynamically (Cite Lesson 00009)
        dummy_input = torch.randn(1, Config.in_channels, 128, 256)
        with torch.no_grad():
            features = self.backbone(dummy_input)
        backbone_out_channels = features.shape[1]

        # 2. Pooling Head (Attention Pooling)
        # Cite Lesson 00007: Attention Pooling over Global Max/Avg
        self.pooling = MultiHeadAttentionPooling(
            in_dim=backbone_out_channels, num_heads=Config.pooling_heads
        )

        # 3. Classifier
        classifier_in_dim = backbone_out_channels * Config.pooling_heads
        self.classifier = nn.Linear(classifier_in_dim, Config.num_classes)

    def forward(self, x):
        """
        Args:
            x: Input spectrogram of shape (batch, 1, n_mels, time)
        Returns:
            logits: (batch, num_classes)
        """
        # 1. Extract CNN Features
        # x shape: (N, 1, 128, T) -> (N, C, F', T')
        x = self.backbone(x)

        # 2. Pool Frequency Dimension
        # Average over the frequency dimension to get time sequence
        # x shape: (N, C, 1, T') -> (N, C, T')
        x = torch.mean(x, dim=2)

        # 3. Prepare for Pooling
        # Permute to (N, T', C)
        x = x.permute(0, 2, 1)

        # 4. Attention Pooling
        # Output: (N, H*C)
        x = self.pooling(x)

        # 5. Classifier
        # Output: (N, num_classes)
        logits = self.classifier(x)

        return logits
