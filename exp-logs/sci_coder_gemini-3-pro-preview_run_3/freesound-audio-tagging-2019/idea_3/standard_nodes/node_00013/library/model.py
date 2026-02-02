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
    CNN Backbone with Multi-Head Attention Pooling.
    Simplified architecture removing Transformer for better stability on mid-sized datasets.
    """

    def __init__(self):
        super(AudioClassifier, self).__init__()

        # 1. Backbone: EfficientNet-B0
        # Cite solution_lesson_node_00010: Standard CNNs are more robust for mid-sized datasets.
        if Config.backbone_name == "efficientnet_b0":
            weights = (
                models.EfficientNet_B0_Weights.DEFAULT if Config.pretrained else None
            )
            effnet = models.efficientnet_b0(weights=weights)
        else:
            # Fallback
            weights = (
                models.EfficientNet_B2_Weights.DEFAULT if Config.pretrained else None
            )
            effnet = models.efficientnet_b2(weights=weights)

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

        # Determine backbone output channels dynamically
        # Cite solution_lesson_node_00009: Decouple custom heads by dynamically retrieving dimensions.
        dummy_input = torch.randn(1, Config.in_channels, 128, 256)
        with torch.no_grad():
            features = self.backbone(dummy_input)
        backbone_out_channels = features.shape[1]

        # 2. Projection Layer
        self.projection = nn.Linear(backbone_out_channels, Config.hidden_dim)
        self.layer_norm = nn.LayerNorm(Config.hidden_dim)
        self.dropout = nn.Dropout(0.2)

        # 3. Pooling Head
        # Cite solution_lesson_node_00007: Attention Pooling handles multi-label temporal events better than Max/Avg.
        self.pooling = MultiHeadAttentionPooling(
            in_dim=Config.hidden_dim, num_heads=Config.pooling_heads
        )

        # 4. Classifier
        classifier_in_dim = Config.hidden_dim * Config.pooling_heads
        self.classifier = nn.Linear(classifier_in_dim, Config.num_classes)

    def forward(self, x):
        """
        Args:
            x: Input spectrogram of shape (batch, 1, n_mels, time)
        Returns:
            logits: (batch, num_classes)
        """
        # 1. Extract CNN Features
        x = self.backbone(x)

        # 2. Pool Frequency Dimension (Average)
        # x shape: (N, C, F, T) -> (N, C, 1, T)
        x = torch.mean(x, dim=2)

        # 3. Prepare for Sequence Processing
        # Permute to (N, T, C)
        x = x.permute(0, 2, 1)

        # 4. Projection
        x = self.projection(x)
        x = self.layer_norm(x)
        x = self.dropout(x)

        # 5. Pooling (Attention)
        x = self.pooling(x)

        # 6. Classifier
        logits = self.classifier(x)

        return logits
