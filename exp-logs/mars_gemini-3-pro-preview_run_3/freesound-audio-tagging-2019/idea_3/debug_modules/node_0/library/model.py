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
    Context-Aware Hybrid Architecture for Audio Tagging.
    Combines EfficientNet-B2 backbone, Transformer Encoder, and Multi-Head Attention Pooling.
    """

    def __init__(self):
        super(AudioClassifier, self).__init__()

        # 1. Backbone: EfficientNet-B2
        # We use the features only, removing the original classifier
        weights = models.EfficientNet_B2_Weights.DEFAULT if Config.pretrained else None
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
        # This preserves the magnitude of the activation
        with torch.no_grad():
            new_conv.weight.data = original_conv.weight.data.sum(dim=1, keepdim=True)

        effnet.features[0][0] = new_conv
        self.backbone = effnet.features

        # Determine backbone output channels dynamically
        # EfficientNet-B2 typically outputs 1408 channels
        dummy_input = torch.randn(1, Config.in_channels, 128, 256)
        with torch.no_grad():
            features = self.backbone(dummy_input)
        backbone_out_channels = features.shape[1]

        # 2. Projection Layer
        # Maps CNN feature dimension to Transformer hidden dimension
        self.projection = nn.Linear(
            backbone_out_channels, Config.transformer_hidden_dim
        )
        self.layer_norm = nn.LayerNorm(Config.transformer_hidden_dim)
        self.dropout = nn.Dropout(Config.transformer_dropout)

        # 3. Positional Encoding
        # Learnable positional embeddings.
        # We assume a max sequence length. With 30s audio and stride 32, seq len is ~81.
        # We set max_len to 200 to be safe.
        self.max_len = 200
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.max_len, Config.transformer_hidden_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.transformer_hidden_dim,
            nhead=Config.transformer_heads,
            dim_feedforward=Config.transformer_hidden_dim * 4,
            dropout=Config.transformer_dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.transformer_layers
        )

        # 5. Pooling Head
        self.pooling = MultiHeadAttentionPooling(
            in_dim=Config.transformer_hidden_dim, num_heads=Config.pooling_heads
        )

        # 6. Classifier
        # Input dimension is hidden_dim * num_pooling_heads
        classifier_in_dim = Config.transformer_hidden_dim * Config.pooling_heads
        self.classifier = nn.Linear(classifier_in_dim, Config.num_classes)

    def forward(self, x):
        """
        Args:
            x: Input spectrogram of shape (batch, 1, n_mels, time)
        Returns:
            logits: (batch, num_classes)
        """
        # x shape: (N, 1, 128, T)

        # 1. Extract CNN Features
        # Output shape: (N, C, F', T') -> e.g., (N, 1408, 4, 80) for B2
        x = self.backbone(x)

        # 2. Pool Frequency Dimension
        # We want to treat this as a sequence over time.
        # Average over the frequency dimension (H)
        # x shape: (N, C, 1, T')
        x = torch.mean(x, dim=2)

        # 3. Prepare for Transformer
        # Permute to (N, T', C)
        x = x.permute(0, 2, 1)

        # Project to hidden dimension
        x = self.projection(x)
        x = self.layer_norm(x)

        # 4. Add Positional Embeddings
        seq_len = x.shape[1]
        # Slice positional embeddings to match sequence length
        if seq_len > self.max_len:
            # In case sequence is longer than expected (unlikely with fixed input), interpolate
            pos_embed = F.interpolate(
                self.pos_embed.permute(0, 2, 1), size=seq_len, mode="linear"
            ).permute(0, 2, 1)
        else:
            pos_embed = self.pos_embed[:, :seq_len, :]

        x = x + pos_embed
        x = self.dropout(x)

        # 5. Transformer Encoder
        # Output: (N, T', E)
        x = self.transformer(x)

        # 6. Pooling
        # Output: (N, H*E)
        x = self.pooling(x)

        # 7. Classifier
        # Output: (N, num_classes)
        logits = self.classifier(x)

        return logits
