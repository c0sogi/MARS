import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchaudio
from library.config import ModelConfig


class MultiHeadAttentionPooling(nn.Module):
    """
    Multi-Head Attention Pooling layer.
    Aggregates a sequence of vectors into a single vector using a learnable query.
    Allows the model to attend to different parts of the sequence simultaneously.
    """

    def __init__(self, in_dim, num_heads):
        super().__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads

        if in_dim % num_heads != 0:
            raise ValueError(
                f"Input dimension {in_dim} must be divisible by num_heads {num_heads}"
            )

        self.head_dim = in_dim // num_heads

        # Projections for Keys and Values (Input sequence)
        self.W_k = nn.Linear(in_dim, in_dim)
        self.W_v = nn.Linear(in_dim, in_dim)

        # Learnable Query vector (shared across batch)
        # Shape: (num_heads, head_dim)
        self.query = nn.Parameter(torch.randn(num_heads, self.head_dim))

        # Output projection
        self.out_proj = nn.Linear(in_dim, in_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.normal_(self.query, mean=0, std=0.02)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input sequence of shape (Batch, Time, Dim)
        Returns:
            torch.Tensor: Pooled output of shape (Batch, Dim)
        """
        B, T, D = x.shape

        # Project Keys and Values
        K = self.W_k(x)  # (B, T, D)
        V = self.W_v(x)  # (B, T, D)

        # Reshape for multi-head attention: (B, T, H, d_head)
        K = K.view(B, T, self.num_heads, self.head_dim)
        V = V.view(B, T, self.num_heads, self.head_dim)

        # Prepare Query: Expand learnable query to batch size
        # Q: (1, 1, H, d_head) -> (B, 1, H, d_head)
        Q = self.query.view(1, 1, self.num_heads, self.head_dim).expand(B, -1, -1, -1)

        # Compute Attention Scores: Q * K^T
        # Q: (B, 1, H, d), K: (B, T, H, d)
        # Result: (B, 1, H, T) -> Squeeze to (B, H, T)
        scores = torch.einsum("bqhd,bthd->bhqt", Q, K).squeeze(2)

        # Scale scores
        scores = scores / (self.head_dim**0.5)

        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)  # (B, H, T)

        # Apply weights to Values
        # Weights: (B, H, T), V: (B, T, H, d)
        # Result: (B, H, d)
        context = torch.einsum("bht,bthd->bhd", attn_weights, V)

        # Concatenate heads: (B, H * d) -> (B, D)
        context = context.reshape(B, D)

        # Final projection
        output = self.out_proj(context)

        return output


class SKResNetConformer(nn.Module):
    """
    Hybrid Architecture:
    1. Backbone: SK-ResNet34 (Pretrained, modified strides for temporal resolution)
    2. Neck: Conformer Encoder (Captures global and local context)
    3. Head: Multi-Head Attention Pooling + Linear Classifier
    """

    def __init__(self):
        super().__init__()

        # --- 1. Backbone ---
        # Load pretrained SK-ResNet34
        self.backbone = timm.create_model(
            ModelConfig.model_name,
            pretrained=ModelConfig.pretrained,
            in_chans=ModelConfig.in_channels,
            features_only=False,
        )

        # Modification A: Remove MaxPool in Stem
        # Standard ResNet stem downsamples by 4 (Conv s2 + MaxPool s2).
        # We replace MaxPool with Identity to keep higher resolution (Downsample by 2 only).
        self.backbone.maxpool = nn.Identity()

        # Modification B: Remove Stride in Layer 3 and Layer 4
        # This prevents the feature map from becoming too small in the time dimension.
        self._remove_stride(self.backbone.layer3)
        self._remove_stride(self.backbone.layer4)

        # Remove original classification head
        self.backbone.global_pool = nn.Identity()
        self.backbone.fc = nn.Identity()

        # --- 2. Neck (Conformer) ---
        self.conformer = torchaudio.models.Conformer(
            input_dim=ModelConfig.backbone_out_dim,
            num_heads=ModelConfig.conformer_heads,
            ffn_dim=ModelConfig.backbone_out_dim * 4,
            num_layers=ModelConfig.conformer_layers,
            depthwise_conv_kernel_size=ModelConfig.conformer_kernel_size,
            dropout=ModelConfig.conformer_dropout,
        )

        # --- 3. Head ---
        self.pooling = MultiHeadAttentionPooling(
            in_dim=ModelConfig.conformer_dim, num_heads=ModelConfig.pooling_heads
        )

        self.classifier = nn.Linear(ModelConfig.conformer_dim, ModelConfig.num_classes)

    def _remove_stride(self, layer):
        """
        Recursively sets the stride of all Conv2d layers in the given module to (1, 1).
        This effectively removes downsampling from residual blocks.
        """
        for module in layer.modules():
            if isinstance(module, nn.Conv2d):
                module.stride = (1, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input spectrograms (B, 3, F, T)
        Returns:
            torch.Tensor: Logits (B, num_classes)
        """
        # --- Backbone Forward ---
        # Use forward_features to get the output of the last convolutional layer
        # Output shape: (B, C, F', T')
        x = self.backbone.forward_features(x)

        # --- Frequency Pooling ---
        # Collapse the frequency dimension, preserving time and channels
        # (B, C, F', T') -> (B, C, 1, T') -> (B, C, T')
        x = torch.mean(x, dim=2)

        # Prepare for Conformer: (B, Time, Dim)
        x = x.permute(0, 2, 1)

        # --- Conformer Forward ---
        # Create lengths tensor (all samples are padded to full length)
        B, T, D = x.shape
        lengths = torch.full((B,), T, device=x.device, dtype=torch.long)

        # Conformer returns (output, lengths)
        x, _ = self.conformer(x, lengths)

        # --- Pooling & Classification ---
        # Aggregate temporal sequence: (B, T, D) -> (B, D)
        x = self.pooling(x)

        # Final logits
        logits = self.classifier(x)

        return logits
