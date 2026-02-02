import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from library.config import NUM_CLASSES


class SingleHeadAttention(nn.Module):
    """
    Single-Head Attention Pooling layer.
    Aggregates spatial/temporal features using a learned attention mechanism.
    Acts as a 'Voice Activity Detector' to focus on relevant parts of the clip
    while suppressing background noise or silence.
    """

    def __init__(self, in_channels, hidden_channels=None):
        super().__init__()
        # Use a bottleneck ratio for the hidden layer (standard practice is reduction by 8 or 16)
        if hidden_channels is None:
            hidden_channels = in_channels // 8

        # Attention mechanism:
        # 1. Project features to a lower dimensional space
        # 2. Apply non-linearity (Tanh)
        # 3. Project to a single scalar score per spatial location
        self.attention_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.Tanh(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Flatten(start_dim=2),  # Output shape: (Batch, 1, H*W)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: Input feature map of shape (Batch, Channels, Height, Width)
        Returns:
            Global feature vector of shape (Batch, Channels)
        """
        B, C, H, W = x.size()

        # 1. Compute Attention Scores
        # The network learns which spatial locations (time/freq) are important
        attn_scores = self.attention_net(x)  # (B, 1, H*W)

        # 2. Normalize scores to sum to 1 across the spatial/temporal grid
        attn_weights = self.softmax(attn_scores)  # (B, 1, H*W)

        # 3. Flatten input features to match attention map
        x_flat = x.view(B, C, -1)  # (B, C, H*W)

        # 4. Weighted Sum
        # Perform batched matrix multiplication: (B, C, N) @ (B, N, 1) -> (B, C, 1)
        # We permute weights to (B, H*W, 1) for the multiplication.
        out = torch.bmm(x_flat, attn_weights.permute(0, 2, 1))

        return out.squeeze(-1)  # (B, C)


class AudioEfficientNet(nn.Module):
    """
    EfficientNet-B0 based model for Audio Classification.

    Architecture:
    - Input: 3-Channel Audio Image (Spectrogram, Delta, Delta-Delta)
    - Backbone: EfficientNet-B0 (Pretrained on ImageNet)
    - Pooling: Single-Head Attention Pooling (Learned localization)
    - Head: Linear Classification Layer
    """

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()

        # 1. Load Pretrained Backbone
        # We use ImageNet weights. The 3-channel audio input (RGB-like) allows
        # us to leverage the pretrained texture and pattern detectors directly.
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)

        # 2. Extract Feature Extractor
        # EfficientNet-B0 structure consists of: features -> avgpool -> classifier
        # We discard the original pooling and classifier, keeping only the convolutional features.
        self.features = self.backbone.features

        # EfficientNet-B0 outputs 1280 channels at the final convolutional layer
        self.out_channels = 1280

        # 3. Define Custom Head
        # Replace global average pooling with Attention Pooling to better handle
        # short commands within the 1-second window.
        self.attention_pool = SingleHeadAttention(self.out_channels)

        # Final classification layer
        self.classifier = nn.Linear(self.out_channels, num_classes)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, 3, n_mels, time)
               Channels correspond to [LogMel, Delta, DeltaDelta]
        Returns:
            Logits of shape (Batch, num_classes)
        """
        # 1. Backbone Feature Extraction
        # Input: (B, 3, H, W) -> Output: (B, 1280, H', W')
        # For 64 mels and ~100 time steps, output spatial dims will be small (e.g., 2x4),
        # but rich in semantic information.
        x = self.features(x)

        # 2. Attention Pooling
        # Aggregates features over time and frequency, focusing on the command.
        # Output: (B, 1280)
        x = self.attention_pool(x)

        # 3. Classification
        # Output: (B, num_classes)
        x = self.classifier(x)

        return x
