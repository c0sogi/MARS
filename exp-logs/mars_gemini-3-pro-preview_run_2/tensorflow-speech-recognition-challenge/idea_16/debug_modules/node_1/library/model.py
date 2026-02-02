import torch
import torch.nn as nn
import timm
from library.config import Config
from library.audio_frontend import DifferentiableFrontend


class SingleHeadAttentionPooling2D(nn.Module):
    """
    2D Attention Pooling Layer.
    Computes a spatial attention map (over Time and Frequency dimensions)
    to aggregate features into a single vector.
    """

    def __init__(self, in_channels):
        super().__init__()
        # 1x1 Conv to compute attention score for each spatial location
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Flatten(),  # Flatten spatial dims (H*W)
            nn.Softmax(dim=1),  # Normalize scores over spatial dims
        )

    def forward(self, x):
        """
        Args:
            x: Feature map of shape (Batch, Channels, Freq, Time)

        Returns:
            Global feature vector of shape (Batch, Channels)
        """
        B, C, H, W = x.size()

        # Compute attention scores: (B, H*W)
        attn_scores = self.attention(x)

        # Reshape for broadcasting: (B, 1, H*W)
        attn_scores = attn_scores.unsqueeze(1)

        # Flatten input features: (B, C, H*W)
        x_flat = x.view(B, C, -1)

        # Weighted sum of features: (B, C)
        out = torch.sum(x_flat * attn_scores, dim=2)

        return out


class EfficientNetV2Speech(nn.Module):
    """
    End-to-End Speech Command Recognition Model.

    Architecture:
    1. DifferentiableFrontend: Waveform -> Spectrogram + Augmentation (GPU)
    2. EfficientNetV2-B0 Backbone: Feature Extraction
    3. SingleHeadAttentionPooling2D: Spatial Aggregation
    4. Linear Classifier: Prediction
    """

    def __init__(self, pretrained=True):
        super().__init__()

        # ==========================================
        # 1. GPU Frontend
        # ==========================================
        self.frontend = DifferentiableFrontend()

        # ==========================================
        # 2. Backbone (EfficientNetV2-B0)
        # ==========================================
        # Load model without classifier (num_classes=0) and without global pooling
        # to preserve spatial dimensions for attention pooling.
        self.backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )

        # ==========================================
        # 3. Input Adaptation (3-channel -> 1-channel)
        # ==========================================
        # The backbone expects RGB images. We modify the first convolution (conv_stem)
        # to accept 1-channel spectrograms.
        if hasattr(self.backbone, "conv_stem"):
            old_layer = self.backbone.conv_stem

            # Create new layer with in_channels=1
            new_layer = nn.Conv2d(
                in_channels=1,
                out_channels=old_layer.out_channels,
                kernel_size=old_layer.kernel_size,
                stride=old_layer.stride,
                padding=old_layer.padding,
                bias=old_layer.bias is not None,
            )

            # Sum-Initialization: Initialize weights by summing RGB channels.
            # This preserves the magnitude of activations.
            with torch.no_grad():
                new_layer.weight.copy_(old_layer.weight.sum(dim=1, keepdim=True))
                if old_layer.bias is not None:
                    new_layer.bias.copy_(old_layer.bias)

            self.backbone.conv_stem = new_layer
        else:
            raise AttributeError(
                f"Backbone {Config.MODEL_NAME} does not have 'conv_stem'. Structure mismatch."
            )

        # ==========================================
        # 4. Pooling and Classifier
        # ==========================================
        # Retrieve feature dimension (1280 for EfficientNetV2-B0)
        self.num_features = self.backbone.num_features

        self.pool = SingleHeadAttentionPooling2D(self.num_features)
        self.fc = nn.Linear(self.num_features, Config.NUM_CLASSES)

    def forward(self, x, noise_bank=None):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Raw waveforms (Batch, Time).
            noise_bank (list, optional): Background noise for augmentation in Frontend.

        Returns:
            torch.Tensor: Logits (Batch, Num_Classes).
        """
        # 1. Frontend: Waveform -> Spectrogram (B, 1, F, T)
        # Includes augmentation (Noise Mixing, SpecAugment) if training
        x = self.frontend(x, noise_bank=noise_bank)

        # 2. Backbone: Spectrogram -> Feature Map (B, C, H, W)
        x = self.backbone(x)

        # 3. Attention Pooling: Feature Map -> Global Vector (B, C)
        x = self.pool(x)

        # 4. Classifier: Global Vector -> Logits (B, Num_Classes)
        x = self.fc(x)

        return x
