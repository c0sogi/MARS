import torch
import torch.nn as nn
import timm
from library.config import ModelConfig, AudioConfig


class AttentionPooling2D(nn.Module):
    """
    Single-Head 2D Attention Pooling layer.

    This layer computes a spatial attention map from the input feature map
    and performs a weighted sum pooling. It allows the model to focus on
    specific time-frequency regions (e.g., voice activity) while ignoring
    background noise.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        # 1x1 Convolution to project features to a single attention score map
        self.attention_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (Batch, Channels, Freq, Time)

        Returns:
            Pooled tensor of shape (Batch, Channels)
        """
        # x: (B, C, H, W)

        # Compute attention scores: (B, 1, H, W)
        attn_logits = self.attention_conv(x)

        # Flatten spatial dimensions: (B, 1, H*W)
        b, c, h, w = x.shape
        attn_logits_flat = attn_logits.view(b, 1, -1)

        # Apply Softmax over the spatial dimensions
        attn_weights_flat = torch.softmax(attn_logits_flat, dim=2)

        # Reshape back to (B, 1, H, W)
        attn_weights = attn_weights_flat.view(b, 1, h, w)

        # Apply attention weights and sum over spatial dimensions
        # (B, C, H, W) * (B, 1, H, W) -> (B, C, H, W) -> Sum -> (B, C)
        out = (x * attn_weights).sum(dim=(2, 3))

        return out


class AudioEfficientNetV2(nn.Module):
    """
    EfficientNetV2-B0 architecture adapted for Audio Spectrograms.

    Features:
    - Sum-Initialized Input Layer (RGB -> 1 Channel)
    - Single-Head 2D Attention Pooling
    - Dropout and Linear Classifier
    """

    def __init__(self, config: ModelConfig = None, num_classes: int = None):
        """
        Args:
            config: ModelConfig instance containing architecture hyperparameters.
            num_classes: Number of output classes. If None, loaded from AudioConfig.
        """
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config

        if num_classes is None:
            num_classes = AudioConfig().num_classes

        # 1. Load Pretrained Backbone
        # We use num_classes=0 and global_pool='' to retrieve the raw feature map
        # without the standard pooling and classification head.
        self.backbone = timm.create_model(
            config.model_name,
            pretrained=config.pretrained,
            num_classes=0,
            global_pool="",
            drop_rate=config.drop_rate,
            drop_path_rate=config.drop_path_rate,
        )

        # 2. Modify Input Layer
        # Adapt the first layer to accept 1-channel input (Spectrogram)
        self._modify_input_layer()

        # 3. Define Head
        # Get the number of output features from the backbone
        self.num_features = self.backbone.num_features

        # Custom Attention Pooling
        self.pool = AttentionPooling2D(self.num_features)

        # Dropout (using rate from config)
        self.dropout = nn.Dropout(p=config.drop_rate)

        # Final Classifier
        self.classifier = nn.Linear(self.num_features, num_classes)

    def _modify_input_layer(self):
        """
        Replaces the first convolutional layer (conv_stem) with a 1-channel version.
        Weights are initialized by summing the original RGB weights to preserve
        activation magnitudes (Sum-Initialization).
        """
        # Ensure the backbone has the expected structure
        if not hasattr(self.backbone, "conv_stem"):
            raise AttributeError(
                f"Backbone {self.config.model_name} does not have 'conv_stem'."
            )

        old_layer = self.backbone.conv_stem

        # Create a new Conv2d layer with in_channels=1
        # We keep all other parameters (out_channels, kernel, stride, padding) same
        new_layer = nn.Conv2d(
            in_channels=self.config.in_channels,  # 1
            out_channels=old_layer.out_channels,
            kernel_size=old_layer.kernel_size,
            stride=old_layer.stride,
            padding=old_layer.padding,
            bias=(old_layer.bias is not None),
        )

        # Initialize weights
        with torch.no_grad():
            # Old weights shape: (Out, 3, K, K)
            # Sum across the channel dimension (dim=1) -> (Out, 1, K, K)
            new_weight = old_layer.weight.sum(dim=1, keepdim=True)
            new_layer.weight.copy_(new_weight)

            # Copy bias if it exists
            if old_layer.bias is not None:
                new_layer.bias.copy_(old_layer.bias)

        # Replace the layer in the backbone
        self.backbone.conv_stem = new_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x: Input spectrograms of shape (Batch, 1, Freq, Time)

        Returns:
            logits: Class logits of shape (Batch, NumClasses)
        """
        # Extract features from backbone
        # Shape: (Batch, Channels, H, W)
        x = self.backbone(x)

        # Apply 2D Attention Pooling
        # Shape: (Batch, Channels)
        x = self.pool(x)

        # Apply Dropout
        x = self.dropout(x)

        # Classification
        # Shape: (Batch, NumClasses)
        x = self.classifier(x)

        return x
