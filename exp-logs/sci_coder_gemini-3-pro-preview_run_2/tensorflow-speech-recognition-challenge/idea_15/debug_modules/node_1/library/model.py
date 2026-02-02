import torch
import torch.nn as nn
import timm
from library.config import Config
from library.audio_frontend import DifferentiableFrontend


class SingleHeadAttentionPooling(nn.Module):
    """
    Implementation of 2D Spatial Attention Pooling.
    Projects features to an attention map, normalizes via Softmax, and computes
    a weighted sum of the features.
    """

    def __init__(self, in_features):
        super().__init__()
        # 1x1 Conv to compute a scalar attention score for each spatial position
        self.attention_gate = nn.Conv2d(
            in_channels=in_features, out_channels=1, kernel_size=1, bias=True
        )
        self.softmax = nn.Softmax(dim=2)  # Applied over flattened spatial dims

    def forward(self, x):
        """
        Args:
            x: Feature map of shape (Batch, Channels, Height, Width)
        Returns:
            Pooled feature vector of shape (Batch, Channels)
        """
        B, C, H, W = x.size()

        # 1. Compute attention scores: (B, C, H, W) -> (B, 1, H, W)
        attn_scores = self.attention_gate(x)

        # 2. Flatten spatial dimensions: (B, 1, H*W)
        attn_scores = attn_scores.view(B, 1, -1)

        # 3. Normalize scores across spatial locations
        attn_weights = self.softmax(attn_scores)

        # 4. Flatten input features: (B, C, H*W)
        x_flat = x.view(B, C, -1)

        # 5. Weighted Sum: sum(Features * Weights) -> (B, C)
        # (B, C, N) * (B, 1, N) -> Broadcast -> Sum over N
        out = (x_flat * attn_weights).sum(dim=2)

        return out


class EfficientNetV2Audio(nn.Module):
    """
    End-to-End Audio Classification Model.
    Pipeline: Raw Waveform -> GPU Frontend -> EfficientNetV2 Backbone -> Attention Pooling -> Classifier
    """

    def __init__(self, background_noise=None):
        super().__init__()

        # ==========================================
        # 1. GPU Frontend
        # ==========================================
        # Handles Augmentation (Noise, SpecAugment) and Feature Extraction (Log-Mel)
        self.frontend = DifferentiableFrontend(background_noise=background_noise)

        # ==========================================
        # 2. Backbone (EfficientNetV2-B0)
        # ==========================================
        # Load pretrained backbone, removing the default classifier and pooling
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=True,
            num_classes=0,  # Remove classifier
            global_pool="",  # Return feature maps (B, C, H, W)
        )

        # Modify the first layer (Stem) to accept 1-channel input (Spectrogram)
        # instead of 3-channel input (RGB Images).
        if hasattr(self.backbone, "conv_stem"):
            old_layer = self.backbone.conv_stem
            new_layer = nn.Conv2d(
                in_channels=1,
                out_channels=old_layer.out_channels,
                kernel_size=old_layer.kernel_size,
                stride=old_layer.stride,
                padding=old_layer.padding,
                bias=old_layer.bias is not None,
            )

            # Initialize weights by summing the original RGB weights
            # This preserves the magnitude of activations.
            with torch.no_grad():
                new_layer.weight.copy_(old_layer.weight.sum(dim=1, keepdim=True))
                if old_layer.bias is not None:
                    new_layer.bias.copy_(old_layer.bias)

            self.backbone.conv_stem = new_layer
        else:
            # Fallback/Safety check if architecture naming changes in timm
            raise AttributeError("Could not find 'conv_stem' in the backbone model.")

        # ==========================================
        # 3. Pooling Head
        # ==========================================
        self.num_features = self.backbone.num_features
        self.pooling = SingleHeadAttentionPooling(self.num_features)

        # ==========================================
        # 4. Classifier
        # ==========================================
        self.classifier = nn.Linear(self.num_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Args:
            x: Raw audio waveforms (Batch, Samples)
        Returns:
            Logits (Batch, Num_Classes)
        """
        # 1. Frontend: Waveform -> Log-Mel Spectrogram (Batch, 1, Freq, Time)
        x = self.frontend(x)

        # 2. Backbone: Spectrogram -> Feature Maps (Batch, C, H, W)
        x = self.backbone(x)

        # 3. Pooling: Feature Maps -> Feature Vector (Batch, C)
        x = self.pooling(x)

        # 4. Classification: Feature Vector -> Logits (Batch, Num_Classes)
        x = self.classifier(x)

        return x
