import torch
import torch.nn as nn
import timm
from library.config import Config
from library.custom_layers import (
    GPUNoiseInjector,
    DifferentiableSpectrogram,
    SpecAugment,
    AttentionPooling,
)


class EfficientNetV2Audio(nn.Module):
    """
    EfficientNetV2-B0 based Audio Classification Model.

    This model implements an end-to-end pipeline that ingests raw waveforms,
    performs GPU-accelerated feature extraction and augmentation, and utilizes
    a pre-trained vision backbone for classification.
    """

    def __init__(
        self, background_noise=None, num_classes=Config.NUM_CLASSES, pretrained=True
    ):
        """
        Args:
            background_noise (torch.Tensor, optional): 1D Tensor containing background noise
                                                       for on-the-fly augmentation.
            num_classes (int): Number of target classes.
            pretrained (bool): Whether to load ImageNet weights for the backbone.
        """
        super().__init__()

        # ==========================================
        # 1. GPU-Native Front-End
        # ==========================================
        # Noise Injection: Mixes background noise into waveforms
        # If no noise tensor is provided (e.g., during inference), use an empty tensor.
        if background_noise is None:
            background_noise = torch.empty(0)

        self.noise_injector = GPUNoiseInjector(background_noise)

        # Spectrogram Generation: Waveform -> Log-Mel Spectrogram + Instance Norm
        self.spectrogram = DifferentiableSpectrogram()

        # SpecAugment: Frequency and Time Masking (Training only)
        self.spec_augment = SpecAugment()

        # ==========================================
        # 2. Backbone (EfficientNetV2-B0)
        # ==========================================
        # We use 'tf_efficientnetv2_b0' as it is a robust identifier in timm
        # for the EfficientNetV2-B0 architecture.
        model_name = "tf_efficientnetv2_b0"

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove the default classification head
            global_pool="",  # Return spatial feature maps (B, C, H, W)
        )

        # Adapt First Layer (RGB 3-channel -> Mono 1-channel)
        # The stem convolution in timm's EfficientNetV2 is named 'conv_stem'.
        if hasattr(self.backbone, "conv_stem"):
            first_conv = self.backbone.conv_stem

            # Create a new 1-channel convolution with the same parameters
            new_conv = nn.Conv2d(
                in_channels=Config.IN_CHANNELS,  # 1
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None,
            )

            # Initialize weights: Sum over the RGB channels of the pretrained weights
            # This preserves the scale of activations for the 1-channel input.
            with torch.no_grad():
                new_conv.weight[:] = first_conv.weight.sum(dim=1, keepdim=True)
                if first_conv.bias is not None:
                    new_conv.bias[:] = first_conv.bias

            self.backbone.conv_stem = new_conv
        else:
            raise AttributeError(
                f"Backbone {model_name} does not have 'conv_stem'. Check timm version."
            )

        # ==========================================
        # 3. Head (Attention Pooling + Classifier)
        # ==========================================
        # Retrieve the number of output features from the backbone (usually 1280 for B0)
        num_features = self.backbone.num_features

        # 2D Attention Pooling acts as a learned Voice Activity Detector
        self.pool = AttentionPooling(num_features)

        self.dropout = nn.Dropout(Config.DROPOUT)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Raw waveforms of shape (Batch, Time).

        Returns:
            torch.Tensor: Class logits of shape (Batch, NumClasses).
        """
        # 1. Spectrogram Generation
        # Input: (B, T) -> Output: (B, 1, F, T)
        x = self.spectrogram(x)

        # 3. SpecAugment (Train only)
        # Input: (B, 1, F, T) -> Output: (B, 1, F, T)
        x = self.spec_augment(x)

        # 4. Backbone Feature Extraction
        # Input: (B, 1, F, T) -> Output: (B, C, H, W)
        x = self.backbone(x)

        # 5. Attention Pooling
        # Aggregates spatial features into a global embedding
        # Input: (B, C, H, W) -> Output: (B, C)
        x = self.pool(x)

        # 6. Classifier
        x = self.dropout(x)
        x = self.classifier(x)

        return x
