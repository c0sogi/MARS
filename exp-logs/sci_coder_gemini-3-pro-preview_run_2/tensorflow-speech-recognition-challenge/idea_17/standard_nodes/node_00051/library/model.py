import torch
import torch.nn as nn
import torchaudio
import timm
from library.config import (
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    WIN_LENGTH,
    N_MELS,
    F_MIN,
    F_MAX,
    MODEL_NAME,
    NUM_CLASSES,
    TIME_MASK_PARAM,
    FREQ_MASK_PARAM,
    MASK_PROB,
    DROP_RATE,
    DROP_PATH_RATE,
)


class LogMelFrontEnd(nn.Module):
    """
    Differentiable Front-End for converting raw waveforms to Log-Mel Spectrograms.
    Includes Instance Normalization to align with vision backbone statistics.
    """

    def __init__(self):
        super().__init__()
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=F_MIN,
            f_max=F_MAX,
            power=2.0,
            normalized=False,
        )
        # Instance Norm to normalize per-sample (B, 1, F, T)
        self.instance_norm = nn.InstanceNorm2d(1, affine=True)

    def forward(self, waveforms):
        """
        Args:
            waveforms (torch.Tensor): Raw audio (B, T)
        Returns:
            torch.Tensor: Normalized Log-Mel Spectrograms (B, 1, F, T)
        """
        # Generate Mel Spectrogram: (B, F, T)
        x = self.mel_spectrogram(waveforms)

        # Logarithm (Log-Mel): Add epsilon for numerical stability
        x = torch.log(x + 1e-9)

        # Add channel dimension: (B, 1, F, T)
        x = x.unsqueeze(1)

        # Apply Instance Normalization
        x = self.instance_norm(x)

        return x


class SpecAugment(nn.Module):
    """
    Applies Time and Frequency Masking for data augmentation.
    Active only during training with a specified probability.
    """

    def __init__(self):
        super().__init__()
        self.time_masking = torchaudio.transforms.TimeMasking(
            time_mask_param=TIME_MASK_PARAM
        )
        self.freq_masking = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=FREQ_MASK_PARAM
        )
        self.prob = MASK_PROB

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Spectrograms (B, 1, F, T)
        """
        if not self.training:
            return x

        # Apply masking probabilistically per batch or sample
        # Here we apply to the batch if a random draw < prob
        if torch.rand(1).item() < self.prob:
            # torchaudio transforms expect (..., F, T)
            # We work on the last two dimensions
            x = self.freq_masking(x)
            x = self.time_masking(x)

        return x


class SingleHeadAttentionPooling(nn.Module):
    """
    2D Attention Pooling to aggregate features spatially.
    Acts as a soft Voice Activity Detector (VAD).
    """

    def __init__(self, in_features):
        super().__init__()
        # 1x1 Conv to compute attention scores
        self.attn_conv = nn.Conv2d(in_features, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Feature maps (B, C, H, W)
        Returns:
            torch.Tensor: Pooled features (B, C)
        """
        B, C, H, W = x.shape

        # Compute attention scores: (B, 1, H, W)
        attn_logits = self.attn_conv(x)

        # Flatten spatial dimensions: (B, 1, H*W)
        attn_logits = attn_logits.view(B, 1, -1)

        # Normalize scores
        attn_weights = self.softmax(attn_logits)

        # Flatten input features: (B, C, H*W)
        x_flat = x.view(B, C, -1)

        # Weighted sum: (B, C, H*W) @ (B, H*W, 1) -> (B, C, 1)
        # Transpose weights to (B, H*W, 1) for matmul
        pooled = torch.bmm(x_flat, attn_weights.transpose(1, 2))

        # Remove last dimension: (B, C)
        return pooled.squeeze(2)


class AudioEfficientNetV2(nn.Module):
    """
    End-to-End Audio Classification Model.
    Pipeline: Waveform -> LogMel -> SpecAugment -> EfficientNetV2 -> AttnPooling -> Classifier
    """

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()

        # 1. Front-End
        self.frontend = LogMelFrontEnd()

        # 2. Augmentation
        self.spec_augment = SpecAugment()

        # 3. Backbone
        # Create EfficientNetV2-B0
        # num_classes=0 removes the default classifier and pooling
        self.backbone = timm.create_model(
            MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=DROP_RATE,
            drop_path_rate=DROP_PATH_RATE,
            global_pool="",  # Disable default pooling to get feature maps
        )

        # 4. Input Adaptation (3 Channels -> 1 Channel)
        # EfficientNet stem: Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # We modify it to accept 1 channel, initializing by summing RGB weights
        old_stem = self.backbone.conv_stem
        new_stem = nn.Conv2d(
            in_channels=1,
            out_channels=old_stem.out_channels,
            kernel_size=old_stem.kernel_size,
            stride=old_stem.stride,
            padding=old_stem.padding,
            bias=old_stem.bias is not None,
        )

        # Sum weights along the channel dimension (dim=1)
        # old_stem.weight shape: (Out, 3, K, K) -> (Out, 1, K, K)
        with torch.no_grad():
            new_stem.weight.copy_(old_stem.weight.sum(dim=1, keepdim=True))
            if old_stem.bias is not None:
                new_stem.bias.copy_(old_stem.bias)

        self.backbone.conv_stem = new_stem

        # 5. Pooling & Classifier
        # Get number of features from the backbone (1280 for B0)
        num_features = self.backbone.num_features

        self.pooling = SingleHeadAttentionPooling(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Raw waveforms (B, T)
        Returns:
            torch.Tensor: Logits (B, NumClasses)
        """
        # 1. Front-End: Waveform -> Spectrogram
        x = self.frontend(x)

        # 2. Augmentation (Training only)
        x = self.spec_augment(x)

        # 3. Backbone: Spectrogram -> Feature Maps
        # Output shape: (B, 1280, H, W)
        x = self.backbone(x)

        # 4. Pooling: Feature Maps -> Feature Vector
        # Output shape: (B, 1280)
        x = self.pooling(x)

        # 5. Classifier: Feature Vector -> Logits
        x = self.classifier(x)

        return x
