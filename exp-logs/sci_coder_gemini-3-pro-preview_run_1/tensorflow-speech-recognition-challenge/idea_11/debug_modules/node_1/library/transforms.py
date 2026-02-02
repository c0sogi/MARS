import torch
import torch.nn as nn
import torchaudio
import numpy as np
from library.config import Config


class AudioTransforms(nn.Module):
    """
    GPU-Accelerated Data Augmentation and Feature Extraction Module.

    This module converts raw audio waveforms into 3-Channel RGB-like tensors
    (Log-Mel Spectrogram + Delta + Delta-Delta) to leverage pretrained vision backbones.
    It applies SpecAugment and Mixup regularization dynamically on the GPU.
    """

    def __init__(self, device=Config.DEVICE):
        super().__init__()
        self.device = device

        # ---------------------------------------------------------------------
        # 1. Feature Extraction Components
        # ---------------------------------------------------------------------
        # Mel Spectrogram: Maps raw audio to time-frequency domain
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=20,
            f_max=Config.SAMPLE_RATE // 2,
            normalized=True,
        ).to(self.device)

        # Amplitude to DB: Converts power spectrogram to logarithmic scale (dB)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=80
        ).to(self.device)

        # ---------------------------------------------------------------------
        # 2. Augmentation Components (SpecAugment)
        # ---------------------------------------------------------------------
        # Time Masking: Masks random segments in the time dimension
        self.time_masking = torchaudio.transforms.TimeMasking(
            time_mask_param=30, p=0.5  # Approx 30% of 1-sec clip
        ).to(self.device)

        # Frequency Masking: Masks random segments in the frequency dimension
        self.freq_masking = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=20, p=0.5  # Approx 15% of mel bands
        ).to(self.device)

    def compute_deltas(self, spec):
        """
        Computes first and second order derivatives (Delta and Delta-Delta)
        of the spectrogram along the time dimension.

        Args:
            spec: (Batch, 1, n_mels, time)

        Returns:
            delta1, delta2 with same shape as spec.
        """
        # torchaudio.functional.compute_deltas operates on the last dimension
        delta1 = torchaudio.functional.compute_deltas(spec)
        delta2 = torchaudio.functional.compute_deltas(delta1)
        return delta1, delta2

    def forward(
        self, waveforms, labels=None, train=False, mixup_alpha=Config.MIXUP_ALPHA
    ):
        """
        Forward pass to convert waveforms to augmented features.

        Args:
            waveforms (Tensor): Raw audio of shape (Batch, Time).
            labels (Tensor, optional): Class indices of shape (Batch).
            train (bool): Whether to apply augmentations (SpecAugment, Mixup).
            mixup_alpha (float): Alpha parameter for Beta distribution in Mixup.

        Returns:
            If Mixup is active:
                (features, labels_a, labels_b, lam)
            Else:
                features
        """
        # Move data to GPU
        x = waveforms.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)

        # ---------------------------------------------------------------------
        # 1. Feature Generation (Mel + Delta + DeltaDelta)
        # ---------------------------------------------------------------------
        with torch.no_grad():
            # Generate Mel Spectrogram: (Batch, n_mels, time)
            mel = self.mel_spectrogram(x)

            # Convert to Log Scale (dB)
            log_mel = self.amplitude_to_db(mel)

            # Add Channel Dimension: (Batch, 1, n_mels, time)
            log_mel = log_mel.unsqueeze(1)

            # Compute Deltas
            delta1, delta2 = self.compute_deltas(log_mel)

            # Stack to create 3-Channel Input: (Batch, 3, n_mels, time)
            features = torch.cat([log_mel, delta1, delta2], dim=1)

        # ---------------------------------------------------------------------
        # 2. SpecAugment (Train Only)
        # ---------------------------------------------------------------------
        if train:
            # Apply masking. We reshape to (Batch*3, F, T) to apply different
            # random masks to each channel (Mel, Delta, Delta2). This acts as
            # a strong regularizer, forcing the model to rely on partial information
            # from different dynamic views.
            B, C, F, T = features.shape
            reshaped = features.view(B * C, F, T)

            masked = self.freq_masking(reshaped)
            masked = self.time_masking(masked)

            features = masked.view(B, C, F, T)

        # ---------------------------------------------------------------------
        # 3. Normalization (Instance Norm)
        # ---------------------------------------------------------------------
        # Standardize each sample to N(0, 1). This handles variations in recording
        # volume and maps the dB values to a range suitable for the CNN.
        instance_mean = features.mean(dim=(2, 3), keepdim=True)
        instance_std = features.std(dim=(2, 3), keepdim=True) + 1e-5
        features = (features - instance_mean) / instance_std

        # ---------------------------------------------------------------------
        # 4. Mixup (Train Only)
        # ---------------------------------------------------------------------
        if train and labels is not None and mixup_alpha > 0:
            # Sample lambda from Beta distribution
            lam = np.random.beta(mixup_alpha, mixup_alpha)

            # Generate permutation indices
            batch_size = features.size(0)
            index = torch.randperm(batch_size).to(self.device)

            # Mix features
            mixed_features = lam * features + (1 - lam) * features[index]

            # Return mixed features and mixup metadata
            return mixed_features, labels, labels[index], lam

        # Inference or Training without Mixup
        return features
