import math
import numpy as np
import torch
import torchaudio
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from library.config import Config

# Ensure reproducibility
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)


class MultiResolutionSTFT:
    """
    Generates a multi-resolution time-frequency representation from raw EEG signals.
    It computes Mel Spectrograms with different window sizes to capture both
    high-frequency transients (spikes) and low-frequency rhythms (delta waves).
    """

    def __init__(self):
        self.sr = Config.SR
        self.n_mels = Config.N_MELS
        self.fmin = Config.FMIN
        self.fmax = Config.FMAX
        self.target_shape = Config.IMG_SIZE_A  # (128, 500)

        # Define window sizes in samples
        self.win_sizes = Config.STFT_WINDOW_SIZES

        # Calculate hop length to approximate the target width
        # We will resize strictly later, but this keeps the aspect ratio reasonable.
        self.hop_length = Config.TOTAL_SAMPLES // self.target_shape[1]

        self.transforms = []
        for win_length in self.win_sizes:
            # n_fft must be a power of 2 and >= win_length
            n_fft = 2 ** math.ceil(math.log2(win_length))
            n_fft = max(n_fft, 32)  # Ensure a reasonable minimum

            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sr,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                f_min=self.fmin,
                f_max=self.fmax,
                center=True,
                pad_mode="reflect",
                power=2.0,
                normalized=True,
            )

            amp_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80.0)

            self.transforms.append((mel_transform, amp_to_db))

    def __call__(self, eeg_data):
        """
        Args:
            eeg_data (np.ndarray): Raw EEG signal of shape (Time, Channels).
                                   Example: (10000, 19)
        Returns:
            np.ndarray: Multi-resolution spectrogram of shape (Freq, Time, Depth).
                        Example: (128, 500, 57)
        """
        # Handle NaNs in raw data by replacing with 0
        eeg_data = np.nan_to_num(eeg_data, nan=0.0)

        # Convert to Tensor: (Channels, Time) for torchaudio
        tensor_data = torch.from_numpy(eeg_data.T).float()

        results = []

        for mel_trans, db_trans in self.transforms:
            # Apply MelSpectrogram: Output (Channels, n_mels, Time)
            # We treat channels as batch dimension for parallel processing
            mels = mel_trans(tensor_data)

            # Convert to dB
            mels_db = db_trans(mels)

            # Resize to target (128, 500)
            # Interpolate expects (Batch, Channels, H, W).
            # We reshape to (Channels, 1, Freq, Time) to treat each channel as an image
            mels_db = mels_db.unsqueeze(1)

            mels_resized = torch.nn.functional.interpolate(
                mels_db, size=self.target_shape, mode="bilinear", align_corners=False
            )

            # Squeeze back to (Channels, Freq, Time) -> (19, 128, 500)
            mels_resized = mels_resized.squeeze(1)

            results.append(mels_resized.numpy())

        # Stack results along the channel dimension
        # List of 3 arrays of shape (19, 128, 500) -> (57, 128, 500)
        combined = np.concatenate(results, axis=0)

        # Transpose to (Freq, Time, Channels) for Albumentations compatibility
        # (57, 128, 500) -> (128, 500, 57)
        combined = np.transpose(combined, (1, 2, 0))

        return combined


class SpecAugment(A.ImageOnlyTransform):
    """
    Applies Time and Frequency Masking to the spectrogram.
    Simulates the SpecAugment paper: https://arxiv.org/abs/1904.08779
    """

    def __init__(
        self,
        num_mask=2,
        freq_masking_max_percentage=0.1,
        time_masking_max_percentage=0.1,
        always_apply=False,
        p=0.5,
    ):
        super(SpecAugment, self).__init__(always_apply, p)
        self.num_mask = num_mask
        self.freq_masking_max_percentage = freq_masking_max_percentage
        self.time_masking_max_percentage = time_masking_max_percentage

    def apply(self, image, **params):
        # Image shape: (Freq, Time, Channels)
        h, w, c = image.shape
        img_aug = image.copy()

        # Frequency Masking (Rows)
        freq_mask_param = int(self.freq_masking_max_percentage * h)
        for _ in range(self.num_mask):
            f = np.random.randint(0, freq_mask_param + 1)
            if f > 0:
                f0 = np.random.randint(0, h - f + 1)
                img_aug[f0 : f0 + f, :, :] = 0  # Mask all channels identically

        # Time Masking (Cols)
        time_mask_param = int(self.time_masking_max_percentage * w)
        for _ in range(self.num_mask):
            t = np.random.randint(0, time_mask_param + 1)
            if t > 0:
                t0 = np.random.randint(0, w - t + 1)
                img_aug[:, t0 : t0 + t, :] = 0  # Mask all channels identically

        return img_aug

    def get_transform_init_args_names(self):
        return (
            "num_mask",
            "freq_masking_max_percentage",
            "time_masking_max_percentage",
        )


class EEGNormalize(A.ImageOnlyTransform):
    """
    Min-Max normalization per instance to [0, 1].
    Robust to outliers and varying signal amplitudes.
    """

    def __init__(self, always_apply=True, p=1.0):
        super(EEGNormalize, self).__init__(always_apply, p)

    def apply(self, image, **params):
        # Image shape: (H, W, C)
        img_min = image.min()
        img_max = image.max()

        epsilon = 1e-6
        if (img_max - img_min) > epsilon:
            image = (image - img_min) / (img_max - img_min)
        else:
            # If flat signal, return zeros
            image = np.zeros_like(image)

        return image


def get_transforms(mode="train", data_type="eeg"):
    """
    Returns the Albumentations composition of transforms.

    Args:
        mode (str): "train" or "valid"/"test".
        data_type (str): "eeg" (Stream A) or "spec" (Stream B).
    """
    transforms = []

    # 1. Normalization (Always apply)
    transforms.append(EEGNormalize(p=1.0))

    # 2. Augmentation (Train only)
    if mode == "train":
        if Config.USE_SPECAUG:
            # Adjust masking intensity based on data type if needed
            transforms.append(
                SpecAugment(
                    num_mask=2,
                    freq_masking_max_percentage=0.1,
                    time_masking_max_percentage=0.1,
                    p=0.5,
                )
            )

    # 3. Convert to Tensor
    # Albumentations ToTensorV2 converts (H, W, C) -> (C, H, W)
    transforms.append(ToTensorV2())

    return A.Compose(transforms)
