import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.config import Config


def quantize_data(data):
    """
    Handles NaNs and clips outliers in raw EEG data.
    Args:
        data (np.ndarray): Raw EEG data.
    Returns:
        np.ndarray: Cleaned data.
    """
    # Fill NaNs with 0
    data = np.nan_to_num(data, nan=0.0, posinf=1024, neginf=-1024)
    # Clip to reasonable EEG range (uV) to suppress artifacts
    data = np.clip(data, -1024, 1024)
    return data


def eeg_to_mel_spec(eeg_data, config):
    """
    Converts raw EEG signals to a Mel Spectrogram image using a bipolar montage.

    Args:
        eeg_data (np.ndarray): Shape (Time, Channels).
        config (Config): Configuration object with parameters.

    Returns:
        np.ndarray: Processed spectrogram image of shape (H, W, 1).
    """
    # 1. Apply Montage
    # Map channel names to indices
    ch_map = {name: i for i, name in enumerate(config.EEG_CHANNELS)}

    montage_signals = []
    for c1, c2 in config.MONTAGE_PAIRS:
        if c1 in ch_map and c2 in ch_map:
            idx1 = ch_map[c1]
            idx2 = ch_map[c2]
            diff = eeg_data[:, idx1] - eeg_data[:, idx2]
            montage_signals.append(diff)
        else:
            # Fallback for missing channels (should not happen in valid dataset)
            montage_signals.append(np.zeros(eeg_data.shape[0], dtype=np.float32))

    # Stack to shape (16, Time)
    signals = np.stack(montage_signals, axis=0)

    # 2. Convert to Tensor for Torchaudio
    signals_tensor = torch.tensor(signals, dtype=torch.float32)

    # 3. Compute Mel Spectrogram
    # Output shape: (16, n_mels, time_steps)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.EEG_SR,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
        f_min=config.FMIN,
        f_max=config.FMAX,
        center=True,
        power=2.0,
    )

    mels = mel_transform(signals_tensor)

    # 4. Log Transform (Amplitude to DB)
    mels = torchaudio.transforms.AmplitudeToDB(top_db=80.0)(mels)

    # 5. Image Assembly
    # We stack the frequency bands of the 16 channels vertically.
    # Current shape: (Channels, Mels, Time) -> (16, 128, T)
    # Target shape: (Channels * Mels, Time) -> (2048, T)
    C, M, T = mels.shape
    mels_image = mels.permute(1, 0, 2).reshape(C * M, T).numpy()

    # 6. Standardization (Instance-level)
    mean = mels_image.mean()
    std = mels_image.std()
    if std > 1e-6:
        mels_image = (mels_image - mean) / std
    else:
        mels_image = mels_image - mean

    # Add channel dimension for Albumentations: (H, W, 1)
    mels_image = mels_image[:, :, np.newaxis]

    return mels_image


def process_kaggle_spec(spec_df, config):
    """
    Processes the pre-computed Kaggle spectrograms.

    Args:
        spec_df (pd.DataFrame): Raw spectrogram data.
        config (Config): Configuration object.

    Returns:
        np.ndarray: Processed spectrogram image (H, W, 1).
    """
    # Fill NaNs
    spec_df = spec_df.fillna(0)

    # Extract values. Kaggle specs are (Time, Frequencies)
    data = spec_df.values

    # Log transform (log1p to handle 0s safely)
    data = np.log1p(data)

    # Transpose to (Freqs, Time) to match EEG spec orientation (Freq on Y-axis)
    data = data.T

    # Standardization
    mean = data.mean()
    std = data.std()
    if std > 1e-6:
        data = (data - mean) / std
    else:
        data = data - mean

    # Add channel dimension
    data = data[:, :, np.newaxis]

    return data


def load_and_process_eeg(
    file_path, start_time, duration, cache_id, config, load_cached_data=False
):
    """
    Loads raw EEG parquet, slices it, processes it into a spectrogram,
    and handles caching logic.

    Args:
        file_path (str): Path to the EEG parquet file.
        start_time (float): Start time in seconds (offset).
        duration (float): Duration in seconds.
        cache_id (str): Unique identifier for caching (e.g., eeg_id_sub_id).
        config (Config): Configuration object.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        np.ndarray: The processed spectrogram.
    """
    cache_path = os.path.join(config.CACHE_DIR, f"{cache_id}.npy")

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception:
            # If load fails, proceed to compute
            pass

    # 2. Compute from Scratch
    try:
        eeg_df = pd.read_parquet(file_path)
    except FileNotFoundError:
        # Return zeros if file missing
        return np.zeros((config.IMG_SIZE[0], config.IMG_SIZE[1], 1), dtype=np.float32)

    # Calculate indices
    sr = config.EEG_SR
    start_idx = int(start_time * sr)
    end_idx = start_idx + int(duration * sr)

    # Handle bounds and padding
    total_len = len(eeg_df)
    expected_len = int(duration * sr)

    if start_idx >= total_len:
        # Window completely outside
        raw_data = np.zeros((expected_len, len(config.EEG_CHANNELS)))
    else:
        # Slice
        if start_idx < 0:
            start_idx = 0
        raw_data = eeg_df.iloc[start_idx:end_idx].values

    # Pad if shorter than expected
    if len(raw_data) < expected_len:
        pad_len = expected_len - len(raw_data)
        raw_data = np.pad(raw_data, ((0, pad_len), (0, 0)), mode="constant")

    # Quantize/Clean
    raw_data = quantize_data(raw_data)

    # Convert to Spectrogram
    spec = eeg_to_mel_spec(raw_data, config)

    # 3. Save to Cache if requested
    # If load_cached_data is True, it implies this dataset (e.g., Validation)
    # should be cached for future speedups.
    if load_cached_data:
        np.save(cache_path, spec)

    return spec


def get_transforms(config, mode="train"):
    """
    Returns Albumentations transforms for the spectrograms.
    """
    h, w = config.IMG_SIZE

    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=h, width=w),
                # Masking for regularization
                A.CoarseDropout(
                    max_holes=config.MASK_NUM_HOLES,
                    max_height=config.MASK_MAX_SIZE,
                    max_width=config.MASK_MAX_SIZE,
                    fill_value=0,
                    p=0.5,
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([A.Resize(height=h, width=w), ToTensorV2()])


def mixup_data(x, y, alpha=0.4, device="cuda"):
    """
    Applies MixUp augmentation to inputs and targets.
    Returns: mixed_inputs, target_a, target_b, lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the loss for mixed inputs.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
