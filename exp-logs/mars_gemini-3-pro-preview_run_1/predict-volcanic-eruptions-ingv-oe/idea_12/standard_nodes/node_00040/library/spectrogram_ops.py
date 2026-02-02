import os
import glob
import numpy as np
import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
from library.config import Config
from library.utils import save_npy, load_npy, seed_everything


class DualResSpectrogramGenerator:
    """
    Generates Dual-Resolution Spectrograms for the Vision Branch.
    Produces a 20-channel tensor combining Wide-Band (Time-Res) and Narrow-Band (Freq-Res) views.
    Applies Global Log-Max Scaling to preserve absolute energy information.
    """

    def __init__(self, global_max=None):
        """
        Args:
            global_max (float): The normalization constant (max absolute spectrogram value).
                                If None, defaults to Config.GLOBAL_MAX_READING.
        """
        self.global_max = (
            global_max if global_max is not None else Config.GLOBAL_MAX_READING
        )

        # Wide-Band Spectrogram (High Time Resolution, Short Window)
        # Captures impulsive shocks
        self.spec_wide = torchaudio.transforms.Spectrogram(
            n_fft=Config.N_FFT_WIDE,
            hop_length=Config.HOP_WIDE,
            power=1.0,  # Magnitude Spectrogram
        )

        # Narrow-Band Spectrogram (High Frequency Resolution, Long Window)
        # Captures harmonic tremors
        self.spec_narrow = torchaudio.transforms.Spectrogram(
            n_fft=Config.N_FFT_NARROW,
            hop_length=Config.HOP_NARROW,
            power=1.0,  # Magnitude Spectrogram
        )

        self.img_size = Config.IMG_SIZE

    def _load_sensor_data(self, file_path):
        """
        Loads sensor data from CSV, handles NaNs, and converts to Tensor.
        """
        try:
            # Load as float32
            df = pd.read_csv(file_path, dtype="float32")

            # Fill NaNs with 0.0 (Silence) to avoid spectral artifacts
            df = df.fillna(0.0)

            # Transpose to (Sensors, Time)
            # df shape is (Time, Sensors), we want (Sensors, Time)
            data = df.values.T
            return torch.tensor(data, dtype=torch.float32)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            # Return silent waveform on error
            return torch.zeros(
                (Config.NUM_SENSORS, Config.SIGNAL_LENGTH), dtype=torch.float32
            )

    def transform(self, waveform):
        """
        Converts raw waveform to normalized dual-resolution spectrogram tensor.

        Args:
            waveform (Tensor): Shape (Sensors, Time)

        Returns:
            Tensor: Shape (20, 256, 256)
        """
        # 1. Generate Spectrograms
        # Output Shape: (Sensors, FreqBins, TimeSteps)
        s_wide = self.spec_wide(waveform)
        s_narrow = self.spec_narrow(waveform)

        # 2. Resize to fixed IMG_SIZE (256, 256)
        # Interpolate expects (Batch, Channels, H, W).
        # We treat Sensors as the Batch dimension for independent resizing.
        # Input: (Sensors, 1, Freq, Time)
        s_wide = s_wide.unsqueeze(1)
        s_narrow = s_narrow.unsqueeze(1)

        # Bilinear interpolation is suitable for spectrograms
        s_wide_resized = F.interpolate(
            s_wide, size=self.img_size, mode="bilinear", align_corners=False
        )
        s_narrow_resized = F.interpolate(
            s_narrow, size=self.img_size, mode="bilinear", align_corners=False
        )

        # Squeeze back to (Sensors, 256, 256)
        s_wide_resized = s_wide_resized.squeeze(1)
        s_narrow_resized = s_narrow_resized.squeeze(1)

        # 3. Stack Channel-wise
        # Concatenate along dim 0 to get 20 channels
        # Channels 0-9: Wide-Band, Channels 10-19: Narrow-Band
        stacked = torch.cat([s_wide_resized, s_narrow_resized], dim=0)

        # 4. Global Log-Max Scaling
        # Formula: log(X + 1) / log(M + 1)
        # This preserves relative energy differences between samples.
        m_val = float(self.global_max)

        # Avoid log(0) issues with log1p
        log_x = torch.log1p(stacked)
        log_m = np.log1p(m_val)

        normalized = log_x / log_m

        return normalized

    def process_file(self, file_path):
        """
        End-to-end processing for a single file.
        Returns numpy array.
        """
        waveform = self._load_sensor_data(file_path)
        tensor = self.transform(waveform)
        return tensor.numpy()


def compute_global_max(metadata_df, sample_size=100):
    """
    Iterates through a subset of the training data to derive the
    Global Max Spectrogram Value used for normalization.

    Args:
        metadata_df (pd.DataFrame): Training metadata.
        sample_size (int): Number of files to sample.

    Returns:
        float: The maximum observed magnitude.
    """
    print(f"Computing global max from {sample_size} samples...")

    # Initialize generator without normalization logic (dummy max)
    # We only need the raw spectrogram values.
    gen = DualResSpectrogramGenerator(global_max=1.0)

    # Sample files
    sample = metadata_df.sample(
        n=min(len(metadata_df), sample_size), random_state=Config.SEED
    )

    max_val = 0.0

    for _, row in sample.iterrows():
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        if not os.path.exists(full_path):
            continue

        waveform = gen._load_sensor_data(full_path)

        # Compute raw magnitudes
        s_wide = gen.spec_wide(waveform)
        s_narrow = gen.spec_narrow(waveform)

        # Find max in this segment
        curr_max = max(s_wide.max().item(), s_narrow.max().item())

        if curr_max > max_val:
            max_val = curr_max

    print(f"Computed Global Max Spectrogram Value: {max_val}")
    return max_val


def generate_dataset_spectrograms(metadata_df, subset_name, load_cached_data=True):
    """
    Generates and caches spectrograms for a specific dataset subset (train/val/test).
    Implements strict caching logic.

    Args:
        metadata_df (pd.DataFrame): Metadata containing 'segment_id' and 'file_path'.
        subset_name (str): Identifier for the subset ('train', 'val', 'test').
        load_cached_data (bool): If True, attempts to use existing files.

    Returns:
        str: Path to the directory containing the processed .npy files.
    """
    # Define cache directory for this subset
    cache_subdir = os.path.join(Config.CACHE_DIR, f"spectrograms_{subset_name}")
    os.makedirs(cache_subdir, exist_ok=True)

    # Check for existing cache
    # We validate cache by checking if the number of files matches the metadata
    existing_files = glob.glob(os.path.join(cache_subdir, "*.npy"))
    if load_cached_data and len(existing_files) >= len(metadata_df):
        # Validate cache content against current Config (Cite debug_lesson_4)
        try:
            sample_shape = load_npy(existing_files[0]).shape
            expected_shape = (
                Config.IN_CHANNELS,
                Config.IMG_SIZE[0],
                Config.IMG_SIZE[1],
            )
            if sample_shape == expected_shape:
                print(
                    f"[{subset_name}] Valid cache found with {len(existing_files)} files. Skipping generation."
                )
                return cache_subdir
            else:
                print(
                    f"[{subset_name}] Cache shape mismatch! Found {sample_shape}, expected {expected_shape}. Invalidating cache."
                )
                load_cached_data = False
        except Exception as e:
            print(f"[{subset_name}] Cache validation failed: {e}. Invalidating cache.")
            load_cached_data = False

    print(f"[{subset_name}] Generating spectrograms for {len(metadata_df)} files...")

    # Handle Global Max Normalization Constant
    # We must use the SAME constant for Train, Val, and Test.
    # Logic: Compute on Train, Save it, Load it for others.
    global_max_path = os.path.join(Config.CACHE_DIR, "global_max_spectrogram.npy")

    if subset_name == "train":
        # For training set, we define the constant
        if load_cached_data and os.path.exists(global_max_path):
            global_max = float(load_npy(global_max_path))
            print(f"Loaded cached Global Max: {global_max}")
        else:
            # Compute from scratch using a subset
            global_max = compute_global_max(metadata_df, sample_size=200)
            save_npy(np.array(global_max), global_max_path)
    else:
        # For val/test, we try to load the constant derived from training
        if os.path.exists(global_max_path):
            global_max = float(load_npy(global_max_path))
        else:
            # Fallback if train hasn't been processed (e.g. inference only mode)
            print("Warning: Global Max file not found. Using Config default.")
            global_max = Config.GLOBAL_MAX_READING

    # Initialize Generator with the determined constant
    generator = DualResSpectrogramGenerator(global_max=global_max)

    # Processing Loop
    for idx, row in metadata_df.iterrows():
        seg_id = str(row["segment_id"])
        file_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_DIR, file_path)
        save_path = os.path.join(cache_subdir, f"{seg_id}.npy")

        # Partial cache check: skip if this specific file exists
        if load_cached_data and os.path.exists(save_path):
            continue

        if os.path.exists(full_path):
            # Generate
            spec_tensor = generator.process_file(full_path)

            # Save as float32 .npy
            save_npy(spec_tensor.astype(np.float32), save_path)
        else:
            print(f"Error: Source file not found: {full_path}")

    print(f"[{subset_name}] Processing complete. Data saved to {cache_subdir}")
    return cache_subdir
