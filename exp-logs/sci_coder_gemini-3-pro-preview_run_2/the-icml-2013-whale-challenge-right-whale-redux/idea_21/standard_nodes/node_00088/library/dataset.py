import os
import torch
import torchaudio
import numpy as np
import pandas as pd
import soundfile as sf
from torch.utils.data import Dataset
from library.config import Config
from library.utils import seed_everything

# Ensure reproducibility
seed_everything(Config.SEED)


class WhaleTransforms:
    """
    Handles stochastic augmentations for the training phase.
    """

    def __init__(self, mode="train"):
        self.mode = mode
        self.transforms = []

        if self.mode == "train":
            # Aggressive SpecAugment as per Golden Recipe
            # Frequency Masking
            for _ in range(Config.SPEC_AUG_FREQ_MASK_NUM):
                self.transforms.append(
                    torchaudio.transforms.FrequencyMasking(
                        freq_mask_param=Config.SPEC_AUG_FREQ_MASK_PARAM
                    )
                )
            # Time Masking
            for _ in range(Config.SPEC_AUG_TIME_MASK_NUM):
                self.transforms.append(
                    torchaudio.transforms.TimeMasking(
                        time_mask_param=Config.SPEC_AUG_TIME_MASK_PARAM
                    )
                )

        self.transforms = torch.nn.Sequential(*self.transforms)

    def __call__(self, x):
        return self.transforms(x)


def compute_spectrogram(filepath):
    """
    Reads audio, pads/crops to fixed length, generates Mel Spectrogram,
    applies dynamic range correction, and performs instance standardization.
    """
    # 1. Load Audio
    # Using soundfile for robust loading, then converting to torch
    try:
        audio, sr = sf.read(filepath)
        waveform = torch.from_numpy(audio).float()
    except Exception as e:
        # Fallback for corrupted files (though metadata check passed)
        # Return a silent waveform of correct length
        print(f"Error loading {filepath}: {e}")
        waveform = torch.zeros(Config.SR * 2)  # Default 2s

    # 2. Fix Length (Pad/Crop to 2.0 seconds / 4000 samples)
    # Analysis showed max duration is 2.0s.
    target_length = int(Config.SR * 2.0)
    current_length = waveform.shape[0]

    if current_length < target_length:
        padding = target_length - current_length
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    elif current_length > target_length:
        waveform = waveform[:target_length]

    # Ensure shape is (1, samples) for torchaudio
    waveform = waveform.unsqueeze(0)

    # 3. Generate Mel Spectrogram
    # Config: N_FFT=1024, HOP=64, MELS=128, No Normalization
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=Config.SR,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        f_min=Config.FMIN,
        f_max=Config.FMAX,
        normalized=Config.NORMALIZED_MEL,
    )
    mel_spec = mel_transform(waveform)

    # 4. Amplitude to DB with Top-DB Clamping
    # This fixes the noise floor
    db_transform = torchaudio.transforms.AmplitudeToDB(top_db=Config.TOP_DB)
    log_mel_spec = db_transform(mel_spec)

    # 5. Instance Standardization
    # Zero-Mean, Unit-Variance per clip
    mean = log_mel_spec.mean()
    std = log_mel_spec.std()
    # Avoid division by zero
    norm_spec = (log_mel_spec - mean) / (std + 1e-6)

    return norm_spec


def load_dataset_data(df, split_name, load_cached_data=True):
    """
    Loads dataset spectrograms and labels.
    Implements strict caching mechanism using .npy files.
    """
    # Define cache paths
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    specs_path = os.path.join(cache_dir, f"{split_name}_data.npy")
    labels_path = os.path.join(cache_dir, f"{split_name}_targets.npy")
    clips_path = os.path.join(cache_dir, f"{split_name}_clips.npy")

    # Check if cache exists and loading is requested
    if load_cached_data and os.path.exists(specs_path) and os.path.exists(labels_path):
        print(f"Loading cached data for {split_name} from {cache_dir}...")
        specs = np.load(specs_path)
        labels = np.load(labels_path)
        # Clips are optional (only needed for test), but load if exists
        if os.path.exists(clips_path):
            clips = np.load(clips_path, allow_pickle=True)
        else:
            clips = df["clip"].values if "clip" in df.columns else np.array([])
        return specs, labels, clips

    print(f"Processing data for {split_name} from scratch...")

    # Debugging limit
    if Config.DEBUG:
        print(f"DEBUG mode: limiting to {Config.DEBUG_SAMPLES} samples.")
        df = df.iloc[: Config.DEBUG_SAMPLES].copy()

    specs_list = []
    labels_list = []
    clips_list = []

    # Process files
    for idx, row in df.iterrows():
        # Construct full path
        full_path = os.path.join(Config.INPUT_ROOT, row["file_path"])

        # Compute features
        spec = compute_spectrogram(full_path)
        # spec shape is (1, n_mels, time). Convert to numpy.
        specs_list.append(spec.numpy())

        # Handle label
        if "label" in row:
            labels_list.append(row["label"])
        else:
            labels_list.append(-1)  # Placeholder for test

        # Handle clip name
        if "clip" in row:
            clips_list.append(row["clip"])
        else:
            clips_list.append("")

    # Stack into arrays
    specs_array = np.stack(specs_list).astype(np.float32)
    labels_array = np.array(labels_list).astype(np.int64)
    clips_array = np.array(clips_list)

    # Save to cache
    print(f"Saving processed data to {cache_dir}...")
    np.save(specs_path, specs_array)
    np.save(labels_path, labels_array)
    np.save(clips_path, clips_array)

    return specs_array, labels_array, clips_array


class WhaleDataset(Dataset):
    """
    PyTorch Dataset for Right Whale Detection.
    Serves pre-processed spectrograms and applies on-the-fly augmentations.
    """

    def __init__(self, df, split_name="train", load_cached_data=True, transform=None):
        self.split_name = split_name
        self.transform = transform

        # Load data (cached or fresh)
        self.specs, self.labels, self.clips = load_dataset_data(
            df, split_name, load_cached_data=load_cached_data
        )

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        # Retrieve pre-computed spectrogram
        # Shape: (1, 128, 63)
        spec = torch.from_numpy(self.specs[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # Apply augmentations (SpecAugment)
        # Note: SpecAugment expects (channel, freq, time) or (freq, time)
        # Our spec is (1, 128, 63), which works with torchaudio transforms
        if self.transform:
            spec = self.transform(spec)

        return spec, label
