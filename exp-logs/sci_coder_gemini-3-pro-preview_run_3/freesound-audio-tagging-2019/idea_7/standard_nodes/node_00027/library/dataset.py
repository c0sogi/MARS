import os
import torch
import torchaudio
import pandas as pd
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset
from typing import Tuple, List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from library.config import Config
from library.utils import set_seed

# Set seed for reproducibility
set_seed(Config.seed)


class AudioDataset(Dataset):
    """
    PyTorch Dataset for Audio Tagging.
    Serves pre-computed spectrograms and applies SpecAugment during training.
    """

    def __init__(
        self, X: np.ndarray, y: np.ndarray, fnames: np.ndarray, phase: str = "train"
    ):
        """
        Args:
            X: Input spectrograms (N, n_mels, time_steps)
            y: Targets (N, num_classes)
            fnames: File names (N,)
            phase: 'train', 'val', or 'test'
        """
        self.X = X
        self.y = y
        self.fnames = fnames
        self.phase = phase

        # SpecAugment Transforms (applied on tensor)
        self.time_masking = torchaudio.transforms.TimeMasking(
            time_mask_param=Config.spec_augment_time_mask
        )
        self.freq_masking = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=Config.spec_augment_freq_mask
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Retrieve pre-processed spectrogram: (n_mels, time)
        spec = self.X[idx]

        # Convert to tensor
        spec_tensor = torch.from_numpy(spec).float()

        # Add channel dimension: (1, n_mels, time)
        spec_tensor = spec_tensor.unsqueeze(0)

        # Apply Augmentation only in training
        if self.phase == "train":
            # Apply SpecAugment
            spec_tensor = self.time_masking(spec_tensor)
            spec_tensor = self.freq_masking(spec_tensor)

        target = torch.from_numpy(self.y[idx]).float()

        return spec_tensor, target


def get_label_mapping() -> Tuple[List[str], Dict[str, int]]:
    """Reads the sample submission to get the correct class order."""
    sub_df = pd.read_csv(Config.sample_submission)
    # Columns are fname, Label1, Label2, ...
    labels = sub_df.columns[1:].tolist()
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    return labels, label_to_idx


def encode_labels(
    label_str: str, label_to_idx: Dict[str, int], num_classes: int
) -> np.ndarray:
    """Converts a comma-separated label string to a multi-hot binary vector."""
    vector = np.zeros(num_classes, dtype=np.float32)
    if pd.isna(label_str) or label_str == "":
        return vector

    for lbl in label_str.split(","):
        if lbl in label_to_idx:
            vector[label_to_idx[lbl]] = 1.0
    return vector


def compute_spectrogram(filepath: str) -> np.ndarray:
    """
    Reads audio, resamples, pads/crops, computes Log-Mel Spectrogram, and normalizes.
    Returns numpy array of shape (n_mels, time_steps).
    """
    # 1. Load Audio
    try:
        wav, sr = sf.read(filepath)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        # Return zeros in case of failure
        expected_len = int(Config.duration * Config.sample_rate / Config.hop_length) + 1
        return np.zeros((Config.n_mels, expected_len), dtype=np.float32)

    # 2. Convert to Tensor
    wav_tensor = torch.from_numpy(wav).float()

    # 3. Ensure Mono
    if wav_tensor.ndim > 1:
        wav_tensor = wav_tensor.mean(dim=1)

    # 4. Resample if necessary
    if sr != Config.sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=Config.sample_rate
        )
        wav_tensor = resampler(wav_tensor)

    # 5. Fix Duration (Pad or Crop)
    current_len = wav_tensor.shape[0]
    target_len = Config.target_length

    if current_len > target_len:
        # Truncate
        wav_tensor = wav_tensor[:target_len]
    elif current_len < target_len:
        # Pad with zeros
        padding = target_len - current_len
        wav_tensor = torch.nn.functional.pad(wav_tensor, (0, padding))

    # 6. Compute Mel Spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=Config.sample_rate,
        n_fft=Config.n_fft,
        hop_length=Config.hop_length,
        n_mels=Config.n_mels,
        f_min=Config.f_min,
        f_max=Config.f_max,
    )

    spec = mel_transform(wav_tensor)  # Shape: (n_mels, time)

    # 7. Amplitude to DB
    db_transform = torchaudio.transforms.AmplitudeToDB()
    spec = db_transform(spec)

    # 8. Per-Instance Normalization
    mean = spec.mean()
    std = spec.std()
    spec = (spec - mean) / (std + 1e-6)

    return spec.numpy()


def _process_one_file(args):
    """Helper for parallel processing."""
    row, label_to_idx, num_classes, input_root = args
    fname = row["fname"]
    rel_path = row["filepath"]
    full_path = os.path.join(input_root, rel_path)

    spec = compute_spectrogram(full_path)

    if "labels" in row and pd.notna(row["labels"]):
        lbl_vec = encode_labels(row["labels"], label_to_idx, num_classes)
    else:
        lbl_vec = np.zeros(num_classes, dtype=np.float32)

    return spec, lbl_vec, fname


def prepare_data_arrays(
    df: pd.DataFrame,
    class_names: List[str],
    label_to_idx: Dict[str, int],
    input_root: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Iterates through the dataframe, processes audio, and prepares X, y, fnames arrays.
    Uses ThreadPoolExecutor for parallel processing.
    """
    num_classes = len(class_names)

    print(f"Processing {len(df)} files with {Config.num_workers} workers...")

    # Prepare arguments for parallel execution
    args_list = []
    for _, row in df.iterrows():
        args_list.append((row, label_to_idx, num_classes, input_root))

    X_list = []
    y_list = []
    fnames_list = []

    # Execute in parallel
    with ThreadPoolExecutor(max_workers=Config.num_workers) as executor:
        results = list(executor.map(_process_one_file, args_list))

    # Unpack results
    for spec, lbl, fname in results:
        X_list.append(spec)
        y_list.append(lbl)
        fnames_list.append(fname)

    X = np.stack(X_list).astype(np.float32)
    y = np.stack(y_list).astype(np.float32)
    fnames = np.array(fnames_list)

    return X, y, fnames


def get_datasets(
    load_cached_data: bool = True,
) -> Tuple[AudioDataset, AudioDataset, AudioDataset]:
    """
    Main function to retrieve Train, Val, and Test datasets.
    Handles caching logic to speed up subsequent runs.
    """
    class_names, label_to_idx = get_label_mapping()

    # Define cache paths
    cache_dir = Config.working_dir
    os.makedirs(cache_dir, exist_ok=True)

    phases = ["train", "val", "test"]
    datasets = {}

    # Load metadata
    meta_dfs = {
        "train": pd.read_csv(Config.train_csv),
        "val": pd.read_csv(Config.val_csv),
        "test": pd.read_csv(Config.test_csv),
    }

    # Handle Debug Mode
    if Config.debug:
        print("DEBUG MODE: Reducing dataset size to 100 samples per phase.")
        for p in phases:
            meta_dfs[p] = meta_dfs[p].head(100)

    for phase in phases:
        print(f"Preparing {phase} dataset...")

        # Cache filenames include debug flag to avoid mixing full/debug caches
        suffix = "_debug" if Config.debug else ""
        x_path = os.path.join(cache_dir, f"{phase}_X{suffix}.npy")
        y_path = os.path.join(cache_dir, f"{phase}_y{suffix}.npy")
        f_path = os.path.join(cache_dir, f"{phase}_fnames{suffix}.npy")

        # 1. Try Load from Cache
        loaded = False
        if load_cached_data and Config.use_cache:
            if (
                os.path.exists(x_path)
                and os.path.exists(y_path)
                and os.path.exists(f_path)
            ):
                print(f"  Loading {phase} from cache...")
                try:
                    X = np.load(x_path)
                    y = np.load(y_path)
                    fnames = np.load(f_path)
                    loaded = True
                except Exception as e:
                    print(f"  Failed to load cache: {e}. Recomputing...")
                    loaded = False
            else:
                print(f"  Cache not found for {phase} at {x_path}.")

        # 2. Compute if not loaded
        if not loaded:
            print(f"  Computing {phase} features from scratch...")
            X, y, fnames = prepare_data_arrays(
                meta_dfs[phase], class_names, label_to_idx, Config.input_root
            )

            # Save to cache
            print(f"  Saving {phase} to cache...")
            np.save(x_path, X)
            np.save(y_path, y)
            np.save(f_path, fnames)

        # 3. Create Dataset
        datasets[phase] = AudioDataset(X, y, fnames, phase=phase)
        print(f"  {phase.capitalize()} dataset ready. Shape: {X.shape}")

    return datasets["train"], datasets["val"], datasets["test"]
