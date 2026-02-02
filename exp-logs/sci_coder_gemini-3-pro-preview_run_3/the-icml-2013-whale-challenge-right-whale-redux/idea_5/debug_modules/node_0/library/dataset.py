import os
import torch
import torchaudio
import torchaudio.transforms as T
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import get_logger

logger = get_logger(__name__)


class AudioPreprocessor:
    """
    Handles the conversion of raw audio files to normalized Log-Mel Spectrograms.
    """

    def __init__(self):
        self.target_sr = Config.SR
        self.target_len = int(Config.SR * Config.DURATION)  # e.g., 4000 samples

        # Define Mel Spectrogram transformation
        self.mel_transform = T.MelSpectrogram(
            sample_rate=Config.SR,
            n_fft=Config.N_FFT,
            win_length=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.FMIN,
            f_max=Config.FMAX,
            center=True,
            power=2.0,
        )

        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)

    def process_file(self, file_path):
        """
        Reads an audio file and returns a processed spectrogram tensor.
        """
        try:
            # Load audio
            waveform, sr = torchaudio.load(file_path)
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return None

        # Resample if necessary
        if sr != self.target_sr:
            resampler = T.Resample(sr, self.target_sr)
            waveform = resampler(waveform)

        # Mix down to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Ensure fixed length (Pad or Crop)
        current_len = waveform.shape[1]
        if current_len < self.target_len:
            pad_amount = self.target_len - current_len
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        elif current_len > self.target_len:
            waveform = waveform[:, : self.target_len]

        # Compute Spectrogram
        mel_spec = self.mel_transform(waveform)
        log_mel_spec = self.amplitude_to_db(mel_spec)

        # Ensure exact time dimension for CNN input
        # Config.IMG_SIZE is (H, W) -> (320, 200)
        target_w = Config.IMG_SIZE[1]
        current_w = log_mel_spec.shape[2]

        if current_w < target_w:
            log_mel_spec = torch.nn.functional.pad(
                log_mel_spec, (0, target_w - current_w)
            )
        elif current_w > target_w:
            log_mel_spec = log_mel_spec[:, :, :target_w]

        # Instance-level Min-Max Normalization
        # Scales data to [0, 1] per sample
        min_val = log_mel_spec.min()
        max_val = log_mel_spec.max()
        if max_val - min_val > 1e-6:
            log_mel_spec = (log_mel_spec - min_val) / (max_val - min_val)
        else:
            log_mel_spec = torch.zeros_like(log_mel_spec)

        return log_mel_spec  # Shape: (1, n_mels, time_steps)


def prepare_dataset_cache(
    metadata_path, cache_name, load_cached_data=True, debug=False, subset_size=None
):
    """
    Loads data from cache or processes raw audio files and saves to cache.
    """
    # Adjust cache name for debug mode to avoid overwriting full dataset cache
    if debug:
        cache_name = f"{cache_name}_debug"

    cache_path = os.path.join(Config.CACHE_DIR, f"{cache_name}.npz")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        logger.info(f"Loading cached data from {cache_path}")
        try:
            data = np.load(cache_path)
            return data["specs"], data["labels"], data["clips"]
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_path}: {e}. Reprocessing...")

    # 2. Process data from scratch
    logger.info(f"Processing data for {cache_name} from {metadata_path}...")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    if debug and subset_size:
        logger.info(f"Debug mode: processing subset of {subset_size} samples.")
        df = df.head(subset_size)

    preprocessor = AudioPreprocessor()

    specs_list = []
    labels_list = []
    clips_list = []

    for idx, row in df.iterrows():
        full_path = os.path.join(Config.INPUT_ROOT, row["file_path"])

        spec = preprocessor.process_file(full_path)

        if spec is not None:
            # Convert to numpy for storage
            specs_list.append(spec.numpy())

            # Handle labels (test set might not have 'label' column)
            if "label" in row:
                labels_list.append(row["label"])
            else:
                labels_list.append(-1)  # Placeholder for test

            clips_list.append(row["clip_name"])

        if (idx + 1) % 2000 == 0:
            logger.info(f"Processed {idx + 1}/{len(df)} files")

    if not specs_list:
        raise RuntimeError("No audio files were successfully processed.")

    # Stack into a single array
    # specs_list elements are (1, H, W). Concatenate on axis 0 -> (N, H, W)
    # We remove the channel dim here to save space, will add back in Dataset
    specs = np.concatenate(specs_list, axis=0)
    labels = np.array(labels_list, dtype=np.float32)
    clips = np.array(clips_list)

    # 3. Save to cache
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    np.savez(cache_path, specs=specs, labels=labels, clips=clips)
    logger.info(f"Saved processed data to {cache_path}. Shape: {specs.shape}")

    return specs, labels, clips


class WhaleDataset(Dataset):
    """
    PyTorch Dataset for Right Whale Detection.
    """

    def __init__(self, specs, labels, clips, is_train=False):
        self.specs = specs
        self.labels = labels
        self.clips = clips
        self.is_train = is_train

        # Augmentations
        self.time_masking = T.TimeMasking(time_mask_param=Config.SPECAUG_TIME_MASK)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=Config.SPECAUG_FREQ_MASK)

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        # Retrieve spectrogram
        # Shape in storage: (H, W) -> (320, 200)
        spec_np = self.specs[idx]

        # Convert to tensor
        spec = torch.tensor(spec_np, dtype=torch.float32)

        # Ensure channel dimension: (H, W) -> (1, H, W)
        if spec.dim() == 2:
            spec = spec.unsqueeze(0)

        # Apply SpecAugment if training
        if self.is_train and Config.SPECAUG:
            spec = self.time_masking(spec)
            spec = self.freq_masking(spec)

        # Replicate channels for ConvNeXt backbone (expects 3 channels)
        # (1, H, W) -> (3, H, W)
        spec = spec.repeat(Config.IN_CHANNELS, 1, 1)

        # Retrieve label and clip name
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        clip_name = self.clips[idx]

        return spec, label, clip_name


def get_dataloaders(debug=False, load_cached_data=True):
    """
    Generates DataLoaders for train, validation, and test sets.
    Handles caching implicitly.
    """
    subset_size = Config.DEBUG_SUBSET_SIZE if debug else None

    # --- Train Loader ---
    train_specs, train_labels, train_clips = prepare_dataset_cache(
        Config.TRAIN_CSV, "train", load_cached_data, debug, subset_size
    )
    train_ds = WhaleDataset(train_specs, train_labels, train_clips, is_train=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    # --- Validation Loader ---
    val_specs, val_labels, val_clips = prepare_dataset_cache(
        Config.VAL_CSV, "val", load_cached_data, debug, subset_size
    )
    val_ds = WhaleDataset(val_specs, val_labels, val_clips, is_train=False)
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- Test Loader ---
    test_specs, test_labels, test_clips = prepare_dataset_cache(
        Config.TEST_CSV, "test", load_cached_data, debug, subset_size
    )
    test_ds = WhaleDataset(test_specs, test_labels, test_clips, is_train=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
