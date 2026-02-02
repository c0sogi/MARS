import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from library.config import Config


class AudioProcessor:
    """
    Handles audio loading, spectrogram generation, and normalization.
    """

    def __init__(self):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SR,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.FMIN,
            f_max=Config.FMAX,
            normalized=False,
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")
        self.target_length = int(Config.SR * Config.DURATION)

    def __call__(self, file_path):
        try:
            # Load audio using soundfile
            audio, sr = sf.read(file_path)

            # Handle multi-channel (take first channel)
            if audio.ndim > 1:
                audio = audio[:, 0]

            # Pad or crop to target length
            if len(audio) < self.target_length:
                pad_width = self.target_length - len(audio)
                audio = np.pad(audio, (0, pad_width), mode="constant")
            else:
                audio = audio[: self.target_length]

            # Convert to tensor
            waveform = torch.from_numpy(audio).float().unsqueeze(0)  # Shape: (1, Time)

            # Generate Spectrogram
            spec = self.mel_transform(waveform)  # Shape: (1, Freq, Time)
            spec = self.amp_to_db(spec)

            # Instance-level Min-Max Normalization
            min_val = spec.min()
            max_val = spec.max()
            delta = max_val - min_val + 1e-6
            spec = (spec - min_val) / delta

            return spec.numpy()  # Return as numpy array for efficient caching

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # Return a zero tensor of expected shape in case of error
            # We assume shape based on duration and hop length
            # Approx time steps: 2000*2 / 20 = 200 + 1
            dummy_waveform = torch.zeros(1, self.target_length)
            dummy_spec = self.mel_transform(dummy_waveform)
            return dummy_spec.numpy()


def load_and_cache_data(df, cache_name, load_cached_data=True, debug=False):
    """
    Loads data from .npz cache if available, otherwise processes raw audio and saves cache.
    """
    cache_dir = os.path.join(Config.WORKING_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{cache_name}.npz")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading {cache_name} from cache: {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            return {
                "images": data["images"],
                "labels": data["labels"],
                "clips": data["clips"],
            }
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing from scratch.")

    # 2. Process from scratch
    print(f"Processing data for {cache_name}...")
    processor = AudioProcessor()

    if debug:
        print(f"Debug mode enabled: processing only {Config.DEBUG_SAMPLES} samples.")
        df = df.iloc[: Config.DEBUG_SAMPLES]

    images = []
    labels = []
    clips = []

    for _, row in df.iterrows():
        full_path = os.path.join(Config.INPUT_ROOT, row["file_path"])
        spec = processor(full_path)
        images.append(spec)
        clips.append(row["clip_name"])

        if "label" in row:
            labels.append(row["label"])
        else:
            labels.append(-1.0)  # Placeholder for test data

    images = np.stack(images).astype(np.float32)
    labels = np.array(labels, dtype=np.float32)
    clips = np.array(clips)

    # Save to cache
    print(f"Saving {cache_name} to {cache_path}")
    np.savez(cache_path, images=images, labels=labels, clips=clips)

    return {"images": images, "labels": labels, "clips": clips}


class WhaleDataset(Dataset):
    def __init__(self, data_dict, targets=None, transform=None):
        self.images = data_dict["images"]
        self.clips = data_dict["clips"]
        self.transform = transform

        # Use provided targets (e.g., soft labels) if available, else use dataset labels
        if targets is not None:
            self.labels = targets
        else:
            self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Convert numpy array to torch tensor
        img = torch.from_numpy(self.images[idx])  # Shape: (1, Freq, Time)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # Apply transforms (e.g., SpecAugment)
        if self.transform:
            img = self.transform(img)

        return img, label, self.clips[idx]


def get_dataloaders(
    load_cached_data=True, debug=Config.DEBUG, pseudo_labels=None, student_mode=False
):
    """
    Creates DataLoaders for train, validation, and test sets.
    If student_mode is True and pseudo_labels are provided, combines Train and Pseudo-Test data.
    """

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Load or Process Data
    train_data = load_and_cache_data(train_df, "train_data", load_cached_data, debug)
    val_data = load_and_cache_data(val_df, "val_data", load_cached_data, debug)
    test_data = load_and_cache_data(test_df, "test_data", load_cached_data, debug)

    # Define Augmentations (SpecAugment) for Training
    train_transform = torch.nn.Sequential(
        torchaudio.transforms.FrequencyMasking(freq_mask_param=Config.FREQ_MASK_PARAM),
        torchaudio.transforms.TimeMasking(time_mask_param=Config.TIME_MASK_PARAM),
    )

    # Create Datasets
    train_ds = WhaleDataset(train_data, transform=train_transform)
    val_ds = WhaleDataset(val_data, transform=None)
    test_ds = WhaleDataset(test_data, transform=None)

    # Handle Student Mode (Self-Training)
    if student_mode and pseudo_labels is not None:
        print("Configuring Student Mode: Injecting Pseudo-Labeled Test Data...")

        # Identify test clips that have pseudo labels
        test_clips = test_data["clips"]
        indices = []
        soft_targets = []

        for i, clip in enumerate(test_clips):
            if clip in pseudo_labels:
                indices.append(i)
                soft_targets.append(pseudo_labels[clip])

        if indices:
            # Create a dataset subset for pseudo-labeled data
            pseudo_images = test_data["images"][indices]
            pseudo_clips = test_data["clips"][indices]
            pseudo_targets = np.array(soft_targets, dtype=np.float32)

            pseudo_data_dict = {
                "images": pseudo_images,
                "clips": pseudo_clips,
                "labels": pseudo_targets,  # Placeholder, real targets passed below
            }

            # Apply training transforms to pseudo-labeled data as well
            pseudo_ds = WhaleDataset(
                pseudo_data_dict, targets=pseudo_targets, transform=train_transform
            )

            # Combine original train and pseudo-labeled test
            train_ds = ConcatDataset([train_ds, pseudo_ds])
            print(
                f"Student Training Set: {len(train_ds)} samples (Original: {len(train_data['images'])}, Pseudo: {len(pseudo_ds)})"
            )
        else:
            print("Warning: No matching pseudo labels found for test data.")

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Essential for BatchNorm stability
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
