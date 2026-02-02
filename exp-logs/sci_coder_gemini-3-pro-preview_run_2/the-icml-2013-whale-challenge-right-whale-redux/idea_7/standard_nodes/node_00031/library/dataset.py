import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
import cv2
from torch.utils.data import Dataset
from library.config import Config
from library.utils import seed_everything

# Ensure deterministic behavior for transforms
seed_everything(Config.SEED)


def get_transforms(phase: str):
    """
    Returns the data augmentation pipeline based on the phase.

    Args:
        phase (str): 'train', 'val', or 'test'.

    Returns:
        torch.nn.Sequential: A sequence of transforms.
    """
    if phase == "train":
        # SpecAugment: Time and Frequency Masking
        # Note: Input is (1, H, W). Torchaudio masks expect (..., Freq, Time).
        # We treat H (224) as Freq and W (224) as Time.
        return torch.nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=20),
            torchaudio.transforms.TimeMasking(time_mask_param=40),
        )
    else:
        # No augmentation for validation/test
        return torch.nn.Sequential()


class WhaleDataset(Dataset):
    """
    PyTorch Dataset for Right Whale Call Detection.
    """

    def __init__(self, data, labels=None, clips=None, phase="train"):
        """
        Args:
            data (np.ndarray): Preprocessed image data of shape (N, H, W).
            labels (np.ndarray, optional): Labels of shape (N,).
            clips (np.ndarray, optional): Clip filenames of shape (N,).
            phase (str): 'train', 'val', or 'test' to determine transforms.
        """
        self.data = data
        self.labels = labels
        self.clips = clips
        self.phase = phase
        self.transform = get_transforms(phase)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load image (H, W) -> Tensor (1, H, W)
        img_arr = self.data[idx]
        img_tensor = torch.tensor(img_arr, dtype=torch.float32).unsqueeze(0)

        # Apply transforms (SpecAugment)
        if self.transform:
            img_tensor = self.transform(img_tensor)

        # Return format depends on availability of labels/clips
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img_tensor, label
        elif self.clips is not None:
            return img_tensor, self.clips[idx]
        else:
            return img_tensor


def preprocess_audio(file_path):
    """
    Loads audio, computes Log-Mel Spectrogram, resizes, and applies instance standardization.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        np.ndarray: Processed spectrogram of shape (H, W).
    """
    try:
        # 1. Load Audio
        audio, sr = sf.read(file_path)

        # Handle multi-channel (though analysis says all are mono)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # 2. Compute Mel Spectrogram
        # We use torchaudio functional for consistency, but need tensor input
        audio_tensor = torch.tensor(audio, dtype=torch.float32)

        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.FMIN,
            f_max=Config.FMAX,
            normalized=Config.NORMALIZED,
        )(audio_tensor)

        # 3. Log Scale (dB)
        log_mel_spec = torchaudio.transforms.AmplitudeToDB(top_db=80)(mel_spec)

        # Convert to numpy for resizing
        spec_np = log_mel_spec.numpy()

        # 4. Resize to Target Image Size (224, 224)
        # cv2.resize expects (width, height)
        spec_resized = cv2.resize(
            spec_np,
            (Config.IMG_SIZE[1], Config.IMG_SIZE[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        # 5. Instance-Wise Standardization (Zero-Mean, Unit-Variance)
        mean = spec_resized.mean()
        std = spec_resized.std()
        spec_norm = (spec_resized - mean) / (std + 1e-6)

        return spec_norm

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        # Return a zero array of correct shape as fallback
        return np.zeros(Config.IMG_SIZE, dtype=np.float32)


def process_and_cache(
    metadata_csv,
    data_cache,
    label_cache=None,
    clip_cache=None,
    load_cached_data=True,
    is_test=False,
):
    """
    Handles the caching logic: loads if exists and requested, else processes and saves.
    """
    # Check if all required caches exist
    caches_exist = os.path.exists(data_cache)
    if label_cache:
        caches_exist = caches_exist and os.path.exists(label_cache)
    if clip_cache:
        caches_exist = caches_exist and os.path.exists(clip_cache)

    if load_cached_data and caches_exist:
        print(f"Loading cached data from {data_cache}...")
        data = np.load(data_cache)
        labels = np.load(label_cache) if label_cache else None
        clips = np.load(clip_cache, allow_pickle=True) if clip_cache else None
        return data, labels, clips

    # Process from scratch
    print(f"Processing data from {metadata_csv}...")
    df = pd.read_csv(metadata_csv)

    # Debugging: subset if Config.DEBUG
    if Config.DEBUG:
        df = df.head(100)
        print("DEBUG Mode: Processing only 100 samples.")

    data_list = []
    label_list = []
    clip_list = []

    for idx, row in df.iterrows():
        # Construct full path
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_ROOT, rel_path)

        # Process
        spec = preprocess_audio(full_path)
        data_list.append(spec)

        if not is_test:
            label_list.append(row["label"])
        else:
            clip_list.append(row["clip"])

    # Convert to numpy arrays
    data_arr = np.array(data_list, dtype=np.float32)

    # Save Data
    os.makedirs(os.path.dirname(data_cache), exist_ok=True)
    np.save(data_cache, data_arr)
    print(f"Saved data to {data_cache}")

    labels_arr = None
    if not is_test:
        labels_arr = np.array(label_list, dtype=np.int64)
        np.save(label_cache, labels_arr)
        print(f"Saved labels to {label_cache}")

    clips_arr = None
    if is_test:
        clips_arr = np.array(clip_list, dtype=object)
        np.save(clip_cache, clips_arr)
        print(f"Saved clips to {clip_cache}")

    return data_arr, labels_arr, clips_arr


def get_datasets(load_cached_data=True):
    """
    Main interface to get Train, Val, and Test datasets.
    Handles caching and preprocessing internally.

    Args:
        load_cached_data (bool): If True, attempts to load from .npy cache.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """

    # --- Train Data ---
    train_data, train_labels, _ = process_and_cache(
        metadata_csv=Config.TRAIN_CSV,
        data_cache=Config.TRAIN_DATA_CACHE,
        label_cache=Config.TRAIN_LABELS_CACHE,
        load_cached_data=load_cached_data,
        is_test=False,
    )

    # --- Val Data ---
    val_data, val_labels, _ = process_and_cache(
        metadata_csv=Config.VAL_CSV,
        data_cache=Config.VAL_DATA_CACHE,
        label_cache=Config.VAL_LABELS_CACHE,
        load_cached_data=load_cached_data,
        is_test=False,
    )

    # --- Test Data ---
    test_data, _, test_clips = process_and_cache(
        metadata_csv=Config.TEST_CSV,
        data_cache=Config.TEST_DATA_CACHE,
        clip_cache=Config.TEST_CLIPS_CACHE,
        load_cached_data=load_cached_data,
        is_test=True,
    )

    # Create Dataset Objects
    train_dataset = WhaleDataset(train_data, labels=train_labels, phase="train")
    val_dataset = WhaleDataset(val_data, labels=val_labels, phase="val")
    test_dataset = WhaleDataset(test_data, clips=test_clips, phase="test")

    print(
        f"Datasets ready: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}"
    )

    return train_dataset, val_dataset, test_dataset
