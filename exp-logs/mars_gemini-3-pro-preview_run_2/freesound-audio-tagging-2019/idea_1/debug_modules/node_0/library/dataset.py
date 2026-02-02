import os
import pandas as pd
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from library.config import Config


def get_classes():
    """
    Retrieves the list of classes from the sample_submission.csv file
    to ensure the correct order for predictions.
    """
    ss_path = os.path.join(Config.INPUT_ROOT, "sample_submission.csv")
    df = pd.read_csv(ss_path)
    # The columns are fname, Label1, Label2, ...
    classes = [c for c in df.columns if c not in ["fname", "file_path"]]
    return classes


class AudioDataset(Dataset):
    def __init__(self, df, classes, mode="train"):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (fname, file_path, labels/encoded_labels).
            classes (list): List of class names.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = df.reset_index(drop=True)
        self.classes = classes
        self.mode = mode

        # Pre-compute label lookup if labels exist
        if "encoded_labels" in self.df.columns:
            self.labels = self.df["encoded_labels"].tolist()
        else:
            self.labels = None

        # Audio transformation components
        # We initialize MelSpectrogram here.
        # Note: We will handle resampling dynamically or assume inputs are close,
        # but strictly we check SR per file.
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.FMIN,
            f_max=Config.FMAX,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row["fname"]
        rel_path = row["file_path"]
        full_path = os.path.join(Config.INPUT_ROOT, rel_path)

        # 1. Load Audio
        try:
            waveform, sr = torchaudio.load(full_path)
        except Exception as e:
            # Fallback for corrupted files (though metadata should be clean)
            # Create a silent waveform of 1 second
            waveform = torch.zeros(1, Config.SAMPLE_RATE)
            sr = Config.SAMPLE_RATE

        # 2. Convert to Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 3. Resample
        if sr != Config.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=Config.SAMPLE_RATE
            )
            waveform = resampler(waveform)

        # 4. Length Adjustment
        current_len = waveform.shape[1]

        if self.mode == "train":
            # Fixed length for training (Random Crop or Pad)
            target_len = Config.AUDIO_LEN
            if current_len > target_len:
                start = np.random.randint(0, current_len - target_len)
                waveform = waveform[:, start : start + target_len]
            elif current_len < target_len:
                pad_amt = target_len - current_len
                waveform = torch.nn.functional.pad(waveform, (0, pad_amt))
        else:
            # Variable length for Val/Test
            # Ensure minimum length for FFT
            if current_len < Config.N_FFT:
                pad_amt = Config.N_FFT - current_len
                waveform = torch.nn.functional.pad(waveform, (0, pad_amt))

        # 5. Compute Log-Mel Spectrogram
        spec = self.mel_spec(waveform)
        spec = self.amplitude_to_db(spec)

        # 6. Instance-wise Normalization
        mean = spec.mean()
        std = spec.std()
        spec = (spec - mean) / (std + 1e-6)

        # 7. Get Label
        if self.labels is not None:
            label_vec = torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            # Dummy label for test
            label_vec = torch.zeros(Config.NUM_CLASSES, dtype=torch.float32)

        return spec, label_vec, fname


def collate_fn(batch):
    """
    Custom collate function to handle variable length spectrograms.
    Pads the time dimension (dim 2) to the maximum length in the batch.
    """
    # batch is a list of tuples: (spec, label, fname)
    # spec shape: (1, n_mels, time)

    # Find maximum time dimension in this batch
    max_time = max([x[0].shape[2] for x in batch])

    specs = []
    labels = []
    fnames = []

    for spec, label, fname in batch:
        current_time = spec.shape[2]
        pad_amt = max_time - current_time
        if pad_amt > 0:
            # Pad the last dimension (time)
            spec = torch.nn.functional.pad(spec, (0, pad_amt))
        specs.append(spec)
        labels.append(label)
        fnames.append(fname)

    return torch.stack(specs), torch.stack(labels), fnames


def get_dataloaders(load_cached_data=True):
    """
    Prepares DataLoaders for train, val, and test sets.
    Implements caching for processed metadata (dataframes with encoded labels).
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "train_processed.parquet")
    val_cache = os.path.join(cache_dir, "val_processed.parquet")
    test_cache = os.path.join(cache_dir, "test_processed.parquet")

    classes = get_classes()

    # --- Load or Create DataFrames ---
    if (
        load_cached_data
        and os.path.exists(train_cache)
        and os.path.exists(val_cache)
        and os.path.exists(test_cache)
    ):
        train_df = pd.read_parquet(train_cache)
        val_df = pd.read_parquet(val_cache)
        test_df = pd.read_parquet(test_cache)
    else:
        # Load raw metadata
        train_df = pd.read_csv(Config.TRAIN_CSV)
        val_df = pd.read_csv(Config.VAL_CSV)
        test_df = pd.read_csv(Config.TEST_CSV)

        # Initialize MultiLabelBinarizer with the specific vocabulary
        mlb = MultiLabelBinarizer(classes=classes)
        # Fit on classes to ensure all columns exist and are ordered correctly
        mlb.fit([classes])

        # Process Train Labels
        train_df["label_list"] = train_df["labels"].apply(lambda x: x.split(","))
        train_encoded = mlb.transform(train_df["label_list"])
        train_df["encoded_labels"] = list(train_encoded)

        # Process Val Labels
        val_df["label_list"] = val_df["labels"].apply(lambda x: x.split(","))
        val_encoded = mlb.transform(val_df["label_list"])
        val_df["encoded_labels"] = list(val_encoded)

        # Save to cache
        train_df.to_parquet(train_cache)
        val_df.to_parquet(val_cache)
        test_df.to_parquet(test_cache)

    # --- Debug / Subsampling ---
    if Config.DEBUG:
        train_df = train_df.head(100)
        val_df = val_df.head(50)
        # test_df = test_df.head(50) # Optional

    if Config.MAX_TRAIN_SAMPLES:
        train_df = train_df.iloc[: Config.MAX_TRAIN_SAMPLES]
    if Config.MAX_VAL_SAMPLES:
        val_df = val_df.iloc[: Config.MAX_VAL_SAMPLES]

    # --- Create Datasets ---
    train_dataset = AudioDataset(train_df, classes, mode="train")
    val_dataset = AudioDataset(val_df, classes, mode="val")
    test_dataset = AudioDataset(test_df, classes, mode="test")

    # --- Create DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
