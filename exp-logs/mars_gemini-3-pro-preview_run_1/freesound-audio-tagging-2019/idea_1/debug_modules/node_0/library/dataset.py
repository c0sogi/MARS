import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed

# Ensure torchaudio uses a backend that supports the environment
# soundfile is generally robust for Linux/Windows
if torchaudio.get_audio_backend() != "soundfile":
    try:
        torchaudio.set_audio_backend("soundfile")
    except:
        pass


class AudioDataset(Dataset):
    """
    Custom Dataset for loading and processing audio files for tagging.
    """

    def __init__(self, df, data_dir, phase="train", transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing metadata (filenames, labels, filepaths).
            data_dir (str): Root directory for audio files.
            phase (str): 'train', 'val', or 'test'. Controls augmentation/cropping logic.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = df
        self.data_dir = data_dir
        self.phase = phase
        self.transform = transform

        # Identify label columns
        # We assume columns starting after 'filepath' or specific known columns are labels.
        # Based on metadata generation, columns are: fname, labels, filepath, [classes...]
        # We can filter by excluding metadata columns.
        meta_cols = [
            "fname",
            "labels",
            "filepath",
            "label_count",
            "duration",
            "sample_rate",
            "n_channels",
        ]
        self.label_cols = [c for c in df.columns if c not in meta_cols]

        # Pre-compute paths and labels
        self.filepaths = df["filepath"].tolist()

        # For test set, labels might be all 0s, but we still load them to keep format consistent
        if len(self.label_cols) > 0:
            self.labels = df[self.label_cols].values.astype(np.float32)
        else:
            # Fallback if no label columns found (unlikely given problem description)
            self.labels = np.zeros((len(df), Config.NUM_CLASSES), dtype=np.float32)

        # Audio Config
        self.target_sr = Config.SR
        self.target_len = int(Config.SR * Config.DURATION)

        # Transforms
        # We initialize MelSpectrogram here.
        # Note: In a multi-worker dataloader, these objects are pickled and sent to workers.
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SR,
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
        # 1. Construct full path
        rel_path = self.filepaths[idx]
        full_path = os.path.join(self.data_dir, rel_path)

        # 2. Load Audio
        # torchaudio.load returns (waveform, sample_rate)
        # waveform shape: (channels, time)
        try:
            waveform, sr = torchaudio.load(full_path)
        except Exception as e:
            # Fallback for corrupted files (though metadata cleaning should have removed them)
            # Return silence
            print(f"Warning: Error loading {full_path}: {e}")
            waveform = torch.zeros(1, self.target_len)
            sr = self.target_sr

        # 3. Resample if necessary
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.target_sr
            )
            waveform = resampler(waveform)

        # 4. Mix to Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 5. Pad or Crop to Target Length
        waveform_len = waveform.shape[1]

        if waveform_len < self.target_len:
            # Pad
            padding = self.target_len - waveform_len
            # Pad at the end
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif waveform_len > self.target_len:
            # Crop
            if self.phase == "train":
                # Random Crop
                start = torch.randint(0, waveform_len - self.target_len, (1,)).item()
            else:
                # Center Crop (Deterministic)
                start = (waveform_len - self.target_len) // 2

            waveform = waveform[:, start : start + self.target_len]

        # 6. Compute Mel Spectrogram
        # Input: (1, time) -> Output: (1, n_mels, time_steps)
        spec = self.mel_spec(waveform)

        # 7. Log Transform (Amplitude to DB)
        spec = self.amplitude_to_db(spec)

        # Optional: Normalize?
        # Standard practice is often just Log-Mel. We can do instance normalization if needed.
        # For this baseline, we stick to Log-Mel.

        # 8. Get Label
        label = self.labels[idx]

        return spec, label


def get_dataframe(phase="train", debug=False, debug_size=None):
    """
    Loads the metadata DataFrame for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.
        debug (bool): If True, subset the data.
        debug_size (int): Number of samples to use in debug mode.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    if phase == "train":
        csv_path = Config.TRAIN_CSV
    elif phase == "val":
        csv_path = Config.VAL_CSV
    elif phase == "test":
        csv_path = Config.TEST_CSV
    else:
        raise ValueError(f"Unknown phase: {phase}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if debug:
        size = debug_size if debug_size is not None else Config.DEBUG_SUBSET_SIZE
        df = df.head(size).copy()
        print(f"[{phase}] Debug mode: using {len(df)} samples.")

    return df


def get_dataloader(phase, batch_size=None, num_workers=None, shuffle=None, debug=False):
    """
    Creates a DataLoader for the specified phase.

    Args:
        phase (str): 'train', 'val', or 'test'.
        batch_size (int, optional): Batch size. Defaults to Config.BATCH_SIZE.
        num_workers (int, optional): Number of workers. Defaults to Config.NUM_WORKERS.
        shuffle (bool, optional): Whether to shuffle. Defaults to True for train, False otherwise.
        debug (bool): Whether to use debug subset.

    Returns:
        DataLoader: PyTorch DataLoader.
    """
    # Set defaults based on Config if not provided
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    if num_workers is None:
        num_workers = Config.NUM_WORKERS
    if shuffle is None:
        shuffle = phase == "train"

    df = get_dataframe(phase, debug=debug)

    dataset = AudioDataset(df=df, data_dir=Config.INPUT_ROOT, phase=phase)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=(phase == "train"),  # Drop last incomplete batch during training
    )

    return loader
