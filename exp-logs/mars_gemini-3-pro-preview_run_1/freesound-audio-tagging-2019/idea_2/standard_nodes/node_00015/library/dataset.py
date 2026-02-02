import os
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from library.config import CFG
from library.utils import set_seed

# Set seed for reproducibility in data processing
set_seed(CFG.seed)


class AudioDataset(Dataset):
    def __init__(self, df, mode="train"):
        """
        Args:
            df (pd.DataFrame): Dataframe containing metadata.
            mode (str): 'train', 'val', or 'test'. Controls augmentation and cropping.
        """
        self.df = df.reset_index(drop=True)
        self.mode = mode

        # Identify label columns: exclude metadata columns
        # Metadata columns: fname, labels, filepath, and potentially analysis cols like duration
        # We assume all other columns are classes
        exclude_cols = {
            "fname",
            "labels",
            "filepath",
            "label_count",
            "duration",
            "sample_rate",
            "n_channels",
        }
        self.label_cols = [c for c in self.df.columns if c not in exclude_cols]

        # Audio Processing / Transforms
        # 1. Resampler: 44.1kHz -> 32kHz
        # We assume input is 44.1kHz based on EDA.
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=44100, new_freq=CFG.sample_rate
        )

        # 2. Spectrogram: Log-Mel
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=CFG.sample_rate,
            n_fft=CFG.n_fft,
            win_length=CFG.n_fft,
            hop_length=CFG.hop_length,
            n_mels=CFG.n_mels,
            f_min=CFG.fmin,
            f_max=CFG.fmax,
            normalized=False,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)

        # 3. Augmentations (Train only)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=CFG.spec_aug_freq_mask
        )
        self.time_mask = torchaudio.transforms.TimeMasking(
            time_mask_param=CFG.spec_aug_time_mask
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(CFG.input_root, row["filepath"])

        # 1. Load Audio
        try:
            # sf.read returns (samples, channels) or (samples,)
            wav, sr = sf.read(file_path)
            wav = wav.astype(np.float32)
        except Exception as e:
            # Fallback for corrupted files (though dataset should be clean)
            # Create a silent signal of 5 seconds
            sr = 44100
            wav = np.zeros(sr * 5, dtype=np.float32)

        # Convert to Mono if necessary
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)

        # Convert to Tensor
        wav_tensor = torch.from_numpy(wav)

        # 2. Resample
        if sr != CFG.sample_rate:
            if sr == 44100:
                wav_tensor = self.resampler(wav_tensor)
            else:
                # Dynamic resampling for unexpected rates
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=CFG.sample_rate
                )
                wav_tensor = resampler(wav_tensor)

        # 3. Length Handling (Crop or Pad)
        target_samples = int(CFG.sample_rate * CFG.duration)
        num_samples = wav_tensor.shape[0]

        if self.mode == "train":
            # Random Crop or Pad to fixed length
            if num_samples > target_samples:
                start = np.random.randint(0, num_samples - target_samples)
                wav_tensor = wav_tensor[start : start + target_samples]
            elif num_samples < target_samples:
                pad_amt = target_samples - num_samples
                # Pad with zeros (silence)
                wav_tensor = torch.nn.functional.pad(wav_tensor, (0, pad_amt))
        else:
            # Val/Test: Keep full length.
            # If extremely short, pad to avoid empty spectrograms?
            # Minimum 1 hop length required.
            if num_samples < CFG.hop_length:
                pad_amt = CFG.hop_length - num_samples + 100
                wav_tensor = torch.nn.functional.pad(wav_tensor, (0, pad_amt))

        # 4. Compute Spectrogram
        # Input: (Time) -> Output: (n_mels, Time)
        spec = self.melspec(wav_tensor)
        spec = self.db_transform(spec)

        # 5. Normalization (Instance Level)
        # Standardize to mean 0, std 1 per sample
        mean = spec.mean()
        std = spec.std()
        spec = (spec - mean) / (std + 1e-6)

        # Add Channel Dimension: (1, n_mels, Time)
        spec = spec.unsqueeze(0)

        # 6. SpecAugment (Train only)
        if self.mode == "train":
            spec = self.freq_mask(spec)
            spec = self.time_mask(spec)

        # 7. Prepare Targets
        # For test set, labels might be all 0s, which is fine
        targets = row[self.label_cols].values.astype(np.float32)
        targets = torch.tensor(targets)

        return {"image": spec, "target": targets, "fname": row["fname"]}


def collate_fn(batch):
    """
    Custom collate function to handle variable length audio in batch.
    Pads the time dimension to the maximum length in the batch.
    """
    # batch is a list of dicts

    # 1. Find max time dimension
    max_time = 0
    for item in batch:
        max_time = max(max_time, item["image"].shape[2])

    images = []
    targets = []
    fnames = []

    for item in batch:
        img = item["image"]  # (1, n_mels, time)
        current_time = img.shape[2]

        # Pad time dimension if necessary
        if current_time < max_time:
            pad_amt = max_time - current_time
            # Pad last dimension (time)
            # value=0 is appropriate as data is standardized (mean~0)
            img = torch.nn.functional.pad(img, (0, pad_amt), value=0)

        images.append(img)
        targets.append(item["target"])
        fnames.append(item["fname"])

    return {
        "image": torch.stack(images),
        "target": torch.stack(targets),
        "fname": fnames,
    }


def get_dataloader(split, load_cached_data=False, debug=CFG.debug):
    """
    Factory function to create DataLoaders.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Placeholder for caching logic (not used for on-the-fly audio).
        debug (bool): If True, subsets data for quick debugging.
    """
    # Select Metadata File
    if split == "train":
        csv_path = CFG.train_csv
        mode = "train"
        shuffle = True
        batch_size = CFG.batch_size
    elif split == "val":
        csv_path = CFG.val_csv
        mode = "val"
        shuffle = False
        batch_size = CFG.inference_batch_size
    elif split == "test":
        csv_path = CFG.test_csv
        mode = "test"
        shuffle = False
        batch_size = CFG.inference_batch_size
    else:
        raise ValueError(f"Unknown split: {split}")

    # Load DataFrame
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Debug Mode
    if debug:
        df = df.sample(n=min(len(df), 100), random_state=CFG.seed).reset_index(
            drop=True
        )
        print(f"DEBUG MODE: Loaded {len(df)} samples for {split}.")

    # Create Dataset
    dataset = AudioDataset(df, mode=mode)

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=CFG.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(split == "train"),  # Drop last incomplete batch only during training
    )

    return loader
