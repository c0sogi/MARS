import os
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset
from library.config import Config


class AudioDataset(Dataset):
    def __init__(self, mode="train"):
        """
        Args:
            mode (str): One of 'train', 'val', 'test'.
        """
        self.mode = mode

        # Load Metadata based on mode
        if mode == "train":
            self.df = pd.read_csv(Config.TRAIN_CSV)
            self.data_dir = Config.INPUT_ROOT
        elif mode == "val":
            self.df = pd.read_csv(Config.VAL_CSV)
            self.data_dir = Config.INPUT_ROOT
        elif mode == "test":
            self.df = pd.read_csv(Config.TEST_CSV)
            self.data_dir = Config.INPUT_ROOT
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Identify label columns (exclude metadata columns)
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

        # Audio Configuration
        self.sr = Config.SR
        self.duration = Config.DURATION
        self.target_samples = int(self.sr * self.duration)

        # Transforms
        # Pre-initialize resampler for common case (44100 -> 32000)
        self.resampler_44k = torchaudio.transforms.Resample(
            orig_freq=44100, new_freq=self.sr
        )

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.FMIN,
            f_max=Config.FMAX,
        )

        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row["fname"]
        filepath = os.path.join(self.data_dir, row["filepath"])

        # 1. Load Audio
        try:
            # sf.read returns (data, samplerate)
            wav, orig_sr = sf.read(filepath)
        except Exception as e:
            # Fallback for read errors: return silent audio
            wav = np.zeros(self.target_samples, dtype=np.float32)
            orig_sr = self.sr

        # Ensure float32
        wav = wav.astype(np.float32)

        # 2. Handle Channels (Mix to Mono)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)

        # 3. Resample if necessary
        wav_t = torch.from_numpy(wav)
        if orig_sr != self.sr:
            if orig_sr == 44100:
                wav_t = self.resampler_44k(wav_t.unsqueeze(0)).squeeze(0)
            else:
                # On-the-fly resampler for other rates
                resampler = torchaudio.transforms.Resample(
                    orig_freq=orig_sr, new_freq=self.sr
                )
                wav_t = resampler(wav_t.unsqueeze(0)).squeeze(0)

        # 4. Crop or Pad (Waveform level)
        curr_len = wav_t.shape[0]

        if self.mode == "train":
            # Strict length requirement for batching in training
            if curr_len < self.target_samples:
                # Pad with zeros at the end
                pad_amt = self.target_samples - curr_len
                wav_t = torch.nn.functional.pad(wav_t, (0, pad_amt))
            elif curr_len > self.target_samples:
                # Random crop
                max_start = curr_len - self.target_samples
                start = np.random.randint(0, max_start)
                wav_t = wav_t[start : start + self.target_samples]
        else:
            # Val/Test: Keep full length, but ensure min length for FFT
            if curr_len < Config.N_FFT:
                pad_amt = Config.N_FFT - curr_len
                wav_t = torch.nn.functional.pad(wav_t, (0, pad_amt))

        # 5. Compute Spectrogram
        # MelSpectrogram expects (Channel, Time) or (Time) -> (Channel, n_mels, time)
        # We add channel dim: (1, Time)
        spec = self.mel_spec(wav_t.unsqueeze(0))
        spec = self.amp_to_db(spec)  # (1, n_mels, time)

        # 7. Get Labels
        labels = row[self.label_cols].values.astype(np.float32)
        labels_t = torch.from_numpy(labels)

        return {"image": spec, "target": labels_t, "fname": fname}


def collate_fn(batch):
    """
    Collate function to handle variable length audio in validation/test.
    Pads the time dimension (dim 2) to the maximum length in the batch.
    """
    # batch is a list of dicts

    # 1. Determine max time length in the current batch
    max_time = 0
    for item in batch:
        max_time = max(max_time, item["image"].shape[2])

    images = []
    targets = []
    fnames = []

    for item in batch:
        img = item["image"]
        target = item["target"]
        fname = item["fname"]

        # Pad if necessary
        current_time = img.shape[2]
        if current_time < max_time:
            pad_amt = max_time - current_time
            # Pad last dimension (time). value=-80.0 (approx silence in dB)
            img = torch.nn.functional.pad(img, (0, pad_amt), value=-80.0)

        images.append(img)
        targets.append(target)
        fnames.append(fname)

    # Stack into batch tensors
    images = torch.stack(images)
    targets = torch.stack(targets)

    return {"image": images, "target": targets, "fname": fnames}
