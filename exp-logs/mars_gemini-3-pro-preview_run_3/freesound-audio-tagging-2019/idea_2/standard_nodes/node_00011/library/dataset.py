import os
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from library.config import Config


def get_class_names():
    """
    Reads the sample submission file to retrieve the list of classes
    in the correct order expected for submission.
    """
    ss_path = os.path.join(Config.input_root, "sample_submission.csv")
    df = pd.read_csv(ss_path)
    # The first column is 'fname', the rest are class labels
    return df.columns[1:].tolist()


class AudioDataset(Dataset):
    def __init__(self, metadata_path, mode="train"):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            mode (str): 'train', 'val', or 'test'. Controls augmentation and label processing.
        """
        super().__init__()
        self.mode = mode
        self.metadata_path = metadata_path

        # Load Metadata
        self.df = pd.read_csv(metadata_path)

        # Handle Debug Mode
        if Config.debug:
            self.df = self.df.iloc[: Config.debug_sample_size].reset_index(drop=True)

        # Class Mapping
        self.classes = get_class_names()
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        # Audio Parameters
        self.sr = Config.sample_rate
        self.duration = Config.duration
        self.target_length = self.sr * self.duration  # Samples

        # Transforms
        # 1. Mel Spectrogram
        self.mel_spec = T.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=Config.n_fft,
            hop_length=Config.hop_length,
            n_mels=Config.n_mels,
            f_min=Config.fmin,
            f_max=Config.fmax,
        )

        # 2. Amplitude to DB (Log Scale)
        self.amplitude_to_db = T.AmplitudeToDB()

        # 3. Augmentations (Train only)
        self.time_masking = T.TimeMasking(time_mask_param=Config.spec_augment_time_mask)
        self.freq_masking = T.FrequencyMasking(
            freq_mask_param=Config.spec_augment_freq_mask
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row["fname"]

        # Construct full path
        # Metadata 'filepath' is relative to Config.input_root
        audio_path = os.path.join(Config.input_root, row["filepath"])

        # 1. Load Audio
        # Load returns (channels, time)
        waveform, sr = torchaudio.load(audio_path)

        # 2. Resample if necessary
        if sr != self.sr:
            resampler = T.Resample(sr, self.sr)
            waveform = resampler(waveform)

        # 3. Convert to Mono
        # If multiple channels, average them
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 4. Adjust Duration (Pad or Truncate)
        # We want exactly self.target_length samples
        current_len = waveform.shape[1]

        if current_len < self.target_length:
            # Pad with silence on the right
            padding = self.target_length - current_len
            # pad tuple is (pad_left, pad_right)
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif current_len > self.target_length:
            # Truncate (keep the beginning)
            waveform = waveform[:, : self.target_length]

        # 5. Generate Log-Mel Spectrogram
        # Input: (1, samples) -> Output: (1, n_mels, time_steps)
        spec = self.mel_spec(waveform)
        spec = self.amplitude_to_db(spec)

        # 6. Apply Augmentation (Train only)
        if self.mode == "train":
            spec = self.time_masking(spec)
            spec = self.freq_masking(spec)

        # 7. Process Labels
        target = torch.zeros(self.num_classes, dtype=torch.float32)

        # Test set has no labels
        if self.mode != "test":
            if pd.notna(row.get("labels")) and row["labels"] != "":
                label_list = row["labels"].split(",")
                for lbl in label_list:
                    if lbl in self.class_to_idx:
                        target[self.class_to_idx[lbl]] = 1.0

        return spec, target, fname
