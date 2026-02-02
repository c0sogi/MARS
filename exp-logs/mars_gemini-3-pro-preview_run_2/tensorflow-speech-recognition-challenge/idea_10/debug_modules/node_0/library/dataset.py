import os
import random
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from library.config import Config

# Fixed Label Mapping
LABELS = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "silence",
    "unknown",
]
LABEL2IDX = {label: idx for idx, label in enumerate(LABELS)}


class SpeechCommandDataset(Dataset):
    def __init__(self, split, load_cached_data=True):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load data from cache if available.
        """
        self.split = split
        self.sample_rate = Config.SAMPLE_RATE
        self.num_samples = Config.NUM_SAMPLES  # 16000

        # Load Metadata
        if split == "train":
            self.df = pd.read_csv(Config.TRAIN_CSV)
        elif split == "val":
            self.df = pd.read_csv(Config.VAL_CSV)
        elif split == "test":
            self.df = pd.read_csv(Config.TEST_CSV)
        else:
            raise ValueError(f"Invalid split: {split}")

        # Initialize Audio Transforms
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=Config.N_FFT,
            win_length=Config.WIN_LENGTH,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            f_min=Config.F_MIN,
            f_max=Config.F_MAX,
            normalized=True,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # Initialize Augmentations
        self.time_masking = torchaudio.transforms.TimeMasking(
            time_mask_param=Config.SPEC_AUG_TIME_MASK_PARAM
        )
        self.freq_masking = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=Config.SPEC_AUG_FREQ_MASK_PARAM
        )

        # Load Background Noise Bank (for mixing and silence class)
        self.noise_bank = {}
        self.noise_keys = []
        if split == "train":
            self._load_background_noise_bank()

        # Caching Logic
        self.waveforms = None  # Will hold regular waveforms (N, 16000)
        self.labels = None  # Will hold labels (N,)
        self.is_background = None  # Boolean array
        self.map_indices = None  # Maps dataset index to waveform array index

        self._load_data(load_cached_data)

    def _load_background_noise_bank(self):
        """Loads all background noise files into memory."""
        if not os.path.exists(Config.BACKGROUND_NOISE_DIR):
            return

        for filename in os.listdir(Config.BACKGROUND_NOISE_DIR):
            if filename.endswith(".wav"):
                path = os.path.join(Config.BACKGROUND_NOISE_DIR, filename)
                try:
                    waveform, sr = torchaudio.load(path)
                    if sr != self.sample_rate:
                        resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                        waveform = resampler(waveform)
                    # Ensure mono
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)

                    self.noise_bank[filename] = waveform
                    self.noise_keys.append(filename)
                except Exception as e:
                    print(f"Warning: Failed to load noise file {filename}: {e}")

    def _load_data(self, load_cached_data):
        """Loads dataset waveforms and labels, using cache if available."""
        cache_prefix = f"{self.split}"
        waveforms_path = os.path.join(
            Config.WORKING_DIR, f"{cache_prefix}_waveforms.npy"
        )
        labels_path = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_labels.npy")
        is_bg_path = os.path.join(
            Config.WORKING_DIR, f"{cache_prefix}_is_background.npy"
        )
        map_idx_path = os.path.join(
            Config.WORKING_DIR, f"{cache_prefix}_map_indices.npy"
        )
        fnames_path = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_fnames.npy")

        # Check if cache exists
        cache_exists = (
            os.path.exists(waveforms_path)
            and os.path.exists(labels_path)
            and os.path.exists(is_bg_path)
            and os.path.exists(map_idx_path)
            and os.path.exists(fnames_path)
        )

        if load_cached_data and cache_exists:
            # print(f"Loading {self.split} data from cache...")
            self.waveforms = np.load(waveforms_path)
            self.labels = np.load(labels_path)
            self.is_background = np.load(is_bg_path)
            self.map_indices = np.load(map_idx_path)
            # We don't necessarily need fnames loaded in memory, but good for verification
        else:
            # print(f"Processing {self.split} data from scratch...")
            os.makedirs(Config.WORKING_DIR, exist_ok=True)

            waveforms_list = []
            labels_list = []
            is_bg_list = []
            map_idx_list = []
            fnames_list = []

            current_waveform_idx = 0

            for idx, row in self.df.iterrows():
                fname = row["fname"]
                label_str = row["label"]

                # Determine Label Index
                label_idx = LABEL2IDX.get(label_str, LABEL2IDX["unknown"])
                labels_list.append(label_idx)
                fnames_list.append(fname)

                # Check if it is a background noise sample (silence class)
                # In train.csv, is_background is True for silence samples
                is_bg = False
                if "is_background" in row and row["is_background"]:
                    is_bg = True

                is_bg_list.append(is_bg)

                if is_bg:
                    # For background samples, we don't store them in the fixed-size waveform array
                    # We store a dummy index (or use the fname to look up in noise_bank later)
                    # We map to -1 to indicate look up in noise_bank
                    map_idx_list.append(-1)
                else:
                    # Regular file: Load, Pad/Crop, Store
                    file_path = os.path.join(Config.INPUT_ROOT, row["file_path"])
                    try:
                        wav, sr = sf.read(file_path)
                        # Handle multi-channel (should be mono)
                        if len(wav.shape) > 1:
                            wav = np.mean(wav, axis=1)

                        # Resample if needed (assuming 16k input, but safety check)
                        # sf.read returns numpy, torchaudio returns tensor.
                        # Using numpy for cache.
                        if sr != self.sample_rate:
                            # Simple linear interp or skip?
                            # Given constraints, assume 16k or rely on robustness.
                            # EDA said all are 16k.
                            pass

                        # Pad or Crop to 16000
                        if len(wav) < self.num_samples:
                            pad_width = self.num_samples - len(wav)
                            wav = np.pad(wav, (0, pad_width), mode="constant")
                        elif len(wav) > self.num_samples:
                            # Center crop
                            start = (len(wav) - self.num_samples) // 2
                            wav = wav[start : start + self.num_samples]

                        waveforms_list.append(wav.astype(np.float32))
                        map_idx_list.append(current_waveform_idx)
                        current_waveform_idx += 1
                    except Exception as e:
                        # Fallback for corrupt files: zeros
                        waveforms_list.append(
                            np.zeros(self.num_samples, dtype=np.float32)
                        )
                        map_idx_list.append(current_waveform_idx)
                        current_waveform_idx += 1

            # Convert to numpy arrays
            self.waveforms = np.array(waveforms_list)
            self.labels = np.array(labels_list, dtype=np.int64)
            self.is_background = np.array(is_bg_list, dtype=bool)
            self.map_indices = np.array(map_idx_list, dtype=np.int64)
            fnames_array = np.array(fnames_list)

            # Save to cache
            np.save(waveforms_path, self.waveforms)
            np.save(labels_path, self.labels)
            np.save(is_bg_path, self.is_background)
            np.save(map_idx_path, self.map_indices)
            np.save(fnames_path, fnames_array)

    def _get_waveform(self, idx):
        """Retrieves waveform for index, handling silence class logic."""
        if self.is_background[idx]:
            # It's a silence sample. Pick random crop from noise bank.
            # The fname in df corresponds to the source file.
            fname = self.df.iloc[idx]["fname"]
            # Sometimes fname in csv might differ from disk name if processed?
            # Metadata script preserves original filename.

            if fname in self.noise_bank:
                noise = self.noise_bank[fname]
            elif len(self.noise_keys) > 0:
                # Fallback
                noise = self.noise_bank[random.choice(self.noise_keys)]
            else:
                return torch.zeros(1, self.num_samples)

            # Random crop
            c_len = noise.shape[1]
            if c_len > self.num_samples:
                start = random.randint(0, c_len - self.num_samples)
                waveform = noise[:, start : start + self.num_samples]
            else:
                # Pad
                waveform = torch.zeros(1, self.num_samples)
                waveform[:, :c_len] = noise
            return waveform
        else:
            # Regular sample
            mapped_idx = self.map_indices[idx]
            wav_np = self.waveforms[mapped_idx]
            return torch.from_numpy(wav_np).unsqueeze(0)  # (1, 16000)

    def _add_background_noise(self, waveform):
        """Mixes background noise into the waveform."""
        if not self.noise_keys:
            return waveform

        # Select random noise file
        noise_key = random.choice(self.noise_keys)
        noise = self.noise_bank[noise_key]

        # Select random segment
        noise_len = noise.shape[1]
        if noise_len > self.num_samples:
            start = random.randint(0, noise_len - self.num_samples)
            noise_segment = noise[:, start : start + self.num_samples]
        else:
            noise_segment = torch.zeros(1, self.num_samples)
            noise_segment[:, :noise_len] = noise

        # Calculate RMS
        signal_rms = waveform.norm(p=2)
        noise_rms = noise_segment.norm(p=2)

        if noise_rms < 1e-6:
            return waveform

        # Target SNR
        snr_db = random.uniform(Config.NOISE_MIN_SNR_DB, Config.NOISE_MAX_SNR_DB)
        snr = 10 ** (snr_db / 20)

        # Scale noise
        scale = signal_rms / (noise_rms * snr + 1e-9)

        # Mix
        mixed = waveform + scale * noise_segment
        return mixed.clamp(-1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 1. Get Waveform
        waveform = self._get_waveform(idx)
        label = self.labels[idx]

        # 2. Waveform Augmentation (Train only)
        # Don't add noise if the label is already silence (index 10)
        if self.split == "train" and label != 10:
            if random.random() < Config.NOISE_INJECTION_PROB:
                waveform = self._add_background_noise(waveform)

        # 3. Compute Spectrogram
        # Input: (1, samples), Output: (1, n_mels, time)
        spec = self.mel_spectrogram(waveform)
        spec = self.amplitude_to_db(spec)

        # 4. Spectrogram Augmentation (Train only)
        if self.split == "train":
            if random.random() < Config.SPEC_AUG_PROB:
                spec = self.time_masking(spec)
                spec = self.freq_masking(spec)

        # Return
        return spec, label


def get_dataloaders(load_cached_data=True):
    """
    Creates DataLoaders for train, val, and test splits.
    Applies WeightedRandomSampler for training to balance classes.
    """
    # 1. Create Datasets
    train_dataset = SpeechCommandDataset("train", load_cached_data=load_cached_data)
    val_dataset = SpeechCommandDataset("val", load_cached_data=load_cached_data)
    test_dataset = SpeechCommandDataset("test", load_cached_data=load_cached_data)

    # 2. Create Weighted Sampler for Training
    # Count class occurrences
    labels = train_dataset.labels
    class_counts = np.bincount(labels, minlength=Config.NUM_CLASSES)

    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1)

    # Calculate weights: inverse frequency
    class_weights = 1.0 / class_counts

    # Assign weight to each sample
    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_dataset), replacement=True
    )

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
