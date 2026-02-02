import os
import random
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from library.config import Config
from library.utils import set_seed

# Ensure reproducible behavior for audio operations
torchaudio.set_audio_backend("soundfile")


class SpeechDataset(Dataset):
    def __init__(
        self,
        df,
        label_encoder,
        noise_files=None,
        mode="train",
        config=Config,
    ):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe containing 'filepath' and 'label'.
            label_encoder (LabelEncoder): Fitted encoder for labels.
            noise_files (list): List of filepaths to background noise audio for injection.
            mode (str): 'train', 'val', or 'test'.
            config (Config): Configuration object.
        """
        self.df = df
        self.label_encoder = label_encoder
        self.noise_files = noise_files if noise_files else []
        self.mode = mode
        self.config = config
        self.sample_rate = config.sample_rate
        self.target_length = int(config.sample_rate * config.duration)

        # Pre-instantiate transforms to avoid overhead
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # SpecAugment
        self.time_masking = torchaudio.transforms.TimeMasking(
            time_mask_param=config.time_mask_param
        )
        self.freq_masking = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=config.freq_mask_param
        )

    def __len__(self):
        return len(self.df)

    def _load_audio(self, filepath, is_silence_label=False):
        """
        Loads audio. If label is silence (background noise file), crops a random segment.
        Otherwise loads and pads/crops to target length.
        """
        full_path = os.path.join(self.config.input_dir, filepath)

        try:
            # Get file info first to handle long noise files efficiently
            info = sf.info(full_path)
            file_len = info.frames

            if is_silence_label and file_len > self.target_length:
                # Random crop from long background noise file
                offset = random.randint(0, file_len - self.target_length)
                waveform, _ = torchaudio.load(
                    full_path, frame_offset=offset, num_frames=self.target_length
                )
            else:
                # Standard load
                waveform, _ = torchaudio.load(full_path)
        except Exception as e:
            # Fallback for corrupted files: return silence
            # print(f"Warning: Failed to load {filepath}: {e}")
            return torch.zeros(1, self.target_length)

        # Convert to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Pad or Crop to target length
        current_len = waveform.shape[1]
        if current_len < self.target_length:
            padding = self.target_length - current_len
            # Pad symmetrically or at end? Simple zero pad at end is standard for commands
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif current_len > self.target_length:
            waveform = waveform[:, : self.target_length]

        return waveform

    def _inject_noise(self, waveform):
        """
        Injects random background noise into the waveform.
        """
        if not self.noise_files or random.random() < 0.2:  # 20% chance to skip noise
            return waveform

        noise_path = random.choice(self.noise_files)
        noise_wave = self._load_audio(noise_path, is_silence_label=True)

        # Calculate SNR
        # SNR = 10 * log10(Signal_Power / Noise_Power)
        # We want to scale noise to achieve target SNR

        signal_power = waveform.pow(2).mean()
        noise_power = noise_wave.pow(2).mean()

        if noise_power == 0 or signal_power == 0:
            return waveform

        target_snr_db = random.uniform(
            self.config.noise_snr_min, self.config.noise_snr_max
        )
        target_snr_linear = 10 ** (target_snr_db / 10)

        # Desired Noise Power = Signal Power / SNR
        desired_noise_power = signal_power / target_snr_linear
        scale = torch.sqrt(desired_noise_power / noise_power)

        noisy_waveform = waveform + (noise_wave * scale)

        # Clamp to valid audio range
        return torch.clamp(noisy_waveform, -1.0, 1.0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row["filepath"]
        label_str = row["label"]

        # 1. Load Audio
        is_silence = label_str == self.config.silence_label
        waveform = self._load_audio(filepath, is_silence_label=is_silence)

        # 2. Augmentation (Train only)
        if self.mode == "train":
            # Noise Injection (only for non-silence classes to avoid double noise)
            if not is_silence:
                waveform = self._inject_noise(waveform)

        # 3. Spectrogram Conversion
        spec = self.mel_spectrogram(waveform)
        spec = self.amplitude_to_db(spec)

        # 4. SpecAugment (Train only)
        if self.mode == "train":
            spec = self.time_masking(spec)
            spec = self.freq_masking(spec)

        # 5. Prepare Label
        # For test set, label might be placeholder, but we still encode it if possible
        # or return a dummy.
        if self.mode == "test":
            target = -1  # Dummy
        else:
            target = self.label_encoder.transform([label_str])[0]

        return spec, target


def get_dataloaders(load_cached_data=True):
    """
    Prepares DataLoaders for Train, Validation, and Test.
    Implements Variance-Aware Balancing and Caching.
    """
    set_seed(Config.seed)
    Config.create_dirs()

    # ---------------------------------------------------------
    # 1. Load Metadata
    # ---------------------------------------------------------
    df_train = pd.read_csv(Config.train_metadata_path)
    df_val = pd.read_csv(Config.val_metadata_path)
    df_test = pd.read_csv(Config.test_metadata_path)

    # ---------------------------------------------------------
    # 2. Variance-Aware Balancing (Train Only)
    # ---------------------------------------------------------
    balanced_train_path = os.path.join(Config.working_dir, "train_balanced.parquet")

    if load_cached_data and os.path.exists(balanced_train_path):
        print(f"Loading cached balanced training data from {balanced_train_path}")
        df_train_balanced = pd.read_parquet(balanced_train_path)
    else:
        print("Balancing training data...")

        # Separate categories
        # Targets: yes, no, up, down, ...
        # Silence: label == 'silence'
        # Auxiliary: Everything else

        target_mask = df_train["label"].isin(Config.target_labels)
        silence_mask = df_train["label"] == Config.silence_label
        aux_mask = (~target_mask) & (~silence_mask)

        df_targets = df_train[target_mask]
        df_silence = df_train[silence_mask]
        df_aux = df_train[aux_mask]

        balanced_dfs = []

        # A. Upsample Targets to ~2000
        target_count = 2000
        for label in Config.target_labels:
            subset = df_targets[df_targets["label"] == label]
            if len(subset) == 0:
                continue
            # Resample with replacement
            resampled = subset.sample(
                n=target_count, replace=True, random_state=Config.seed
            )
            balanced_dfs.append(resampled)

        # B. Upsample Silence to ~2000
        # Silence has very few files, but we crop them randomly in Dataset.
        # So we just duplicate the rows.
        if len(df_silence) > 0:
            resampled_silence = df_silence.sample(
                n=target_count, replace=True, random_state=Config.seed
            )
            balanced_dfs.append(resampled_silence)

        # C. Keep Auxiliary as is (Natural Distribution)
        # This preserves variance for the "Unknown" class
        balanced_dfs.append(df_aux)

        df_train_balanced = (
            pd.concat(balanced_dfs)
            .sample(frac=1, random_state=Config.seed)
            .reset_index(drop=True)
        )

        # Cache
        df_train_balanced.to_parquet(balanced_train_path)
        print(f"Balanced training data saved. Total samples: {len(df_train_balanced)}")

    # ---------------------------------------------------------
    # 3. Label Encoding
    # ---------------------------------------------------------
    # We fit the encoder on ALL labels present in the balanced training set.
    # This includes 'bed', 'bird', 'yes', 'silence', etc.
    unique_labels = sorted(df_train_balanced["label"].unique())
    le = LabelEncoder()
    le.fit(unique_labels)

    # Update Config.num_classes to match the actual number of classes found
    Config.num_classes = len(le.classes_)
    print(f"Updated Config.num_classes to {Config.num_classes}")

    # Verify we cover validation labels too (should be subset of train usually)
    # If val has labels not in train (unlikely with this split), they will cause error.
    # The metadata split ensures strat/group split, so classes should be covered.

    # ---------------------------------------------------------
    # 4. Noise Files for Injection
    # ---------------------------------------------------------
    # Extract paths of background noise files from the original train df
    # usually marked as silence.
    noise_df = df_train[df_train["label"] == Config.silence_label]
    noise_files = noise_df["filepath"].tolist()

    # ---------------------------------------------------------
    # 5. Create Datasets
    # ---------------------------------------------------------
    # Debug mode: reduce size
    if Config.debug:
        df_train_balanced = df_train_balanced.iloc[: Config.debug_sample_size]
        df_val = df_val.iloc[: Config.debug_sample_size]
        df_test = df_test.iloc[: Config.debug_sample_size]

    train_dataset = SpeechDataset(
        df_train_balanced, le, noise_files=noise_files, mode="train", config=Config
    )

    val_dataset = SpeechDataset(df_val, le, noise_files=None, mode="val", config=Config)

    test_dataset = SpeechDataset(
        df_test,
        le,  # Note: Test labels are placeholders, but we pass LE for consistency
        noise_files=None,
        mode="test",
        config=Config,
    )

    # ---------------------------------------------------------
    # 6. Create DataLoaders
    # ---------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for training stability
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, le
