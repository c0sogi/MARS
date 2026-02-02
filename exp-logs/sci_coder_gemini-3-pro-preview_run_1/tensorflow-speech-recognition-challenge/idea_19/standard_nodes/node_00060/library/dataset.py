import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import get_fine_grained_labels
from library.audio_transforms import AudioProcessor


class SpeechDataset(Dataset):
    def __init__(self, split="train", mode="train", load_cached_data=True):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            mode (str): 'train' (enables augmentation) or 'infer' (disables augmentation).
            load_cached_data (bool): Whether to try loading cached balanced metadata.
        """
        self.split = split
        self.mode = mode
        self.is_training = mode == "train"

        # Initialize Audio Processor
        self.processor = AudioProcessor()

        # Label Mapping (Fine-grained)
        self.fine_labels = get_fine_grained_labels()
        self.label_to_idx = {label: idx for idx, label in enumerate(self.fine_labels)}
        self.idx_to_label = {idx: label for idx, label in enumerate(self.fine_labels)}

        # Load and Prepare Data
        self.data = self._load_data(load_cached_data)

    def _load_data(self, load_cached_data):
        """
        Loads metadata, recovers fine-grained labels, and performs balancing for training.
        """
        cache_path = os.path.join(Config.CACHE_DIR, f"{self.split}_balanced.parquet")

        # 1. Try Cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                # print(f"Loaded cached {self.split} data with {len(df)} records.")
                return df.to_dict("records")
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # 2. Load Source Metadata
        if self.split == "train":
            csv_path = Config.TRAIN_CSV
        elif self.split == "val":
            csv_path = Config.VAL_CSV
        else:
            csv_path = Config.TEST_CSV

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # 3. Recover Fine-Grained Labels (for Train/Val)
        if self.split in ["train", "val"]:
            # Extract parent folder name from filepath: train/audio/bed/file.wav -> bed
            # Note: Windows/Linux path compatibility handled by os.path.split usually,
            # but filepath in csv is likely forward slash.
            def extract_label(fp):
                parts = fp.replace("\\", "/").split("/")
                # Structure is input/train/audio/label/file.wav or train/audio/label/file.wav
                # We look for the folder before the filename
                if len(parts) >= 2:
                    return parts[-2]
                return "unknown"

            # Apply extraction
            df["fine_label"] = df["filepath"].apply(extract_label)

            # Map _background_noise_ folder to 'silence' explicitly if not already
            # (The metadata script maps it to 'silence' in 'label' col, but filepath still has _background_noise_)
            df.loc[df["filepath"].str.contains("_background_noise_"), "fine_label"] = (
                Config.SILENCE_LABEL
            )
        else:
            # Test set has no labels, use dummy
            df["fine_label"] = Config.UNKNOWN_LABEL

        # 4. Balancing (Only for Train split)
        if self.split == "train":
            df = self._balance_training_data(df)

        # 5. Save Cache
        if self.split == "train":  # Only cache complex training setups
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df.to_parquet(cache_path, index=False)

        return df.to_dict("records")

    def _balance_training_data(self, df):
        """
        Implements Variance-Aware Target-Centric Balancing.
        - Target commands: Upsample to ~2000.
        - Silence: Upsample to ~2000.
        - Aux/Unknown: Keep natural counts.
        """
        target_count = 2000
        balanced_dfs = []

        # Group by fine-grained label
        groups = df.groupby("fine_label")

        for label, group in groups:
            count = len(group)

            if label in Config.TARGET_LABELS:
                # Upsample targets
                if count < target_count:
                    # Resample with replacement
                    resampled = group.sample(
                        n=target_count, replace=True, random_state=Config.SEED
                    )
                    balanced_dfs.append(resampled)
                else:
                    # If naturally more (rare), take all or downsample? Usually they are ~1700.
                    # We'll just keep them all if > 2000, or sample 2000?
                    # Prompt says "Upsample... to ~2000". We'll cap min at 2000.
                    balanced_dfs.append(group)

            elif label == Config.SILENCE_LABEL:
                # Upsample silence (usually very few files)
                # We need many samples because we generate random crops in __getitem__
                if count > 0:
                    resampled = group.sample(
                        n=target_count, replace=True, random_state=Config.SEED
                    )
                    balanced_dfs.append(resampled)

            else:
                # Auxiliary classes - Keep natural distribution
                balanced_dfs.append(group)

        return (
            pd.concat(balanced_dfs, ignore_index=True)
            .sample(frac=1, random_state=Config.SEED)
            .reset_index(drop=True)
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        filepath = os.path.join(Config.INPUT_ROOT, record["filepath"])
        fine_label = record["fine_label"]

        # ---------------------------------------------------------
        # Audio Loading & Processing
        # ---------------------------------------------------------

        # Special handling for 'silence' during training to ensure variety
        if self.is_training and fine_label == Config.SILENCE_LABEL:
            # Instead of loading the specific file (which might be repeated 500 times),
            # we generate a random silence clip from the loaded background noises.
            if self.processor.noises:
                noise_idx = np.random.randint(0, len(self.processor.noises))
                noise_wave = self.processor.noises[noise_idx]

                # Random crop of 1 second (16000 samples)
                if noise_wave.shape[1] > Config.NUM_SAMPLES:
                    start = np.random.randint(
                        0, noise_wave.shape[1] - Config.NUM_SAMPLES
                    )
                    waveform = noise_wave[:, start : start + Config.NUM_SAMPLES]
                else:
                    # Pad if too short
                    padding = Config.NUM_SAMPLES - noise_wave.shape[1]
                    waveform = torch.nn.functional.pad(noise_wave, (0, padding))

                # Process: No extra noise injection needed for pure noise, but spectrogram needed
                spectrogram = self.processor.get_spectrogram(waveform)
                # Apply SpecAugment
                spectrogram = self.processor.apply_spec_augment(spectrogram)
            else:
                # Fallback if no noises loaded
                spectrogram = self.processor.process_audio(
                    filepath, is_training=True, should_augment=True
                )
        else:
            # Standard processing
            # For validation/test, we don't augment. For train, we do.
            spectrogram = self.processor.process_audio(
                filepath, is_training=self.is_training, should_augment=self.is_training
            )

        # ---------------------------------------------------------
        # Label Processing
        # ---------------------------------------------------------
        label_idx = self.label_to_idx.get(
            fine_label, 0
        )  # Default to 0 if issue, though shouldn't happen

        # Return filename for submission mapping
        fname = os.path.basename(filepath)

        return spectrogram, label_idx, fname


def get_dataloader(
    split,
    batch_size=Config.BATCH_SIZE,
    mode="train",
    shuffle=True,
    num_workers=Config.NUM_WORKERS,
):
    """
    Factory function to create DataLoaders.
    """
    dataset = SpeechDataset(split=split, mode=mode)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=(
            mode == "train"
        ),  # Drop last incomplete batch during training for stability
    )

    return loader
