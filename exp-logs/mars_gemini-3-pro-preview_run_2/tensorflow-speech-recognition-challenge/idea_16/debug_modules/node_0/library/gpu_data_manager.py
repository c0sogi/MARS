import os
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from library.config import Config
from library.utils import logger


class GPUDataset:
    """
    Manages the dataset on the GPU to eliminate CPU-GPU transfer bottlenecks.
    Handles loading, caching, and batching of audio data.
    """

    def __init__(self, mode="train", device=Config.DEVICE, load_cached_data=True):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            device (str): Device to load data onto ('cuda' or 'cpu').
            load_cached_data (bool): Whether to attempt loading from cache.
        """
        self.mode = mode
        self.device = device
        self.cache_dir = Config.WORKING_DIR

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Define cache filenames
        self.cache_paths = {
            "waveforms": os.path.join(self.cache_dir, f"{mode}_waveforms.npy"),
            "labels": os.path.join(self.cache_dir, f"{mode}_labels.npy"),
            "noise_bank": os.path.join(self.cache_dir, "background_noise.npy"),
        }

        # Load data
        self.waveforms, self.labels, self.noise_bank = self._load_data(load_cached_data)

        # Move to GPU
        logger.info(f"Moving {self.mode} dataset to {self.device}...")
        self.waveforms = torch.from_numpy(self.waveforms).float().to(self.device)
        self.labels = torch.from_numpy(self.labels).long().to(self.device)

        # Handle noise bank for training augmentation
        if self.mode == "train" and self.noise_bank is not None:
            # noise_bank is a list of arrays (different lengths), keep as list of tensors
            self.noise_bank = [
                torch.from_numpy(n).float().to(self.device) for n in self.noise_bank
            ]
        else:
            self.noise_bank = []

        # Setup sampling weights for training
        if self.mode == "train":
            self.weights = self._calculate_sampling_weights()
            logger.info(
                f"Initialized weighted sampling for {len(self.waveforms)} training samples."
            )

    def _load_data(self, load_cached):
        """
        Loads data from cache or processes it from scratch.
        """
        # 1. Try Loading from Cache
        if load_cached:
            if os.path.exists(self.cache_paths["waveforms"]) and os.path.exists(
                self.cache_paths["labels"]
            ):
                logger.info(f"Loading {self.mode} data from cache: {self.cache_dir}")
                waveforms = np.load(self.cache_paths["waveforms"])
                labels = np.load(self.cache_paths["labels"])

                noise_bank = None
                if self.mode == "train" and os.path.exists(
                    self.cache_paths["noise_bank"]
                ):
                    # Load noise bank (saved as object array of numpy arrays)
                    noise_bank = np.load(
                        self.cache_paths["noise_bank"], allow_pickle=True
                    )

                return waveforms, labels, noise_bank
            else:
                logger.info(
                    f"Cache not found for {self.mode}. Processing from scratch..."
                )

        # 2. Process from Scratch
        if self.mode == "train":
            csv_path = Config.TRAIN_CSV
        elif self.mode == "val":
            csv_path = Config.VAL_CSV
        else:
            csv_path = Config.TEST_CSV

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        waveforms_list = []
        labels_list = []
        noise_bank_list = []

        # Pre-calculate target length
        target_len = Config.NUM_SAMPLES

        # Helper to process one file
        def process_file(rel_path):
            full_path = os.path.join(Config.INPUT_ROOT, rel_path)
            try:
                # Load audio
                wav, sr = sf.read(full_path)

                # Pad or Crop
                if len(wav) < target_len:
                    padding = target_len - len(wav)
                    wav = np.pad(wav, (0, padding), "constant")
                elif len(wav) > target_len:
                    wav = wav[:target_len]

                return wav.astype(np.float32)
            except Exception as e:
                logger.warning(f"Failed to read {full_path}: {e}")
                return np.zeros(target_len, dtype=np.float32)

        # Iterate through metadata
        logger.info(f"Processing {len(df)} files for {self.mode} set...")

        # Special handling for Train to manage background noise
        if self.mode == "train":
            # Separate background noise files
            df_noise = df[df["is_background"] == True]
            df_commands = df[df["is_background"] == False]

            # 1. Process Standard Commands
            for _, row in df_commands.iterrows():
                wav = process_file(row["file_path"])
                waveforms_list.append(wav)
                labels_list.append(
                    Config.LABEL2ID.get(row["label"], Config.LABEL2ID["unknown"])
                )

            # 2. Process Background Noise
            # We need to:
            #   a) Add full noise to noise_bank for augmentation
            #   b) Slice noise into 1s clips for 'silence' class samples

            # Target number of silence samples to match other classes (approx 1700 total)
            # There are usually 5-6 noise files.
            silence_samples_per_file = 350

            for _, row in df_noise.iterrows():
                full_path = os.path.join(Config.INPUT_ROOT, row["file_path"])
                try:
                    raw_noise, _ = sf.read(full_path)
                    raw_noise = raw_noise.astype(np.float32)

                    # Add to bank
                    noise_bank_list.append(raw_noise)

                    # Slice for 'silence' class
                    # Stride to get distinct samples
                    max_start = len(raw_noise) - target_len
                    if max_start > 0:
                        starts = np.linspace(
                            0, max_start, silence_samples_per_file, dtype=int
                        )
                        for start in starts:
                            clip = raw_noise[start : start + target_len]
                            waveforms_list.append(clip)
                            labels_list.append(Config.LABEL2ID["silence"])

                except Exception as e:
                    logger.warning(f"Failed to process noise file {full_path}: {e}")

        else:
            # Val / Test processing (simple)
            for _, row in df.iterrows():
                wav = process_file(row["file_path"])
                waveforms_list.append(wav)
                # For test, label is placeholder, but we load it anyway
                label_str = row["label"] if "label" in row else "unknown"
                labels_list.append(
                    Config.LABEL2ID.get(label_str, Config.LABEL2ID["unknown"])
                )

        # Convert to numpy arrays
        waveforms = np.stack(waveforms_list)
        labels = np.array(labels_list, dtype=np.int64)

        # Save to cache
        logger.info(f"Saving {self.mode} data to cache...")
        np.save(self.cache_paths["waveforms"], waveforms)
        np.save(self.cache_paths["labels"], labels)

        if self.mode == "train":
            # Save noise bank as object array because lengths differ
            noise_bank_np = np.array(noise_bank_list, dtype=object)
            np.save(self.cache_paths["noise_bank"], noise_bank_np)
            return waveforms, labels, noise_bank_list

        return waveforms, labels, None

    def _calculate_sampling_weights(self):
        """
        Calculates weights for WeightedRandomSampler to handle class imbalance.
        """
        # Count samples per class
        class_counts = torch.bincount(self.labels, minlength=Config.NUM_CLASSES)

        # Avoid division by zero
        class_counts = class_counts.float()
        class_counts[class_counts == 0] = 1.0

        # Inverse frequency weights
        class_weights = 1.0 / class_counts

        # Assign weight to each sample
        sample_weights = class_weights[self.labels]

        return sample_weights

    def get_batch(self, batch_size):
        """
        Returns a batch of data.
        For training: uses weighted random sampling.
        """
        if self.mode == "train":
            # Sample indices based on weights
            indices = torch.multinomial(self.weights, batch_size, replacement=True)
            return self.waveforms[indices], self.labels[indices]
        else:
            raise NotImplementedError("For validation/test, use get_iterator()")

    def get_iterator(self, batch_size):
        """
        Generator that yields batches sequentially for validation/testing.
        """
        num_samples = len(self.waveforms)
        indices = torch.arange(num_samples, device=self.device)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]

            yield self.waveforms[batch_indices], self.labels[batch_indices]

    def __len__(self):
        return len(self.waveforms)
