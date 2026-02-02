import os
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from library.config import Config


class GPUResidentDataset:
    """
    A dataset class that loads the entire dataset into GPU memory for high-throughput training.
    Handles caching, preprocessing (pad/crop), and weighted sampling.
    """

    def __init__(self, mode="train", load_cached_data=True):
        self.mode = mode
        self.device = Config.DEVICE

        # Data containers
        self.waveforms = None  # Tensor: (N, 16000)
        self.labels = None  # Tensor: (N,)
        self.fnames = None  # Numpy Array: (N,) - Test only
        self.background_noise = None  # Tensor: (M, 16000) - Train only
        self.sample_weights = None  # Tensor: (N,) - Train only

        # Configure paths based on mode
        if self.mode == "train":
            self.csv_path = Config.TRAIN_CSV
            self.wav_cache = Config.CACHE_TRAIN_WAVEFORMS
            self.lbl_cache = Config.CACHE_TRAIN_LABELS
            self.noise_cache = Config.CACHE_BACKGROUND_NOISE
        elif self.mode == "val":
            self.csv_path = Config.VAL_CSV
            self.wav_cache = Config.CACHE_VAL_WAVEFORMS
            self.lbl_cache = Config.CACHE_VAL_LABELS
        elif self.mode == "test":
            self.csv_path = Config.TEST_CSV
            self.wav_cache = Config.CACHE_TEST_WAVEFORMS
            self.lbl_cache = Config.CACHE_TEST_FNAMES

        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # 1. Load Main Data (Waveforms + Labels/Fnames)
        self._load_data(load_cached_data)

        # 2. Load Background Noise (Train only)
        if self.mode == "train":
            self._load_background_noise(load_cached_data)

        # 3. Move Data to GPU
        self.to_gpu()

        # 4. Compute Sampling Weights (Train only)
        if self.mode == "train":
            self._compute_sampling_weights()

    def _load_data(self, load_cached):
        """
        Loads data from cache or processes from scratch.
        """
        cache_exists = os.path.exists(self.wav_cache) and os.path.exists(self.lbl_cache)

        if load_cached and cache_exists:
            print(f"[{self.mode}] Loading cached data from {Config.WORKING_DIR}...")
            self.waveforms = np.load(self.wav_cache)

            if self.mode == "test":
                self.fnames = np.load(self.lbl_cache)
                # Create dummy labels for compatibility
                self.labels = np.zeros(len(self.waveforms), dtype=np.int64)
            else:
                self.labels = np.load(self.lbl_cache)
        else:
            print(f"[{self.mode}] Processing data from scratch...")
            if not os.path.exists(self.csv_path):
                raise FileNotFoundError(f"Metadata file not found: {self.csv_path}")

            df = pd.read_csv(self.csv_path)
            num_samples = len(df)

            # Pre-allocate memory
            waveforms = np.zeros((num_samples, Config.NUM_SAMPLES), dtype=np.float32)

            if self.mode == "test":
                fnames = []
            else:
                labels = np.zeros(num_samples, dtype=np.int64)

            # Process files
            for i, row in df.iterrows():
                file_path = os.path.join(Config.INPUT_ROOT, row["file_path"])

                # Read Audio
                try:
                    wav, sr = sf.read(file_path)
                except Exception as e:
                    print(
                        f"Warning: Failed to read {file_path}, using silence. Error: {e}"
                    )
                    wav = np.zeros(Config.NUM_SAMPLES, dtype=np.float32)

                # Pad or Crop to fixed length
                if len(wav) > Config.NUM_SAMPLES:
                    wav = wav[: Config.NUM_SAMPLES]
                elif len(wav) < Config.NUM_SAMPLES:
                    padding = Config.NUM_SAMPLES - len(wav)
                    wav = np.pad(wav, (0, padding), "constant")

                waveforms[i] = wav

                # Process Label/Fname
                if self.mode == "test":
                    fnames.append(row["fname"])
                else:
                    label_str = row["label"]
                    labels[i] = Config.LABEL2ID.get(
                        label_str, Config.LABEL2ID["unknown"]
                    )

            # Save to Cache
            np.save(self.wav_cache, waveforms)
            self.waveforms = waveforms

            if self.mode == "test":
                fnames = np.array(fnames)
                np.save(self.lbl_cache, fnames)
                self.fnames = fnames
                self.labels = np.zeros(len(waveforms), dtype=np.int64)
            else:
                np.save(self.lbl_cache, labels)
                self.labels = labels

            print(f"[{self.mode}] Processed and cached {num_samples} samples.")

    def _load_background_noise(self, load_cached):
        """
        Loads background noise files, splits them into 1s chunks, and caches them.
        """
        if load_cached and os.path.exists(self.noise_cache):
            print(f"[{self.mode}] Loading cached background noise bank...")
            self.background_noise = np.load(self.noise_cache)
        else:
            print(f"[{self.mode}] Processing background noise bank...")
            noise_dir = os.path.join(Config.TRAIN_AUDIO_DIR, "_background_noise_")
            noise_chunks = []

            if os.path.exists(noise_dir):
                for filename in os.listdir(noise_dir):
                    if not filename.endswith(".wav"):
                        continue

                    path = os.path.join(noise_dir, filename)
                    try:
                        wav, sr = sf.read(path)
                    except Exception:
                        continue

                    # Split into 1-second non-overlapping chunks
                    num_chunks = len(wav) // Config.NUM_SAMPLES
                    for i in range(num_chunks):
                        chunk = wav[
                            i * Config.NUM_SAMPLES : (i + 1) * Config.NUM_SAMPLES
                        ]
                        noise_chunks.append(chunk)

            if not noise_chunks:
                # Fallback to a single silent clip if no noise found
                noise_chunks.append(np.zeros(Config.NUM_SAMPLES, dtype=np.float32))

            self.background_noise = np.array(noise_chunks, dtype=np.float32)
            np.save(self.noise_cache, self.background_noise)
            print(
                f"[{self.mode}] Cached {len(self.background_noise)} background noise clips."
            )

    def to_gpu(self):
        """
        Transfers all data tensors to the configured GPU device.
        """
        if self.waveforms is not None:
            self.waveforms = torch.from_numpy(self.waveforms).to(self.device)

        if self.labels is not None:
            self.labels = torch.from_numpy(self.labels).to(self.device)

        if self.background_noise is not None:
            self.background_noise = torch.from_numpy(self.background_noise).to(
                self.device
            )

    def _compute_sampling_weights(self):
        """
        Computes sample weights for balanced sampling (1 / class_frequency).
        """
        labels_cpu = self.labels.cpu().numpy()
        # Count occurrences of each class
        class_counts = np.bincount(labels_cpu, minlength=Config.NUM_CLASSES)

        # Avoid division by zero
        class_counts = np.maximum(class_counts, 1)

        # Calculate weights: inverse frequency
        class_weights = 1.0 / class_counts

        # Map weights to each sample
        sample_weights = class_weights[labels_cpu]

        # Convert to tensor and move to GPU
        self.sample_weights = torch.from_numpy(sample_weights).float().to(self.device)

    def get_batch_indices(self, batch_size):
        """
        Generates a batch of indices using weighted random sampling on the GPU.
        """
        if self.sample_weights is None:
            # Fallback to uniform sampling if weights are not available (e.g. val/test)
            return torch.randint(
                0, len(self.waveforms), (batch_size,), device=self.device
            )

        # Weighted sampling with replacement
        return torch.multinomial(self.sample_weights, batch_size, replacement=True)

    def __len__(self):
        return len(self.waveforms)
