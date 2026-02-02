import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
import torchvision.transforms as VT
from joblib import Parallel, delayed
from library.config import Config


class SpectrogramGenerator:
    """
    Handles the generation, processing, and caching of Log-Mel Spectrograms
    from raw sensor data for Branch B (2D-CNN).
    """

    def __init__(self):
        self.working_dir = Config.WORKING_DIR
        self.input_dir = Config.INPUT_DIR
        self.sensors = Config.SENSORS
        self.sample_rate = Config.SAMPLE_RATE

        # Spectrogram Parameters
        self.n_fft = Config.N_FFT
        self.hop_length = Config.HOP_LENGTH
        self.n_mels = Config.N_MELS
        self.f_min = Config.F_MIN
        self.f_max = Config.F_MAX
        self.img_size = Config.IMG_SIZE  # (Height, Width)

        # Target Scaling
        self.target_scaling = Config.TARGET_SCALING

        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)

    def _get_transforms(self):
        """
        Initializes the transformation pipeline.
        Created locally to ensure thread/process safety during parallel execution.
        """
        # 1. Mel Spectrogram
        mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            power=2.0,
        )

        # 2. Amplitude to DB (Log scaling)
        db_transform = T.AmplitudeToDB(stype="power", top_db=80)

        # 3. Resize to fixed dimensions (Height, Width)
        resize_transform = VT.Resize(self.img_size, antialias=True)

        return mel_transform, db_transform, resize_transform

    def _process_file(self, segment_id, file_path, target):
        """
        Worker function to process a single CSV file into a spectrogram tensor.
        """
        full_path = os.path.join(self.input_dir, file_path)

        try:
            # Instantiate transforms locally for the worker process
            mel_transform, db_transform, resize_transform = self._get_transforms()

            # Check file existence
            if not os.path.exists(full_path):
                return None, None, None

            # Read CSV with float32 to save memory
            df = pd.read_csv(full_path, dtype="float32")
            df = df.fillna(0)

            # Extract waveforms: Shape (Time, Sensors) -> (Sensors, Time)
            waveforms = []
            for sensor in self.sensors:
                if sensor in df.columns:
                    waveforms.append(df[sensor].values)
                else:
                    # Pad with zeros if sensor is missing
                    waveforms.append(np.zeros(len(df), dtype=np.float32))

            # Stack to (10, Time)
            waveform_tensor = torch.tensor(np.stack(waveforms), dtype=torch.float32)

            # Compute Spectrogram: (10, n_mels, time_steps)
            # MelSpectrogram treats the first dim as channel/batch if configured,
            # but here we pass (Channels, Time) and it processes each channel.
            spec = mel_transform(waveform_tensor)

            # Convert to DB (Log scale)
            spec = db_transform(spec)

            # Resize to fixed size (10, Height, Width)
            spec = resize_transform(spec)

            # Convert to numpy (float32) for storage
            spec_np = spec.numpy().astype(np.float32)

            return segment_id, spec_np, target

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None, None, None

    def get_dataset(self, dataset_type="train", load_cached_data=True):
        """
        Main method to retrieve the spectrogram dataset.
        Manages caching and parallel processing.

        Args:
            dataset_type (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load from cache if available.

        Returns:
            X (np.ndarray): Spectrogram tensor (N, 10, H, W).
            y (np.ndarray or None): Target values (N,).
            ids (np.ndarray): Segment IDs (N,).
        """
        cache_X_path = os.path.join(
            self.working_dir, f"spectrograms_X_{dataset_type}.npy"
        )
        cache_y_path = os.path.join(
            self.working_dir, f"spectrograms_y_{dataset_type}.npy"
        )
        cache_ids_path = os.path.join(
            self.working_dir, f"spectrograms_ids_{dataset_type}.npy"
        )

        # 1. Try Load Cache
        if (
            load_cached_data
            and os.path.exists(cache_X_path)
            and os.path.exists(cache_ids_path)
        ):
            print(
                f"Loading cached spectrograms for {dataset_type} from {self.working_dir}..."
            )
            X = np.load(cache_X_path)
            ids = np.load(cache_ids_path)

            if dataset_type != "test" and os.path.exists(cache_y_path):
                y = np.load(cache_y_path)
            else:
                y = None

            return X, y, ids

        # 2. Compute from Scratch
        print(f"Generating spectrograms for {dataset_type}...")

        # Identify Metadata File
        if dataset_type == "train":
            meta_path = Config.TRAIN_METADATA
        elif dataset_type == "val":
            meta_path = Config.VAL_METADATA
        elif dataset_type == "test":
            meta_path = Config.TEST_METADATA
        else:
            raise ValueError(f"Invalid dataset_type: {dataset_type}")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        df_meta = pd.read_csv(meta_path)

        # Prepare arguments for parallel processing
        tasks = []
        for _, row in df_meta.iterrows():
            seg_id = row["segment_id"]
            f_path = row["file_path"]
            # Target is 0 for test, actual value for train/val
            target = row["time_to_eruption"] if "time_to_eruption" in row else 0
            tasks.append((seg_id, f_path, target))

        # Execute Parallel Processing
        # Using n_jobs=-1 to maximize throughput for IO/CPU bound task
        results = Parallel(n_jobs=-1, verbose=0)(
            delayed(self._process_file)(s, f, t) for s, f, t in tasks
        )

        # Filter out any failed processings
        results = [r for r in results if r[0] is not None]

        if not results:
            raise RuntimeError(
                f"No data generated for {dataset_type}. Check input files."
            )

        # Unpack results
        ids_list, X_list, y_list = zip(*results)

        # Stack into arrays
        X = np.stack(X_list)  # Shape: (N, 10, H, W)
        ids = np.array(ids_list)
        y = np.array(y_list, dtype=np.float32)

        # Apply Target Scaling (Log1p) for Train/Val
        if dataset_type != "test":
            if self.target_scaling == "log1p":
                y = np.log1p(y)
        else:
            y = None

        # 3. Save to Cache
        print(f"Saving spectrograms to {self.working_dir}...")
        np.save(cache_X_path, X)
        np.save(cache_ids_path, ids)
        if y is not None:
            np.save(cache_y_path, y)

        return X, y, ids
