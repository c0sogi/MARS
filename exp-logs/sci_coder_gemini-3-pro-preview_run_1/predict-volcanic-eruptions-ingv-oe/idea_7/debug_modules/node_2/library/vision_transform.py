import os
import numpy as np
import pandas as pd
import librosa
import warnings

from library.config import (
    INPUT_DIR,
    SENSOR_COLS,
    SAMPLING_RATE,
    SPEC_N_FFT,
    SPEC_HOP_LENGTH,
    SPEC_N_MELS,
    SPEC_FMIN,
    SPEC_FMAX,
    SEED,
    CACHE_DIR,
)
from library.utils import CacheManager, seed_everything

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class VisionTransformer:
    """
    Handles the conversion of raw sensor data into vision-ready 10-channel Log-Mel Spectrograms.
    Implements per-sample standardization and caching.
    """

    def __init__(self, cache_dir=CACHE_DIR):
        self.cache_manager = CacheManager(cache_dir=cache_dir)
        seed_everything(SEED)

    def _compute_spectrogram(self, df):
        """
        Computes 10-channel Log-Mel Spectrogram from the sensor dataframe.

        Args:
            df: DataFrame of shape (60001, 10) containing sensor readings.

        Returns:
            np.ndarray: Shape (10, n_mels, time_steps)
        """
        specs = []

        for col in SENSOR_COLS:
            # Ensure float32
            y = df[col].values.astype(np.float32)

            # Compute Mel Spectrogram
            S = librosa.feature.melspectrogram(
                y=y,
                sr=SAMPLING_RATE,
                n_fft=SPEC_N_FFT,
                hop_length=SPEC_HOP_LENGTH,
                n_mels=SPEC_N_MELS,
                fmin=SPEC_FMIN,
                fmax=SPEC_FMAX,
            )

            # Convert to log scale (dB).
            # We use the max of the current sensor as reference to normalize the peak to 0 dB.
            # This emphasizes texture over absolute loudness for this branch.
            max_val = np.max(S)
            ref_val = max_val if max_val > 0 else 1.0

            S_db = librosa.power_to_db(S, ref=ref_val)
            specs.append(S_db)

        # Stack into (10, n_mels, time)
        img = np.stack(specs, axis=0)
        return img

    def _apply_instance_standardization(self, img):
        """
        Applies per-sample standardization: (x - mean) / std.

        Args:
            img: np.ndarray of shape (10, n_mels, time)

        Returns:
            np.ndarray: Normalized tensor.
        """
        mu = np.mean(img)
        sigma = np.std(img)

        # Avoid division by zero
        if sigma < 1e-8:
            sigma = 1e-8

        img_norm = (img - mu) / sigma
        return img_norm

    def generate_spectrograms(self, metadata_df, data_type, load_cached_data=True):
        """
        Generates or loads spectrograms for the given metadata.

        Args:
            metadata_df: DataFrame containing 'segment_id' and 'file_path'.
            data_type: 'train', 'val', or 'test'.
            load_cached_data: Whether to attempt loading from cache.

        Returns:
            Tuple (X, y, ids):
                X: (N, 10, n_mels, time) float32 array
                y: (N,) float32 array (targets)
                ids: (N,) int64 array (segment_ids)
        """
        # Define cache parameters
        cache_params = {
            "data_type": data_type,
            "num_samples": len(metadata_df),
            "n_mels": SPEC_N_MELS,
            "n_fft": SPEC_N_FFT,
            "hop_length": SPEC_HOP_LENGTH,
            "transform": "log_mel_instance_norm_v1",
        }

        # Attempt to load from cache
        if load_cached_data:
            X_loaded = self.cache_manager.load(
                f"{data_type}_X", params=cache_params, ext=".npy"
            )
            y_loaded = self.cache_manager.load(
                f"{data_type}_y", params=cache_params, ext=".npy"
            )
            ids_loaded = self.cache_manager.load(
                f"{data_type}_ids", params=cache_params, ext=".npy"
            )

            if X_loaded is not None and y_loaded is not None and ids_loaded is not None:
                print(f"Loaded {data_type} spectrograms from cache.")
                return X_loaded, y_loaded, ids_loaded

        print(f"Generating {data_type} spectrograms for {len(metadata_df)} segments...")

        X_list = []
        y_list = []
        ids_list = []

        for _, row in metadata_df.iterrows():
            segment_id = row["segment_id"]
            file_path = row["file_path"]
            target = row["time_to_eruption"]

            full_path = os.path.join(INPUT_DIR, file_path)

            try:
                # Load data (use float32 to save memory/time)
                df = pd.read_csv(full_path, dtype="float32")

                # Fill NaNs (Sensor 2 often has NaNs)
                df = df.fillna(0)

                # Compute Spectrogram
                img = self._compute_spectrogram(df)

                # Apply Standardization
                img = self._apply_instance_standardization(img)

                X_list.append(img)
                y_list.append(target)
                ids_list.append(segment_id)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                # In case of error, we skip.
                # For test set, this might be critical, but assuming data integrity based on metadata check.
                continue

        # Convert to numpy arrays
        X = np.stack(X_list).astype(np.float32)
        y = np.array(y_list, dtype=np.float32)
        ids = np.array(ids_list, dtype=np.int64)

        # Save to cache
        self.cache_manager.save(X, f"{data_type}_X", params=cache_params, ext=".npy")
        self.cache_manager.save(y, f"{data_type}_y", params=cache_params, ext=".npy")
        self.cache_manager.save(
            ids, f"{data_type}_ids", params=cache_params, ext=".npy"
        )

        print(f"Saved {data_type} spectrograms to cache. Shape: {X.shape}")

        return X, y, ids


def get_vision_data(train_meta, val_meta, test_meta, load_cached_data=True):
    """
    Wrapper to generate vision data for all splits.

    Returns:
        (X_train, y_train, ids_train), (X_val, y_val, ids_val), (X_test, y_test, ids_test)
    """
    transformer = VisionTransformer()

    print("Processing Train Vision Data...")
    train_data = transformer.generate_spectrograms(
        train_meta, "train", load_cached_data
    )

    print("Processing Val Vision Data...")
    val_data = transformer.generate_spectrograms(val_meta, "val", load_cached_data)

    print("Processing Test Vision Data...")
    test_data = transformer.generate_spectrograms(test_meta, "test", load_cached_data)

    return train_data, val_data, test_data
