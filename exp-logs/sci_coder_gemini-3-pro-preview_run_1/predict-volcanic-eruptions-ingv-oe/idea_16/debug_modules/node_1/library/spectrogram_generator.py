import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import cv2
from library.config import Config
from library.utils import CacheManager


class SpectrogramProcessor:
    """
    Handles the generation of Log-Mel Spectrograms for the Vision Branch.
    Implements Global Log-Max Scaling and caching mechanisms.
    """

    def __init__(self):
        self.config = Config
        self.cache_manager = CacheManager()
        self.output_dir = self.config.SPECTROGRAM_CACHE_DIR
        self.global_max_file = os.path.join(
            self.config.WORKING_DIR, "global_max_spectrogram.npy"
        )

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def compute_global_max(self, train_metadata_path, load_cached_data=True):
        """
        Scans the training set to find the absolute maximum value across all sensors.
        Used for Global Log-Max Scaling to preserve absolute energy levels.
        """
        # 1. Check Cache
        if load_cached_data and os.path.exists(self.global_max_file):
            if self.cache_manager.is_cache_valid(self.global_max_file):
                print(f"Loading cached global max from {self.global_max_file}")
                return float(np.load(self.global_max_file))

        print("Computing global max from training data (this may take a while)...")

        if not os.path.exists(train_metadata_path):
            # Fallback to Config default if metadata missing
            print(
                f"Warning: Train metadata not found at {train_metadata_path}. Using Config default."
            )
            return self.config.GLOBAL_MAX_READING

        df_meta = pd.read_csv(train_metadata_path)
        global_max = 0.0

        # Iterate through all training files to find the absolute max
        count = 0
        for _, row in df_meta.iterrows():
            file_path = os.path.join(self.config.INPUT_DIR, row["file_path"])
            if os.path.exists(file_path):
                try:
                    # Load data, float32
                    df = pd.read_csv(file_path, dtype="float32")
                    # Compute max absolute value
                    current_max = df.abs().max().max()
                    if current_max > global_max:
                        global_max = current_max
                except Exception:
                    pass

            count += 1
            if count % 500 == 0:
                print(f"Scanned {count}/{len(df_meta)} files for global max...")

        print(f"Global Max computed: {global_max}")

        # Save to cache
        np.save(self.global_max_file, global_max)
        self.cache_manager.update_cache_metadata(self.global_max_file)

        return float(global_max)

    def _process_single_segment(self, df_segment, global_max):
        """
        Generates a 10-channel Log-Mel Spectrogram for a single segment.
        Returns: numpy array of shape (10, 224, 224)
        """
        # Fill NaNs with 0
        df_segment = df_segment.fillna(0)

        spectrograms = []
        target_h, target_w = self.config.IMG_SIZE

        # Scaling factor: log(M_global + 1)
        log_global_max = np.log1p(global_max)
        if log_global_max < 1e-6:
            log_global_max = 1.0

        for sensor_col in self.config.SENSOR_COLS:
            if sensor_col in df_segment.columns:
                signal = df_segment[sensor_col].values.astype(np.float32)
            else:
                signal = np.zeros(self.config.SIGNAL_LENGTH, dtype=np.float32)

            # Compute Mel Spectrogram
            # We use n_mels=target_h (224) to match the target image height
            sig_tensor = torch.from_numpy(signal).float()

            transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.config.SAMPLE_RATE,
                n_fft=self.config.N_FFT,
                hop_length=self.config.HOP_LENGTH,
                n_mels=target_h,
                f_min=0.0,
                f_max=self.config.SAMPLE_RATE / 2.0,
            )

            melspec = transform(sig_tensor).numpy()

            # Convert Power to Magnitude (approximate amplitude for scaling)
            mag_spec = np.sqrt(melspec)

            # Global Log-Max Scaling
            # Formula: X_norm = log(X + 1) / log(M_global + 1)
            norm_spec = np.log1p(mag_spec) / log_global_max

            # Resize to (224, 224)
            # melspec shape is (n_mels, time_steps) -> (224, ~235)
            # cv2.resize expects (width, height)
            resized_spec = cv2.resize(norm_spec, (target_w, target_h))

            spectrograms.append(resized_spec)

        # Stack to (10, 224, 224)
        return np.stack(spectrograms, axis=0)

    def generate_dataset(self, metadata_path, load_cached_data=True):
        """
        Generates spectrograms for all segments in the provided metadata file.
        Saves results as .npy files in the cache directory.
        """
        if not os.path.exists(metadata_path):
            print(f"Metadata file not found: {metadata_path}")
            return

        # Always compute/load global max from TRAIN metadata to ensure consistent scaling
        train_meta_path = self.config.TRAIN_METADATA_PATH
        global_max = self.compute_global_max(
            train_meta_path, load_cached_data=load_cached_data
        )

        df_meta = pd.read_csv(metadata_path)
        print(
            f"Generating spectrograms for {len(df_meta)} segments from {os.path.basename(metadata_path)}..."
        )

        count = 0
        generated_count = 0

        for _, row in df_meta.iterrows():
            seg_id = str(row[self.config.SEGMENT_ID_COL])
            file_path = os.path.join(self.config.INPUT_DIR, row["file_path"])
            save_path = os.path.join(self.output_dir, f"{seg_id}.npy")

            # Check cache for individual file
            if load_cached_data and os.path.exists(save_path):
                # We assume if file exists it's valid to save time
                count += 1
                continue

            if not os.path.exists(file_path):
                continue

            try:
                df_seg = pd.read_csv(file_path, dtype="float32")
                spec_tensor = self._process_single_segment(df_seg, global_max)

                # Save as float32
                np.save(save_path, spec_tensor.astype(np.float32))
                generated_count += 1

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

            count += 1
            if count % 500 == 0:
                print(f"Processed {count}/{len(df_meta)}...")

        print(f"Finished. Generated {generated_count} new spectrograms.")
