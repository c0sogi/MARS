import os
import glob
import numpy as np
import pandas as pd
import librosa
import cv2
from scipy.stats import kurtosis, skew
from joblib import Parallel, delayed
import warnings

from library.config import Config
from library.utils import seed_everything

# Suppress warnings
warnings.filterwarnings("ignore")


class FeatureEngineer:
    """
    Handles feature engineering for both Tabular (LGBM) and Vision (CNN) branches.
    Implements Energy-Partitioned Tabular Features and Dual-Resolution Spectrograms with Scalar Fusion.
    """

    def __init__(self):
        self.config = Config
        seed_everything(self.config.SEED)

    def _read_sensor_data(self, file_path):
        """Reads sensor data from CSV, handling types and missing values."""
        full_path = os.path.join(self.config.INPUT_DIR, file_path)
        # Load as float32 to save memory and handle potential NaNs
        df = pd.read_csv(full_path, dtype="float32")
        # Fill NaNs with 0 (assumption: sensor dropout implies 0 signal)
        df = df.fillna(0)
        return df

    # ==========================================
    # Tabular Feature Extraction
    # ==========================================
    def _extract_sensor_tabular(self, series, sensor_name):
        """Extracts robust energy and spectral features for a single sensor series."""
        x = series.values
        res = {}

        # --- 1. Time / Energy Domain ---
        # Basic Stats
        res[f"{sensor_name}_mean"] = np.mean(x)
        res[f"{sensor_name}_std"] = np.std(x)
        res[f"{sensor_name}_min"] = np.min(x)
        res[f"{sensor_name}_max"] = np.max(x)
        res[f"{sensor_name}_kurtosis"] = kurtosis(x)
        res[f"{sensor_name}_skew"] = skew(x)

        # Quantiles (Distribution shape)
        q_vals = [0.01, 0.05, 0.95, 0.99]
        qs = np.quantile(x, q_vals)
        for q, v in zip(q_vals, qs):
            res[f"{sensor_name}_q{int(q*100)}"] = v

        # Absolute Quantiles (Magnitude/Energy distribution)
        abs_x = np.abs(x)
        abs_q_vals = [0.5, 0.95, 0.99]
        abs_qs = np.quantile(abs_x, abs_q_vals)
        for q, v in zip(abs_q_vals, abs_qs):
            res[f"{sensor_name}_abs_q{int(q*100)}"] = v

        # Crest Factor (Impulsiveness: Peak / RMS)
        rms = np.sqrt(np.mean(x**2))
        res[f"{sensor_name}_rms"] = rms
        res[f"{sensor_name}_crest_factor"] = np.max(abs_x) / (rms + 1e-9)

        # Zero Crossing Rate (Raw count)
        zcr = ((x[:-1] * x[1:]) < 0).sum()
        res[f"{sensor_name}_zcr"] = zcr

        # --- 2. Spectral Domain ---
        # FFT for Subband Energy
        fft_x = np.fft.rfft(x)
        fft_mag = np.abs(fft_x)
        freqs = np.fft.rfftfreq(len(x), d=1 / self.config.SAMPLING_RATE)

        # Log-Subband Energies (Granular spectral intensity)
        for f_low, f_high in self.config.SUBBAND_FREQS:
            mask = (freqs >= f_low) & (freqs < f_high)
            if mask.sum() > 0:
                energy = np.sum(fft_mag[mask] ** 2)
                res[f"{sensor_name}_subband_{f_low}_{f_high}_energy"] = np.log1p(energy)
            else:
                res[f"{sensor_name}_subband_{f_low}_{f_high}_energy"] = 0.0

        # Robust MFCCs
        # Ensure array is contiguous for librosa
        x_cont = np.ascontiguousarray(x)
        mfccs = librosa.feature.mfcc(
            y=x_cont,
            sr=self.config.SAMPLING_RATE,
            n_mfcc=self.config.MFCC_N,
            n_fft=1024,
            hop_length=512,
        )
        # mfccs shape: (n_mfcc, t)

        # Compute robust stats over time for each coefficient
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        mfcc_q05 = np.quantile(mfccs, 0.05, axis=1)
        mfcc_q95 = np.quantile(mfccs, 0.95, axis=1)

        # Skip 0th coeff if desired, or keep. We use 1 to N-1 based on config MFCC_N=13
        for i in range(1, self.config.MFCC_N):
            res[f"{sensor_name}_mfcc_{i}_mean"] = mfcc_mean[i]
            res[f"{sensor_name}_mfcc_{i}_std"] = mfcc_std[i]
            res[f"{sensor_name}_mfcc_{i}_q05"] = mfcc_q05[i]
            res[f"{sensor_name}_mfcc_{i}_q95"] = mfcc_q95[i]

        return res

    def _process_tabular_row(self, row):
        """Worker function to process a single segment for tabular data."""
        segment_id = int(row["segment_id"])
        file_path = row["file_path"]

        try:
            df_sensor = self._read_sensor_data(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

        features = {}
        features["segment_id"] = segment_id
        features["time_to_eruption"] = row["time_to_eruption"]

        sensor_cols = [f"sensor_{i}" for i in range(1, self.config.NUM_SENSORS + 1)]

        # Spatial Correlation (Flattened upper triangle of correlation matrix)
        if len(sensor_cols) > 1:
            corr_mat = df_sensor[sensor_cols].corr().values
            iu = np.triu_indices(len(sensor_cols), k=1)
            corr_vals = corr_mat[iu]
            for idx, val in enumerate(corr_vals):
                features[f"spatial_corr_{idx}"] = val

        # Per-Sensor Features
        for sensor in sensor_cols:
            if sensor in df_sensor.columns:
                s_feats = self._extract_sensor_tabular(df_sensor[sensor], sensor)
                features.update(s_feats)

        return features

    def process_tabular(self, split="train", load_cached_data=True):
        """
        Generates or loads tabular features for the specified split.
        """
        # Determine paths
        if split == "train":
            meta_path = self.config.TRAIN_METADATA_PATH
            cache_path = self.config.TABULAR_TRAIN_CACHE
        elif split == "val":
            meta_path = self.config.VAL_METADATA_PATH
            cache_path = self.config.TABULAR_VAL_CACHE
        elif split == "test":
            meta_path = self.config.TEST_METADATA_PATH
            cache_path = self.config.TABULAR_TEST_CACHE
        else:
            raise ValueError(f"Unknown split: {split}")

        # 1. Check Cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached tabular features from {cache_path}")
            return pd.read_parquet(cache_path)

        # 2. Load Metadata
        print(f"Generating tabular features for {split}...")
        df_meta = pd.read_csv(meta_path)

        if self.config.DEBUG:
            df_meta = df_meta.head(self.config.DEBUG_SAMPLE_SIZE)

        # 3. Parallel Extraction
        results = Parallel(n_jobs=self.config.NUM_WORKERS, backend="loky")(
            delayed(self._process_tabular_row)(row) for _, row in df_meta.iterrows()
        )

        # Filter out failures
        results = [r for r in results if r is not None]

        # 4. Save Cache
        df_features = pd.DataFrame(results)
        print(f"Saving tabular features to {cache_path}")
        df_features.to_parquet(cache_path, index=False)

        return df_features

    # ==========================================
    # Vision Feature Extraction (Spectrograms + Scalars)
    # ==========================================
    def _generate_spectrogram(self, signal, n_fft, hop_length):
        """Generates a Log-Mel Spectrogram."""
        y = signal.astype(np.float32)

        S = librosa.feature.melspectrogram(
            y=y,
            sr=self.config.SAMPLING_RATE,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=self.config.N_MELS,
            fmin=self.config.FMIN,
            fmax=self.config.FMAX,
        )

        # Log-scale: log(S + 1)
        S_log = np.log1p(S)
        return S_log

    def _normalize_spectrogram(self, S_log):
        """Applies Global Log-Max Scaling to normalize to [0, 1] range based on fixed constant."""
        denom = np.log1p(self.config.GLOBAL_MAX_MAGNITUDE)
        return S_log / denom

    def _resize_image(self, img):
        """Resizes spectrogram to the target input size (H, W)."""
        # cv2.resize expects (width, height)
        target_size = (self.config.IMG_SIZE[1], self.config.IMG_SIZE[0])
        resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        return resized

    def _process_vision_row(self, row, save_dir):
        """Worker function to process a single segment for vision data (Image + Scalars)."""
        segment_id = int(row["segment_id"])
        file_path = row["file_path"]
        save_path = os.path.join(save_dir, f"{segment_id}.npy")

        try:
            df_sensor = self._read_sensor_data(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return

        sensor_cols = [f"sensor_{i}" for i in range(1, self.config.NUM_SENSORS + 1)]

        stacked_images = []
        scalars = []

        for sensor in sensor_cols:
            if sensor in df_sensor.columns:
                sig = df_sensor[sensor].values
            else:
                sig = np.zeros(self.config.SIGNAL_LENGTH, dtype=np.float32)

            # --- 1. Dual-Resolution Spectrograms ---
            # Short Window (High Time Resolution)
            spec_short = self._generate_spectrogram(
                sig, self.config.N_FFT_SHORT, self.config.HOP_LENGTH_SHORT
            )
            spec_short = self._normalize_spectrogram(spec_short)
            spec_short = self._resize_image(spec_short)

            # Long Window (High Frequency Resolution)
            spec_long = self._generate_spectrogram(
                sig, self.config.N_FFT_LONG, self.config.HOP_LENGTH_LONG
            )
            spec_long = self._normalize_spectrogram(spec_long)
            spec_long = self._resize_image(spec_long)

            stacked_images.append(spec_short)
            stacked_images.append(spec_long)

            # --- 2. Normalized Scalar Fusion Features ---
            # Log-Total-Energy
            energy = np.sum(sig**2)
            log_energy = np.log1p(energy)

            # Global Max (Absolute)
            abs_sig = np.abs(sig)
            glob_max = np.max(abs_sig)

            # Crest Factor
            rms = np.sqrt(np.mean(sig**2))
            crest = glob_max / (rms + 1e-9)

            scalars.extend([log_energy, glob_max, crest])

        # Stack images: Shape (20, 128, 128)
        image_tensor = np.stack(stacked_images, axis=0).astype(np.float32)

        # Scalars vector: Shape (30,)
        scalar_vector = np.array(scalars, dtype=np.float32)

        # Save as dictionary
        data = {
            "image": image_tensor,
            "scalars": scalar_vector,
            "target": row["time_to_eruption"],
        }

        np.save(save_path, data)

    def process_vision(self, split="train", load_cached_data=True):
        """
        Generates vision features (Spectrograms + Scalars) and saves them as .npy files.
        """
        # Determine paths
        if split == "train":
            meta_path = self.config.TRAIN_METADATA_PATH
            save_dir = self.config.SPECTROGRAM_TRAIN_DIR
        elif split == "val":
            meta_path = self.config.VAL_METADATA_PATH
            save_dir = self.config.SPECTROGRAM_VAL_DIR
        elif split == "test":
            meta_path = self.config.TEST_METADATA_PATH
            save_dir = self.config.SPECTROGRAM_TEST_DIR
        else:
            raise ValueError(f"Unknown split: {split}")

        # Load Metadata
        df_meta = pd.read_csv(meta_path)
        if self.config.DEBUG:
            df_meta = df_meta.head(self.config.DEBUG_SAMPLE_SIZE)

        # Check Cache (Check if directory is populated)
        os.makedirs(save_dir, exist_ok=True)
        existing_files = glob.glob(os.path.join(save_dir, "*.npy"))

        # If we have enough files and caching is enabled, skip
        if load_cached_data and len(existing_files) >= len(df_meta):
            print(
                f"Vision features for {split} appear cached in {save_dir}. Skipping generation."
            )
            return

        print(f"Generating vision features for {split} into {save_dir}...")

        # Parallel Generation
        Parallel(n_jobs=self.config.NUM_WORKERS, backend="loky")(
            delayed(self._process_vision_row)(row, save_dir)
            for _, row in df_meta.iterrows()
        )
        print(f"Completed vision feature generation for {split}.")
