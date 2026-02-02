import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from library.config import Config
from library.utils import seed_everything


class FeatureEngineer:
    """
    Handles signal processing and feature extraction for the Magnitude-Modulated Hybrid Ensemble.
    Generates:
    1. Energy-Partitioned Tabular Features (for LightGBM).
    2. Global-Scaled Spectrograms (for EfficientNet).
    3. Global Energy Scalars (for FiLM Modulation).
    """

    def __init__(self):
        seed_everything(Config.SEED)
        self.device = torch.device("cpu")  # CPU sufficient for data loading/prep

        # Spectrogram Transform
        # Output shape: (n_mels, time) -> (256, ~235)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=Config.SAMPLING_RATE,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.IMG_SIZE[0],
            power=2.0,
        ).to(self.device)

        # MFCC Transform for Tabular Features
        self.mfcc_transform = T.MFCC(
            sample_rate=Config.SAMPLING_RATE,
            n_mfcc=13,
            melkwargs={
                "n_fft": Config.N_FFT,
                "hop_length": Config.HOP_LENGTH,
                "n_mels": 40,
            },
        ).to(self.device)

    def _load_signal(self, file_path: str) -> pd.DataFrame:
        """
        Loads a sensor data CSV file.
        """
        try:
            # Load as float32 to handle potential NaNs and memory
            df = pd.read_csv(file_path, dtype="float32")
            # Fill NaNs with 0 (sensor dropout)
            df = df.fillna(0.0)
            return df
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def extract_tabular_features(self, df_signal: pd.DataFrame) -> dict:
        """
        Extracts statistical and spectral features for the Tabular Branch.
        Includes Time Domain stats, Robust MFCCs, Sub-band Energies, and Spatial Correlation.
        """
        features = {}
        sensors = df_signal.columns
        data_np = df_signal.values

        # -------------------------------------------------------
        # 1. Time Domain Statistics (Vectorized)
        # -------------------------------------------------------
        means = np.mean(data_np, axis=0)
        stds = np.std(data_np, axis=0)
        mins = np.min(data_np, axis=0)
        maxs = np.max(data_np, axis=0)

        # Absolute Quantiles (Robust to sign, captures magnitude distribution)
        abs_data = np.abs(data_np)
        q_levels = [0.05, 0.25, 0.50, 0.75, 0.95]
        quantiles = np.quantile(abs_data, q_levels, axis=0)  # Shape: (5, n_sensors)

        # Zero Crossing Rate (Raw)
        # (n_samples-1, n_sensors) boolean array
        zcr = ((data_np[:-1, :] * data_np[1:, :]) < 0).sum(axis=0)

        for i, sensor in enumerate(sensors):
            features[f"{sensor}_mean"] = means[i]
            features[f"{sensor}_std"] = stds[i]
            features[f"{sensor}_min"] = mins[i]
            features[f"{sensor}_max"] = maxs[i]
            features[f"{sensor}_zcr"] = zcr[i]

            # Quantiles
            for j, q in enumerate(q_levels):
                features[f"{sensor}_abs_q{int(q*100)}"] = quantiles[j, i]

            # Crest Factor: Peak / RMS
            rms = np.sqrt(np.mean(data_np[:, i] ** 2))
            peak = np.max(abs_data[:, i])
            features[f"{sensor}_crest"] = peak / (rms + 1e-9)

        # -------------------------------------------------------
        # 2. Spectral Features (Robust MFCCs & Sub-band Energy)
        # -------------------------------------------------------
        data_tensor = torch.tensor(data_np.T, dtype=torch.float32).to(self.device)

        # MFCC
        mfcc = self.mfcc_transform(data_tensor)  # (n_sensors, n_mfcc, time)

        # Robust Stats on MFCCs (Mean, Std, Q05, Q95)
        mfcc_mean = torch.mean(mfcc, dim=2).numpy()
        mfcc_std = torch.std(mfcc, dim=2).numpy()
        mfcc_q05 = torch.quantile(mfcc, 0.05, dim=2).numpy()
        mfcc_q95 = torch.quantile(mfcc, 0.95, dim=2).numpy()

        for i, sensor in enumerate(sensors):
            # Use all 13 coefficients
            for c in range(13):
                features[f"{sensor}_mfcc_mean_{c}"] = mfcc_mean[i, c]
                features[f"{sensor}_mfcc_std_{c}"] = mfcc_std[i, c]
                features[f"{sensor}_mfcc_q05_{c}"] = mfcc_q05[i, c]
                features[f"{sensor}_mfcc_q95_{c}"] = mfcc_q95[i, c]

        # Sub-band Energies
        # Compute FFT magnitude
        fft = torch.fft.rfft(data_tensor, dim=1)
        mag = torch.abs(fft)  # (n_sensors, freq_bins)
        n_bins = mag.shape[1]

        # Split into 5 bands
        bands = np.array_split(np.arange(n_bins), 5)
        for b_idx, band_idxs in enumerate(bands):
            # Log-Sum-Squared Energy in band
            energy = torch.sum(mag[:, band_idxs] ** 2, dim=1).numpy()
            for i, sensor in enumerate(sensors):
                features[f"{sensor}_band_energy_{b_idx}"] = np.log1p(energy[i])

        # -------------------------------------------------------
        # 3. Spatial Correlation
        # -------------------------------------------------------
        if data_np.shape[1] > 1:
            corr_matrix = np.corrcoef(data_np.T)
            # Extract upper triangle excluding diagonal
            upper_indices = np.triu_indices(data_np.shape[1], k=1)
            corr_values = corr_matrix[upper_indices]
            features["spatial_corr_mean"] = np.mean(corr_values)
            features["spatial_corr_std"] = np.std(corr_values)
        else:
            features["spatial_corr_mean"] = 0.0
            features["spatial_corr_std"] = 0.0

        return features

    def extract_scalars(self, df_signal: pd.DataFrame) -> dict:
        """
        Extracts global energy scalars for the FiLM modulation module.
        Returns a dictionary mapping scalar names to values.
        """
        scalars = {}
        data_np = df_signal.values
        sensors = df_signal.columns

        for i, sensor in enumerate(sensors):
            x = data_np[:, i]
            # Log-Energy
            log_energy = np.log1p(np.sum(x**2))
            # Max Absolute Amplitude
            max_val = np.max(np.abs(x))
            # Crest Factor
            rms = np.sqrt(np.mean(x**2))
            crest = max_val / (rms + 1e-9)

            scalars[f"scalar_{sensor}_log_energy"] = log_energy
            scalars[f"scalar_{sensor}_max"] = max_val
            scalars[f"scalar_{sensor}_crest"] = crest

        return scalars

    def generate_spectrogram(self, df_signal: pd.DataFrame) -> np.ndarray:
        """
        Generates a Single-Resolution Log-Mel Spectrogram with Global Log-Max Scaling.
        Output Shape: (n_sensors, 256, 256)
        """
        # (n_sensors, n_samples)
        data_tensor = torch.tensor(df_signal.values.T, dtype=torch.float32).to(
            self.device
        )

        # 1. Mel Spectrogram
        spec = self.mel_transform(data_tensor)  # (n_sensors, 256, time)

        # 2. Log Scaling (Log1p)
        spec = torch.log1p(spec)

        # 3. Resize to fixed square size (256, 256)
        # Unsqueeze for interpolation: (N, C, H, W) -> (n_sensors, 1, 256, time)
        spec = spec.unsqueeze(1)
        spec = F.interpolate(
            spec, size=Config.IMG_SIZE, mode="bilinear", align_corners=False
        )
        spec = spec.squeeze(1)  # (n_sensors, 256, 256)

        # 4. Global Log-Max Scaling
        # Formula: X_norm = log(X + 1) / log(M_global + 1)
        # Note: 'spec' is already log(X+1).
        denom = np.log1p(Config.GLOBAL_MAX_CONST)
        spec = spec / denom

        return spec.numpy()

    def process_dataset(
        self, metadata_path: str, output_dir_name: str, load_cached_data: bool = True
    ) -> pd.DataFrame:
        """
        Main processing loop.
        1. Checks for cached features.
        2. If not found, iterates metadata, computes features/spectrograms.
        3. Saves results to disk.

        Args:
            metadata_path: Path to the metadata CSV (train/val/test).
            output_dir_name: Subdirectory name in working dir to store artifacts.
            load_cached_data: Whether to attempt loading from cache.

        Returns:
            pd.DataFrame: The tabular features dataframe (including scalars and targets).
        """
        # Setup paths
        cache_dir = os.path.join(Config.WORKING_DIR, output_dir_name)
        feat_path = os.path.join(cache_dir, "features.parquet")
        spec_dir = os.path.join(cache_dir, "spectrograms")

        os.makedirs(spec_dir, exist_ok=True)

        # Load Metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        df_meta = pd.read_csv(metadata_path)

        # Check Cache
        if load_cached_data and os.path.exists(feat_path):
            # Simple validation: check if spectrogram count matches metadata count
            cached_specs = len([f for f in os.listdir(spec_dir) if f.endswith(".npy")])
            if cached_specs >= len(df_meta):
                print(f"Loading cached features from {feat_path}")
                return pd.read_parquet(feat_path)
            else:
                print("Cache incomplete. Reprocessing...")

        print(f"Processing {len(df_meta)} segments from {metadata_path}...")

        tabular_records = []

        # Processing Loop
        for idx, row in df_meta.iterrows():
            seg_id = int(row["segment_id"])
            file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

            # Load Signal
            df_signal = self._load_signal(file_path)
            if df_signal is None:
                continue

            # 1. Extract Tabular Features
            feats = self.extract_tabular_features(df_signal)
            feats["segment_id"] = seg_id

            # Add target if available
            if "time_to_eruption" in row:
                feats["time_to_eruption"] = row["time_to_eruption"]

            # 2. Extract Scalars (for FiLM)
            scalars = self.extract_scalars(df_signal)
            feats.update(scalars)

            tabular_records.append(feats)

            # 3. Generate and Save Spectrogram
            spec = self.generate_spectrogram(df_signal)
            spec_save_path = os.path.join(spec_dir, f"{seg_id}.npy")
            np.save(spec_save_path, spec)

            if (idx + 1) % 500 == 0:
                print(f"Processed {idx + 1}/{len(df_meta)} segments.")

        # Save Tabular Features
        df_features = pd.DataFrame(tabular_records)
        df_features.to_parquet(feat_path, index=False)
        print(f"Saved features to {feat_path}")

        return df_features
