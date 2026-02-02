import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import scipy.stats as stats
from sklearn.decomposition import PCA
from library.config import Config
from library.utils import save_parquet, load_parquet, save_npy, load_npy


class FeatureProcessor:
    """
    Handles feature engineering for the Seismic Eruption Prediction pipeline.
    Implements the 'Latent-Source Cepstral Stacking' strategy.
    Generates both Tabular features (including PCA latent source and MFCCs)
    and Spectrograms for the Vision branch.
    """

    def __init__(self):
        self.config = Config()
        self.working_dir = self.config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)

        # Initialize Torchaudio Transforms
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.config.SAMPLING_RATE,
            n_mfcc=self.config.N_MFCC,
            melkwargs={
                "n_fft": self.config.N_FFT,
                "hop_length": self.config.HOP_LENGTH,
                "n_mels": self.config.N_MELS,
                "f_min": float(self.config.FMIN),
                "f_max": float(self.config.FMAX),
                "center": True,
            },
        )

        self.melspec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.SAMPLING_RATE,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH,
            n_mels=self.config.N_MELS,
            f_min=float(self.config.FMIN),
            f_max=float(self.config.FMAX),
            center=True,
        )

    def _compute_pca_source(self, sensor_data):
        """
        Extracts the First Principal Component (PC1) from the 10 sensors
        to represent the 'Latent Source' or common mode signal.

        Args:
            sensor_data (np.ndarray): Shape (n_samples, 10)

        Returns:
            np.ndarray: Shape (n_samples,) representing the PC1 signal.
        """
        # Fit PCA on the segment
        pca = PCA(
            n_components=self.config.N_PCA_COMPONENTS, random_state=self.config.SEED
        )
        # Normalize roughly before PCA to ensure scale doesn't dominate (though data is int16 normalized already)
        # We use the raw data directly as per strategy description for "Latent Source"
        pc1 = pca.fit_transform(sensor_data)
        return pc1.flatten()

    def _extract_tabular_features(self, df_segment):
        """
        Computes a comprehensive set of tabular features for a single segment.
        Includes Time-domain, Frequency-domain (MFCC), and Spatial features.

        Args:
            df_segment (pd.DataFrame): Raw sensor data (60001, 10).

        Returns:
            dict: Dictionary of extracted features.
        """
        features = {}

        # 1. Prepare Signals: 10 Sensors + 1 Latent Source
        sensor_cols = [c for c in df_segment.columns if "sensor" in c]
        sensor_data = df_segment[sensor_cols].values

        # Handle NaNs if any (fill with 0 as per robust strategy)
        if np.isnan(sensor_data).any():
            sensor_data = np.nan_to_num(sensor_data, nan=0.0)

        # Compute Latent Source (PC1)
        latent_source = self._compute_pca_source(sensor_data)

        # Dictionary of signals to process
        signals = {col: sensor_data[:, i] for i, col in enumerate(sensor_cols)}
        signals["latent_source"] = latent_source

        # 2. Iterate over all signals (Sensors + Latent)
        for sig_name, sig in signals.items():
            # --- Time Domain Statistics ---
            features[f"{sig_name}_mean"] = np.mean(sig)
            features[f"{sig_name}_std"] = np.std(sig)
            features[f"{sig_name}_min"] = np.min(sig)
            features[f"{sig_name}_max"] = np.max(sig)
            features[f"{sig_name}_skew"] = stats.skew(sig)
            features[f"{sig_name}_kurt"] = stats.kurtosis(sig)

            # Quantiles
            q_vals = np.quantile(sig, [0.01, 0.05, 0.50, 0.95, 0.99])
            features[f"{sig_name}_q01"] = q_vals[0]
            features[f"{sig_name}_q05"] = q_vals[1]
            features[f"{sig_name}_q50"] = q_vals[2]
            features[f"{sig_name}_q95"] = q_vals[3]
            features[f"{sig_name}_q99"] = q_vals[4]

            # --- Energy Domain (Absolute Stats) ---
            abs_sig = np.abs(sig)
            features[f"{sig_name}_abs_mean"] = np.mean(abs_sig)
            features[f"{sig_name}_abs_max"] = np.max(abs_sig)
            features[f"{sig_name}_abs_q05"] = np.quantile(abs_sig, 0.05)
            features[f"{sig_name}_abs_q50"] = np.quantile(abs_sig, 0.50)
            features[f"{sig_name}_abs_q95"] = np.quantile(abs_sig, 0.95)

            # --- Structural (Zero Crossing Rate) ---
            # Raw ZCR without centering
            zcr = ((sig[:-1] * sig[1:]) < 0).sum()
            features[f"{sig_name}_zcr"] = zcr

            # --- Cepstral Features (MFCCs) ---
            # Extract MFCCs using Torchaudio
            sig_tensor = torch.tensor(sig.astype(np.float32), dtype=torch.float32)
            mfcc_tensor = self.mfcc_transform(sig_tensor)
            mfcc = mfcc_tensor.numpy()

            # Aggregate MFCCs using Robust Statistics only (Mean, Std, Q05, Q95)
            # Exclude Min/Max to avoid noise artifacts
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            mfcc_q05 = np.quantile(mfcc, 0.05, axis=1)
            mfcc_q95 = np.quantile(mfcc, 0.95, axis=1)

            for i in range(self.config.N_MFCC):
                features[f"{sig_name}_mfcc_{i}_mean"] = mfcc_mean[i]
                features[f"{sig_name}_mfcc_{i}_std"] = mfcc_std[i]
                features[f"{sig_name}_mfcc_{i}_q05"] = mfcc_q05[i]
                features[f"{sig_name}_mfcc_{i}_q95"] = mfcc_q95[i]

            # --- Spectral Features (FFT) ---
            # Compute FFT
            fft_val = np.fft.rfft(sig)
            fft_mag = np.abs(fft_val)
            # Frequencies: 0 to Nyquist
            freqs = np.fft.rfftfreq(len(sig), d=1 / self.config.SAMPLING_RATE)

            # Normalize magnitude for weighted avg
            mag_sum = np.sum(fft_mag) + 1e-12

            # Centroid
            centroid = np.sum(freqs * fft_mag) / mag_sum
            features[f"{sig_name}_spec_centroid"] = centroid

            # Bandwidth
            bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * fft_mag) / mag_sum)
            features[f"{sig_name}_spec_bandwidth"] = bandwidth

            # Dominant Frequency
            dom_freq = freqs[np.argmax(fft_mag)]
            features[f"{sig_name}_dom_freq"] = dom_freq

        # 3. Spatial Features (Correlation Matrix)
        # Compute correlation between physical sensors only
        df_sensors = pd.DataFrame(sensor_data, columns=sensor_cols)
        corr_matrix = df_sensors.corr().abs()

        # Extract upper triangle (excluding diagonal)
        # 10 sensors -> 45 pairs
        for i in range(len(sensor_cols)):
            for j in range(i + 1, len(sensor_cols)):
                col_i = sensor_cols[i]
                col_j = sensor_cols[j]
                val = corr_matrix.loc[col_i, col_j]
                features[f"corr_{col_i}_{col_j}"] = val

        return features

    def _generate_spectrogram(self, df_segment):
        """
        Generates a stacked Log-Mel Spectrogram for the Vision Branch.

        Args:
            df_segment (pd.DataFrame): Raw sensor data.

        Returns:
            np.ndarray: Shape (10, n_mels, time_steps)
        """
        sensor_cols = [c for c in df_segment.columns if "sensor" in c]
        specs = []

        for col in sensor_cols:
            sig = df_segment[col].values.astype(np.float32)
            if np.isnan(sig).any():
                sig = np.nan_to_num(sig, nan=0.0)

            # Compute Mel Spectrogram using Torchaudio
            sig_tensor = torch.tensor(sig, dtype=torch.float32)
            melspec_tensor = self.melspec_transform(sig_tensor)

            # Convert to Log Scale (dB) manually
            # 10 * log10(S / max(S))
            melspec_tensor = torch.clamp(melspec_tensor, min=1e-10)
            log_melspec = 10.0 * torch.log10(melspec_tensor)
            log_melspec = log_melspec - torch.max(log_melspec)

            log_melspec = log_melspec.numpy()

            # Normalize to roughly [0, 1] or standard range could be done here,
            # but usually Batch Norm in CNN handles it.
            # We keep it as dB values for now.
            # Standardize per image to help convergence
            mean = log_melspec.mean()
            std = log_melspec.std() + 1e-6
            log_melspec = (log_melspec - mean) / std

            specs.append(log_melspec)

        # Stack to (Channels, Freq, Time)
        # Shape: (10, 128, T)
        return np.stack(specs, axis=0)

    def process_data(self, metadata_path, dataset_name, load_cached_data=True):
        """
        Main driver to process a dataset (train/val/test).
        Handles caching, loading raw data, feature extraction, and saving.

        Args:
            metadata_path (str): Path to the metadata CSV.
            dataset_name (str): Name tag for the dataset (e.g., 'train', 'val', 'test').
            load_cached_data (bool): Whether to attempt loading from cache.

        Returns:
            tuple: (tabular_features_df, spectrograms_array, targets_array)
        """
        # Define Cache Paths
        cache_feat_path = os.path.join(
            self.working_dir, f"{dataset_name}_features.parquet"
        )
        cache_spec_path = os.path.join(
            self.working_dir, f"{dataset_name}_spectrograms.npy"
        )
        cache_target_path = os.path.join(
            self.working_dir, f"{dataset_name}_targets.npy"
        )

        # 1. Try to Load from Cache
        if load_cached_data:
            if (
                os.path.exists(cache_feat_path)
                and os.path.exists(cache_spec_path)
                and os.path.exists(cache_target_path)
            ):

                print(f"[{dataset_name}] Loading features from cache...")
                df_features = load_parquet(cache_feat_path)
                spectrograms = load_npy(cache_spec_path)
                targets = load_npy(cache_target_path)
                return df_features, spectrograms, targets
            else:
                print(
                    f"[{dataset_name}] Cache not found or incomplete. Processing from scratch..."
                )
        else:
            print(f"[{dataset_name}] Force processing from scratch...")

        # 2. Load Metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        df_meta = pd.read_csv(metadata_path)

        # Containers
        all_features = []
        all_spectrograms = []
        all_targets = []

        # 3. Iterate and Process
        # Note: No progress bar as requested
        print(f"[{dataset_name}] Processing {len(df_meta)} segments...")

        for _, row in df_meta.iterrows():
            segment_id = int(row["segment_id"])
            target = row["time_to_eruption"]
            file_rel_path = row["file_path"]

            full_path = os.path.join(self.config.INPUT_DIR, file_rel_path)

            if not os.path.exists(full_path):
                # Should not happen based on metadata check, but for safety
                continue

            # Load Raw Data
            # Use float32 to handle NaNs and memory
            try:
                df_segment = pd.read_csv(full_path, dtype="float32")
            except Exception as e:
                print(f"Error reading {full_path}: {e}")
                continue

            # Fill NaNs
            df_segment = df_segment.fillna(0)

            # A. Extract Tabular Features
            feats = self._extract_tabular_features(df_segment)
            feats["segment_id"] = segment_id  # Track ID
            all_features.append(feats)

            # B. Extract Spectrograms
            spec = self._generate_spectrogram(df_segment)
            all_spectrograms.append(spec)

            # C. Store Target
            all_targets.append(target)

        # 4. Aggregate and Save
        df_features_final = pd.DataFrame(all_features)

        # Ensure segment_id is first column
        cols = ["segment_id"] + [
            c for c in df_features_final.columns if c != "segment_id"
        ]
        df_features_final = df_features_final[cols]

        spectrograms_final = np.stack(all_spectrograms, axis=0)  # (N, 10, 128, T)
        targets_final = np.array(all_targets, dtype=np.float32)

        print(f"[{dataset_name}] Saving to cache...")
        save_parquet(df_features_final, cache_feat_path)
        save_npy(spectrograms_final, cache_spec_path)
        save_npy(targets_final, cache_target_path)

        print(
            f"[{dataset_name}] Processing complete. "
            f"Tabular shape: {df_features_final.shape}, "
            f"Spec shape: {spectrograms_final.shape}"
        )

        return df_features_final, spectrograms_final, targets_final
