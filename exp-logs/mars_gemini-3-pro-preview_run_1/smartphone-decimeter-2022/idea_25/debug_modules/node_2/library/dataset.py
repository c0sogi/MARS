import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from library.config import Config


def get_feature_columns():
    """
    Generates the list of feature column names based on the Config stratification.
    Matches the naming convention in preprocessing.py.
    """
    feature_cols = []

    # Stratified features
    for stratum in Config.STRATA:
        for raw_field in Config.STRATUM_RAW_FIELDS:
            for stat in Config.STRATUM_STATS:
                col_name = f"{stratum}_{raw_field}_{stat}"
                feature_cols.append(col_name)

    # Global context features
    feature_cols.extend(Config.GLOBAL_FEATURES)

    return feature_cols


class GNSSScaler:
    """
    Wrapper for StandardScaler to support JSON/NPY persistence without pickle.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None

    def fit(self, X, feature_names):
        self.scaler.fit(X)
        self.feature_names = feature_names

    def transform(self, X):
        return self.scaler.transform(X)

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        params = {
            "mean": self.scaler.mean_.tolist(),
            "scale": self.scaler.scale_.tolist(),
            "var": self.scaler.var_.tolist(),
            "n_samples_seen": int(self.scaler.n_samples_seen_),
            "feature_names": self.feature_names,
        }
        with open(os.path.join(save_dir, "scaler_params.json"), "w") as f:
            json.dump(params, f)

    def load(self, load_dir):
        path = os.path.join(load_dir, "scaler_params.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler params not found at {path}")

        with open(path, "r") as f:
            params = json.load(f)

        self.scaler.mean_ = np.array(params["mean"])
        self.scaler.scale_ = np.array(params["scale"])
        self.scaler.var_ = np.array(params["var"])
        self.scaler.n_samples_seen_ = params["n_samples_seen"]
        self.feature_names = params["feature_names"]


class GNSSSequenceDataset(Dataset):
    def __init__(self, split="train", scaler=None, load_cached_data=True):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            scaler (GNSSScaler, optional): Pre-fitted scaler. If None and split='train', fits a new one.
            load_cached_data (bool): Whether to load data from cache.
        """
        self.split = split
        self.feature_cols = get_feature_columns()
        self.seq_len = Config.TRAIN_SEQUENCE_LENGTH

        # 1. Load Data
        self.df = self._load_data(split, load_cached_data)

        # 2. Handle Scaling
        if scaler is None:
            if split == "train":
                print("Fitting new scaler on training data...")
                self.scaler = GNSSScaler()
                # Fit on all data (flattened)
                X_all = self.df[self.feature_cols].values
                # Handle NaNs/Infs just in case, though preprocessing should have handled it
                X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
                self.scaler.fit(X_all, self.feature_cols)
                # Save scaler for inference
                self.scaler.save(Config.WORKING_DIR)
            else:
                # Try to load if not provided
                print("Loading scaler from disk...")
                self.scaler = GNSSScaler()
                self.scaler.load(Config.WORKING_DIR)
        else:
            self.scaler = scaler

        # 3. Pre-process sequences
        # Group by drive to ensure continuity
        self.sequences = []
        self._create_sequences()

    def _load_data(self, split, load_cached_data):
        cache_path = os.path.join(Config.CACHE_DIR, f"{split}_processed.parquet")

        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading {split} data from {cache_path}")
            df = pd.read_parquet(cache_path)
        else:
            # If cache is missing or force reload, we assume the preprocessing module
            # has been run externally or we run it here.
            # Per instructions, we import the class.
            from library.preprocessing import GNSSPreprocessor

            preprocessor = GNSSPreprocessor()
            df = preprocessor.process_data(
                split=split, load_cached_data=load_cached_data
            )

        # Sort to ensure temporal order
        df = df.sort_values(
            by=["drive_id", "phone_name", "UnixTimeMillis"]
        ).reset_index(drop=True)
        return df

    def _create_sequences(self):
        """
        Identifies valid windows for training/inference.
        """
        # Group by unique drive+phone
        groups = self.df.groupby(["drive_id", "phone_name"])

        stride = (
            self.seq_len // 2
            if self.split == "train"
            else (self.seq_len - Config.INFERENCE_OVERLAP)
        )

        for (drive_id, phone_name), group in groups:
            indices = group.index.values
            n_samples = len(indices)

            # Generate start indices
            # We ensure we cover the whole drive.
            # If drive is shorter than seq_len, we take one window (padded later).
            if n_samples <= self.seq_len:
                self.sequences.append(
                    {
                        "drive_id": drive_id,
                        "phone_name": phone_name,
                        "start_idx": indices[0],
                        "end_idx": indices[-1] + 1,  # Exclusive
                        "length": n_samples,
                        "pad_needed": self.seq_len - n_samples,
                    }
                )
            else:
                # Sliding window
                for start in range(0, n_samples, stride):
                    end = min(start + self.seq_len, n_samples)

                    # If we are at the end and the last chunk is very small,
                    # we might want to shift back to get a full window if possible,
                    # or just pad. Here we shift back if possible to minimize padding.
                    if (end - start) < self.seq_len and start > 0:
                        start = max(0, end - self.seq_len)

                    self.sequences.append(
                        {
                            "drive_id": drive_id,
                            "phone_name": phone_name,
                            "start_idx": indices[start],
                            "end_idx": indices[end - 1] + 1,
                            "length": end - start,
                            "pad_needed": self.seq_len - (end - start),
                        }
                    )

                    if end == n_samples:
                        break

        print(f"Generated {len(self.sequences)} sequences for split {self.split}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_info = self.sequences[idx]

        # Extract data window
        # iloc is slow, but indices are contiguous integers from reset_index
        # so slicing is faster if we rely on the underlying numpy array or slice
        # However, df is one big dataframe. The indices in seq_info are global df indices.
        # Slicing via loc/iloc with range is efficient.

        data_slice = self.df.iloc[seq_info["start_idx"] : seq_info["end_idx"]]

        # 1. Features
        X_raw = data_slice[self.feature_cols].values
        X_raw = np.nan_to_num(X_raw, nan=0.0)
        X_scaled = self.scaler.transform(X_raw)

        # Padding
        pad_len = seq_info["pad_needed"]
        if pad_len > 0:
            # Pad with zeros
            padding = np.zeros((pad_len, X_scaled.shape[1]), dtype=np.float32)
            X_scaled = np.vstack([X_scaled, padding])

        # Transpose to (Channels, Length) for Conv1d
        # Shape: (C, L)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).transpose(0, 1)

        # 2. Metadata (for reconstruction)
        timestamps = data_slice["UnixTimeMillis"].values
        if pad_len > 0:
            # Pad timestamps with -1 or last value? -1 indicates invalid
            timestamps = np.concatenate(
                [timestamps, np.full(pad_len, -1, dtype=np.int64)]
            )

        # WLS Baseline (needed for offset reconstruction)
        wls_lat = data_slice["WlsLatitudeDegrees"].values
        wls_lon = data_slice["WlsLongitudeDegrees"].values
        if pad_len > 0:
            wls_lat = np.concatenate([wls_lat, np.zeros(pad_len)])
            wls_lon = np.concatenate([wls_lon, np.zeros(pad_len)])

        metadata = {
            "drive_id": seq_info["drive_id"],
            "phone_name": seq_info["phone_name"],
            "UnixTimeMillis": torch.tensor(timestamps, dtype=torch.int64),
            "WlsLatitudeDegrees": torch.tensor(wls_lat, dtype=torch.float64),
            "WlsLongitudeDegrees": torch.tensor(wls_lon, dtype=torch.float64),
            "pad_mask": torch.tensor(
                [1] * seq_info["length"] + [0] * pad_len, dtype=torch.bool
            ),
        }

        # 3. Targets (if train/val)
        if self.split in ["train", "val"]:
            # Targets are delta east, delta north
            y_east = data_slice["target_east"].values
            y_north = data_slice["target_north"].values

            y = np.stack([y_east, y_north], axis=1)  # (L, 2)

            if pad_len > 0:
                y = np.vstack([y, np.zeros((pad_len, 2))])

            # Transpose to (2, L)
            y_tensor = torch.tensor(y, dtype=torch.float32).transpose(0, 1)

            # Decimated Deep Supervision Targets
            # Scales: 1 (full), 1/2, 1/4, 1/8
            targets = {}
            # Scale 0 (Full resolution)
            targets["scale_0"] = y_tensor

            # Decimate
            # We assume the model outputs at these resolutions.
            # Decimation simply takes every k-th sample.
            # NOTE: Ensure the length is divisible or handle truncation.
            # Since TRAIN_SEQUENCE_LENGTH is 256 (power of 2), simple slicing works.

            targets["scale_1"] = y_tensor[:, ::2]  # 1/2
            targets["scale_2"] = y_tensor[:, ::4]  # 1/4
            targets["scale_3"] = y_tensor[:, ::8]  # 1/8

            return X_tensor, targets, metadata

        else:
            # Inference mode
            return X_tensor, metadata
